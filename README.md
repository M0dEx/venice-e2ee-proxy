# Venice E2EE OpenAI Proxy

A local OpenAI-compatible proxy for Venice.ai E2EE models.

It lets OpenAI-style clients call Venice E2EE chat models without learning Venice's TEE/E2EE request format. The proxy accepts normal `/v1/chat/completions` requests, verifies the model attestation envelope, encrypts the prompt for Venice, sends the request upstream, decrypts the response, and returns OpenAI-shaped JSON or SSE.

The Venice API key lives on the proxy. Clients talk to the proxy as if it were an OpenAI-compatible base URL.

## Why this exists

Venice E2EE is useful, but it is not a drop-in OpenAI endpoint:

- requests need Venice TEE headers and encrypted message content
- responses arrive as encrypted SSE chunks
- model attestation has to be fetched and checked before key use
- E2EE models do not expose native server-side OpenAI tool calls, because the tool definitions are encrypted

This proxy handles that glue locally.

The main extra feature is tool-call emulation. When a client sends OpenAI `tools`, the proxy adds a model-specific controller prompt, decrypts the model output, parses tool calls with `vllm-tool-parser`, validates the function name and JSON arguments against the requested tools, and returns OpenAI-style `tool_calls`.

Prompt/parser formats are selected by model id:

- GLM models: GLM XML format
- Qwen models: Qwen XML-wrapped JSON format
- everything else: Hermes-style JSON format

This is not the same as native upstream function calling, but it makes common OpenAI tool clients usable with Venice E2EE models.

## What is supported

Endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`

`/v1/models` proxies Venice's model list and only returns text models that advertise both E2EE and TEE attestation support.

`/v1/chat/completions` supports:

- streaming and non-streaming OpenAI chat responses
- text-only `system`, `developer`, `user`, `assistant`, and `tool` messages
- string content and text-only content parts
- `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, and `stop`
- `stream_options.include_usage`
- Venice reasoning fields: `reasoning` and `reasoning_effort`
- OpenAI function tools via local emulation
- session reuse through the configured session-id header (`X-Venice-Proxy-Session-Id` by default) or `metadata.session_id`

## Build

Requirements:

- recent stable Rust with edition 2024 support
- a C toolchain for the Rust dependencies used by the release build
- network access when Cargo fetches the git dependency `vllm-tool-parser`

Fetch and build:

```bash
cargo fetch
cargo build --release --locked
```

Install from this checkout:

```bash
cargo install --path . --locked
```

The binary requires one positional argument: a TOML config path.

## Configure

Start from `config/default.toml`. It contains all current config fields.

Do not put a real Venice API key in a committed config file. Prefer the environment override:

```bash
VENICE_E2EE_PROXY__VENICE__API_KEY=... cargo run -- config/default.toml
```

Useful config sections:

- `[server]`: bind host and port. Defaults in `config/default.toml` are `0.0.0.0:8080`.
- `[venice]`: Venice base URL, API key, and request timeout.
- `[session]`: in-memory attestation/model-key reuse policy and session-id header.
- `[attestation]`: local attestation policy gates.
- `[e2ee]`: E2EE codec settings.
- `[tools]`: tool emulation mode, retry count, marker timeout, max parsed output size, and schema validation.

Any nested config value can be overridden with `VENICE_E2EE_PROXY__...` environment variables. Examples:

```bash
VENICE_E2EE_PROXY__SERVER__PORT=9000
VENICE_E2EE_PROXY__LOGGING__LEVEL=venice_e2ee_proxy=debug,tower_http=warn
VENICE_E2EE_PROXY__TOOLS__ENABLED=false
```

Durations use strings such as `30s`, `10m`, or `1h`.

## Run locally

```bash
VENICE_E2EE_PROXY__VENICE__API_KEY=... cargo run -- config/default.toml
```

Or with the release binary:

```bash
VENICE_E2EE_PROXY__VENICE__API_KEY=... ./target/release/venice-e2ee-proxy config/default.toml
```

List supported E2EE models:

```bash
curl http://localhost:8080/v1/models
```

Send a chat request:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Venice-Proxy-Session-Id: local-dev' \
  -d '{
    "model": "<model-from-/v1/models>",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "stream": false
  }'
```

For OpenAI SDKs, set the base URL to:

```text
http://localhost:8080/v1
```

The client `Authorization` header is not used by the proxy. The upstream Venice API key comes from the proxy config or environment.

## Docker

Build the image:

```bash
docker build -t venice-e2ee-proxy:local .
```

Run with the bundled default config:

```bash
docker run --rm -p 8080:8080 \
  -e VENICE_E2EE_PROXY__VENICE__API_KEY=... \
  venice-e2ee-proxy:local
```

Run with your own config:

```bash
docker run --rm -p 8080:8080 \
  -e VENICE_E2EE_PROXY__VENICE__API_KEY=... \
  -v /absolute/path/to/config.toml:/etc/venice-e2ee-proxy/config.toml:ro \
  venice-e2ee-proxy:local
```

The image entrypoint runs:

```text
venice-e2ee-proxy /etc/venice-e2ee-proxy/config.toml
```

## Deploy

This service is just an HTTP proxy. Put it somewhere your OpenAI-compatible client can reach, set the Venice API key as an environment variable, and point the client base URL at `/v1` on the proxy.

Keep these deployment details in mind:

- The proxy does not implement client authentication, TLS termination, rate limits, or tenant isolation. Do not expose it directly to the public internet unless something in front of it handles that.
- Sessions and attestation state are in memory. They do not survive restarts and are not shared across replicas.
- The proxy instance key is generated at startup by default. Leave `keys.generate_proxy_instance_key_on_startup = true`; chat requests fail without an instance key.
- If you run more than one replica, use sticky sessions or expect each replica to fetch and cache attestation independently.

## Caveats

- This is not the full OpenAI API. Unknown chat fields are rejected, and only the endpoints listed above exist.
- Message content is text-only. Vision, audio, image inputs, and other multimodal content are not supported.
- Venice web search and Venice system prompt injection are intentionally rejected for E2EE requests.
- `metadata` is accepted for session ids, but it is not forwarded upstream.
- Tool calls are emulated with prompts and parsers. They depend on the model following the requested format. Non-streaming tool requests can retry with correction prompts; streaming tool-call parsing cannot retry after bad output and will fail the stream.
- Tool schema validation supports the subset used by this proxy: object/array/string/integer/number/boolean/null types, `properties`, `required`, `items`, `additionalProperties`, and `enum`.
- Attestation support is intentionally conservative. The verifier checks the Venice attestation envelope, nonce, model key binding, signing-address shape, debug policy, and local TDX/NVIDIA policy gates. Full Intel DCAP/QVL and NVIDIA NRAS verifier backends are not linked. If you configure those as required, requests fail closed.
- The checked-in `config/default.toml` relaxes attestation with `require_tdx = false` and `require_nvidia = "never"` so the proxy can run with the current verifier limitations. Tighten this only when the verifier support you need is present.
