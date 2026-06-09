# Venice E2EE OpenAI Proxy

Local OpenAI-compatible proxy for Venice.AI E2EE models.

The repository implements a local HTTP proxy shell, typed configuration loading, route registration, shared OpenAI-style errors, safe response-header helpers, the Venice HTTP client, Venice model mapping, `GET /v1/models`, proxy key/session lifecycle, the Venice E2EE codec, attestation fetch/policy checks, request-side chat normalization/E2EE request construction, streaming and buffered non-streaming encrypted chat, marker-based tool-call emulation, and mocked proxy integration tests.

## Stack

- Language/runtime: Rust, using the Cargo package manager.
- HTTP/runtime: async Rust with `tokio` and `axum` for the OpenAI-compatible HTTP server.
- Upstream/client direction: typed JSON with `serde`, `reqwest` for Venice HTTP calls, and `toml`/environment configuration.
- Crate layout: one binary entrypoint in `src/main.rs` plus a library surface in `src/lib.rs` for implementation modules.
- Dependency policy: keep dependencies minimal and add new crates only when they support implemented behavior.

## Commands

Use direct Cargo commands only.

| Purpose | Command |
| --- | --- |
| Install/fetch dependencies | `cargo fetch` |
| Local development entrypoint | `VENICE_E2EE_PROXY__VENICE__API_KEY=... cargo run -- config/default.toml` |
| Local development entrypoint with custom config | `VENICE_E2EE_PROXY__VENICE__API_KEY=... cargo run -- path/to/config.toml` |
| Build container image | `docker build -t venice-e2ee-proxy:local .` |
| Run container image | `docker run --rm -p 8080:8080 -e VENICE_E2EE_PROXY__VENICE__API_KEY=... venice-e2ee-proxy:local` |
| Format code | `cargo fmt` |
| Check formatting | `cargo fmt --check` |
| Lint | `cargo clippy --all-targets --all-features -- -D warnings` |
| Typecheck | `cargo check --all-targets --all-features` |
| Unit tests | `cargo test --lib` |
| Mocked models integration tests | `cargo test --test models` |
| Mocked proxy integration tests | `cargo test --test proxy_integration` |
| All tests | `cargo test --all-targets --all-features` |
| Baseline validation | Run `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --all-targets --all-features` |

The proxy requires a TOML config path. Configure the Venice API key either as `venice.api_key` in TOML or with the Figment environment override `VENICE_E2EE_PROXY__VENICE__API_KEY`. Configure stdout tracing with `logging.level`; it accepts simple levels like `info`/`debug` or full tracing filter directives like `venice_e2ee_proxy=debug,tower_http=warn`, and can be overridden with `VENICE_E2EE_PROXY__LOGGING__LEVEL`. The checked-in `config/default.toml` exposes all default values and is copied into the image at `/etc/venice-e2ee-proxy/config.toml`; mount over that path to provide a container config file. Use direct Cargo commands only; this project intentionally does not use a Makefile.


## Module boundaries

The module boundaries are:

- `src/config`: configuration loading and validation.
- `src/http`: HTTP server, route wiring, shared headers, and route errors.
- `src/venice`: Venice upstream API client and model mapping.
- `src/keys`: startup proxy-instance key management.
- `src/sessions`: per-agent-session lifecycle and attestation/model-key state.
- `src/e2ee`: Venice E2EE encryption/decryption codec.
- `src/attestation`: attestation fetch, verification policy, and fail-closed checks.
- `src/openai`: OpenAI-compatible request/response formatting.
- `src/tools`: OpenAI-style tool-call emulation.

Implementation should continue using these modules rather than creating parallel subsystems.

## Attestation v0.1 notes

- `src/attestation` generates a fresh 32-byte nonce and fetches `/tee/attestation` per verification call; it does not maintain an internal cache. Successful results are intended to be cached only by session state according to the session TTL/request limits.
- Basic envelope checks, secp256k1 signing-key normalization, Ethereum-style signing-address checks, debug-policy gates, and TDX/NVIDIA policy surfaces are implemented fail-closed.
- Full Intel DCAP/QVL and NVIDIA NRAS cryptographic verifier backends are not linked in v0.1. When `attestation.require_tdx = true` or NVIDIA evidence is required/present under verification policy, the verifier returns a structured `attestation_verifier_unavailable` error rather than allowing encrypted chat.
- Measurement allowlists are intentionally not implemented for v0.1.

## Tests

- Unit tests in `src/config` cover defaults, validation, and safe Venice API key lookup.
- Unit tests in `src/venice` cover Venice-to-OpenAI model mapping, missing optional metadata defaults, malformed upstream model payloads, and API-key redaction in debug output.
- Unit tests in `src/attestation` cover valid evidence, missing fields, debug evidence, required TDX/NVIDIA failures, malformed upstream evidence, and upstream fetch failures.
- Unit tests in `src/openai/chat` cover chat message normalization, unsupported request shapes, Venice parameter policy, and encrypted Venice request construction.
- Unit tests in `src/http` cover route registration, streaming and buffered encrypted chat success/fail-closed paths, tool-call response/retry handling, chat request construction gating, unknown routes/methods, Axum JSON extractor rejections for malformed/non-object JSON, and safe header helpers.
- Unit tests in `src/main.rs` cover the optional config path CLI shape.
- `src/lib.rs` still verifies the module boundary list.
- `tests/models.rs` verifies mocked Venice success, authentication failures, server errors, malformed payloads, and upstream timeout handling for `GET /v1/models`.
- `tests/proxy_integration.rs` provides a mocked Venice harness covering model listing, streaming and non-streaming chat, attestation success/failure, encrypted response success/failure, tool-call emulation with correction retry headers, fail-closed startup/upstream/protocol paths, and proxy metadata response-header expectations.

