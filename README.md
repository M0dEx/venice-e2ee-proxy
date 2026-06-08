# Venice E2EE OpenAI Proxy

Local OpenAI-compatible proxy for Venice.AI E2EE models.


## Stack

- Language/runtime: Rust, using the Cargo package manager.
- HTTP/runtime: async Rust with `tokio` and `axum` for the OpenAI-compatible HTTP server.
- Upstream/client direction: typed JSON with `serde`, `reqwest` for Venice HTTP calls, and `toml`/environment configuration.
- Crate layout: one binary entrypoint in `src/main.rs` plus a library surface in `src/lib.rs` for implementation modules.

## Commands

Use direct Cargo commands only.

| Purpose | Command |
| --- | --- |
| Install/fetch dependencies | `cargo fetch` |
| Local development entrypoint with defaults | `cargo run` |
| Local development entrypoint with config | `cargo run -- path/to/config.toml` |
| Format code | `cargo fmt` |
| Check formatting | `cargo fmt --check` |
| Lint | `cargo clippy --all-targets --all-features -- -D warnings` |
| Typecheck | `cargo check --all-targets --all-features` |
| Unit tests | `cargo test --lib` |
| Baseline integration test | `cargo test --test baseline` |
| Mocked models integration tests | `cargo test --test models` |
| All tests | `cargo test --all-targets --all-features` |
| Baseline validation | Run `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --all-targets --all-features` |


## Module boundaries


- `src/config`: configuration loading and validation.
- `src/http`: HTTP server, route wiring, shared headers, and route errors.
- `src/venice`: Venice upstream API client and model mapping.
- `src/keys`: startup proxy-instance key management.
- `src/sessions`: per-agent-session lifecycle and attestation/model-key state.
- `src/e2ee`: Venice E2EE encryption/decryption codec.
- `src/attestation`: attestation fetch, verification policy, and fail-closed checks.
- `src/openai`: OpenAI-compatible request/response formatting.
- `src/tools`: OpenAI-style tool-call emulation.


## Attestation v0.1 notes

- `src/attestation` generates a fresh 32-byte nonce and fetches `/tee/attestation` per verification call; it does not maintain an internal cache. Successful results are intended to be cached only by session state according to the session TTL/request limits.
- Basic envelope checks, secp256k1 signing-key normalization, Ethereum-style signing-address checks, debug-policy gates, and TDX/NVIDIA policy surfaces are implemented fail-closed.
- Full Intel DCAP/QVL and NVIDIA NRAS cryptographic verifier backends are not linked in v0.1. When `attestation.require_tdx = true` or NVIDIA evidence is required/present under verification policy, the verifier returns a structured `attestation_verifier_unavailable` error rather than allowing encrypted chat.

## Tests

- Unit tests in `src/config` cover defaults, validation, and safe Venice API key lookup.
- Unit tests in `src/venice` cover Venice-to-OpenAI model mapping, missing optional metadata defaults, malformed upstream model payloads, and API-key redaction in debug output.
- Unit tests in `src/attestation` cover valid evidence, missing fields, debug evidence, required TDX/NVIDIA failures, malformed upstream evidence, and upstream fetch failures.
- Unit tests in `src/http` cover route registration, unknown routes/methods, Axum JSON extractor rejections for malformed/non-object JSON, and safe header helpers.
- Unit tests in `src/main.rs` cover the optional config path CLI shape.
- `src/lib.rs` still verifies the module boundary list.
- `tests/baseline.rs` verifies the Cargo integration test harness is wired.
- `tests/models.rs` verifies mocked Venice success, authentication failures, server errors, malformed payloads, and upstream timeout handling for `GET /v1/models`.

