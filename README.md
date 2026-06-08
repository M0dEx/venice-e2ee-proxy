# Venice E2EE OpenAI Proxy

Local OpenAI-compatible proxy for Venice.AI E2EE models.


## Stack

- Language/runtime: Rust, using the Cargo package manager.
- HTTP/runtime: async Rust with `tokio` and `axum` for the OpenAI-compatible HTTP server.
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
| Integration tests | `cargo test --test baseline` |
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


## Tests

- Unit tests in `src/config` cover defaults, validation, and safe Venice API key lookup.
- Unit tests in `src/http` cover route registration, unknown routes/methods, Axum JSON extractor rejections for malformed/non-object JSON, and safe header helpers.
- Unit tests in `src/main.rs` cover the optional config path CLI shape.
- `src/lib.rs` still verifies the module boundary list.
- `tests/baseline.rs` verifies the Cargo integration test harness is wired.

