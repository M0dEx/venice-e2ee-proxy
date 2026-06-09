//! Library surface for the Venice E2EE OpenAI-compatible proxy.
//!
//! Exposes modules for the local HTTP proxy, Venice upstream client, E2EE,
//! attestation, sessions, OpenAI-compatible formatting, and tool-call emulation.

pub mod attestation;
pub mod config;
pub mod e2ee;
pub mod http;
pub mod keys;
pub mod openai;
pub mod sessions;
pub mod tools;
pub mod venice;

/// Implementation module boundaries for the proxy.
pub const MODULE_BOUNDARIES: &[&str] = &[
    "config",
    "http",
    "venice",
    "keys",
    "sessions",
    "e2ee",
    "attestation",
    "openai",
    "tools",
];

#[cfg(test)]
mod tests {
    use super::MODULE_BOUNDARIES;

    #[test]
    fn documents_expected_initial_module_boundaries() {
        assert_eq!(
            MODULE_BOUNDARIES,
            [
                "config",
                "http",
                "venice",
                "keys",
                "sessions",
                "e2ee",
                "attestation",
                "openai",
                "tools",
            ]
        );
    }
}
