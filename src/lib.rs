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
pub(crate) mod util;
pub mod venice;
