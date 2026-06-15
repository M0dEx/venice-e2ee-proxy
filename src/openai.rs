//! OpenAI-compatible request and response formatting.
//!
//! Includes the typed model-list response used by `GET /v1/models` and the
//! shared OpenAI-style error envelope used by fail-closed validation responses.

use serde::{Deserialize, Serialize};

pub mod chat;

/// OpenAI-compatible model-list response envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

impl ModelListResponse {
    /// Builds a model-list response from already-normalized model objects.
    pub fn new(data: Vec<ModelObject>) -> Self {
        Self {
            object: "list".to_owned(),
            data,
        }
    }
}

/// OpenAI-compatible model object with Venice metadata preserved for clients
/// that need it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    pub name: String,
    pub info: ModelInfo,
    pub venice: VeniceModelMetadata,
}

impl ModelObject {
    /// Builds an OpenAI-compatible model object with mirrored `id` and `name` fields.
    pub fn new(
        id: impl Into<String>,
        created: i64,
        owned_by: impl Into<String>,
        capabilities: ModelCapabilities,
        venice: VeniceModelMetadata,
    ) -> Self {
        let id = id.into();

        Self {
            name: id.clone(),
            id,
            object: "model".to_owned(),
            created,
            owned_by: owned_by.into(),
            info: ModelInfo {
                meta: ModelMeta { capabilities },
            },
            venice,
        }
    }
}

/// OpenAI-compatible model information wrapper containing model metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub meta: ModelMeta,
}

/// OpenAI-compatible model metadata containing capability flags.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMeta {
    pub capabilities: ModelCapabilities,
}

/// Capability flags exposed in the OpenAI-compatible model metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub function_calling: bool,
    pub builtin_tools: bool,
    pub web_search: bool,
    pub code_interpreter: bool,
    pub vision: bool,
    pub reasoning: bool,
    pub reasoning_effort: bool,
}

/// Venice-specific model metadata preserved alongside the OpenAI-compatible model object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VeniceModelMetadata {
    pub id: String,
    #[serde(rename = "supportsE2EE")]
    pub supports_e2ee: bool,
    #[serde(rename = "supportsTeeAttestation")]
    pub supports_tee_attestation: bool,
    #[serde(rename = "supportsReasoning")]
    pub supports_reasoning: bool,
    #[serde(rename = "supportsReasoningEffort")]
    pub supports_reasoning_effort: bool,
}

impl VeniceModelMetadata {
    /// Builds Venice metadata from the upstream model id and supported feature flags.
    pub fn new(
        id: impl Into<String>,
        supports_e2ee: bool,
        supports_tee_attestation: bool,
        supports_reasoning: bool,
        supports_reasoning_effort: bool,
    ) -> Self {
        Self {
            id: id.into(),
            supports_e2ee,
            supports_tee_attestation,
            supports_reasoning,
            supports_reasoning_effort,
        }
    }
}

/// OpenAI-compatible error response envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorObject,
}

impl ErrorResponse {
    /// Builds an OpenAI-compatible error response from message, type, and code strings.
    pub fn new(
        message: impl Into<String>,
        error_type: impl Into<String>,
        code: impl Into<String>,
    ) -> Self {
        Self {
            error: ErrorObject {
                message: message.into(),
                kind: error_type.into(),
                code: code.into(),
            },
        }
    }
}

/// OpenAI-compatible error object containing the client-facing message, type, and code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorObject {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub code: String,
}
