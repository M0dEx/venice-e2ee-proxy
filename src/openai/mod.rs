//! OpenAI-compatible request and response formatting.
//!
//! placeholders and fail-closed validation responses.

use serde::{Deserialize, Serialize};

/// OpenAI-compatible error response envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorObject,
}

impl ErrorResponse {
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

/// OpenAI-compatible error object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorObject {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub code: String,
}
