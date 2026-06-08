//! Venice upstream API client and model mapping.
//!
//! filtering, and mapping into the OpenAI-compatible `/v1/models` response.

use std::{fmt, sync::Arc, time::Duration};

use reqwest::{
    Url,
    header::{ACCEPT, CONTENT_TYPE},
};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::{
    config::{ConfigError, ProxyConfig},
    openai::{
        ModelCapabilities, ModelListResponse, ModelObject, VeniceModelMetadata,
        chat::VeniceE2eeChatRequest,
    },
};

pub const HEADER_VENICE_TEE_CLIENT_PUB_KEY: &str = "X-Venice-TEE-Client-Pub-Key";
pub const HEADER_VENICE_TEE_MODEL_PUB_KEY: &str = "X-Venice-TEE-Model-Pub-Key";
pub const HEADER_VENICE_TEE_SIGNING_ALGO: &str = "X-Venice-TEE-Signing-Algo";

pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone)]
pub struct VeniceClient {
    http: reqwest::Client,
    base_url: Url,
    api_key: Arc<str>,
}

impl VeniceClient {
    pub fn from_config(config: &ProxyConfig) -> Result<Self, VeniceClientError> {
        let api_key = config.venice_api_key_from_env()?;
        Self::new(
            &config.venice.base_url,
            api_key.expose_secret(),
            DEFAULT_REQUEST_TIMEOUT,
        )
    }

    pub fn new(
        base_url: impl AsRef<str>,
        api_key: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self, VeniceClientError> {
        let base_url = parse_base_url(base_url.as_ref())?;
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(VeniceClientError::client_build)?;

        Ok(Self {
            http,
            base_url,
            api_key: Arc::from(api_key.into()),
        })
    }

    pub async fn list_models(&self) -> Result<ModelListResponse, VeniceClientError> {
        let url = self.models_url()?;
        let response = self
            .http
            .get(url)
            .bearer_auth(self.api_key.as_ref())
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(VeniceClientError::request_failure)?;

        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(VeniceClientError::Authentication {
                status: status.as_u16(),
            });
        }
        if !status.is_success() {
            return Err(VeniceClientError::UpstreamStatus {
                status: status.as_u16(),
            });
        }

        let body = response
            .bytes()
            .await
            .map_err(VeniceClientError::request_failure)?;
        parse_model_list_response(&body)
    }

    pub async fn create_chat_completion_stream(
        &self,
        request: &VeniceE2eeChatRequest,
        client_public_key_hex: &str,
        model_public_key_hex: &str,
    ) -> Result<reqwest::Response, VeniceClientError> {
        let url = self.chat_completions_url()?;
        let response = self
            .http
            .post(url)
            .bearer_auth(self.api_key.as_ref())
            .header(ACCEPT, "text/event-stream")
            .header(CONTENT_TYPE, "application/json")
            .header(HEADER_VENICE_TEE_CLIENT_PUB_KEY, client_public_key_hex)
            .header(HEADER_VENICE_TEE_MODEL_PUB_KEY, model_public_key_hex)
            .header(HEADER_VENICE_TEE_SIGNING_ALGO, "ecdsa")
            .json(request)
            .send()
            .await
            .map_err(VeniceClientError::request_failure)?;

        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(VeniceClientError::Authentication {
                status: status.as_u16(),
            });
        }
        if !status.is_success() {
            return Err(VeniceClientError::UpstreamStatus {
                status: status.as_u16(),
            });
        }

        Ok(response)
    }

    pub async fn fetch_attestation_evidence(
        &self,
        model_id: &str,
        nonce: &str,
    ) -> Result<Value, VeniceClientError> {
        let url = self.attestation_url(model_id, nonce)?;
        let response = self
            .http
            .get(url)
            .bearer_auth(self.api_key.as_ref())
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(VeniceClientError::request_failure)?;

        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(VeniceClientError::Authentication {
                status: status.as_u16(),
            });
        }
        if !status.is_success() {
            return Err(VeniceClientError::UpstreamStatus {
                status: status.as_u16(),
            });
        }

        response
            .json::<Value>()
            .await
            .map_err(VeniceClientError::malformed_attestation_payload)
    }

    fn models_url(&self) -> Result<Url, VeniceClientError> {
        self.base_url
            .join("models")
            .map_err(|source| VeniceClientError::EndpointUrl {
                message: source.to_string(),
            })
    }

    fn chat_completions_url(&self) -> Result<Url, VeniceClientError> {
        self.base_url
            .join("chat/completions")
            .map_err(|source| VeniceClientError::EndpointUrl {
                message: source.to_string(),
            })
    }

    fn attestation_url(&self, model_id: &str, nonce: &str) -> Result<Url, VeniceClientError> {
        let mut url = self.base_url.join("tee/attestation").map_err(|source| {
            VeniceClientError::EndpointUrl {
                message: source.to_string(),
            }
        })?;
        url.query_pairs_mut()
            .append_pair("model", model_id)
            .append_pair("nonce", nonce);
        Ok(url)
    }
}

impl fmt::Debug for VeniceClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VeniceClient")
            .field("base_url", &self.base_url)
            .field("api_key", &"[redacted]")
            .finish_non_exhaustive()
    }
}

fn parse_base_url(value: &str) -> Result<Url, VeniceClientError> {
    let mut url = Url::parse(value).map_err(|source| VeniceClientError::InvalidBaseUrl {
        base_url: value.to_owned(),
        message: source.to_string(),
    })?;

    if !url.path().ends_with('/') {
        let path = format!("{}/", url.path());
        url.set_path(&path);
    }

    Ok(url)
}

fn parse_model_list_response(body: &[u8]) -> Result<ModelListResponse, VeniceClientError> {
    let payload: VeniceModelListPayload =
        serde_json::from_slice(body).map_err(VeniceClientError::malformed_payload)?;
    Ok(payload.into_openai_model_list())
}

#[derive(Debug, Error)]
pub enum VeniceClientError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("invalid Venice base URL {base_url}: {message}")]
    InvalidBaseUrl { base_url: String, message: String },
    #[error("failed to build Venice HTTP client: {message}")]
    ClientBuild { message: String },
    #[error("failed to build Venice models URL: {message}")]
    EndpointUrl { message: String },
    #[error("Venice upstream authentication failed with status {status}")]
    Authentication { status: u16 },
    #[error("Venice upstream returned status {status}")]
    UpstreamStatus { status: u16 },
    #[error("Venice upstream request timed out")]
    Timeout,
    #[error("Venice upstream request failed: {message}")]
    Request { message: String },
    #[error("Venice upstream returned malformed model payload: {message}")]
    MalformedPayload { message: String },
    #[error("Venice upstream returned malformed attestation payload: {message}")]
    MalformedAttestationPayload { message: String },
}

impl VeniceClientError {
    pub fn api_error_type(&self) -> &'static str {
        match self {
            Self::Config(_)
            | Self::InvalidBaseUrl { .. }
            | Self::ClientBuild { .. }
            | Self::EndpointUrl { .. } => "proxy_configuration_error",
            Self::Authentication { .. } => "proxy_upstream_authentication_error",
            Self::UpstreamStatus { .. }
            | Self::Timeout
            | Self::Request { .. }
            | Self::MalformedPayload { .. }
            | Self::MalformedAttestationPayload { .. } => "proxy_upstream_error",
        }
    }

    pub fn api_error_code(&self) -> &'static str {
        match self {
            Self::Config(ConfigError::MissingApiKeyEnv { .. }) => "venice_api_key_missing",
            Self::Config(ConfigError::UnreadableApiKeyEnv { .. }) => "venice_api_key_unreadable",
            Self::Config(_)
            | Self::InvalidBaseUrl { .. }
            | Self::ClientBuild { .. }
            | Self::EndpointUrl { .. } => "venice_client_configuration_failed",
            Self::Authentication { .. } => "upstream_authentication_failed",
            Self::UpstreamStatus { .. } => "upstream_status_error",
            Self::Timeout => "upstream_timeout",
            Self::Request { .. } => "upstream_unavailable",
            Self::MalformedPayload { .. } | Self::MalformedAttestationPayload { .. } => {
                "upstream_malformed_response"
            }
        }
    }

    fn client_build(source: reqwest::Error) -> Self {
        Self::ClientBuild {
            message: source.to_string(),
        }
    }

    fn request_failure(source: reqwest::Error) -> Self {
        if source.is_timeout() {
            Self::Timeout
        } else {
            Self::Request {
                message: source.to_string(),
            }
        }
    }

    fn malformed_payload(source: serde_json::Error) -> Self {
        Self::MalformedPayload {
            message: source.to_string(),
        }
    }

    fn malformed_attestation_payload(source: reqwest::Error) -> Self {
        Self::MalformedAttestationPayload {
            message: source.to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct VeniceModelListPayload {
    data: Vec<VeniceModel>,
}

impl VeniceModelListPayload {
    fn into_openai_model_list(self) -> ModelListResponse {
        let data = self
            .data
            .into_iter()
            .filter_map(VeniceModel::into_openai_model_if_supported)
            .collect();

        ModelListResponse::new(data)
    }
}

#[derive(Debug, Deserialize)]
struct VeniceModel {
    id: String,
    #[serde(default)]
    created: Option<i64>,
    #[serde(default)]
    owned_by: Option<String>,
    #[serde(rename = "type")]
    model_type: String,
    model_spec: VeniceModelSpec,
}

impl VeniceModel {
    fn into_openai_model_if_supported(self) -> Option<ModelObject> {
        let capabilities = self.model_spec.capabilities;
        if self.model_type != "text"
            || !capabilities.supports_e2ee
            || !capabilities.supports_tee_attestation
        {
            return None;
        }

        let venice = VeniceModelMetadata::new(
            self.id.clone(),
            capabilities.supports_e2ee,
            capabilities.supports_tee_attestation,
        );
        let openai_capabilities = capabilities.to_openai_capabilities();

        Some(ModelObject::new(
            self.id,
            self.created.unwrap_or(0),
            self.owned_by.unwrap_or_else(|| "venice.ai".to_owned()),
            openai_capabilities,
            venice,
        ))
    }
}

#[derive(Debug, Deserialize)]
struct VeniceModelSpec {
    capabilities: VeniceCapabilities,
}

#[derive(Debug, Deserialize)]
struct VeniceCapabilities {
    #[serde(rename = "supportsE2EE")]
    supports_e2ee: bool,
    #[serde(rename = "supportsTeeAttestation")]
    supports_tee_attestation: bool,
    #[serde(default, rename = "supportsFunctionCalling")]
    supports_function_calling: Option<bool>,
    #[serde(default, rename = "supportsBuiltinTools")]
    supports_builtin_tools: Option<bool>,
    #[serde(default, rename = "supportsWebSearch")]
    supports_web_search: Option<bool>,
    #[serde(default, rename = "supportsCodeInterpreter")]
    supports_code_interpreter: Option<bool>,
    #[serde(default, rename = "supportsVision")]
    supports_vision: Option<bool>,
}

impl VeniceCapabilities {
    fn to_openai_capabilities(&self) -> ModelCapabilities {
        let web_search = self.supports_web_search.unwrap_or(false);
        let code_interpreter = self.supports_code_interpreter.unwrap_or(false);
        let builtin_tools = self
            .supports_builtin_tools
            .unwrap_or(web_search || code_interpreter);

        ModelCapabilities {
            function_calling: self.supports_function_calling.unwrap_or(false),
            builtin_tools,
            web_search,
            code_interpreter,
            vision: self.supports_vision.unwrap_or(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_supported_venice_text_models_to_openai_shape() {
        let body = br#"
        {
          "data": [
            {
              "id": "e2ee-qwen3-5-122b-a10b",
              "created": 1727966436,
              "owned_by": "venice.ai",
              "type": "text",
              "model_spec": {
                "capabilities": {
                  "supportsE2EE": true,
                  "supportsTeeAttestation": true,
                  "supportsFunctionCalling": true,
                  "supportsBuiltinTools": true,
                  "supportsWebSearch": true,
                  "supportsCodeInterpreter": true,
                  "supportsVision": false
                }
              }
            },
            {
              "id": "non-e2ee-text",
              "type": "text",
              "model_spec": {
                "capabilities": {
                  "supportsE2EE": false,
                  "supportsTeeAttestation": true
                }
              }
            },
            {
              "id": "e2ee-image",
              "type": "image",
              "model_spec": {
                "capabilities": {
                  "supportsE2EE": true,
                  "supportsTeeAttestation": true
                }
              }
            }
          ]
        }
        "#;

        let response = parse_model_list_response(body).expect("valid model payload should parse");

        assert_eq!(response.object, "list");
        assert_eq!(response.data.len(), 1);
        let model = &response.data[0];
        assert_eq!(model.id, "e2ee-qwen3-5-122b-a10b");
        assert_eq!(model.object, "model");
        assert_eq!(model.created, 1727966436);
        assert_eq!(model.owned_by, "venice.ai");
        assert_eq!(model.name, "e2ee-qwen3-5-122b-a10b");
        assert!(model.info.meta.capabilities.function_calling);
        assert!(model.info.meta.capabilities.builtin_tools);
        assert!(model.info.meta.capabilities.web_search);
        assert!(model.info.meta.capabilities.code_interpreter);
        assert!(!model.info.meta.capabilities.vision);
        assert_eq!(model.venice.id, "e2ee-qwen3-5-122b-a10b");
        assert!(model.venice.supports_e2ee);
        assert!(model.venice.supports_tee_attestation);
    }

    #[test]
    fn missing_optional_capability_metadata_defaults_to_false() {
        let body = br#"
        {
          "data": [
            {
              "id": "e2ee-minimal",
              "type": "text",
              "model_spec": {
                "capabilities": {
                  "supportsE2EE": true,
                  "supportsTeeAttestation": true
                }
              }
            }
          ]
        }
        "#;

        let response =
            parse_model_list_response(body).expect("minimal capability payload should parse");
        let model = response
            .data
            .first()
            .expect("supported model should be present");

        assert_eq!(model.created, 0);
        assert_eq!(model.owned_by, "venice.ai");
        assert_eq!(model.info.meta.capabilities, ModelCapabilities::default());
    }

    #[test]
    fn malformed_model_payload_is_reported() {
        let body = br#"
        {
          "data": [
            {
              "id": "missing-required-attestation-flag",
              "type": "text",
              "model_spec": {
                "capabilities": {
                  "supportsE2EE": true
                }
              }
            }
          ]
        }
        "#;

        let error = parse_model_list_response(body).expect_err("malformed payload should fail");

        assert!(matches!(error, VeniceClientError::MalformedPayload { .. }));
        assert_eq!(error.api_error_code(), "upstream_malformed_response");
    }

    #[test]
    fn client_debug_output_redacts_api_key() {
        let client = VeniceClient::new(
            "https://api.venice.ai/api/v1",
            "super-secret-test-key",
            DEFAULT_REQUEST_TIMEOUT,
        )
        .expect("client should build");

        let debug = format!("{client:?}");
        assert!(debug.contains("api.venice.ai"));
        assert!(debug.contains("/api/v1/"));
        assert!(debug.contains("[redacted]"));
        assert!(!debug.contains("super-secret-test-key"));
    }
}
