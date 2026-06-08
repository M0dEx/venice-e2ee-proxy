//! HTTP server, route wiring, shared headers, and route errors.
//!

use std::{io, sync::Arc};

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Map, Value};
use thiserror::Error;
use tokio::net::TcpListener;

use crate::{
    config::ProxyConfig,
    keys::ProxyInstanceKey,
    openai::ErrorResponse,
    sessions::SessionManager,
    venice::{VeniceClient, VeniceClientError},
};

pub const HEADER_PROXY_E2EE: &str = "X-Venice-Proxy-E2EE";
pub const HEADER_PROXY_ATTESTATION_MODE: &str = "X-Venice-Proxy-Attestation-Mode";
pub const HEADER_PROXY_ATTESTED_MODEL: &str = "X-Venice-Proxy-Attested-Model";
pub const HEADER_PROXY_TEE_PROVIDER: &str = "X-Venice-Proxy-TEE-Provider";
pub const HEADER_PROXY_TDX_VERIFIED: &str = "X-Venice-Proxy-TDX-Verified";
pub const HEADER_PROXY_TDX_DEBUG: &str = "X-Venice-Proxy-TDX-Debug";
pub const HEADER_PROXY_NVIDIA_VERIFIED: &str = "X-Venice-Proxy-NVIDIA-Verified";
pub const HEADER_PROXY_KEY_BINDING: &str = "X-Venice-Proxy-Key-Binding";
pub const HEADER_PROXY_SESSION_ID: &str = "X-Venice-Proxy-Session-Id";
pub const HEADER_PROXY_SESSION_SCOPE: &str = "X-Venice-Proxy-Session-Scope";
pub const HEADER_PROXY_TOOL_MODE: &str = "X-Venice-Proxy-Tool-Mode";
pub const HEADER_PROXY_TOOL_RETRIES: &str = "X-Venice-Proxy-Tool-Retries";
pub const HEADER_PROXY_ERROR_CODE: &str = "X-Venice-Proxy-Error-Code";

#[derive(Debug, Clone)]
pub struct AppState {
    config: Arc<ProxyConfig>,
    venice_client: VeniceClient,
    proxy_instance_key: Option<ProxyInstanceKey>,
    session_manager: SessionManager,
}

impl AppState {
    pub fn new(config: ProxyConfig) -> Result<Self, VeniceClientError> {
        let venice_client = VeniceClient::from_config(&config)?;
        Ok(Self::from_parts(config, venice_client))
    }

    pub fn from_parts(config: ProxyConfig, venice_client: VeniceClient) -> Self {
        let proxy_instance_key = ProxyInstanceKey::generate_from_config(&config.keys);
        let session_manager = SessionManager::new(config.session.clone());

        Self {
            config: Arc::new(config),
            venice_client,
            proxy_instance_key,
            session_manager,
        }
    }

    pub fn config(&self) -> &ProxyConfig {
        &self.config
    }

    pub fn venice_client(&self) -> &VeniceClient {
        &self.venice_client
    }

    pub fn proxy_instance_key(&self) -> Option<&ProxyInstanceKey> {
        self.proxy_instance_key.as_ref()
    }

    pub fn session_manager(&self) -> &SessionManager {
        &self.session_manager
    }
}

/// Builds the HTTP router using the configured Venice API key environment
/// variable.
pub fn router(config: ProxyConfig) -> Result<Router, VeniceClientError> {
    Ok(router_from_state(AppState::new(config)?))
}

/// Builds the HTTP router with an already-constructed Venice client.
///
/// This keeps route tests deterministic without mutating process-wide
/// environment variables.
pub fn router_with_venice_client(config: ProxyConfig, venice_client: VeniceClient) -> Router {
    router_from_state(AppState::from_parts(config, venice_client))
}

fn router_from_state(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models).fallback(method_not_allowed))
        .route(
            "/v1/chat/completions",
            post(create_chat_completion).fallback(method_not_allowed),
        )
        .fallback(not_found)
        .with_state(state)
}

/// Serves an already-built router on an already-bound listener.
pub async fn serve(listener: TcpListener, router: Router) -> io::Result<()> {
    axum::serve(listener, router).await
}

async fn list_models(State(state): State<AppState>) -> Result<Response, ProxyError> {
    let models = state.venice_client().list_models().await?;
    let mut response = Json(models).into_response();
    ProxyMetadataHeaders::from_config(state.config()).apply(response.headers_mut());
    Ok(response)
}

async fn create_chat_completion(
    State(_state): State<AppState>,
    Json(body): Json<ChatCompletionPlaceholderRequest>,
) -> ProxyError {
    let _accepted_field_count = body.len();

    ProxyError::NotImplemented {
        message: "POST /v1/chat/completions is registered but not available yet."
            .to_owned(),
    }
}

type ChatCompletionPlaceholderRequest = Map<String, Value>;

async fn method_not_allowed(method: Method, uri: Uri) -> ProxyError {
    ProxyError::MethodNotAllowed { method, uri }
}

async fn not_found(uri: Uri) -> ProxyError {
    ProxyError::NotFound { uri }
}

#[derive(Debug, Error)]
pub enum ProxyError {
    #[error(transparent)]
    Venice(#[from] VeniceClientError),
    #[error("{message}")]
    NotImplemented { message: String },
    #[error("method {method} is not supported for {uri}")]
    MethodNotAllowed { method: Method, uri: Uri },
    #[error("route {uri} was not found")]
    NotFound { uri: Uri },
}

impl ProxyError {
    fn status(&self) -> StatusCode {
        match self {
            Self::Venice(_) => StatusCode::BAD_GATEWAY,
            Self::NotImplemented { .. } => StatusCode::NOT_IMPLEMENTED,
            Self::MethodNotAllowed { .. } => StatusCode::METHOD_NOT_ALLOWED,
            Self::NotFound { .. } => StatusCode::NOT_FOUND,
        }
    }

    fn error_type(&self) -> &'static str {
        match self {
            Self::Venice(error) => error.api_error_type(),
            Self::NotImplemented { .. } => "proxy_not_implemented",
            Self::MethodNotAllowed { .. } | Self::NotFound { .. } => "invalid_request_error",
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Venice(error) => error.api_error_code(),
            Self::NotImplemented { .. } => "not_implemented",
            Self::MethodNotAllowed { .. } => "method_not_allowed",
            Self::NotFound { .. } => "not_found",
        }
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = self.status();
        let error_code = self.code();
        let body = ErrorResponse::new(self.to_string(), self.error_type(), error_code);
        let mut response = (status, Json(body)).into_response();
        apply_error_headers(response.headers_mut(), error_code);
        response
    }
}

///
/// Fields are optional so handlers never claim E2EE, attestation, key-binding,
/// or session verification that has not happened yet.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProxyMetadataHeaders {
    pub e2ee: Option<String>,
    pub attestation_mode: Option<String>,
    pub attested_model: Option<String>,
    pub tee_provider: Option<String>,
    pub tdx_verified: Option<bool>,
    pub tdx_debug: Option<bool>,
    pub nvidia_verified: Option<String>,
    pub key_binding: Option<bool>,
    pub session_id: Option<String>,
    pub session_scope: Option<String>,
    pub tool_mode: Option<String>,
    pub tool_retries: Option<u32>,
}

impl ProxyMetadataHeaders {
    /// Creates safe non-assertive metadata from config for later handlers to
    /// extend once they have real verification/session state.
    pub fn from_config(config: &ProxyConfig) -> Self {
        Self {
            attestation_mode: Some(config.attestation.mode.as_str().to_owned()),
            tool_mode: Some(config.tools.mode.as_str().to_owned()),
            ..Self::default()
        }
    }

    pub fn apply(&self, headers: &mut HeaderMap) {
        insert_optional_header(headers, HEADER_PROXY_E2EE, self.e2ee.as_deref());
        insert_optional_header(
            headers,
            HEADER_PROXY_ATTESTATION_MODE,
            self.attestation_mode.as_deref(),
        );
        insert_optional_header(
            headers,
            HEADER_PROXY_ATTESTED_MODEL,
            self.attested_model.as_deref(),
        );
        insert_optional_header(
            headers,
            HEADER_PROXY_TEE_PROVIDER,
            self.tee_provider.as_deref(),
        );
        insert_optional_bool_header(headers, HEADER_PROXY_TDX_VERIFIED, self.tdx_verified);
        insert_optional_bool_header(headers, HEADER_PROXY_TDX_DEBUG, self.tdx_debug);
        insert_optional_header(
            headers,
            HEADER_PROXY_NVIDIA_VERIFIED,
            self.nvidia_verified.as_deref(),
        );
        insert_optional_bool_header(headers, HEADER_PROXY_KEY_BINDING, self.key_binding);
        insert_optional_header(headers, HEADER_PROXY_SESSION_ID, self.session_id.as_deref());
        insert_optional_header(
            headers,
            HEADER_PROXY_SESSION_SCOPE,
            self.session_scope.as_deref(),
        );
        insert_optional_header(headers, HEADER_PROXY_TOOL_MODE, self.tool_mode.as_deref());
        if let Some(tool_retries) = self.tool_retries {
            insert_header(
                headers,
                HEADER_PROXY_TOOL_RETRIES,
                &tool_retries.to_string(),
            );
        }
    }
}

pub fn apply_error_headers(headers: &mut HeaderMap, error_code: &str) {
    insert_header(headers, HEADER_PROXY_ERROR_CODE, error_code);
}

fn insert_optional_header(headers: &mut HeaderMap, name: &'static str, value: Option<&str>) {
    if let Some(value) = value {
        insert_header(headers, name, value);
    }
}

fn insert_optional_bool_header(headers: &mut HeaderMap, name: &'static str, value: Option<bool>) {
    if let Some(value) = value {
        insert_header(headers, name, if value { "true" } else { "false" });
    }
}

fn insert_header(headers: &mut HeaderMap, name: &'static str, value: &str) {
    let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
        return;
    };
    let Ok(value) = HeaderValue::from_str(value) else {
        return;
    };
    headers.insert(name, value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    fn test_app() -> Router {
        router_with_venice_client(ProxyConfig::default(), test_venice_client())
    }

    fn test_venice_client() -> VeniceClient {
        VeniceClient::new(
            "http://127.0.0.1:1/api/v1",
            "test-api-key",
            Duration::from_secs(1),
        )
        .expect("test Venice client should build")
    }

    #[test]
    fn app_state_initializes_key_and_session_managers_from_config() {
        let state = AppState::from_parts(ProxyConfig::default(), test_venice_client());

        let key = state
            .proxy_instance_key()
            .expect("default config should generate startup key");
        assert_eq!(key.public_key_hex().len(), 130);
        assert!(state.session_manager().is_empty().unwrap());

        let mut config = ProxyConfig::default();
        config.keys.generate_proxy_instance_key_on_startup = false;
        let state = AppState::from_parts(config, test_venice_client());
        assert!(state.proxy_instance_key().is_none());
    }

    async fn error_body(response: Response) -> ErrorResponse {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should buffer");
        serde_json::from_slice(&bytes).expect("response should be OpenAI-style error JSON")
    }

    #[tokio::test]
    async fn chat_route_returns_not_implemented_placeholder() {
        let response = test_app()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"example","messages":[]}"#))
                    .expect("request should build"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            response.headers().get(HEADER_PROXY_ERROR_CODE).unwrap(),
            "not_implemented"
        );
        let body = error_body(response).await;
        assert_eq!(body.error.kind, "proxy_not_implemented");
        assert_eq!(body.error.code, "not_implemented");
    }

    #[tokio::test]
    async fn unknown_route_returns_openai_style_not_found() {
        let response = test_app()
            .oneshot(
                Request::builder()
                    .uri("/v1/unknown")
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            response.headers().get(HEADER_PROXY_ERROR_CODE).unwrap(),
            "not_found"
        );
        let body = error_body(response).await;
        assert_eq!(body.error.kind, "invalid_request_error");
        assert_eq!(body.error.code, "not_found");
    }

    #[tokio::test]
    async fn unsupported_method_returns_openai_style_method_error() {
        let response = test_app()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/models")
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(
            response.headers().get(HEADER_PROXY_ERROR_CODE).unwrap(),
            "method_not_allowed"
        );
        let body = error_body(response).await;
        assert_eq!(body.error.kind, "invalid_request_error");
        assert_eq!(body.error.code, "method_not_allowed");
    }

    #[tokio::test]
    async fn malformed_chat_json_uses_axum_extractor_rejection() {
        let response = test_app()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("{"))
                    .expect("request should build"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(response.headers().get(HEADER_PROXY_ERROR_CODE).is_none());
    }

    #[tokio::test]
    async fn non_object_chat_json_uses_axum_extractor_rejection() {
        let response = test_app()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("[]"))
                    .expect("request should build"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert!(response.headers().get(HEADER_PROXY_ERROR_CODE).is_none());
    }

    #[test]
    fn metadata_header_helper_only_emits_safe_config_headers_by_default() {
        let config = ProxyConfig::default();
        let metadata = ProxyMetadataHeaders::from_config(&config);
        let mut headers = HeaderMap::new();

        metadata.apply(&mut headers);

        assert_eq!(
            headers.get(HEADER_PROXY_ATTESTATION_MODE).unwrap(),
            "independent"
        );
        assert_eq!(headers.get(HEADER_PROXY_TOOL_MODE).unwrap(), "emulated");
        assert!(headers.get(HEADER_PROXY_E2EE).is_none());
        assert!(headers.get(HEADER_PROXY_KEY_BINDING).is_none());
    }
}
