use std::time::Duration;

use axum::{
    Json, Router,
    body::Body,
    http::{Request, StatusCode, header::AUTHORIZATION},
    response::{IntoResponse, Response},
    routing::get,
};
use serde::de::DeserializeOwned;
use serde_json::json;
use tokio::net::TcpListener;
use tower::ServiceExt;
use venice_e2ee_proxy::{
    config::ProxyConfig,
    http::{
        self, HEADER_PROXY_ATTESTATION_MODE, HEADER_PROXY_E2EE, HEADER_PROXY_ERROR_CODE,
        HEADER_PROXY_TOOL_MODE,
    },
    openai::{ErrorResponse, ModelListResponse},
    venice::VeniceClient,
};

const TEST_API_KEY: &str = "test-api-key";

#[tokio::test]
async fn get_models_returns_filtered_openai_model_list() {
    let base_url =
        spawn_mock_venice(Router::new().route("/api/v1/models", get(successful_models))).await;
    let app = proxy_app(base_url, Duration::from_secs(1));

    let response = request_models(app).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(HEADER_PROXY_ATTESTATION_MODE)
            .unwrap(),
        "independent"
    );
    assert_eq!(
        response.headers().get(HEADER_PROXY_TOOL_MODE).unwrap(),
        "emulated"
    );
    assert!(response.headers().get(HEADER_PROXY_E2EE).is_none());

    let body: ModelListResponse = response_json(response).await;
    assert_eq!(body.object, "list");
    assert_eq!(body.data.len(), 1);

    let model = &body.data[0];
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
    assert!(model.info.meta.capabilities.reasoning);
    assert!(model.info.meta.capabilities.reasoning_effort);
    assert_eq!(model.venice.id, "e2ee-qwen3-5-122b-a10b");
    assert!(model.venice.supports_e2ee);
    assert!(model.venice.supports_tee_attestation);
    assert!(model.venice.supports_reasoning);
    assert!(model.venice.supports_reasoning_effort);
}

#[tokio::test]
async fn get_models_fails_closed_on_upstream_authentication_errors() {
    for upstream_status in [StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN] {
        let base_url = spawn_mock_venice(Router::new().route(
            "/api/v1/models",
            get(move || async move { upstream_status }),
        ))
        .await;
        let app = proxy_app(base_url, Duration::from_secs(1));

        let response = request_models(app).await;

        assert_proxy_error(
            response,
            "proxy_upstream_authentication_error",
            "upstream_authentication_failed",
        )
        .await;
    }
}

#[tokio::test]
async fn get_models_fails_closed_on_upstream_server_error() {
    let base_url = spawn_mock_venice(Router::new().route(
        "/api/v1/models",
        get(|| async { StatusCode::INTERNAL_SERVER_ERROR }),
    ))
    .await;
    let app = proxy_app(base_url, Duration::from_secs(1));

    let response = request_models(app).await;

    assert_proxy_error(response, "proxy_upstream_error", "upstream_status_error").await;
}

#[tokio::test]
async fn get_models_fails_closed_on_malformed_upstream_payload() {
    let base_url = spawn_mock_venice(Router::new().route(
        "/api/v1/models",
        get(|| async {
            Json(json!({
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
            }))
        }),
    ))
    .await;
    let app = proxy_app(base_url, Duration::from_secs(1));

    let response = request_models(app).await;

    assert_proxy_error(
        response,
        "proxy_upstream_error",
        "upstream_malformed_response",
    )
    .await;
}

#[tokio::test]
async fn get_models_fails_closed_on_upstream_timeout() {
    let base_url = spawn_mock_venice(Router::new().route("/api/v1/models", get(slow_models))).await;
    let app = proxy_app(base_url, Duration::from_millis(20));

    let response = request_models(app).await;

    assert_proxy_error(response, "proxy_upstream_error", "upstream_timeout").await;
}

async fn successful_models(headers: axum::http::HeaderMap) -> Response {
    if headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        != Some("Bearer test-api-key")
    {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    Json(json!({
        "object": "list",
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
                        "supportsVision": false,
                        "supportsReasoning": true,
                        "supportsReasoningEffort": true
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
                "id": "e2ee-without-attestation",
                "type": "text",
                "model_spec": {
                    "capabilities": {
                        "supportsE2EE": true,
                        "supportsTeeAttestation": false
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
    }))
    .into_response()
}

async fn slow_models() -> impl IntoResponse {
    tokio::time::sleep(Duration::from_millis(200)).await;
    Json(json!({ "data": [] }))
}

fn proxy_app(base_url: String, timeout: Duration) -> Router {
    let client = VeniceClient::new(base_url, TEST_API_KEY, timeout)
        .expect("test Venice client should build");
    http::router_with_venice_client(ProxyConfig::default(), client)
}

async fn spawn_mock_venice(app: Router) -> String {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("mock Venice listener should bind");
    let addr = listener
        .local_addr()
        .expect("mock Venice listener should have local address");

    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("mock Venice server should run");
    });

    format!("http://{addr}/api/v1")
}

async fn request_models(app: Router) -> Response {
    app.oneshot(
        Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await
    .expect("request should complete")
}

async fn assert_proxy_error(response: Response, expected_type: &str, expected_code: &str) {
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        response.headers().get(HEADER_PROXY_ERROR_CODE).unwrap(),
        expected_code
    );

    let body: ErrorResponse = response_json(response).await;
    assert_eq!(body.error.kind, expected_type);
    assert_eq!(body.error.code, expected_code);
}

async fn response_json<T: DeserializeOwned>(response: Response) -> T {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body should buffer");
    serde_json::from_slice(&bytes).expect("response should be JSON")
}
