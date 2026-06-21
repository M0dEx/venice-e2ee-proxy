use std::{
    collections::{HashMap, VecDeque},
    convert::Infallible,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{Query, State},
    http::{HeaderMap, Method, Request, StatusCode, header::AUTHORIZATION},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower::ServiceExt;
use venice_e2ee_proxy::{
    config::{ConfigError, NvidiaRequirement, ProxyConfig},
    e2ee::{E2eeCodec, MIN_PACKED_PAYLOAD_LEN},
    http::{
        self, HEADER_PROXY_ATTESTATION_MODE, HEADER_PROXY_ATTESTED_MODEL, HEADER_PROXY_E2EE,
        HEADER_PROXY_ERROR_CODE, HEADER_PROXY_KEY_BINDING, HEADER_PROXY_NVIDIA_VERIFIED,
        HEADER_PROXY_SESSION_ID, HEADER_PROXY_SESSION_SCOPE, HEADER_PROXY_TDX_DEBUG,
        HEADER_PROXY_TDX_VERIFIED, HEADER_PROXY_TEE_PROVIDER, HEADER_PROXY_TOOL_MODE,
        HEADER_PROXY_TOOL_RETRIES,
    },
    keys::ProxyInstanceKey,
    openai::ErrorResponse,
    venice::{
        HEADER_VENICE_TEE_CLIENT_PUB_KEY, HEADER_VENICE_TEE_MODEL_PUB_KEY,
        HEADER_VENICE_TEE_SIGNING_ALGO, VeniceClient,
    },
};

const TEST_API_KEY: &str = "test-api-key";
const TEST_MODEL: &str = "e2ee-test";
const TEST_SESSION_ID: &str = "proxy-integration-session";

#[tokio::test]
async fn streaming_chat_completion_decrypts_split_sse_and_sets_verified_headers() {
    let mock = MockVeniceServer::spawn(MockVeniceOptions::with_attempts(vec![vec![
        MockStreamFrame::SplitText("Hello"),
        MockStreamFrame::Text(" world"),
        MockStreamFrame::Finish("stop"),
        MockStreamFrame::Done,
    ]]))
    .await;
    let app = proxy_app(&mock.base_url, Duration::from_secs(1));

    let response = request_chat(
        app,
        TEST_SESSION_ID,
        json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_verified_chat_headers(&response, TEST_SESSION_ID, None);

    let body = response_text(response).await;
    let data = sse_data(&body);
    assert_eq!(data.len(), 4);

    let first: Value = serde_json::from_str(data[0]).expect("first chunk should be JSON");
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["model"], TEST_MODEL);
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(first["choices"][0]["delta"]["content"], "Hello");

    let second: Value = serde_json::from_str(data[1]).expect("second chunk should be JSON");
    assert!(second["choices"][0]["delta"].get("role").is_none());
    assert_eq!(second["choices"][0]["delta"]["content"], " world");

    let final_chunk: Value = serde_json::from_str(data[2]).expect("final chunk should be JSON");
    assert_eq!(final_chunk["choices"][0]["delta"], json!({}));
    assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
    assert_eq!(data[3], "[DONE]");
}

#[tokio::test]
async fn streaming_chat_completion_decrypts_reasoning_content() {
    let mock = MockVeniceServer::spawn(MockVeniceOptions::with_attempts(vec![vec![
        MockStreamFrame::Reasoning("Thinking..."),
        MockStreamFrame::Text("Final answer"),
        MockStreamFrame::Finish("stop"),
        MockStreamFrame::Done,
    ]]))
    .await;
    let app = proxy_app(&mock.base_url, Duration::from_secs(1));

    let response = request_chat(
        app,
        TEST_SESSION_ID,
        json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "reasoning": {"effort": "high"}
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_verified_chat_headers(&response, TEST_SESSION_ID, None);

    let body = response_text(response).await;
    let data = sse_data(&body);
    assert_eq!(data.len(), 4);

    let reasoning: Value = serde_json::from_str(data[0]).expect("reasoning chunk should be JSON");
    assert_eq!(reasoning["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(
        reasoning["choices"][0]["delta"]["reasoning_content"],
        "Thinking..."
    );
    assert!(reasoning["choices"][0]["delta"].get("content").is_none());

    let answer: Value = serde_json::from_str(data[1]).expect("answer chunk should be JSON");
    assert!(answer["choices"][0]["delta"].get("role").is_none());
    assert_eq!(answer["choices"][0]["delta"]["content"], "Final answer");

    let final_chunk: Value = serde_json::from_str(data[2]).expect("final chunk should be JSON");
    assert_eq!(final_chunk["choices"][0]["delta"], json!({}));
    assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
    assert_eq!(data[3], "[DONE]");
}

#[tokio::test]
async fn non_streaming_chat_completion_buffers_decrypted_reasoning_content() {
    let mock = MockVeniceServer::spawn(MockVeniceOptions::with_attempts(vec![vec![
        MockStreamFrame::Reasoning("Think "),
        MockStreamFrame::Reasoning("carefully."),
        MockStreamFrame::Text("Final answer"),
        MockStreamFrame::Finish("stop"),
        MockStreamFrame::Done,
    ]]))
    .await;
    let app = proxy_app(&mock.base_url, Duration::from_secs(1));

    let response = request_chat(
        app,
        TEST_SESSION_ID,
        json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "reasoning_effort": "medium",
            "venice_parameters": {"strip_thinking_response": false}
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_verified_chat_headers(&response, TEST_SESSION_ID, None);

    let body: Value = response_json(response).await;
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(
        body["choices"][0]["message"]["reasoning_content"],
        "Think carefully."
    );
    assert_eq!(body["choices"][0]["message"]["content"], "Final answer");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn plaintext_reasoning_content_fails_closed() {
    let mock = MockVeniceServer::spawn(MockVeniceOptions::with_attempts(vec![vec![
        MockStreamFrame::Raw("data: {\"id\":\"chatcmpl-upstream-test\",\"object\":\"chat.completion.chunk\",\"created\":1717171717,\"model\":\"e2ee-test\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"plaintext thinking\"},\"finish_reason\":null}]}\n\n"),
        MockStreamFrame::Done,
    ]]))
    .await;
    let app = proxy_app(&mock.base_url, Duration::from_secs(1));

    let response = request_chat(
        app,
        TEST_SESSION_ID,
        json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
        }),
    )
    .await;

    assert_proxy_error(
        response,
        StatusCode::BAD_GATEWAY,
        "proxy_e2ee_error",
        "e2ee_response_decryption_failed",
    )
    .await;
}

#[tokio::test]
async fn non_streaming_chat_completion_buffers_decrypted_response_and_usage() {
    let mock = MockVeniceServer::spawn(MockVeniceOptions::with_attempts(vec![vec![
        MockStreamFrame::Text("Hello"),
        MockStreamFrame::Text(" world"),
        MockStreamFrame::Finish("stop"),
        MockStreamFrame::Usage,
        MockStreamFrame::Done,
    ]]))
    .await;
    let app = proxy_app(&mock.base_url, Duration::from_secs(1));

    let response = request_chat(
        app,
        TEST_SESSION_ID,
        json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_verified_chat_headers(&response, TEST_SESSION_ID, None);

    let body: Value = response_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["id"], "chatcmpl-upstream-test");
    assert_eq!(body["model"], TEST_MODEL);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 1);
    assert_eq!(body["usage"]["completion_tokens"], 2);
    assert_eq!(body["usage"]["total_tokens"], 3);
}

#[tokio::test]
async fn tool_call_emulation_retries_invalid_marker_then_returns_openai_tool_call() {
    let mock = MockVeniceServer::spawn(MockVeniceOptions::with_attempts(vec![
        vec![
            MockStreamFrame::Text(
                r#"<tool_call>{"name":"unknown","arguments":{"query":"example"}}</tool_call>"#,
            ),
            MockStreamFrame::Done,
        ],
        vec![
            MockStreamFrame::Text(
                r#"<tool_call>{"name":"search_web","arguments":{"query":"example"}}</tool_call>"#,
            ),
            MockStreamFrame::Done,
        ],
    ]))
    .await;
    let app = proxy_app(&mock.base_url, Duration::from_secs(1));

    let response = request_chat(
        app,
        TEST_SESSION_ID,
        json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "search"}],
            "stream": false,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": false
                    }
                }
            }]
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_verified_chat_headers(&response, TEST_SESSION_ID, Some("1"));

    let body: Value = response_json(response).await;
    assert_eq!(body["choices"][0]["message"]["content"], Value::Null);
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");

    let tool_call = &body["choices"][0]["message"]["tool_calls"][0];
    assert!(tool_call["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(tool_call["type"], "function");
    assert_eq!(tool_call["function"]["name"], "search_web");
    assert_eq!(tool_call["function"]["arguments"], r#"{"query":"example"}"#);

    assert_eq!(mock.chat_count(), 2);
}

#[tokio::test]
async fn tool_call_emulation_returns_multiple_openai_tool_calls() {
    let mock = MockVeniceServer::spawn(MockVeniceOptions::with_attempts(vec![vec![
        MockStreamFrame::Text(
            r#"<tool_call>{"name":"search_web","arguments":{"query":"first"}}</tool_call>"#,
        ),
        MockStreamFrame::Text(
            r#"<tool_call>{"name":"search_web","arguments":{"query":"second"}}</tool_call>"#,
        ),
        MockStreamFrame::Done,
    ]]))
    .await;
    let app = proxy_app(&mock.base_url, Duration::from_secs(1));

    let response = request_chat(
        app,
        TEST_SESSION_ID,
        json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "search"}],
            "stream": false,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": false
                    }
                }
            }]
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_verified_chat_headers(&response, TEST_SESSION_ID, None);

    let body: Value = response_json(response).await;
    assert_eq!(body["choices"][0]["message"]["content"], Value::Null);
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");

    let tool_calls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls should be an array");
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0]["function"]["name"], "search_web");
    assert_eq!(
        tool_calls[0]["function"]["arguments"],
        r#"{"query":"first"}"#
    );
    assert_eq!(tool_calls[1]["function"]["name"], "search_web");
    assert_eq!(
        tool_calls[1]["function"]["arguments"],
        r#"{"query":"second"}"#
    );
    assert_ne!(tool_calls[0]["id"], tool_calls[1]["id"]);
}

#[tokio::test]
async fn missing_api_key_fails_closed_before_router_starts() {
    let error = http::router(ProxyConfig::default())
        .expect_err("router startup must require an upstream API key");

    assert_eq!(error.api_error_type(), "proxy_configuration_error");
    assert_eq!(error.api_error_code(), "venice_api_key_missing");
}

#[test]
fn invalid_config_fails_closed_before_router_starts() {
    let error = ProxyConfig::from_toml_str(
        r#"
        [venice]
        base_url = "not-a-url"
        "#,
    )
    .expect_err("invalid config must not load");

    assert!(matches!(
        error,
        ConfigError::InvalidValue {
            field: "venice.base_url",
            ..
        }
    ));
}

#[tokio::test]
async fn upstream_unavailable_returns_fail_closed_error() {
    let base_url = unused_local_base_url().await;
    let app = proxy_app(&base_url, Duration::from_millis(100));

    let response = request_models(app).await;

    assert_proxy_error(
        response,
        StatusCode::BAD_GATEWAY,
        "proxy_upstream_error",
        "upstream_unavailable",
    )
    .await;
}

#[tokio::test]
async fn malformed_upstream_chat_response_returns_fail_closed_error() {
    let mock = MockVeniceServer::spawn(MockVeniceOptions::with_attempts(vec![vec![
        MockStreamFrame::Raw("data: {\"choices\":\"bad\"}\n\n"),
    ]]))
    .await;
    let app = proxy_app(&mock.base_url, Duration::from_secs(1));

    let response = request_chat(
        app,
        TEST_SESSION_ID,
        json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
        }),
    )
    .await;

    assert_proxy_error(
        response,
        StatusCode::BAD_GATEWAY,
        "proxy_upstream_error",
        "upstream_malformed_response",
    )
    .await;
}

#[tokio::test]
async fn attestation_failure_returns_error_and_does_not_call_chat_upstream() {
    let mock = MockVeniceServer::spawn(MockVeniceOptions {
        attestation_verified: false,
        chat_attempts: VecDeque::from([vec![MockStreamFrame::Text("should not be called")]]),
    })
    .await;
    let app = proxy_app(&mock.base_url, Duration::from_secs(1));

    let response = request_chat(
        app,
        TEST_SESSION_ID,
        json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
        }),
    )
    .await;

    assert_proxy_error(
        response,
        StatusCode::BAD_GATEWAY,
        "proxy_attestation_error",
        "attestation_upstream_not_verified",
    )
    .await;

    assert_eq!(mock.chat_count(), 0);
}

#[tokio::test]
async fn decryption_failure_returns_fail_closed_error() {
    let mock = MockVeniceServer::spawn(MockVeniceOptions::with_attempts(vec![vec![
        MockStreamFrame::TextForWrongRecipient("secret"),
        MockStreamFrame::Done,
    ]]))
    .await;
    let app = proxy_app(&mock.base_url, Duration::from_secs(1));

    let response = request_chat(
        app,
        TEST_SESSION_ID,
        json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
        }),
    )
    .await;

    assert_proxy_error(
        response,
        StatusCode::BAD_GATEWAY,
        "proxy_e2ee_error",
        "e2ee_response_decryption_failed",
    )
    .await;
}

#[derive(Debug, Clone)]
struct MockVeniceOptions {
    attestation_verified: bool,
    chat_attempts: VecDeque<Vec<MockStreamFrame>>,
}

impl MockVeniceOptions {
    fn with_attempts(attempts: Vec<Vec<MockStreamFrame>>) -> Self {
        Self {
            chat_attempts: VecDeque::from(attempts),
            ..Self::default()
        }
    }
}

impl Default for MockVeniceOptions {
    fn default() -> Self {
        Self {
            attestation_verified: true,
            chat_attempts: VecDeque::new(),
        }
    }
}

#[derive(Debug, Clone)]
enum MockStreamFrame {
    Text(&'static str),
    Reasoning(&'static str),
    SplitText(&'static str),
    TextForWrongRecipient(&'static str),
    Finish(&'static str),
    Usage,
    Done,
    Raw(&'static str),
}

#[derive(Debug, Clone)]
struct MockVeniceServer {
    base_url: String,
    state: MockVeniceState,
}

impl MockVeniceServer {
    async fn spawn(options: MockVeniceOptions) -> Self {
        let state = MockVeniceState::new(options);
        let app = Router::new()
            .route("/api/v1/tee/attestation", get(mock_attestation))
            .route("/api/v1/chat/completions", post(mock_chat_completion))
            .with_state(state.clone());
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

        Self {
            base_url: format!("http://{addr}/api/v1"),
            state,
        }
    }

    fn chat_count(&self) -> usize {
        self.state.chat_count()
    }
}

#[derive(Debug, Clone)]
struct MockVeniceState {
    inner: Arc<MockVeniceStateInner>,
}

#[derive(Debug)]
struct MockVeniceStateInner {
    model_public_key: String,
    attestation_verified: bool,
    chat_attempts: Mutex<VecDeque<Vec<MockStreamFrame>>>,
    chat_count: Mutex<usize>,
}

impl MockVeniceState {
    fn new(options: MockVeniceOptions) -> Self {
        Self {
            inner: Arc::new(MockVeniceStateInner {
                model_public_key: ProxyInstanceKey::generate().public_key_hex().to_owned(),
                attestation_verified: options.attestation_verified,
                chat_attempts: Mutex::new(options.chat_attempts),
                chat_count: Mutex::new(0),
            }),
        }
    }

    fn record_chat(&self) {
        *self
            .inner
            .chat_count
            .lock()
            .expect("mock chat count mutex should not be poisoned") += 1;
    }

    fn chat_count(&self) -> usize {
        *self
            .inner
            .chat_count
            .lock()
            .expect("mock chat count mutex should not be poisoned")
    }

    fn next_chat_attempt(&self) -> Vec<MockStreamFrame> {
        let mut attempts = self
            .inner
            .chat_attempts
            .lock()
            .expect("mock chat attempts mutex should not be poisoned");

        if attempts.len() > 1 {
            attempts.pop_front().expect("attempts length checked above")
        } else {
            attempts.front().cloned().unwrap_or_default()
        }
    }

    fn model_public_key(&self) -> &str {
        &self.inner.model_public_key
    }

    fn attestation_verified(&self) -> bool {
        self.inner.attestation_verified
    }
}

async fn mock_attestation(
    State(state): State<MockVeniceState>,
    Query(query): Query<HashMap<String, String>>,
    headers: HeaderMap,
) -> Response {
    if !is_authorized(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    Json(json!({
        "api_version": "aci/1",
        "attestation": {
            "tee_type": "tdx",
            "evidence": {}
        },
        "verified": state.attestation_verified(),
        "nonce": query.get("nonce").cloned().unwrap_or_default(),
        "model": query.get("model").cloned().unwrap_or_default(),
        "tee_provider": "phala",
        "debug": false,
        "signing_public_key": state.model_public_key(),
    }))
    .into_response()
}

async fn mock_chat_completion(
    State(state): State<MockVeniceState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    state.record_chat();

    if !is_authorized(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let Some(client_public_key) = headers
        .get(HEADER_VENICE_TEE_CLIENT_PUB_KEY)
        .and_then(|value| value.to_str().ok())
    else {
        return (StatusCode::BAD_REQUEST, "missing client key").into_response();
    };

    if headers
        .get(HEADER_VENICE_TEE_MODEL_PUB_KEY)
        .and_then(|value| value.to_str().ok())
        != Some(state.model_public_key())
    {
        return (StatusCode::BAD_REQUEST, "wrong model key").into_response();
    }

    if headers
        .get(HEADER_VENICE_TEE_SIGNING_ALGO)
        .and_then(|value| value.to_str().ok())
        != Some("ecdsa")
    {
        return (StatusCode::BAD_REQUEST, "wrong signing algorithm").into_response();
    }

    if body.get("stream").and_then(Value::as_bool) != Some(true) {
        return (StatusCode::BAD_REQUEST, "upstream request must stream").into_response();
    }

    if body.get("model").and_then(Value::as_str) != Some(TEST_MODEL) {
        return (StatusCode::BAD_REQUEST, "wrong model").into_response();
    }

    if !messages_are_encrypted(&body) {
        return (StatusCode::BAD_REQUEST, "messages must be encrypted").into_response();
    }

    let frames = state.next_chat_attempt();
    mock_sse_response(&frames, client_public_key)
}

fn is_authorized(headers: &HeaderMap) -> bool {
    headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        == Some("Bearer test-api-key")
}

fn messages_are_encrypted(body: &Value) -> bool {
    body.get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            !messages.is_empty()
                && messages.iter().all(|message| {
                    message.get("role").and_then(Value::as_str).is_some()
                        && message
                            .get("content")
                            .and_then(Value::as_str)
                            .is_some_and(is_encrypted_hex)
                })
        })
}

fn is_encrypted_hex(value: &str) -> bool {
    value.len() >= MIN_PACKED_PAYLOAD_LEN * 2
        && value.len().is_multiple_of(2)
        && value.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn mock_sse_response(frames: &[MockStreamFrame], client_public_key: &str) -> Response {
    let chunks = render_mock_sse_chunks(frames, client_public_key);
    let stream = async_stream::stream! {
        for chunk in chunks {
            yield Ok::<Bytes, Infallible>(Bytes::from(chunk));
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(stream))
        .expect("mock SSE response should build")
}

fn render_mock_sse_chunks(frames: &[MockStreamFrame], client_public_key: &str) -> Vec<String> {
    let codec = E2eeCodec::default();
    let mut chunks = Vec::new();

    for frame in frames {
        match frame {
            MockStreamFrame::Text(content) => {
                chunks.push(encrypted_content_event(&codec, content, client_public_key));
            }
            MockStreamFrame::Reasoning(content) => {
                chunks.push(encrypted_reasoning_event(
                    &codec,
                    content,
                    client_public_key,
                ));
            }
            MockStreamFrame::SplitText(content) => {
                let event = encrypted_content_event(&codec, content, client_public_key);
                let split = event.len() / 2;

                chunks.push(event[..split].to_owned());
                chunks.push(event[split..].to_owned());
            }
            MockStreamFrame::TextForWrongRecipient(content) => {
                let wrong_key = ProxyInstanceKey::generate();

                chunks.push(encrypted_content_event(
                    &codec,
                    content,
                    wrong_key.public_key_hex(),
                ));
            }
            MockStreamFrame::Finish(reason) => {
                chunks.push(format!("data: {}\n\n", upstream_finish_chunk(reason)));
            }
            MockStreamFrame::Usage => {
                chunks.push(format!("data: {}\n\n", upstream_usage_chunk()));
            }
            MockStreamFrame::Done => chunks.push("data: [DONE]\n\n".to_owned()),
            MockStreamFrame::Raw(raw) => chunks.push((*raw).to_owned()),
        }
    }

    chunks
}

fn encrypted_content_event(codec: &E2eeCodec, content: &str, client_public_key: &str) -> String {
    let encrypted = codec
        .encrypt_content(content, client_public_key)
        .expect("mock response content should encrypt")
        .into_hex();
    format!("data: {}\n\n", upstream_content_chunk(encrypted))
}

fn encrypted_reasoning_event(codec: &E2eeCodec, content: &str, client_public_key: &str) -> String {
    let encrypted = codec
        .encrypt_content(content, client_public_key)
        .expect("mock response reasoning content should encrypt")
        .into_hex();
    format!("data: {}\n\n", upstream_reasoning_content_chunk(encrypted))
}

fn upstream_content_chunk(encrypted_content: String) -> Value {
    json!({
        "id": "chatcmpl-upstream-test",
        "object": "chat.completion.chunk",
        "created": 1_717_171_717,
        "model": TEST_MODEL,
        "choices": [{
            "index": 0,
            "delta": { "content": encrypted_content },
            "finish_reason": null,
        }],
    })
}

fn upstream_reasoning_content_chunk(encrypted_content: String) -> Value {
    json!({
        "id": "chatcmpl-upstream-test",
        "object": "chat.completion.chunk",
        "created": 1_717_171_717,
        "model": TEST_MODEL,
        "choices": [{
            "index": 0,
            "delta": { "reasoning_content": encrypted_content },
            "finish_reason": null,
        }],
    })
}

fn upstream_finish_chunk(reason: &str) -> Value {
    json!({
        "id": "chatcmpl-upstream-test",
        "object": "chat.completion.chunk",
        "created": 1_717_171_717,
        "model": TEST_MODEL,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": reason,
        }],
    })
}

fn upstream_usage_chunk() -> Value {
    json!({
        "id": "chatcmpl-upstream-test",
        "object": "chat.completion.chunk",
        "created": 1_717_171_717,
        "model": TEST_MODEL,
        "choices": [],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
    })
}

fn proxy_app(base_url: &str, timeout: Duration) -> Router {
    let client = VeniceClient::new(base_url, TEST_API_KEY, timeout)
        .expect("test Venice client should build");
    http::router_with_venice_client(chat_test_config(), client)
}

fn chat_test_config() -> ProxyConfig {
    let mut config = ProxyConfig::default();
    config.attestation.require_tdx = false;
    config.attestation.require_nvidia = NvidiaRequirement::Never;
    config
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

async fn request_chat(app: Router, session_id: &str, body: Value) -> Response {
    app.oneshot(
        Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(HEADER_PROXY_SESSION_ID, session_id)
            .body(Body::from(body.to_string()))
            .expect("request should build"),
    )
    .await
    .expect("request should complete")
}

async fn response_json<T: DeserializeOwned>(response: Response) -> T {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body should buffer");
    serde_json::from_slice(&bytes).expect("response should be JSON")
}

async fn response_text(response: Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body should buffer");
    String::from_utf8(bytes.to_vec()).expect("response body should be UTF-8")
}

async fn assert_proxy_error(
    response: Response,
    expected_status: StatusCode,
    expected_type: &str,
    expected_code: &str,
) {
    assert_eq!(response.status(), expected_status);
    assert_eq!(
        response.headers().get(HEADER_PROXY_ERROR_CODE).unwrap(),
        expected_code
    );

    let body: ErrorResponse = response_json(response).await;
    assert_eq!(body.error.kind, expected_type);
    assert_eq!(body.error.code, expected_code);
}

fn assert_verified_chat_headers(response: &Response, session_id: &str, tool_retries: Option<&str>) {
    assert_eq!(
        response.headers().get(HEADER_PROXY_E2EE).unwrap(),
        "verified"
    );
    assert_eq!(
        response
            .headers()
            .get(HEADER_PROXY_ATTESTATION_MODE)
            .unwrap(),
        "independent"
    );
    assert_eq!(
        response.headers().get(HEADER_PROXY_ATTESTED_MODEL).unwrap(),
        TEST_MODEL
    );
    assert_eq!(
        response.headers().get(HEADER_PROXY_TEE_PROVIDER).unwrap(),
        "phala"
    );
    assert_eq!(
        response.headers().get(HEADER_PROXY_TDX_DEBUG).unwrap(),
        "false"
    );
    assert_eq!(
        response
            .headers()
            .get(HEADER_PROXY_NVIDIA_VERIFIED)
            .unwrap(),
        "not-present"
    );
    assert_eq!(
        response.headers().get(HEADER_PROXY_KEY_BINDING).unwrap(),
        "true"
    );
    assert_eq!(
        response.headers().get(HEADER_PROXY_SESSION_ID).unwrap(),
        session_id
    );
    assert_eq!(
        response.headers().get(HEADER_PROXY_SESSION_SCOPE).unwrap(),
        "agent"
    );
    assert_eq!(
        response.headers().get(HEADER_PROXY_TOOL_MODE).unwrap(),
        "emulated"
    );

    // The test config intentionally disables strict TDX verification because the
    // v0.1 proxy has no linked DCAP/QVL verifier. Do not claim TDX verification
    // in this relaxed mocked-integration path.
    assert!(response.headers().get(HEADER_PROXY_TDX_VERIFIED).is_none());

    match tool_retries {
        Some(expected) => assert_eq!(
            response.headers().get(HEADER_PROXY_TOOL_RETRIES).unwrap(),
            expected
        ),
        None => assert!(response.headers().get(HEADER_PROXY_TOOL_RETRIES).is_none()),
    }
}

fn sse_data(body: &str) -> Vec<&str> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect()
}

async fn unused_local_base_url() -> String {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("unused listener should bind");
    let addr = listener
        .local_addr()
        .expect("unused listener should have local address");
    drop(listener);
    format!("http://{addr}/api/v1")
}
