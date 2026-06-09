//! HTTP server, route wiring, shared headers, and route errors.
//!
//! Routes include Venice-backed model listing, encrypted chat request
//! construction, response transformation, and OpenAI-compatible errors/headers.

use std::{
    io,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::net::TcpListener;
use tracing::{debug, error, info, warn};

use crate::{
    attestation::{AttestationError, AttestationVerifier},
    config::{NvidiaRequirement, ProxyConfig},
    e2ee::{E2eeCodec, E2eeCodecError},
    keys::ProxyInstanceKey,
    openai::{
        ErrorResponse,
        chat::{ChatCompletionRequest, ChatConstructionError, ChatRequestError},
    },
    sessions::{AttestedModelState, SessionContext, SessionError, SessionManager, SessionRequest},
    tools::{ToolEmulationContext, ToolOutputClassification, ValidatedToolCall},
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
    attestation_verifier: AttestationVerifier,
}

impl AppState {
    pub fn new(config: ProxyConfig) -> Result<Self, VeniceClientError> {
        let venice_client = VeniceClient::from_config(&config)?;
        Ok(Self::from_parts(config, venice_client))
    }

    pub fn from_parts(config: ProxyConfig, venice_client: VeniceClient) -> Self {
        let proxy_instance_key = ProxyInstanceKey::generate_from_config(&config.keys);
        let session_manager = SessionManager::new(config.session.clone());
        let attestation_verifier = AttestationVerifier::from_config(&config, venice_client.clone());

        Self {
            config: Arc::new(config),
            venice_client,
            proxy_instance_key,
            session_manager,
            attestation_verifier,
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

    pub fn attestation_verifier(&self) -> &AttestationVerifier {
        &self.attestation_verifier
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
    info!(route = "/v1/models", "listing Venice models");
    let models = state.venice_client().list_models().await?;
    let mut response = Json(models).into_response();
    ProxyMetadataHeaders::from_config(state.config()).apply(response.headers_mut());
    info!(route = "/v1/models", "Venice models response proxied");
    Ok(response)
}

async fn create_chat_completion(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ProxyError> {
    let request = ChatCompletionRequest::parse(&body)?;
    let proxy_instance_key = state
        .proxy_instance_key()
        .ok_or(ProxyError::ProxyInstanceKeyUnavailable)?;

    let session_resolution = state
        .session_manager()
        .get_or_create(SessionRequest::new(&request.model, &headers).with_body(&body))?;
    let session_created = session_resolution.created;
    let session_replaced_expired = session_resolution.replaced_expired;
    let session_scope = session_resolution.session.scope;
    let session = ensure_attested_session(&state, session_resolution.session).await?;
    let model_public_key = session
        .attested_model_public_key
        .as_deref()
        .ok_or(ProxyError::MissingAttestedModelKey)?;

    let codec =
        E2eeCodec::from_config(&state.config().e2ee).map_err(ChatConstructionError::E2ee)?;
    let tool_context = ToolEmulationContext::from_request(&state.config().tools, &request)?;
    let metadata = ProxyMetadataHeaders::for_verified_chat(state.config(), &session);

    info!(
        route = "/v1/chat/completions",
        model = %request.model,
        stream = request.stream,
        message_count = request.messages.len(),
        tool_count = request.tools.len(),
        tool_mode = tool_context.is_some(),
        session_created,
        session_replaced_expired = ?session_replaced_expired,
        session_scope = %session_scope,
        "chat completion request accepted"
    );

    if let Some(tool_context) = tool_context {
        info!(model = %request.model, "using tool-emulated chat completion");
        return openai_tool_emulated_chat_response(
            &state,
            &request,
            &tool_context,
            codec,
            proxy_instance_key.clone(),
            model_public_key,
            metadata,
        )
        .await;
    }

    let prepared = request.into_venice_e2ee_request(&codec, model_public_key)?;
    info!(
        model = %request.model,
        client_stream = prepared.client_stream,
        "forwarding encrypted chat completion to Venice"
    );

    let upstream = state
        .venice_client()
        .create_chat_completion_stream(
            &prepared.upstream,
            proxy_instance_key.public_key_hex(),
            model_public_key,
        )
        .await?;

    if prepared.client_stream {
        info!(model = %request.model, "streaming chat completion response to client");
        Ok(openai_chat_sse_response(
            upstream,
            codec,
            proxy_instance_key.clone(),
            request.model,
            request.stream_options.include_usage.unwrap_or(false),
            metadata,
        ))
    } else {
        info!(model = %request.model, "buffering chat completion response for client");
        openai_chat_buffered_response(
            upstream,
            codec,
            proxy_instance_key.clone(),
            request.model,
            metadata,
        )
        .await
    }
}

async fn ensure_attested_session(
    state: &AppState,
    session: SessionContext,
) -> Result<SessionContext, ProxyError> {
    if session.attested_model_public_key.is_some() {
        info!(model = %session.model_id, session_scope = %session.scope, "using cached model attestation");
        return Ok(session);
    }

    info!(model = %session.model_id, session_scope = %session.scope, "fetching model attestation");
    let attestation = state
        .attestation_verifier()
        .verify_model_attestation(&session.model_id)
        .await?;

    info!(
        model = %attestation.model_id,
        tee_provider = attestation.tee_provider.as_deref().unwrap_or("unknown"),
        tdx_verified = attestation.tdx.verified,
        nvidia_verified = attestation.nvidia.verified.as_header_value(),
        key_binding = attestation.key_binding,
        "model attestation verified"
    );

    let state_update = AttestedModelState {
        model_public_key: attestation.model_public_key,
        attestation_report: attestation.attestation_report,
        verified_at: attestation.verified_at,
    };

    Ok(state
        .session_manager()
        .set_attested_model_state(&session.session_key, state_update)?)
}

async fn openai_chat_buffered_response(
    upstream: reqwest::Response,
    codec: E2eeCodec,
    proxy_instance_key: ProxyInstanceKey,
    fallback_model: String,
    metadata: ProxyMetadataHeaders,
) -> Result<Response, ProxyError> {
    let completion =
        buffer_openai_chat_completion(upstream, codec, proxy_instance_key, fallback_model).await?;
    let mut response = Json(completion).into_response();
    metadata.apply(response.headers_mut());
    Ok(response)
}

async fn openai_tool_emulated_chat_response(
    state: &AppState,
    request: &ChatCompletionRequest,
    tool_context: &ToolEmulationContext,
    codec: E2eeCodec,
    proxy_instance_key: ProxyInstanceKey,
    model_public_key: &str,
    metadata: ProxyMetadataHeaders,
) -> Result<Response, ProxyError> {
    info!(
        model = %request.model,
        max_retries = tool_context.max_retries(),
        "starting tool-emulated chat completion"
    );
    if request.stream {
        let controller_messages = [tool_context.controller_message()];
        let prepared = request.into_venice_e2ee_request_with_messages(
            &codec,
            model_public_key,
            &controller_messages,
            &[],
        )?;
        let upstream = state
            .venice_client()
            .create_chat_completion_stream(
                &prepared.upstream,
                proxy_instance_key.public_key_hex(),
                model_public_key,
            )
            .await?;

        return Ok(openai_tool_emulated_chat_sse_response(
            upstream,
            tool_context.clone(),
            codec,
            proxy_instance_key,
            request.model.clone(),
            request.stream_options.include_usage.unwrap_or(false),
            metadata,
        ));
    }

    let mut retries = 0;
    let mut correction: Option<(String, String)> = None;

    loop {
        let controller_messages = [tool_context.controller_message()];
        let correction_messages: Vec<_> = correction
            .as_ref()
            .map(|(validation_error, invalid_output)| {
                tool_context.correction_message(validation_error, invalid_output)
            })
            .into_iter()
            .collect();
        let prepared = request.into_venice_e2ee_request_with_messages(
            &codec,
            model_public_key,
            &controller_messages,
            &correction_messages,
        )?;
        let upstream = state
            .venice_client()
            .create_chat_completion_stream(
                &prepared.upstream,
                proxy_instance_key.public_key_hex(),
                model_public_key,
            )
            .await?;

        let completion = match tokio::time::timeout(
            tool_context.marker_timeout(),
            buffer_openai_chat_completion(
                upstream,
                codec.clone(),
                proxy_instance_key.clone(),
                request.model.clone(),
            ),
        )
        .await
        {
            Ok(completion) => completion?,
            Err(_) => {
                let validation_error = format!(
                    "tool call marker did not close within {} ms",
                    tool_context.config().tool_call_marker_timeout_ms
                );
                if retries >= tool_context.max_retries() {
                    return Err(ProxyError::ToolCallRetryExhausted {
                        max_retries: tool_context.max_retries(),
                        last_validation_error: validation_error,
                    });
                }
                warn!(
                    model = %request.model,
                    retry = retries + 1,
                    max_retries = tool_context.max_retries(),
                    "tool call marker timed out; retrying with correction"
                );
                retries += 1;
                correction = Some((validation_error, String::new()));
                continue;
            }
        };
        let assistant_content = completion
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .unwrap_or_default();

        match tool_context.classify_assistant_output(assistant_content) {
            ToolOutputClassification::NormalText => {
                info!(model = %request.model, retries, "tool emulation produced normal text");
                let mut metadata = metadata.clone();
                if retries > 0 {
                    metadata.tool_retries = Some(retries);
                }
                return Ok(if request.stream {
                    openai_chat_sse_response_from_completion(
                        completion,
                        request.stream_options.include_usage.unwrap_or(false),
                        metadata,
                    )
                } else {
                    let mut response = Json(completion).into_response();
                    metadata.apply(response.headers_mut());
                    response
                });
            }
            ToolOutputClassification::ToolCall(tool_call) => {
                info!(
                    model = %request.model,
                    tool_name = %tool_call.name,
                    retries,
                    "tool emulation produced tool call"
                );
                let mut metadata = metadata.clone();
                if retries > 0 {
                    metadata.tool_retries = Some(retries);
                }
                return Ok(if request.stream {
                    openai_tool_call_sse_response(completion, tool_call, metadata)
                } else {
                    let body = openai_tool_call_completion(completion, tool_call);
                    let mut response = Json(body).into_response();
                    metadata.apply(response.headers_mut());
                    response
                });
            }
            ToolOutputClassification::InvalidToolCall {
                error,
                invalid_output,
            } => {
                if retries >= tool_context.max_retries() {
                    warn!(
                        model = %request.model,
                        max_retries = tool_context.max_retries(),
                        validation_error = %error,
                        "tool call validation failed and retries were exhausted"
                    );
                    return Err(ProxyError::ToolCallRetryExhausted {
                        max_retries: tool_context.max_retries(),
                        last_validation_error: error.to_string(),
                    });
                }
                warn!(
                    model = %request.model,
                    retry = retries + 1,
                    max_retries = tool_context.max_retries(),
                    validation_error = %error,
                    "tool call validation failed; retrying with correction"
                );
                retries += 1;
                correction = Some((error.to_string(), invalid_output));
            }
        }
    }
}

fn openai_chat_sse_response_from_completion(
    completion: Value,
    include_usage_requested: bool,
    metadata: ProxyMetadataHeaders,
) -> Response {
    let mut events = Vec::new();
    let choice = completion
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .cloned()
        .unwrap_or(Value::Null);
    let index = choice.get("index").and_then(Value::as_u64).unwrap_or(0);
    let content = choice
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !content.is_empty() {
        events.push(openai_chunk_from_completion(
            &completion,
            index,
            json!({"role": "assistant", "content": content}),
            Value::Null,
        ));
    }
    let finish_reason = choice
        .get("finish_reason")
        .cloned()
        .unwrap_or_else(|| Value::String("stop".to_owned()));
    events.push(openai_chunk_from_completion(
        &completion,
        index,
        json!({}),
        finish_reason,
    ));
    if include_usage_requested
        && let Some(usage) = completion.get("usage")
        && !usage.is_null()
    {
        events.push(json!({
            "id": string_field(&completion, "id").unwrap_or("chatcmpl-local"),
            "object": "chat.completion.chunk",
            "created": integer_field(&completion, "created").unwrap_or_else(unix_timestamp_now),
            "model": string_field(&completion, "model").unwrap_or("unknown"),
            "choices": [],
            "usage": usage,
        }));
    }

    sse_response_from_json_events(events, metadata)
}

fn openai_tool_call_sse_response(
    completion: Value,
    tool_call: ValidatedToolCall,
    metadata: ProxyMetadataHeaders,
) -> Response {
    let choice = completion
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .cloned()
        .unwrap_or(Value::Null);
    let index = choice.get("index").and_then(Value::as_u64).unwrap_or(0);
    let events = vec![
        openai_chunk_from_completion(
            &completion,
            index,
            json!({
                "role": "assistant",
                "tool_calls": [tool_call.to_openai_streaming_value()],
            }),
            Value::Null,
        ),
        openai_chunk_from_completion(
            &completion,
            index,
            json!({}),
            Value::String("tool_calls".to_owned()),
        ),
    ];

    sse_response_from_json_events(events, metadata)
}

fn sse_response_from_json_events(events: Vec<Value>, metadata: ProxyMetadataHeaders) -> Response {
    let stream = async_stream::stream! {
        for event in events {
            yield Ok::<Event, io::Error>(Event::default().data(event.to_string()));
        }
        yield Ok::<Event, io::Error>(Event::default().data("[DONE]"));
    };
    let mut response = Sse::new(stream).into_response();
    metadata.apply(response.headers_mut());
    response
}

fn openai_tool_call_completion(completion: Value, tool_call: ValidatedToolCall) -> Value {
    let choice = completion
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .cloned()
        .unwrap_or(Value::Null);
    let index = choice.get("index").and_then(Value::as_u64).unwrap_or(0);

    json!({
        "id": string_field(&completion, "id").unwrap_or("chatcmpl-local"),
        "object": "chat.completion",
        "created": integer_field(&completion, "created").unwrap_or_else(unix_timestamp_now),
        "model": string_field(&completion, "model").unwrap_or("unknown"),
        "choices": [{
            "index": index,
            "message": {
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [tool_call.to_openai_value()],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": completion.get("usage").cloned().unwrap_or(Value::Null),
    })
}

fn openai_chunk_from_completion(
    completion: &Value,
    index: u64,
    delta: Value,
    finish_reason: Value,
) -> Value {
    json!({
        "id": string_field(completion, "id").unwrap_or("chatcmpl-local"),
        "object": "chat.completion.chunk",
        "created": integer_field(completion, "created").unwrap_or_else(unix_timestamp_now),
        "model": string_field(completion, "model").unwrap_or("unknown"),
        "choices": [{
            "index": index,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    })
}

async fn buffer_openai_chat_completion(
    mut upstream: reqwest::Response,
    codec: E2eeCodec,
    proxy_instance_key: ProxyInstanceKey,
    fallback_model: String,
) -> Result<Value, ChatStreamError> {
    info!(model = %fallback_model, "buffering upstream chat stream");
    let mut parser = SseEventParser::default();
    let mut transformer =
        OpenAiChatCompletionBuffer::new(codec, proxy_instance_key, fallback_model.clone());
    let mut upstream_done = false;
    let mut chunk_count = 0_u64;
    let mut event_count = 0_u64;

    while let Some(chunk) = upstream
        .chunk()
        .await
        .map_err(ChatStreamError::upstream_stream)?
    {
        chunk_count += 1;
        let chunk = std::str::from_utf8(&chunk).map_err(ChatStreamError::invalid_utf8)?;
        let events = parser.push(chunk)?;
        event_count += events.len() as u64;
        debug!(
            model = %fallback_model,
            chunk_count,
            parsed_events = events.len(),
            total_events = event_count,
            "parsed buffered upstream SSE chunk"
        );

        for event in events {
            if transformer.handle_event(event)? {
                upstream_done = true;
                break;
            }
        }

        if upstream_done {
            break;
        }
    }

    if !upstream_done {
        warn!(
            model = %fallback_model,
            chunk_count,
            event_count,
            "buffered upstream stream ended before DONE"
        );
        parser.finish()?;
        return Err(ChatStreamError::malformed_event(
            "upstream stream ended before data: [DONE]",
        ));
    }

    let completion = transformer.into_response()?;
    info!(
        model = %fallback_model,
        chunk_count,
        event_count,
        "buffered upstream chat stream transformed"
    );
    Ok(completion)
}

fn openai_chat_sse_response(
    upstream: reqwest::Response,
    codec: E2eeCodec,
    proxy_instance_key: ProxyInstanceKey,
    fallback_model: String,
    include_usage_requested: bool,
    metadata: ProxyMetadataHeaders,
) -> Response {
    let stream = openai_chat_event_stream(
        upstream,
        codec,
        proxy_instance_key,
        fallback_model,
        include_usage_requested,
    );
    let mut response = Sse::new(stream).into_response();
    metadata.apply(response.headers_mut());
    response
}

fn openai_chat_event_stream(
    mut upstream: reqwest::Response,
    codec: E2eeCodec,
    proxy_instance_key: ProxyInstanceKey,
    fallback_model: String,
    include_usage_requested: bool,
) -> impl futures_core::Stream<Item = Result<Event, axum::BoxError>> {
    async_stream::try_stream! {
        info!(
            model = %fallback_model,
            include_usage_requested,
            "starting upstream chat SSE transformation"
        );
        let mut parser = SseEventParser::default();
        let mut transformer = OpenAiChatStreamTransformer::new(
            codec,
            proxy_instance_key,
            fallback_model.clone(),
            include_usage_requested,
        );
        let mut upstream_done = false;
        let mut chunk_count = 0_u64;
        let mut event_count = 0_u64;
        let mut output_count = 0_u64;

        while let Some(chunk) = upstream
            .chunk()
            .await
            .map_err(ChatStreamError::upstream_stream)
            .map_err(box_chat_stream_error)?
        {
            chunk_count += 1;
            let chunk = std::str::from_utf8(&chunk)
                .map_err(ChatStreamError::invalid_utf8)
                .map_err(box_chat_stream_error)?;
            let events = parser.push(chunk).map_err(box_chat_stream_error)?;
            event_count += events.len() as u64;
            debug!(
                model = %fallback_model,
                chunk_count,
                parsed_events = events.len(),
                total_events = event_count,
                "parsed streaming upstream SSE chunk"
            );

            for event in events {
                let outputs = transformer.handle_event(event).map_err(box_chat_stream_error)?;
                output_count += outputs.len() as u64;
                debug!(
                    model = %fallback_model,
                    emitted_outputs = outputs.len(),
                    total_outputs = output_count,
                    "transformed streaming upstream SSE event"
                );

                for output in outputs {
                    match output {
                        StreamOutput::Json(value) => yield Event::default().data(value.to_string()),
                        StreamOutput::Done => {
                            upstream_done = true;
                            info!(
                                model = %fallback_model,
                                chunk_count,
                                event_count,
                                output_count,
                                "completed upstream chat SSE transformation"
                            );
                            yield Event::default().data("[DONE]");
                            break;
                        }
                    }
                }

                if upstream_done {
                    break;
                }
            }

            if upstream_done {
                break;
            }
        }

        if !upstream_done {
            warn!(
                model = %fallback_model,
                chunk_count,
                event_count,
                output_count,
                "streaming upstream stream ended before DONE"
            );
            parser.finish().map_err(box_chat_stream_error)?;
            Err::<(), axum::BoxError>(box_chat_stream_error(ChatStreamError::malformed_event(
                "upstream stream ended before data: [DONE]",
            )))?;
        }
    }
}

fn openai_tool_emulated_chat_sse_response(
    upstream: reqwest::Response,
    tool_context: ToolEmulationContext,
    codec: E2eeCodec,
    proxy_instance_key: ProxyInstanceKey,
    fallback_model: String,
    include_usage_requested: bool,
    metadata: ProxyMetadataHeaders,
) -> Response {
    let stream = openai_tool_emulated_chat_event_stream(
        upstream,
        tool_context,
        codec,
        proxy_instance_key,
        fallback_model,
        include_usage_requested,
    );
    let mut response = Sse::new(stream).into_response();
    metadata.apply(response.headers_mut());
    response
}

fn openai_tool_emulated_chat_event_stream(
    mut upstream: reqwest::Response,
    tool_context: ToolEmulationContext,
    codec: E2eeCodec,
    proxy_instance_key: ProxyInstanceKey,
    fallback_model: String,
    include_usage_requested: bool,
) -> impl futures_core::Stream<Item = Result<Event, axum::BoxError>> {
    async_stream::try_stream! {
        info!(
            model = %fallback_model,
            include_usage_requested,
            "starting tool-emulated upstream chat SSE transformation"
        );
        let mut parser = SseEventParser::default();
        let mut transformer = OpenAiToolEmulatedChatStreamTransformer::new(
            tool_context,
            codec,
            proxy_instance_key,
            fallback_model.clone(),
            include_usage_requested,
        );
        let mut upstream_done = false;
        let mut chunk_count = 0_u64;
        let mut event_count = 0_u64;
        let mut output_count = 0_u64;

        while let Some(chunk) = upstream
            .chunk()
            .await
            .map_err(ChatStreamError::upstream_stream)
            .map_err(box_chat_stream_error)?
        {
            chunk_count += 1;
            let chunk = std::str::from_utf8(&chunk)
                .map_err(ChatStreamError::invalid_utf8)
                .map_err(box_chat_stream_error)?;
            let events = parser.push(chunk).map_err(box_chat_stream_error)?;
            event_count += events.len() as u64;
            debug!(
                model = %fallback_model,
                chunk_count,
                parsed_events = events.len(),
                total_events = event_count,
                "parsed tool-emulated upstream SSE chunk"
            );

            for event in events {
                let outputs = transformer.handle_event(event).map_err(box_chat_stream_error)?;
                output_count += outputs.len() as u64;
                debug!(
                    model = %fallback_model,
                    emitted_outputs = outputs.len(),
                    total_outputs = output_count,
                    "transformed tool-emulated upstream SSE event"
                );

                for output in outputs {
                    match output {
                        StreamOutput::Json(value) => yield Event::default().data(value.to_string()),
                        StreamOutput::Done => {
                            upstream_done = true;
                            info!(
                                model = %fallback_model,
                                chunk_count,
                                event_count,
                                output_count,
                                "completed tool-emulated upstream chat SSE transformation"
                            );
                            yield Event::default().data("[DONE]");
                            break;
                        }
                    }
                }

                if upstream_done {
                    break;
                }
            }

            if upstream_done {
                break;
            }
        }

        if !upstream_done {
            warn!(
                model = %fallback_model,
                chunk_count,
                event_count,
                output_count,
                "tool-emulated upstream stream ended before DONE"
            );
            parser.finish().map_err(box_chat_stream_error)?;
            Err::<(), axum::BoxError>(box_chat_stream_error(ChatStreamError::malformed_event(
                "upstream stream ended before data: [DONE]",
            )))?;
        }
    }
}

fn box_chat_stream_error(error: ChatStreamError) -> axum::BoxError {
    error!(error = %error, "chat stream transformation failed");
    Box::new(error)
}

#[derive(Debug, Default)]
struct SseEventParser {
    buffer: String,
}

impl SseEventParser {
    fn push(&mut self, chunk: &str) -> Result<Vec<RawSseEvent>, ChatStreamError> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        while let Some((boundary_start, boundary_len)) = sse_event_boundary(&self.buffer) {
            let raw = self.buffer[..boundary_start].to_owned();
            self.buffer.drain(..boundary_start + boundary_len);
            if let Some(event) = parse_sse_event(&raw)? {
                events.push(event);
            }
        }

        debug!(
            chunk_bytes = chunk.len(),
            buffered_bytes = self.buffer.len(),
            parsed_events = events.len(),
            "SSE parser processed upstream chunk"
        );
        Ok(events)
    }

    fn finish(&self) -> Result<(), ChatStreamError> {
        if self.buffer.trim().is_empty() {
            Ok(())
        } else {
            warn!(
                buffered_bytes = self.buffer.len(),
                "upstream SSE stream ended with incomplete event"
            );
            Err(ChatStreamError::malformed_event(
                "upstream stream ended with an incomplete SSE event",
            ))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RawSseEvent {
    event: Option<String>,
    data: String,
}

struct OpenAiChatCompletionBuffer {
    codec: E2eeCodec,
    proxy_instance_key: ProxyInstanceKey,
    fallback_id: String,
    fallback_created: i64,
    fallback_model: String,
    id: Option<String>,
    created: Option<i64>,
    model: Option<String>,
    choice_index: Option<u64>,
    saw_encrypted_content: bool,
    content: String,
    finish_reason: Option<Value>,
    usage: Option<Value>,
}

impl OpenAiChatCompletionBuffer {
    fn new(codec: E2eeCodec, proxy_instance_key: ProxyInstanceKey, fallback_model: String) -> Self {
        Self {
            codec,
            proxy_instance_key,
            fallback_id: format!("chatcmpl-local-{}", uuid::Uuid::new_v4()),
            fallback_created: unix_timestamp_now(),
            fallback_model,
            id: None,
            created: None,
            model: None,
            choice_index: None,
            saw_encrypted_content: false,
            content: String::new(),
            finish_reason: None,
            usage: None,
        }
    }

    fn handle_event(&mut self, event: RawSseEvent) -> Result<bool, ChatStreamError> {
        let event_type = event.event.as_deref().unwrap_or("message");
        let is_done = event.data.trim() == "[DONE]";
        debug!(event_type, is_done, "buffering upstream SSE event");

        if event.event.as_deref() == Some("error") {
            warn!("upstream SSE error event while buffering response");
            return Err(ChatStreamError::upstream_event(event.data));
        }

        if is_done {
            info!("received upstream DONE while buffering response");
            if !self.saw_encrypted_content {
                self.codec
                    .decrypt_response_content(None, self.proxy_instance_key.private_key())
                    .map_err(ChatStreamError::decryption)?;
            }
            if self.finish_reason.is_none() {
                self.finish_reason = Some(Value::String("stop".to_owned()));
            }
            return Ok(true);
        }

        debug!("parsing buffered upstream chat JSON chunk");
        let value: Value =
            serde_json::from_str(&event.data).map_err(ChatStreamError::json_event)?;
        if let Some(error) = value.get("error") {
            warn!("upstream JSON error chunk while buffering response");
            return Err(ChatStreamError::upstream_event(error.to_string()));
        }

        self.record_metadata(&value);

        let Some(choices) = value.get("choices").and_then(Value::as_array) else {
            warn!("buffered upstream chat chunk is missing choices array");
            return Err(ChatStreamError::malformed_event(
                "upstream chat chunk is missing choices array",
            ));
        };
        debug!(
            choice_count = choices.len(),
            "parsed buffered upstream chat chunk"
        );

        if choices.is_empty() {
            return self.handle_usage_chunk(&value).map(|()| false);
        }
        if choices.len() != 1 {
            warn!(
                choice_count = choices.len(),
                "unexpected buffered upstream choice count"
            );
            return Err(ChatStreamError::malformed_event(format!(
                "expected exactly one upstream choice, got {}",
                choices.len(),
            )));
        }

        self.handle_choice_chunk(&choices[0])?;
        Ok(false)
    }

    fn handle_usage_chunk(&mut self, value: &Value) -> Result<(), ChatStreamError> {
        let Some(usage) = value.get("usage") else {
            warn!("buffered upstream chunk has no choices and no usage");
            return Err(ChatStreamError::malformed_event(
                "upstream chunk has no choices and no usage",
            ));
        };

        info!("buffered upstream usage chunk");
        self.usage = Some(usage.clone());
        Ok(())
    }

    fn handle_choice_chunk(&mut self, choice: &Value) -> Result<(), ChatStreamError> {
        let choice = choice.as_object().ok_or_else(|| {
            ChatStreamError::malformed_event("upstream choice must be a JSON object")
        })?;
        let index = normalized_choice_index(choice.get("index"))?;
        match self.choice_index {
            Some(existing) if existing != index => {
                return Err(ChatStreamError::malformed_event(
                    "upstream choice index changed while buffering a completion",
                ));
            }
            None => self.choice_index = Some(index),
            Some(_) => {}
        }

        let finish_reason = normalized_finish_reason(choice.get("finish_reason"))?;
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        let content = encrypted_delta_content(delta)?;
        debug!(
            choice_index = index,
            has_encrypted_content = content.is_some(),
            has_finish_reason = !finish_reason.is_null(),
            "transforming buffered upstream choice chunk"
        );

        if let Some(content) = content {
            let decrypted = self
                .codec
                .decrypt_response_content(Some(content), self.proxy_instance_key.private_key())
                .map_err(ChatStreamError::decryption)?;
            self.saw_encrypted_content = true;
            debug!(
                choice_index = index,
                has_decrypted_content = decrypted.is_some(),
                "decrypted buffered upstream content chunk"
            );
            if let Some(content) = decrypted {
                self.content.push_str(&content);
            }
        }

        if !finish_reason.is_null() {
            self.finish_reason = Some(finish_reason);
        }

        Ok(())
    }

    fn record_metadata(&mut self, value: &Value) {
        if self.id.is_none()
            && let Some(id) = string_field(value, "id")
        {
            self.id = Some(id.to_owned());
        }
        if self.created.is_none()
            && let Some(created) = integer_field(value, "created")
        {
            self.created = Some(created);
        }
        if self.model.is_none()
            && let Some(model) = string_field(value, "model")
        {
            self.model = Some(model.to_owned());
        }
    }

    fn into_response(self) -> Result<Value, ChatStreamError> {
        Ok(json!({
            "id": self.id.unwrap_or(self.fallback_id),
            "object": "chat.completion",
            "created": self.created.unwrap_or(self.fallback_created),
            "model": self.model.unwrap_or(self.fallback_model),
            "choices": [{
                "index": self.choice_index.unwrap_or(0),
                "message": {
                    "role": "assistant",
                    "content": self.content,
                },
                "finish_reason": self.finish_reason.unwrap_or_else(|| Value::String("stop".to_owned())),
            }],
            "usage": self.usage.unwrap_or(Value::Null),
        }))
    }
}

fn sse_event_boundary(buffer: &str) -> Option<(usize, usize)> {
    ["\r\n\r\n", "\n\n", "\r\r"]
        .into_iter()
        .filter_map(|delimiter| buffer.find(delimiter).map(|index| (index, delimiter.len())))
        .min_by_key(|(index, _)| *index)
}

fn parse_sse_event(raw: &str) -> Result<Option<RawSseEvent>, ChatStreamError> {
    let mut event = None;
    let mut data_lines = Vec::new();
    let mut saw_non_comment_field = false;

    for line in raw.lines() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        saw_non_comment_field = true;
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => event = Some(value.to_owned()),
            "data" => data_lines.push(value.to_owned()),
            "id" | "retry" => {}
            other => {
                warn!(field = other, "unsupported upstream SSE field");
                return Err(ChatStreamError::malformed_event(format!(
                    "unsupported upstream SSE field {other:?}",
                )));
            }
        }
    }

    if data_lines.is_empty() {
        return if saw_non_comment_field {
            warn!("upstream SSE event did not contain a data field");
            Err(ChatStreamError::malformed_event(
                "upstream SSE event did not contain a data field",
            ))
        } else {
            debug!("ignored upstream SSE comment or heartbeat event");
            Ok(None)
        };
    }

    debug!(
        event_type = event.as_deref().unwrap_or("message"),
        data_line_count = data_lines.len(),
        "parsed upstream SSE event"
    );

    Ok(Some(RawSseEvent {
        event,
        data: data_lines.join("\n"),
    }))
}

struct OpenAiChatStreamTransformer {
    codec: E2eeCodec,
    proxy_instance_key: ProxyInstanceKey,
    fallback_id: String,
    fallback_created: i64,
    fallback_model: String,
    include_usage_requested: bool,
    sent_role: bool,
    sent_final_finish: bool,
}

impl OpenAiChatStreamTransformer {
    fn new(
        codec: E2eeCodec,
        proxy_instance_key: ProxyInstanceKey,
        fallback_model: String,
        include_usage_requested: bool,
    ) -> Self {
        Self {
            codec,
            proxy_instance_key,
            fallback_id: format!("chatcmpl-local-{}", uuid::Uuid::new_v4()),
            fallback_created: unix_timestamp_now(),
            fallback_model,
            include_usage_requested,
            sent_role: false,
            sent_final_finish: false,
        }
    }

    fn handle_event(&mut self, event: RawSseEvent) -> Result<Vec<StreamOutput>, ChatStreamError> {
        let event_type = event.event.as_deref().unwrap_or("message");
        let is_done = event.data.trim() == "[DONE]";
        debug!(
            event_type,
            is_done, "transforming streaming upstream SSE event"
        );

        if event.event.as_deref() == Some("error") {
            warn!("upstream SSE error event while streaming response");
            return Err(ChatStreamError::upstream_event(event.data));
        }

        if is_done {
            info!("received upstream DONE while streaming response");
            let mut output = Vec::new();
            if !self.sent_final_finish {
                debug!("synthesizing final streaming finish chunk before DONE");
                output.push(StreamOutput::Json(self.finish_chunk(None)?));
                self.sent_final_finish = true;
            }
            output.push(StreamOutput::Done);
            return Ok(output);
        }

        debug!("parsing streaming upstream chat JSON chunk");
        let value: Value =
            serde_json::from_str(&event.data).map_err(ChatStreamError::json_event)?;
        if let Some(error) = value.get("error") {
            warn!("upstream JSON error chunk while streaming response");
            return Err(ChatStreamError::upstream_event(error.to_string()));
        }

        let Some(choices) = value.get("choices").and_then(Value::as_array) else {
            warn!("streaming upstream chat chunk is missing choices array");
            return Err(ChatStreamError::malformed_event(
                "upstream chat chunk is missing choices array",
            ));
        };
        debug!(
            choice_count = choices.len(),
            "parsed streaming upstream chat chunk"
        );

        if choices.is_empty() {
            return self.handle_usage_chunk(&value);
        }
        if choices.len() != 1 {
            warn!(
                choice_count = choices.len(),
                "unexpected streaming upstream choice count"
            );
            return Err(ChatStreamError::malformed_event(format!(
                "expected exactly one upstream choice, got {}",
                choices.len(),
            )));
        }

        self.handle_choice_chunk(&value, &choices[0])
    }

    fn handle_choice_chunk(
        &mut self,
        value: &Value,
        choice: &Value,
    ) -> Result<Vec<StreamOutput>, ChatStreamError> {
        let choice = choice.as_object().ok_or_else(|| {
            ChatStreamError::malformed_event("upstream choice must be a JSON object")
        })?;
        let finish_reason = normalized_finish_reason(choice.get("finish_reason"))?;
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        let content = encrypted_delta_content(delta)?;
        debug!(
            has_encrypted_content = content.is_some(),
            has_finish_reason = !finish_reason.is_null(),
            "transforming streaming upstream choice chunk"
        );

        let mut output = Vec::new();
        if content.is_none() {
            if !finish_reason.is_null() {
                output.push(StreamOutput::Json(self.chunk_with_choice(
                    value,
                    choice.get("index"),
                    json!({}),
                    finish_reason,
                )?));
                self.sent_final_finish = true;
            }
            return Ok(output);
        }

        let decrypted = self
            .codec
            .decrypt_response_content(content, self.proxy_instance_key.private_key())
            .map_err(ChatStreamError::decryption)?;
        debug!(
            has_decrypted_content = decrypted.is_some(),
            "decrypted streaming upstream content chunk"
        );

        if let Some(content) = decrypted {
            let mut delta = serde_json::Map::new();
            if !self.sent_role {
                delta.insert("role".to_owned(), Value::String("assistant".to_owned()));
                self.sent_role = true;
            }
            delta.insert("content".to_owned(), Value::String(content));

            let final_finish = !finish_reason.is_null();
            let content_finish_reason = if final_finish {
                Value::Null
            } else {
                finish_reason.clone()
            };
            output.push(StreamOutput::Json(self.chunk_with_choice(
                value,
                choice.get("index"),
                Value::Object(delta),
                content_finish_reason,
            )?));
            if final_finish {
                output.push(StreamOutput::Json(self.chunk_with_choice(
                    value,
                    choice.get("index"),
                    json!({}),
                    finish_reason,
                )?));
                self.sent_final_finish = true;
            }
            return Ok(output);
        }

        Ok(output)
    }

    fn handle_usage_chunk(&self, value: &Value) -> Result<Vec<StreamOutput>, ChatStreamError> {
        let Some(usage) = value.get("usage") else {
            warn!("streaming upstream chunk has no choices and no usage");
            return Err(ChatStreamError::malformed_event(
                "upstream chunk has no choices and no usage",
            ));
        };

        // If a client requests include_usage but Venice omits a usage event,
        // this streaming path omits usage rather than synthesizing unverifiable
        // token counts.
        if !self.include_usage_requested {
            debug!("streaming upstream usage chunk ignored because client did not request usage");
            return Ok(Vec::new());
        }

        info!("streaming upstream usage chunk forwarded");
        Ok(vec![StreamOutput::Json(json!({
            "id": string_field(value, "id").unwrap_or(&self.fallback_id),
            "object": string_field(value, "object").unwrap_or("chat.completion.chunk"),
            "created": integer_field(value, "created").unwrap_or(self.fallback_created),
            "model": string_field(value, "model").unwrap_or(&self.fallback_model),
            "choices": [],
            "usage": usage,
        }))])
    }

    fn finish_chunk(&self, upstream: Option<&Value>) -> Result<Value, ChatStreamError> {
        self.chunk_with_choice(
            upstream.unwrap_or(&Value::Null),
            None,
            json!({}),
            Value::String("stop".to_owned()),
        )
    }

    fn chunk_with_choice(
        &self,
        upstream: &Value,
        index: Option<&Value>,
        delta: Value,
        finish_reason: Value,
    ) -> Result<Value, ChatStreamError> {
        let index = normalized_choice_index(index)?;

        Ok(json!({
            "id": string_field(upstream, "id").unwrap_or(&self.fallback_id),
            "object": string_field(upstream, "object").unwrap_or("chat.completion.chunk"),
            "created": integer_field(upstream, "created").unwrap_or(self.fallback_created),
            "model": string_field(upstream, "model").unwrap_or(&self.fallback_model),
            "choices": [{
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }))
    }
}

struct OpenAiToolEmulatedChatStreamTransformer {
    tool_context: ToolEmulationContext,
    codec: E2eeCodec,
    proxy_instance_key: ProxyInstanceKey,
    fallback_id: String,
    fallback_created: i64,
    fallback_model: String,
    include_usage_requested: bool,
    sent_role: bool,
    sent_final_finish: bool,
    text_buffer: String,
    tool_buffer: Option<String>,
}

impl OpenAiToolEmulatedChatStreamTransformer {
    fn new(
        tool_context: ToolEmulationContext,
        codec: E2eeCodec,
        proxy_instance_key: ProxyInstanceKey,
        fallback_model: String,
        include_usage_requested: bool,
    ) -> Self {
        Self {
            tool_context,
            codec,
            proxy_instance_key,
            fallback_id: format!("chatcmpl-local-{}", uuid::Uuid::new_v4()),
            fallback_created: unix_timestamp_now(),
            fallback_model,
            include_usage_requested,
            sent_role: false,
            sent_final_finish: false,
            text_buffer: String::new(),
            tool_buffer: None,
        }
    }

    fn handle_event(&mut self, event: RawSseEvent) -> Result<Vec<StreamOutput>, ChatStreamError> {
        let event_type = event.event.as_deref().unwrap_or("message");
        let is_done = event.data.trim() == "[DONE]";
        debug!(
            event_type,
            is_done, "transforming tool-emulated streaming upstream SSE event"
        );

        if event.event.as_deref() == Some("error") {
            warn!("upstream SSE error event while streaming tool-emulated response");
            return Err(ChatStreamError::upstream_event(event.data));
        }

        if is_done {
            info!("received upstream DONE while streaming tool-emulated response");
            return self.finish_stream(None);
        }

        let value: Value =
            serde_json::from_str(&event.data).map_err(ChatStreamError::json_event)?;
        if let Some(error) = value.get("error") {
            warn!("upstream JSON error chunk while streaming tool-emulated response");
            return Err(ChatStreamError::upstream_event(error.to_string()));
        }

        let Some(choices) = value.get("choices").and_then(Value::as_array) else {
            warn!("tool-emulated upstream chat chunk is missing choices array");
            return Err(ChatStreamError::malformed_event(
                "upstream chat chunk is missing choices array",
            ));
        };

        if choices.is_empty() {
            return self.handle_usage_chunk(&value);
        }
        if choices.len() != 1 {
            warn!(
                choice_count = choices.len(),
                "unexpected tool-emulated upstream choice count"
            );
            return Err(ChatStreamError::malformed_event(format!(
                "expected exactly one upstream choice, got {}",
                choices.len(),
            )));
        }

        self.handle_choice_chunk(&value, &choices[0])
    }

    fn handle_choice_chunk(
        &mut self,
        value: &Value,
        choice: &Value,
    ) -> Result<Vec<StreamOutput>, ChatStreamError> {
        let choice = choice.as_object().ok_or_else(|| {
            ChatStreamError::malformed_event("upstream choice must be a JSON object")
        })?;
        let index = normalized_choice_index(choice.get("index"))?;
        let finish_reason = normalized_finish_reason(choice.get("finish_reason"))?;
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        let content = encrypted_delta_content(delta)?;

        let mut output = Vec::new();
        if let Some(content) = content {
            let decrypted = self
                .codec
                .decrypt_response_content(Some(content), self.proxy_instance_key.private_key())
                .map_err(ChatStreamError::decryption)?;
            if let Some(content) = decrypted {
                output.extend(self.handle_decrypted_content(value, index, &content)?);
            }
        }

        if !finish_reason.is_null() && !self.sent_final_finish {
            output.extend(self.flush_all_text(value, index)?);
            if !self.sent_final_finish {
                output.push(StreamOutput::Json(self.chunk_with_choice(
                    value,
                    index,
                    json!({}),
                    finish_reason,
                )?));
                self.sent_final_finish = true;
            }
        }

        Ok(output)
    }

    fn handle_decrypted_content(
        &mut self,
        upstream: &Value,
        index: u64,
        content: &str,
    ) -> Result<Vec<StreamOutput>, ChatStreamError> {
        if self.sent_final_finish {
            return Ok(Vec::new());
        }

        if self.tool_buffer.is_some() {
            if let Some(buffer) = self.tool_buffer.as_mut() {
                buffer.push_str(content);
            }
            return self.try_emit_tool_call(upstream, index);
        }

        self.text_buffer.push_str(content);
        let marker_start = self.tool_context.config().marker_start.clone();
        if let Some(marker_index) = self.text_buffer.find(&marker_start) {
            let before_marker = self.text_buffer[..marker_index].to_owned();
            let marker_and_after = self.text_buffer[marker_index..].to_owned();
            self.text_buffer.clear();

            let mut output = self.emit_text(upstream, index, before_marker)?;
            self.tool_buffer = Some(marker_and_after);
            output.extend(self.try_emit_tool_call(upstream, index)?);
            return Ok(output);
        }

        self.flush_safe_text(upstream, index)
    }

    fn try_emit_tool_call(
        &mut self,
        upstream: &Value,
        index: u64,
    ) -> Result<Vec<StreamOutput>, ChatStreamError> {
        let Some(buffer) = self.tool_buffer.as_ref() else {
            return Ok(Vec::new());
        };
        if buffer.len() > self.tool_context.config().tool_call_max_bytes {
            return Err(ChatStreamError::malformed_event(format!(
                "tool call marker exceeded max size of {} bytes",
                self.tool_context.config().tool_call_max_bytes
            )));
        }

        let marker_end = self.tool_context.config().marker_end.clone();
        let Some(end_start) = buffer.find(&marker_end) else {
            return Ok(Vec::new());
        };
        let end = end_start + marker_end.len();
        let marker = buffer[..end].to_owned();
        let trailing = buffer[end..].trim().to_owned();
        if !trailing.is_empty() {
            warn!("discarding text after streamed tool call marker");
        }

        let tool_call = self
            .tool_context
            .validate_marker(&marker)
            .map_err(|error| {
                ChatStreamError::malformed_event(format!("tool call marker is invalid: {error}"))
            })?;
        self.tool_buffer = None;
        self.text_buffer.clear();
        self.emit_tool_call(upstream, index, tool_call)
    }

    fn flush_safe_text(
        &mut self,
        upstream: &Value,
        index: u64,
    ) -> Result<Vec<StreamOutput>, ChatStreamError> {
        let marker_start = self.tool_context.config().marker_start.as_str();
        let keep_len = marker_prefix_suffix_len(&self.text_buffer, marker_start);
        let flush_len = self.text_buffer.len().saturating_sub(keep_len);
        if flush_len == 0 {
            return Ok(Vec::new());
        }

        let text = self.text_buffer[..flush_len].to_owned();
        self.text_buffer.drain(..flush_len);
        self.emit_text(upstream, index, text)
    }

    fn flush_all_text(
        &mut self,
        upstream: &Value,
        index: u64,
    ) -> Result<Vec<StreamOutput>, ChatStreamError> {
        if self.tool_buffer.is_some() {
            return Err(ChatStreamError::malformed_event(
                "upstream stream ended with an incomplete tool call marker",
            ));
        }
        let text = std::mem::take(&mut self.text_buffer);
        self.emit_text(upstream, index, text)
    }

    fn emit_text(
        &mut self,
        upstream: &Value,
        index: u64,
        text: String,
    ) -> Result<Vec<StreamOutput>, ChatStreamError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut delta = serde_json::Map::new();
        if !self.sent_role {
            delta.insert("role".to_owned(), Value::String("assistant".to_owned()));
            self.sent_role = true;
        }
        delta.insert("content".to_owned(), Value::String(text));

        Ok(vec![StreamOutput::Json(self.chunk_with_choice(
            upstream,
            index,
            Value::Object(delta),
            Value::Null,
        )?)])
    }

    fn emit_tool_call(
        &mut self,
        upstream: &Value,
        index: u64,
        tool_call: ValidatedToolCall,
    ) -> Result<Vec<StreamOutput>, ChatStreamError> {
        let mut delta = serde_json::Map::new();
        if !self.sent_role {
            delta.insert("role".to_owned(), Value::String("assistant".to_owned()));
            self.sent_role = true;
        }
        delta.insert(
            "tool_calls".to_owned(),
            Value::Array(vec![tool_call.to_openai_streaming_value()]),
        );

        self.sent_final_finish = true;
        Ok(vec![
            StreamOutput::Json(self.chunk_with_choice(
                upstream,
                index,
                Value::Object(delta),
                Value::Null,
            )?),
            StreamOutput::Json(self.chunk_with_choice(
                upstream,
                index,
                json!({}),
                Value::String("tool_calls".to_owned()),
            )?),
        ])
    }

    fn handle_usage_chunk(&self, value: &Value) -> Result<Vec<StreamOutput>, ChatStreamError> {
        let Some(usage) = value.get("usage") else {
            warn!("tool-emulated upstream chunk has no choices and no usage");
            return Err(ChatStreamError::malformed_event(
                "upstream chunk has no choices and no usage",
            ));
        };

        if !self.include_usage_requested || self.sent_final_finish {
            return Ok(Vec::new());
        }

        Ok(vec![StreamOutput::Json(json!({
            "id": string_field(value, "id").unwrap_or(&self.fallback_id),
            "object": string_field(value, "object").unwrap_or("chat.completion.chunk"),
            "created": integer_field(value, "created").unwrap_or(self.fallback_created),
            "model": string_field(value, "model").unwrap_or(&self.fallback_model),
            "choices": [],
            "usage": usage,
        }))])
    }

    fn finish_stream(
        &mut self,
        upstream: Option<&Value>,
    ) -> Result<Vec<StreamOutput>, ChatStreamError> {
        let upstream = upstream.unwrap_or(&Value::Null);
        let mut output = self.flush_all_text(upstream, 0)?;
        if !self.sent_final_finish {
            output.push(StreamOutput::Json(self.chunk_with_choice(
                upstream,
                0,
                json!({}),
                Value::String("stop".to_owned()),
            )?));
            self.sent_final_finish = true;
        }
        output.push(StreamOutput::Done);
        Ok(output)
    }

    fn chunk_with_choice(
        &self,
        upstream: &Value,
        index: u64,
        delta: Value,
        finish_reason: Value,
    ) -> Result<Value, ChatStreamError> {
        Ok(json!({
            "id": string_field(upstream, "id").unwrap_or(&self.fallback_id),
            "object": string_field(upstream, "object").unwrap_or("chat.completion.chunk"),
            "created": integer_field(upstream, "created").unwrap_or(self.fallback_created),
            "model": string_field(upstream, "model").unwrap_or(&self.fallback_model),
            "choices": [{
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }))
    }
}

fn marker_prefix_suffix_len(buffer: &str, marker: &str) -> usize {
    if buffer.is_empty() || marker.is_empty() {
        return 0;
    }

    let mut best = 0;
    for (index, _) in buffer.char_indices() {
        let suffix = &buffer[index..];
        if suffix.len() < marker.len() && marker.starts_with(suffix) {
            best = best.max(suffix.len());
        }
    }
    best
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StreamOutput {
    Json(Value),
    Done,
}

fn normalized_choice_index(index: Option<&Value>) -> Result<u64, ChatStreamError> {
    match index {
        Some(Value::Number(number)) => number.as_u64().ok_or_else(|| {
            ChatStreamError::malformed_event("upstream choice index must be a non-negative integer")
        }),
        Some(_) => Err(ChatStreamError::malformed_event(
            "upstream choice index must be a non-negative integer",
        )),
        None => Ok(0),
    }
}

fn normalized_finish_reason(value: Option<&Value>) -> Result<Value, ChatStreamError> {
    match value {
        Some(Value::Null) | None => Ok(Value::Null),
        Some(Value::String(reason)) => Ok(Value::String(reason.clone())),
        Some(_) => Err(ChatStreamError::malformed_event(
            "upstream finish_reason must be a string or null",
        )),
    }
}

fn encrypted_delta_content(delta: &Value) -> Result<Option<&str>, ChatStreamError> {
    match delta.get("content") {
        Some(Value::Null) => {
            debug!("ignoring null upstream delta.content");
            Ok(None)
        }
        Some(Value::String(content)) if content.is_empty() => {
            debug!("ignoring empty upstream delta.content");
            Ok(None)
        }
        Some(Value::String(content)) => Ok(Some(content.as_str())),
        Some(_) => Err(ChatStreamError::malformed_event(
            "upstream delta.content must be a string or null",
        )),
        None => Ok(None),
    }
}

fn string_field<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    value.get(field).and_then(Value::as_str)
}

fn integer_field(value: &Value, field: &str) -> Option<i64> {
    value.get(field).and_then(Value::as_i64)
}

fn unix_timestamp_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

async fn method_not_allowed(method: Method, uri: Uri) -> ProxyError {
    ProxyError::MethodNotAllowed { method, uri }
}

async fn not_found(uri: Uri) -> ProxyError {
    ProxyError::NotFound { uri }
}

#[derive(Debug, Error)]
pub enum ChatStreamError {
    #[error("Venice upstream stream failed: {message}")]
    UpstreamStream { message: String },
    #[error("Venice upstream stream emitted an error event: {message}")]
    UpstreamEvent { message: String },
    #[error("Venice upstream stream event is malformed: {message}")]
    MalformedEvent { message: String },
    #[error("failed to decrypt Venice E2EE response chunk: {source}")]
    Decryption { source: E2eeCodecError },
}

impl ChatStreamError {
    fn upstream_stream(source: reqwest::Error) -> Self {
        Self::UpstreamStream {
            message: source.to_string(),
        }
    }

    fn upstream_event(message: impl Into<String>) -> Self {
        Self::UpstreamEvent {
            message: message.into(),
        }
    }

    fn malformed_event(message: impl Into<String>) -> Self {
        Self::MalformedEvent {
            message: message.into(),
        }
    }

    fn invalid_utf8(source: std::str::Utf8Error) -> Self {
        Self::MalformedEvent {
            message: format!("upstream SSE bytes are not valid UTF-8: {source}"),
        }
    }

    fn json_event(source: serde_json::Error) -> Self {
        Self::MalformedEvent {
            message: format!("upstream SSE data is not valid JSON: {source}"),
        }
    }

    fn decryption(source: E2eeCodecError) -> Self {
        match source {
            E2eeCodecError::MissingEncryptedContent
            | E2eeCodecError::MalformedEncryptedPayload { .. }
            | E2eeCodecError::AuthenticationFailed
            | E2eeCodecError::UnsupportedCodecShape { .. }
            | E2eeCodecError::InvalidPlaintextUtf8 => Self::Decryption { source },
            other => Self::Decryption { source: other },
        }
    }

    fn api_error_type(&self) -> &'static str {
        match self {
            Self::UpstreamStream { .. }
            | Self::UpstreamEvent { .. }
            | Self::MalformedEvent { .. } => "proxy_upstream_error",
            Self::Decryption { .. } => "proxy_e2ee_error",
        }
    }

    fn api_error_code(&self) -> &'static str {
        match self {
            Self::UpstreamStream { .. } => "upstream_stream_error",
            Self::UpstreamEvent { .. } => "upstream_stream_error",
            Self::MalformedEvent { .. } => "upstream_malformed_response",
            Self::Decryption { .. } => "e2ee_response_decryption_failed",
        }
    }
}

#[derive(Debug, Error)]
pub enum ProxyError {
    #[error(transparent)]
    Venice(#[from] VeniceClientError),
    #[error(transparent)]
    Attestation(#[from] AttestationError),
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error(transparent)]
    ChatRequest(#[from] ChatRequestError),
    #[error(transparent)]
    ChatConstruction(#[from] ChatConstructionError),
    #[error(transparent)]
    ChatStream(#[from] ChatStreamError),
    #[error("The model failed to produce a valid tool call after correction attempts.")]
    ToolCallRetryExhausted {
        max_retries: u32,
        last_validation_error: String,
    },
    #[error(
        "proxy instance key is unavailable; keys.generate_proxy_instance_key_on_startup must be enabled for E2EE chat requests"
    )]
    ProxyInstanceKeyUnavailable,
    #[error("session does not contain an attested model public key after attestation verification")]
    MissingAttestedModelKey,
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
            Self::Attestation(error) if error.verifier_unavailable() => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            Self::Attestation(_) => StatusCode::BAD_GATEWAY,
            Self::Session(
                SessionError::MissingSessionIdentifier | SessionError::InvalidHeaderValue { .. },
            ) => StatusCode::BAD_REQUEST,
            Self::Session(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::ChatRequest(_) => StatusCode::BAD_REQUEST,
            Self::ChatConstruction(_)
            | Self::ChatStream(_)
            | Self::ToolCallRetryExhausted { .. } => StatusCode::BAD_GATEWAY,
            Self::ProxyInstanceKeyUnavailable | Self::MissingAttestedModelKey => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            Self::NotImplemented { .. } => StatusCode::NOT_IMPLEMENTED,
            Self::MethodNotAllowed { .. } => StatusCode::METHOD_NOT_ALLOWED,
            Self::NotFound { .. } => StatusCode::NOT_FOUND,
        }
    }

    fn error_type(&self) -> &'static str {
        match self {
            Self::Venice(error) => error.api_error_type(),
            Self::Attestation(error) => error.api_error_type(),
            Self::Session(
                SessionError::MissingSessionIdentifier | SessionError::InvalidHeaderValue { .. },
            ) => "invalid_request_error",
            Self::Session(_) => "proxy_session_error",
            Self::ChatRequest(_) => "invalid_request_error",
            Self::ChatConstruction(_) => "proxy_e2ee_error",
            Self::ChatStream(error) => error.api_error_type(),
            Self::ToolCallRetryExhausted { .. } => "proxy_tool_call_error",
            Self::ProxyInstanceKeyUnavailable => "proxy_configuration_error",
            Self::MissingAttestedModelKey => "proxy_attestation_error",
            Self::NotImplemented { .. } => "proxy_not_implemented",
            Self::MethodNotAllowed { .. } | Self::NotFound { .. } => "invalid_request_error",
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Venice(error) => error.api_error_code(),
            Self::Attestation(error) => error.api_error_code(),
            Self::Session(SessionError::MissingSessionIdentifier) => "session_identifier_missing",
            Self::Session(SessionError::InvalidHeaderValue { .. }) => "invalid_session_header",
            Self::Session(_) => "session_error",
            Self::ChatRequest(error) => error.api_error_code(),
            Self::ChatConstruction(error) => error.api_error_code(),
            Self::ChatStream(error) => error.api_error_code(),
            Self::ToolCallRetryExhausted { .. } => "invalid_tool_call",
            Self::ProxyInstanceKeyUnavailable => "proxy_instance_key_unavailable",
            Self::MissingAttestedModelKey => "attestation_failed",
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
        let error_type = self.error_type();
        if status.is_server_error() {
            error!(
                status = status.as_u16(),
                error_code,
                error_type,
                error = %self,
                "proxy request failed"
            );
        } else {
            warn!(
                status = status.as_u16(),
                error_code,
                error_type,
                error = %self,
                "proxy request rejected"
            );
        }

        let mut response = if let Self::ToolCallRetryExhausted {
            max_retries,
            last_validation_error,
        } = &self
        {
            let body = json!({
                "error": {
                    "message": self.to_string(),
                    "type": error_type,
                    "code": error_code,
                    "details": {
                        "max_retries": max_retries,
                        "last_validation_error": last_validation_error,
                    },
                }
            });
            (status, Json(body)).into_response()
        } else {
            let body = ErrorResponse::new(self.to_string(), error_type, error_code);
            (status, Json(body)).into_response()
        };
        apply_error_headers(response.headers_mut(), error_code);
        response
    }
}

/// Safe proxy metadata headers.
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
    /// Creates safe non-assertive metadata from config before a route has
    /// verification/session state.
    pub fn from_config(config: &ProxyConfig) -> Self {
        Self {
            attestation_mode: Some(config.attestation.mode.as_str().to_owned()),
            tool_mode: Some(config.tools.mode.as_str().to_owned()),
            ..Self::default()
        }
    }

    pub fn for_verified_chat(config: &ProxyConfig, session: &SessionContext) -> Self {
        let evidence = session
            .attestation_report
            .as_ref()
            .and_then(|report| report.get("attestation"))
            .and_then(Value::as_object);
        let tee_provider = evidence
            .and_then(|evidence| evidence.get("tee_provider"))
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let tdx_debug = evidence.and_then(|evidence| {
            evidence
                .get("debug")
                .or_else(|| evidence.get("tdx_debug"))
                .and_then(Value::as_bool)
        });
        let nvidia_payload_present = evidence
            .and_then(|evidence| evidence.get("nvidia_payload"))
            .is_some_and(|value| !value.is_null());
        let nvidia_verified = match (config.attestation.require_nvidia, nvidia_payload_present) {
            (_, false) => "not-present",
            (NvidiaRequirement::Never, true) => "ignored",
            (_, true) => "verified",
        }
        .to_owned();

        Self {
            e2ee: Some("verified".to_owned()),
            attestation_mode: Some(config.attestation.mode.as_str().to_owned()),
            attested_model: Some(session.model_id.clone()),
            tee_provider: Some(tee_provider),
            tdx_verified: config.attestation.require_tdx.then_some(true),
            tdx_debug,
            nvidia_verified: Some(nvidia_verified),
            key_binding: Some(true),
            session_id: Some(session.agent_session_id.clone()),
            session_scope: Some(session.scope.as_str().to_owned()),
            tool_mode: Some(config.tools.mode.as_str().to_owned()),
            tool_retries: None,
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
    use std::{
        collections::{HashMap, VecDeque},
        sync::{Arc, Mutex},
        time::Duration,
    };

    use axum::{
        body::Body,
        extract::Query,
        http::Request,
        routing::{get, post},
    };
    use serde_json::json;

    use crate::config::NvidiaRequirement;
    use tower::ServiceExt;

    fn test_app() -> Router {
        router_with_venice_client(ProxyConfig::default(), test_venice_client())
    }

    fn test_venice_client() -> VeniceClient {
        test_venice_client_for_base_url("http://127.0.0.1:1/api/v1")
    }

    fn test_venice_client_for_base_url(base_url: impl AsRef<str>) -> VeniceClient {
        VeniceClient::new(base_url.as_ref(), "test-api-key", Duration::from_secs(1))
            .expect("test Venice client should build")
    }

    fn chat_config_with_basic_test_attestation() -> ProxyConfig {
        let mut config = ProxyConfig::default();
        config.attestation.require_tdx = false;
        config.attestation.require_nvidia = NvidiaRequirement::Never;
        config
    }

    #[test]
    fn app_state_initializes_key_and_session_managers_from_config() {
        let state = AppState::from_parts(ProxyConfig::default(), test_venice_client());

        let key = state
            .proxy_instance_key()
            .expect("default config should generate startup key");
        assert_eq!(key.public_key_hex().len(), 130);
        assert!(state.session_manager().is_empty().unwrap());
        assert_eq!(
            state.attestation_verifier().policy(),
            &ProxyConfig::default().attestation
        );

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
    async fn chat_route_ignores_upstream_role_only_chunk_before_encrypted_content() {
        let response = streaming_chat_response(
            "chat-route-role-only",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
            vec![
                MockStreamFrame::Role,
                MockStreamFrame::Text("Hello"),
                MockStreamFrame::Finish("stop"),
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body(response).await;
        let data = sse_data(&body);
        assert_eq!(data.len(), 3);
        let first: Value = serde_json::from_str(data[0]).expect("first chunk should be JSON");
        assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(first["choices"][0]["delta"]["content"], "Hello");
        assert_eq!(data[2], "[DONE]");
    }

    #[tokio::test]
    async fn chat_route_streams_decrypted_normal_assistant_text() {
        let response = streaming_chat_response(
            "chat-route-test",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
            vec![
                MockStreamFrame::NullContent,
                MockStreamFrame::EmptyContent,
                MockStreamFrame::Text("Hello"),
                MockStreamFrame::Finish("stop"),
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(HEADER_PROXY_E2EE).unwrap(),
            "verified"
        );
        assert_eq!(
            response.headers().get(HEADER_PROXY_ATTESTED_MODEL).unwrap(),
            "e2ee-test"
        );

        let body = response_body(response).await;
        let data = sse_data(&body);
        assert_eq!(data.len(), 3);

        let first: Value = serde_json::from_str(data[0]).expect("first chunk should be JSON");
        assert_eq!(first["object"], "chat.completion.chunk");
        assert_eq!(first["model"], "e2ee-test");
        assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(first["choices"][0]["delta"]["content"], "Hello");
        assert!(first["choices"][0]["finish_reason"].is_null());

        let final_chunk: Value = serde_json::from_str(data[1]).expect("final chunk should be JSON");
        assert_eq!(final_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
        assert_eq!(data[2], "[DONE]");
    }

    #[tokio::test]
    async fn chat_route_streams_multiple_decrypted_content_chunks() {
        let response = streaming_chat_response(
            "chat-route-multiple-chunks",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
            vec![
                MockStreamFrame::Text("Hello"),
                MockStreamFrame::Text(" world"),
                MockStreamFrame::Finish("stop"),
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body(response).await;
        let data = sse_data(&body);
        let first: Value = serde_json::from_str(data[0]).expect("first chunk should be JSON");
        let second: Value = serde_json::from_str(data[1]).expect("second chunk should be JSON");

        assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(first["choices"][0]["delta"]["content"], "Hello");
        assert!(second["choices"][0]["delta"].get("role").is_none());
        assert_eq!(second["choices"][0]["delta"]["content"], " world");
        assert_eq!(data.last().copied(), Some("[DONE]"));
    }

    #[tokio::test]
    async fn chat_route_passes_through_usage_chunk_when_requested_and_upstream_provides_it() {
        let response = streaming_chat_response(
            "chat-route-usage",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true}}"#,
            vec![
                MockStreamFrame::Text("Hello"),
                MockStreamFrame::Finish("stop"),
                MockStreamFrame::Usage,
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body(response).await;
        let data = sse_data(&body);
        assert_eq!(data.len(), 4);
        let usage_chunk: Value = serde_json::from_str(data[2]).expect("usage chunk should be JSON");
        assert_eq!(usage_chunk["choices"], json!([]));
        assert_eq!(usage_chunk["usage"]["total_tokens"], 3);
        assert_eq!(data[3], "[DONE]");
    }

    #[tokio::test]
    async fn chat_route_returns_buffered_non_streaming_completion() {
        let response = chat_response(
            "chat-route-non-streaming-success",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
            vec![
                MockStreamFrame::NullContent,
                MockStreamFrame::EmptyContent,
                MockStreamFrame::Text("Hello"),
                MockStreamFrame::Text(" world"),
                MockStreamFrame::Finish("stop"),
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(HEADER_PROXY_E2EE).unwrap(),
            "verified"
        );
        let body = json_body(response).await;
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["id"], "chatcmpl-upstream-test");
        assert_eq!(body["created"], 1_717_171_717);
        assert_eq!(body["model"], "e2ee-test");
        assert_eq!(body["choices"][0]["index"], 0);
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert!(body["usage"].is_null());
    }

    #[tokio::test]
    async fn chat_route_treats_omitted_stream_as_buffered_non_streaming() {
        let response = chat_response(
            "chat-route-omitted-stream",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}]}"#,
            vec![MockStreamFrame::Text("Hello"), MockStreamFrame::Done],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn chat_route_streams_tool_call_chunks() {
        let response = streaming_chat_response(
            "chat-route-tool-stream",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"search"}],"stream":true,"tools":[{"type":"function","function":{"name":"search_web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}}]}"#,
            vec![
                MockStreamFrame::Text("<tool_call>\n{\"name\":\"search_web\",\"arguments\":{\"query\":\"example\"}}\n</tool_call>"),
                MockStreamFrame::Finish("stop"),
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body(response).await;
        let data = sse_data(&body);
        assert_eq!(data.len(), 3);
        let tool_chunk: Value = serde_json::from_str(data[0]).expect("tool chunk should be JSON");
        assert_eq!(tool_chunk["choices"][0]["delta"]["role"], "assistant");
        let tool_call = &tool_chunk["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tool_call["index"], 0);
        assert!(tool_call["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(tool_call["type"], "function");
        assert_eq!(tool_call["function"]["name"], "search_web");
        assert_eq!(tool_call["function"]["arguments"], r#"{"query":"example"}"#);
        assert!(tool_chunk["choices"][0]["finish_reason"].is_null());
        let final_chunk: Value = serde_json::from_str(data[1]).expect("final chunk should be JSON");
        assert_eq!(final_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(final_chunk["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(data[2], "[DONE]");
    }

    #[tokio::test]
    async fn chat_route_streams_text_then_buffers_and_emits_tool_call() {
        let response = streaming_chat_response(
            "chat-route-tool-stream-mixed-text",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"search"}],"stream":true,"tools":[{"type":"function","function":{"name":"search_web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}}]}"#,
            vec![
                MockStreamFrame::NullContent,
                MockStreamFrame::EmptyContent,
                MockStreamFrame::Text("I'll check that. "),
                MockStreamFrame::Text("<tool_call>{\"name\":\"search_web\",\"arguments\":{\"query\":\"example\"}}"),
                MockStreamFrame::Text("</tool_call>"),
                MockStreamFrame::Finish("stop"),
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body(response).await;
        let data = sse_data(&body);
        assert_eq!(data.len(), 4);
        let text_chunk: Value = serde_json::from_str(data[0]).expect("text chunk should be JSON");
        assert_eq!(text_chunk["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            text_chunk["choices"][0]["delta"]["content"],
            "I'll check that. "
        );
        assert!(
            text_chunk["choices"][0]["delta"]
                .get("tool_calls")
                .is_none()
        );

        let tool_chunk: Value = serde_json::from_str(data[1]).expect("tool chunk should be JSON");
        assert!(tool_chunk["choices"][0]["delta"].get("role").is_none());
        assert!(tool_chunk["choices"][0]["delta"].get("content").is_none());
        let tool_call = &tool_chunk["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tool_call["function"]["name"], "search_web");
        assert_eq!(tool_call["function"]["arguments"], r#"{"query":"example"}"#);

        let final_chunk: Value = serde_json::from_str(data[2]).expect("final chunk should be JSON");
        assert_eq!(final_chunk["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(data[3], "[DONE]");
    }

    #[tokio::test]
    async fn chat_route_returns_non_streaming_tool_call_body_from_mixed_text() {
        let response = chat_response(
            "chat-route-tool-non-stream-mixed-text",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"search"}],"stream":false,"tools":[{"type":"function","function":{"name":"search_web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]}"#,
            vec![
                MockStreamFrame::Text("I'll check that. <tool_call>{\"name\":\"search_web\",\"arguments\":{\"query\":\"example\"}}</tool_call>"),
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        let tool_call = &body["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tool_call["function"]["name"], "search_web");
        assert_eq!(tool_call["function"]["arguments"], r#"{"query":"example"}"#);
    }

    #[tokio::test]
    async fn chat_route_returns_non_streaming_tool_call_body() {
        let response = chat_response(
            "chat-route-tool-non-stream",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"search"}],"stream":false,"tools":[{"type":"function","function":{"name":"search_web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]}"#,
            vec![
                MockStreamFrame::Text("<tool_call>{\"name\":\"search_web\",\"arguments\":{\"query\":\"example\"}}</tool_call>"),
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["object"], "chat.completion");
        assert!(body["choices"][0]["message"]["content"].is_null());
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        let tool_call = &body["choices"][0]["message"]["tool_calls"][0];
        assert!(tool_call["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(tool_call["type"], "function");
        assert_eq!(tool_call["function"]["name"], "search_web");
        assert_eq!(tool_call["function"]["arguments"], r#"{"query":"example"}"#);
    }

    #[tokio::test]
    async fn chat_route_tool_mode_leaves_normal_text_unaffected() {
        let response = streaming_chat_response(
            "chat-route-tool-normal-text",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":true,"tools":[{"type":"function","function":{"name":"search_web","parameters":{"type":"object"}}}]}"#,
            vec![
                MockStreamFrame::Text("Hello without tools"),
                MockStreamFrame::Finish("stop"),
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body(response).await;
        let data = sse_data(&body);
        let first: Value = serde_json::from_str(data[0]).expect("first chunk should be JSON");
        assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            first["choices"][0]["delta"]["content"],
            "Hello without tools"
        );
        assert!(first["choices"][0]["delta"].get("tool_calls").is_none());
        assert_eq!(data.last().copied(), Some("[DONE]"));
    }

    #[tokio::test]
    async fn chat_route_treats_marker_like_non_protocol_text_as_normal_text() {
        let response = streaming_chat_response(
            "chat-route-tool-marker-like-text",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":true,"tools":[{"type":"function","function":{"name":"search_web","parameters":{"type":"object"}}}]}"#,
            vec![
                MockStreamFrame::Text("<tool_cal>{not actually a marker}"),
                MockStreamFrame::Finish("stop"),
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body(response).await;
        let data = sse_data(&body);
        let first: Value = serde_json::from_str(data[0]).expect("first chunk should be JSON");
        assert_eq!(
            first["choices"][0]["delta"]["content"],
            "<tool_cal>{not actually a marker}"
        );
        assert!(first["choices"][0]["delta"].get("tool_calls").is_none());
    }

    #[tokio::test]
    async fn chat_route_retries_invalid_tool_call_and_returns_success() {
        let response = chat_response_sequence(
            "chat-route-tool-retry-success",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"search"}],"stream":false,"tools":[{"type":"function","function":{"name":"search_web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]}"#,
            vec![
                vec![
                    MockStreamFrame::Text("<tool_call>{\"name\":\"unknown\",\"arguments\":{\"query\":\"example\"}}</tool_call>"),
                    MockStreamFrame::Done,
                ],
                vec![
                    MockStreamFrame::Text("<tool_call>{\"name\":\"search_web\",\"arguments\":{\"query\":\"example\"}}</tool_call>"),
                    MockStreamFrame::Done,
                ],
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(HEADER_PROXY_TOOL_RETRIES).unwrap(),
            "1"
        );
        let body = json_body(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "search_web"
        );
    }

    #[tokio::test]
    async fn chat_route_returns_retry_failure_error_shape() {
        let response = chat_response(
            "chat-route-tool-retry-failure",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"search"}],"stream":false,"tools":[{"type":"function","function":{"name":"search_web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]}"#,
            vec![
                MockStreamFrame::Text("<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>"),
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            response.headers().get(HEADER_PROXY_ERROR_CODE).unwrap(),
            "invalid_tool_call"
        );
        let body = json_body(response).await;
        assert_eq!(body["error"]["type"], "proxy_tool_call_error");
        assert_eq!(body["error"]["code"], "invalid_tool_call");
        assert_eq!(body["error"]["details"]["max_retries"], 2);
        assert!(
            body["error"]["details"]["last_validation_error"]
                .as_str()
                .unwrap()
                .contains("unknown tool name")
        );
    }

    #[tokio::test]
    async fn chat_route_non_streaming_fails_closed_on_upstream_error_response() {
        let response = chat_response_with_upstream_status(
            "chat-route-non-streaming-upstream-error",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
            StatusCode::INTERNAL_SERVER_ERROR,
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            response.headers().get(HEADER_PROXY_ERROR_CODE).unwrap(),
            "upstream_status_error"
        );
        let body = error_body(response).await;
        assert_eq!(body.error.kind, "proxy_upstream_error");
        assert_eq!(body.error.code, "upstream_status_error");
    }

    #[tokio::test]
    async fn chat_route_non_streaming_fails_closed_on_malformed_upstream_payload() {
        let response = chat_response(
            "chat-route-non-streaming-malformed",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
            vec![MockStreamFrame::Raw("data: {\"choices\":\"bad\"}\n\n")],
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            response.headers().get(HEADER_PROXY_ERROR_CODE).unwrap(),
            "upstream_malformed_response"
        );
        let body = error_body(response).await;
        assert_eq!(body.error.kind, "proxy_upstream_error");
        assert_eq!(body.error.code, "upstream_malformed_response");
    }

    #[tokio::test]
    async fn chat_route_non_streaming_fails_closed_on_missing_encrypted_content() {
        let response = chat_response(
            "chat-route-non-streaming-missing-content",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
            vec![MockStreamFrame::Finish("stop"), MockStreamFrame::Done],
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            response.headers().get(HEADER_PROXY_ERROR_CODE).unwrap(),
            "e2ee_response_decryption_failed"
        );
        let body = error_body(response).await;
        assert_eq!(body.error.kind, "proxy_e2ee_error");
        assert_eq!(body.error.code, "e2ee_response_decryption_failed");
    }

    #[tokio::test]
    async fn chat_route_non_streaming_fails_closed_on_decryption_failure() {
        let response = chat_response(
            "chat-route-non-streaming-decryption-failure",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
            vec![MockStreamFrame::TextForWrongRecipient(" secret"), MockStreamFrame::Done],
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            response.headers().get(HEADER_PROXY_ERROR_CODE).unwrap(),
            "e2ee_response_decryption_failed"
        );
        let body = error_body(response).await;
        assert_eq!(body.error.kind, "proxy_e2ee_error");
        assert_eq!(body.error.code, "e2ee_response_decryption_failed");
    }

    #[tokio::test]
    async fn chat_route_non_streaming_passes_through_usage_when_available() {
        let response = chat_response(
            "chat-route-non-streaming-usage",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
            vec![
                MockStreamFrame::Text("Hello"),
                MockStreamFrame::Finish("stop"),
                MockStreamFrame::Usage,
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["choices"][0]["message"]["content"], "Hello");
        assert_eq!(body["usage"]["prompt_tokens"], 1);
        assert_eq!(body["usage"]["completion_tokens"], 2);
        assert_eq!(body["usage"]["total_tokens"], 3);
    }

    #[tokio::test]
    async fn chat_route_fails_closed_on_upstream_stream_error_event() {
        let response = streaming_chat_response(
            "chat-route-upstream-error",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
            vec![MockStreamFrame::Error("model failed")],
        )
        .await;

        assert_stream_body_fails(response).await;
    }

    #[tokio::test]
    async fn chat_route_fails_closed_on_malformed_upstream_event() {
        let response = streaming_chat_response(
            "chat-route-malformed-event",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
            vec![MockStreamFrame::Raw("data: {\"choices\":\n\n")],
        )
        .await;

        assert_stream_body_fails(response).await;
    }

    #[tokio::test]
    async fn chat_route_fails_closed_on_decryption_failure_mid_stream() {
        let response = streaming_chat_response(
            "chat-route-decryption-failure",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
            vec![
                MockStreamFrame::Text("Hello"),
                MockStreamFrame::TextForWrongRecipient(" secret"),
                MockStreamFrame::Done,
            ],
        )
        .await;

        assert_stream_body_fails(response).await;
    }

    #[tokio::test]
    async fn chat_route_synthesizes_final_finish_chunk_before_done_when_needed() {
        let response = streaming_chat_response(
            "chat-route-final-done",
            r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
            vec![MockStreamFrame::Text("Hello"), MockStreamFrame::Done],
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body(response).await;
        let data = sse_data(&body);
        assert_eq!(data.len(), 3);
        let final_chunk: Value = serde_json::from_str(data[1]).expect("final chunk should be JSON");
        assert_eq!(final_chunk["choices"][0]["delta"], json!({}));
        assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
        assert_eq!(data[2], "[DONE]");
    }

    #[tokio::test]
    async fn chat_route_attestation_failure_prevents_request_construction() {
        let model_public_key = ProxyInstanceKey::generate().public_key_hex().to_owned();
        let base_url = spawn_attestation_server(model_public_key, false).await;
        let app = router_with_venice_client(
            chat_config_with_basic_test_attestation(),
            test_venice_client_for_base_url(base_url),
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header(HEADER_PROXY_SESSION_ID, "chat-route-attestation-failure")
                    .body(Body::from(
                        r#"{"model":"e2ee-test","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
                    ))
                    .expect("request should build"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            response.headers().get(HEADER_PROXY_ERROR_CODE).unwrap(),
            "attestation_upstream_not_verified"
        );
        let body = error_body(response).await;
        assert_eq!(body.error.kind, "proxy_attestation_error");
        assert_eq!(body.error.code, "attestation_upstream_not_verified");
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
    async fn non_object_chat_json_returns_structured_invalid_request() {
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

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers().get(HEADER_PROXY_ERROR_CODE).unwrap(),
            "invalid_request"
        );
        let body = error_body(response).await;
        assert_eq!(body.error.kind, "invalid_request_error");
        assert_eq!(body.error.code, "invalid_request");
    }

    #[derive(Debug, Clone)]
    enum MockStreamFrame {
        Role,
        NullContent,
        EmptyContent,
        Text(&'static str),
        TextForWrongRecipient(&'static str),
        Finish(&'static str),
        Usage,
        Done,
        Error(&'static str),
        Raw(&'static str),
    }

    async fn streaming_chat_response(
        session_id: &'static str,
        request_body: &'static str,
        frames: Vec<MockStreamFrame>,
    ) -> Response {
        chat_response(session_id, request_body, frames).await
    }

    async fn chat_response(
        session_id: &'static str,
        request_body: &'static str,
        frames: Vec<MockStreamFrame>,
    ) -> Response {
        let model_public_key = ProxyInstanceKey::generate().public_key_hex().to_owned();
        let base_url = spawn_streaming_venice_server(model_public_key, true, frames).await;
        request_chat(session_id, request_body, base_url).await
    }

    async fn chat_response_sequence(
        session_id: &'static str,
        request_body: &'static str,
        attempts: Vec<Vec<MockStreamFrame>>,
    ) -> Response {
        let model_public_key = ProxyInstanceKey::generate().public_key_hex().to_owned();
        let base_url =
            spawn_streaming_venice_server_sequence(model_public_key, true, attempts).await;
        request_chat(session_id, request_body, base_url).await
    }

    async fn chat_response_with_upstream_status(
        session_id: &'static str,
        request_body: &'static str,
        upstream_status: StatusCode,
    ) -> Response {
        let model_public_key = ProxyInstanceKey::generate().public_key_hex().to_owned();
        let base_url =
            spawn_venice_server_with_chat_status(model_public_key, upstream_status).await;
        request_chat(session_id, request_body, base_url).await
    }

    async fn request_chat(
        session_id: &'static str,
        request_body: &'static str,
        base_url: String,
    ) -> Response {
        let app = router_with_venice_client(
            chat_config_with_basic_test_attestation(),
            test_venice_client_for_base_url(base_url),
        );

        app.oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(HEADER_PROXY_SESSION_ID, session_id)
                .body(Body::from(request_body))
                .expect("request should build"),
        )
        .await
        .expect("request should complete")
    }

    async fn json_body(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should buffer");
        serde_json::from_slice(&bytes).expect("response should be JSON")
    }

    async fn response_body(response: Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should buffer");
        String::from_utf8(bytes.to_vec()).expect("response body should be UTF-8")
    }

    async fn assert_stream_body_fails(response: Response) {
        assert_eq!(response.status(), StatusCode::OK);
        let result = axum::body::to_bytes(response.into_body(), usize::MAX).await;
        assert!(
            result.is_err(),
            "stream body should fail closed instead of completing successfully"
        );
    }

    fn sse_data(body: &str) -> Vec<&str> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect()
    }

    async fn spawn_streaming_venice_server(
        model_public_key: String,
        verified: bool,
        frames: Vec<MockStreamFrame>,
    ) -> String {
        spawn_streaming_venice_server_sequence(model_public_key, verified, vec![frames]).await
    }

    async fn spawn_streaming_venice_server_sequence(
        model_public_key: String,
        verified: bool,
        attempts: Vec<Vec<MockStreamFrame>>,
    ) -> String {
        let chat_attempts = Arc::new(Mutex::new(VecDeque::from(attempts)));
        let attestation_key = model_public_key.clone();
        let app = Router::new()
            .route(
                "/api/v1/tee/attestation",
                get(move |Query(query): Query<HashMap<String, String>>| {
                    let model_public_key = attestation_key.clone();
                    async move {
                        Json(json!({
                            "attestation": {
                                "verified": verified,
                                "nonce": query.get("nonce").cloned().unwrap_or_default(),
                                "model": query.get("model").cloned().unwrap_or_default(),
                                "tee_provider": "tdx",
                                "signing_key": model_public_key,
                            }
                        }))
                    }
                }),
            )
            .route(
                "/api/v1/chat/completions",
                post(move |headers: HeaderMap, Json(body): Json<Value>| {
                    let chat_attempts = chat_attempts.clone();
                    async move {
                        let Some(client_public_key) = headers
                            .get(crate::venice::HEADER_VENICE_TEE_CLIENT_PUB_KEY)
                            .and_then(|value| value.to_str().ok())
                        else {
                            return (
                                StatusCode::BAD_REQUEST,
                                [("content-type", "text/plain")],
                                "missing client key".to_owned(),
                            );
                        };
                        if body.get("stream").and_then(Value::as_bool) != Some(true) {
                            return (
                                StatusCode::BAD_REQUEST,
                                [("content-type", "text/plain")],
                                "upstream request must stream".to_owned(),
                            );
                        }
                        let messages = body.get("messages").and_then(Value::as_array);
                        if messages.is_none_or(|messages| {
                            messages.is_empty()
                                || !messages.iter().all(|message| {
                                    message.get("role").and_then(Value::as_str).is_some()
                                        && message
                                            .get("content")
                                            .and_then(Value::as_str)
                                            .is_some_and(|content| {
                                                !content.is_empty()
                                                    && content
                                                        .chars()
                                                        .all(|ch| ch.is_ascii_hexdigit())
                                            })
                                })
                        }) {
                            return (
                                StatusCode::BAD_REQUEST,
                                [("content-type", "text/plain")],
                                "messages must be encrypted message objects".to_owned(),
                            );
                        }

                        let frames = {
                            let mut attempts = chat_attempts
                                .lock()
                                .expect("mock chat attempts mutex should not be poisoned");
                            if attempts.len() > 1 {
                                attempts.pop_front().expect("attempts length checked above")
                            } else {
                                attempts.front().cloned().unwrap_or_default()
                            }
                        };

                        (
                            StatusCode::OK,
                            [("content-type", "text/event-stream")],
                            render_mock_sse(&frames, client_public_key),
                        )
                    }
                }),
            );
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

    async fn spawn_venice_server_with_chat_status(
        model_public_key: String,
        upstream_status: StatusCode,
    ) -> String {
        let attestation_key = model_public_key.clone();
        let app = Router::new()
            .route(
                "/api/v1/tee/attestation",
                get(move |Query(query): Query<HashMap<String, String>>| {
                    let model_public_key = attestation_key.clone();
                    async move {
                        Json(json!({
                            "attestation": {
                                "verified": true,
                                "nonce": query.get("nonce").cloned().unwrap_or_default(),
                                "model": query.get("model").cloned().unwrap_or_default(),
                                "tee_provider": "tdx",
                                "signing_key": model_public_key,
                            }
                        }))
                    }
                }),
            )
            .route(
                "/api/v1/chat/completions",
                post(move || async move { upstream_status }),
            );
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

    fn render_mock_sse(frames: &[MockStreamFrame], client_public_key: &str) -> String {
        let codec = E2eeCodec::default();
        let mut output = String::new();
        for frame in frames {
            match frame {
                MockStreamFrame::Role => {
                    output.push_str(&format!("data: {}\n\n", upstream_role_chunk()));
                }
                MockStreamFrame::NullContent => {
                    output.push_str(&format!("data: {}\n\n", upstream_null_content_chunk()));
                }
                MockStreamFrame::EmptyContent => {
                    output.push_str(&format!(
                        "data: {}\n\n",
                        upstream_content_chunk(String::new())
                    ));
                }
                MockStreamFrame::Text(content) => {
                    let encrypted = codec
                        .encrypt_content(content, client_public_key)
                        .expect("mock content should encrypt")
                        .into_hex();
                    output.push_str(&format!("data: {}\n\n", upstream_content_chunk(encrypted)));
                }
                MockStreamFrame::TextForWrongRecipient(content) => {
                    let wrong_key = ProxyInstanceKey::generate();
                    let encrypted = codec
                        .encrypt_content(content, wrong_key.public_key_hex())
                        .expect("mock content should encrypt")
                        .into_hex();
                    output.push_str(&format!("data: {}\n\n", upstream_content_chunk(encrypted)));
                }
                MockStreamFrame::Finish(reason) => {
                    output.push_str(&format!("data: {}\n\n", upstream_finish_chunk(reason)));
                }
                MockStreamFrame::Usage => {
                    output.push_str(&format!("data: {}\n\n", upstream_usage_chunk()));
                }
                MockStreamFrame::Done => output.push_str("data: [DONE]\n\n"),
                MockStreamFrame::Error(message) => {
                    output.push_str(&format!(
                        "event: error\ndata: {}\n\n",
                        json!({ "message": message })
                    ));
                }
                MockStreamFrame::Raw(raw) => output.push_str(raw),
            }
        }
        output
    }

    fn upstream_role_chunk() -> Value {
        json!({
            "id": "chatcmpl-upstream-test",
            "object": "chat.completion.chunk",
            "created": 1_717_171_717,
            "model": "e2ee-test",
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant" },
                "finish_reason": null,
            }],
        })
    }

    fn upstream_content_chunk(encrypted_content: String) -> Value {
        json!({
            "id": "chatcmpl-upstream-test",
            "object": "chat.completion.chunk",
            "created": 1_717_171_717,
            "model": "e2ee-test",
            "choices": [{
                "index": 0,
                "delta": { "content": encrypted_content },
                "finish_reason": null,
            }],
        })
    }

    fn upstream_null_content_chunk() -> Value {
        json!({
            "id": "chatcmpl-upstream-test",
            "object": "chat.completion.chunk",
            "created": 1_717_171_717,
            "model": "e2ee-test",
            "choices": [{
                "index": 0,
                "delta": { "content": Value::Null },
                "finish_reason": null,
            }],
        })
    }

    fn upstream_finish_chunk(reason: &str) -> Value {
        json!({
            "id": "chatcmpl-upstream-test",
            "object": "chat.completion.chunk",
            "created": 1_717_171_717,
            "model": "e2ee-test",
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
            "model": "e2ee-test",
            "choices": [],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        })
    }

    async fn spawn_attestation_server(model_public_key: String, verified: bool) -> String {
        let app = Router::new().route(
            "/api/v1/tee/attestation",
            get(move |Query(query): Query<HashMap<String, String>>| {
                let model_public_key = model_public_key.clone();
                async move {
                    Json(json!({
                        "attestation": {
                            "verified": verified,
                            "nonce": query.get("nonce").cloned().unwrap_or_default(),
                            "model": query.get("model").cloned().unwrap_or_default(),
                            "signing_key": model_public_key,
                        }
                    }))
                }
            }),
        );
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("mock attestation listener should bind");
        let addr = listener
            .local_addr()
            .expect("mock attestation listener should have local address");

        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("mock attestation server should run");
        });

        format!("http://{addr}/api/v1")
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
