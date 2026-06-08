use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::e2ee::{E2eeCodec, E2eeCodecError};

#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<NormalizedChatMessage>,
    pub stream: bool,
    pub stream_options: OpenAiStreamOptions,
    pub venice_parameters: VeniceParameters,
    pub passthrough: OpenAiPassthroughFields,
}

impl ChatCompletionRequest {
    pub fn parse(value: &Value) -> Result<Self, ChatRequestError> {
        let object = value
            .as_object()
            .ok_or_else(|| ChatRequestError::invalid("request body must be a JSON object"))?;
        reject_unknown_fields(
            object,
            &[
                "model",
                "messages",
                "stream",
                "stream_options",
                "temperature",
                "top_p",
                "max_tokens",
                "max_completion_tokens",
                "stop",
                "tools",
                "tool_choice",
                "metadata",
                "venice_parameters",
            ],
            "request",
        )?;

        validate_ignored_client_only_fields(object)?;

        let model = required_non_empty_string(object, "model")?.to_owned();
        let messages_value = object
            .get("messages")
            .ok_or(ChatRequestError::MissingField { field: "messages" })?;
        let messages = normalize_messages(messages_value)?;
        let stream = optional_bool(object, "stream")?.unwrap_or(false);
        let stream_options = OpenAiStreamOptions::parse(object.get("stream_options"))?;
        let venice_parameters = VeniceParameters::parse(object.get("venice_parameters"))?;
        let passthrough = OpenAiPassthroughFields::parse(object)?;

        Ok(Self {
            model,
            messages,
            stream,
            stream_options,
            venice_parameters,
            passthrough,
        })
    }

    pub fn into_venice_e2ee_request(
        &self,
        codec: &E2eeCodec,
        model_public_key_hex: &str,
    ) -> Result<PreparedVeniceChatRequest, ChatConstructionError> {
        let encrypted_messages = codec
            .encrypt_json_payload(&self.messages, model_public_key_hex)
            .map_err(ChatConstructionError::E2ee)?;

        Ok(PreparedVeniceChatRequest {
            client_stream: self.stream,
            upstream: VeniceE2eeChatRequest {
                model: self.model.clone(),
                messages: encrypted_messages.into_hex(),
                stream: true,
                stream_options: VeniceStreamOptions {
                    include_usage: self.stream_options.include_usage.unwrap_or(true),
                },
                venice_parameters: self.venice_parameters.clone(),
                temperature: self.passthrough.temperature.clone(),
                top_p: self.passthrough.top_p.clone(),
                max_tokens: self.passthrough.max_tokens,
                max_completion_tokens: self.passthrough.max_completion_tokens,
                stop: self.passthrough.stop.clone(),
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedChatMessage {
    pub role: String,
    pub content: String,
}

impl NormalizedChatMessage {
    fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiStreamOptions {
    #[serde(default, deserialize_with = "deserialize_optional_bool_reject_null")]
    pub include_usage: Option<bool>,
}

impl OpenAiStreamOptions {
    fn parse(value: Option<&Value>) -> Result<Self, ChatRequestError> {
        let Some(value) = value else {
            return Ok(Self::default());
        };
        let object = value.as_object().ok_or_else(|| {
            ChatRequestError::invalid_field("stream_options", "stream_options must be an object")
        })?;
        reject_unknown_fields(object, &["include_usage"], "stream_options")?;
        if let Some(include_usage) = object.get("include_usage")
            && !include_usage.is_boolean()
        {
            return Err(ChatRequestError::invalid_field(
                "include_usage",
                format!("expected boolean, got {}", json_kind(include_usage)),
            ));
        }
        deserialize_typed_field("stream_options", value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VeniceParameters {
    pub include_venice_system_prompt: bool,
    pub enable_web_search: String,
}

impl Default for VeniceParameters {
    fn default() -> Self {
        Self {
            include_venice_system_prompt: false,
            enable_web_search: "off".to_owned(),
        }
    }
}

impl VeniceParameters {
    fn parse(value: Option<&Value>) -> Result<Self, ChatRequestError> {
        let Some(value) = value else {
            return Ok(Self::default());
        };
        let object = value.as_object().ok_or_else(|| {
            ChatRequestError::invalid_field(
                "venice_parameters",
                "venice_parameters must be an object",
            )
        })?;
        reject_unknown_fields(
            object,
            &["include_venice_system_prompt", "enable_web_search"],
            "venice_parameters",
        )?;
        validate_raw_venice_parameter_types(object)?;

        let raw: RawVeniceParameters = deserialize_typed_field("venice_parameters", value)?;
        let include_venice_system_prompt = raw.include_venice_system_prompt.unwrap_or(false);
        if include_venice_system_prompt {
            return Err(ChatRequestError::UnsupportedVeniceParameter {
                field: "venice_parameters.include_venice_system_prompt",
                message: "Venice system prompt injection is disabled for E2EE requests".to_owned(),
            });
        }

        let enable_web_search = match raw.enable_web_search {
            None => "off".to_owned(),
            Some(RawVeniceWebSearch::String(value)) if value == "off" => "off".to_owned(),
            Some(RawVeniceWebSearch::Bool(false)) => "off".to_owned(),
            Some(RawVeniceWebSearch::String(value)) => {
                return Err(ChatRequestError::UnsupportedVeniceParameter {
                    field: "venice_parameters.enable_web_search",
                    message: format!(
                        "Venice web search is out of scope for E2EE requests; expected \"off\", got {value:?}"
                    ),
                });
            }
            Some(RawVeniceWebSearch::Bool(true)) => {
                return Err(ChatRequestError::UnsupportedVeniceParameter {
                    field: "venice_parameters.enable_web_search",
                    message: "Venice web search is out of scope for E2EE requests".to_owned(),
                });
            }
        };

        Ok(Self {
            include_venice_system_prompt,
            enable_web_search,
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct OpenAiPassthroughFields {
    pub temperature: Option<Value>,
    pub top_p: Option<Value>,
    pub max_tokens: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub stop: Option<StopSequence>,
}

impl OpenAiPassthroughFields {
    fn parse(object: &Map<String, Value>) -> Result<Self, ChatRequestError> {
        Ok(Self {
            temperature: optional_number(object, "temperature")?,
            top_p: optional_number(object, "top_p")?,
            max_tokens: optional_u64(object, "max_tokens")?,
            max_completion_tokens: optional_u64(object, "max_completion_tokens")?,
            stop: optional_stop(object)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedVeniceChatRequest {
    pub client_stream: bool,
    pub upstream: VeniceE2eeChatRequest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VeniceE2eeChatRequest {
    pub model: String,
    pub messages: String,
    pub stream: bool,
    pub stream_options: VeniceStreamOptions,
    pub venice_parameters: VeniceParameters,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    String(String),
    Strings(Vec<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VeniceStreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawVeniceParameters {
    #[serde(default, deserialize_with = "deserialize_optional_bool_reject_null")]
    include_venice_system_prompt: Option<bool>,
    #[serde(
        default,
        deserialize_with = "deserialize_optional_web_search_reject_null"
    )]
    enable_web_search: Option<RawVeniceWebSearch>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
enum RawVeniceWebSearch {
    String(String),
    Bool(bool),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAssistantToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: RawAssistantToolFunction,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAssistantToolFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTextContentPart {
    #[serde(rename = "type")]
    kind: TextContentPartType,
    text: String,
}

#[derive(Debug, Clone, Deserialize)]
enum TextContentPartType {
    #[serde(rename = "text")]
    Text,
}

#[derive(Debug, Error)]
pub enum ChatRequestError {
    #[error("missing required field {field}")]
    MissingField { field: &'static str },
    #[error("invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("invalid field {field}: {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },
    #[error("unsupported request field {field}")]
    UnsupportedField { field: String },
    #[error("unsupported message role {role:?}")]
    UnsupportedMessageRole { role: String },
    #[error("unsupported message content at {path}: {message}")]
    UnsupportedMessageContent { path: String, message: String },
    #[error("invalid assistant tool-call history: {message}")]
    InvalidToolCallHistory { message: String },
    #[error("unsupported Venice parameter {field}: {message}")]
    UnsupportedVeniceParameter {
        field: &'static str,
        message: String,
    },
}

impl ChatRequestError {
    pub fn api_error_code(&self) -> &'static str {
        match self {
            Self::MissingField { .. } | Self::InvalidRequest { .. } | Self::InvalidField { .. } => {
                "invalid_request"
            }
            Self::UnsupportedField { .. } => "unsupported_request_field",
            Self::UnsupportedMessageRole { .. } => "unsupported_message_role",
            Self::UnsupportedMessageContent { .. } => "unsupported_message_content",
            Self::InvalidToolCallHistory { .. } => "invalid_tool_call_history",
            Self::UnsupportedVeniceParameter { .. } => "unsupported_venice_parameter",
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    fn invalid_field(field: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidField {
            field,
            message: message.into(),
        }
    }

    fn unsupported_content(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self::UnsupportedMessageContent {
            path: path.into(),
            message: message.into(),
        }
    }

    fn invalid_tool_history(message: impl Into<String>) -> Self {
        Self::InvalidToolCallHistory {
            message: message.into(),
        }
    }
}

#[derive(Debug, Error)]
pub enum ChatConstructionError {
    #[error(transparent)]
    E2ee(#[from] E2eeCodecError),
}

impl ChatConstructionError {
    pub fn api_error_code(&self) -> &'static str {
        match self {
            Self::E2ee(_) => "e2ee_request_encryption_failed",
        }
    }
}

fn normalize_messages(value: &Value) -> Result<Vec<NormalizedChatMessage>, ChatRequestError> {
    let messages = value
        .as_array()
        .ok_or_else(|| ChatRequestError::invalid_field("messages", "messages must be an array"))?;
    if messages.is_empty() {
        return Err(ChatRequestError::invalid_field(
            "messages",
            "messages must include at least one message",
        ));
    }

    messages
        .iter()
        .enumerate()
        .map(|(index, value)| normalize_message(index, value))
        .collect()
}

fn normalize_message(
    index: usize,
    value: &Value,
) -> Result<NormalizedChatMessage, ChatRequestError> {
    let object = value.as_object().ok_or_else(|| {
        ChatRequestError::invalid_field("messages", format!("message {index} must be an object"))
    })?;
    let role = required_non_empty_string(object, "role")?;

    match role {
        "system" | "developer" | "user" => {
            reject_unknown_fields(object, &["role", "content"], "message")?;
            let content = required_content_text(
                object.get("content"),
                &format!("messages[{index}].content"),
            )?;
            Ok(NormalizedChatMessage::new(role, content))
        }
        "assistant" => normalize_assistant_message(index, object),
        "tool" => normalize_tool_result_message(index, object),
        other => Err(ChatRequestError::UnsupportedMessageRole {
            role: other.to_owned(),
        }),
    }
}

fn normalize_assistant_message(
    index: usize,
    object: &Map<String, Value>,
) -> Result<NormalizedChatMessage, ChatRequestError> {
    reject_unknown_fields(
        object,
        &["role", "content", "tool_calls"],
        "assistant message",
    )?;
    let content = optional_content_text(
        object.get("content"),
        &format!("messages[{index}].content"),
        true,
    )?;
    let tool_calls = normalize_assistant_tool_calls(object.get("tool_calls"))?;

    if content.as_deref().unwrap_or_default().is_empty() && tool_calls.is_none() {
        return Err(ChatRequestError::invalid_field(
            "messages",
            "assistant messages must include string content or a supported tool_calls history entry",
        ));
    }

    let normalized_content = match (content, tool_calls) {
        (Some(content), Some(tool_calls)) if !content.is_empty() => {
            format!("{content}\n\n{tool_calls}")
        }
        (Some(content), _) => content,
        (None, Some(tool_calls)) => tool_calls,
        (None, None) => unreachable!("empty assistant messages are rejected above"),
    };

    Ok(NormalizedChatMessage::new("assistant", normalized_content))
}

fn normalize_tool_result_message(
    index: usize,
    object: &Map<String, Value>,
) -> Result<NormalizedChatMessage, ChatRequestError> {
    reject_unknown_fields(object, &["role", "tool_call_id", "content"], "tool message")?;
    let tool_call_id = required_non_empty_string(object, "tool_call_id")?;
    let content =
        required_content_text(object.get("content"), &format!("messages[{index}].content"))?;
    let normalized = format!(
        "<tool_result id=\"{}\">\n{}\n</tool_result>\n\nUse the tool result above to continue the answer.",
        xml_escape_attr(tool_call_id),
        content,
    );

    // Venice E2EE compatibility: present prior tool output as user-visible context.
    Ok(NormalizedChatMessage::new("user", normalized))
}

fn normalize_assistant_tool_calls(
    value: Option<&Value>,
) -> Result<Option<String>, ChatRequestError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.is_array() {
        return Err(ChatRequestError::invalid_tool_history(
            "assistant tool_calls must be an array",
        ));
    }
    let tool_calls: Vec<RawAssistantToolCall> = serde_json::from_value(value.clone()).map_err(
        |source| {
            ChatRequestError::invalid_tool_history(format!(
                "assistant tool_calls must be an array of supported function tool call objects: {source}"
            ))
        },
    )?;
    if tool_calls.is_empty() {
        return Err(ChatRequestError::invalid_tool_history(
            "assistant tool_calls must not be empty when provided",
        ));
    }
    if tool_calls.len() > 1 {
        return Err(ChatRequestError::invalid_tool_history(
            "parallel assistant tool_calls are not supported in the E2EE proxy MVP",
        ));
    }

    let tool_call = tool_calls
        .first()
        .expect("tool_calls is known to contain one element");
    let id = non_empty_typed_string(&tool_call.id, "tool_call.id")?;
    if tool_call.kind != "function" {
        return Err(ChatRequestError::invalid_tool_history(format!(
            "only function tool calls are supported, got {:?}",
            tool_call.kind
        )));
    }
    let name = non_empty_typed_string(&tool_call.function.name, "tool_call.function.name")?;
    let arguments = non_empty_typed_string(
        &tool_call.function.arguments,
        "tool_call.function.arguments",
    )?;
    let parsed_arguments: Value = serde_json::from_str(arguments).map_err(|source| {
        ChatRequestError::invalid_tool_history(format!(
            "tool_call.function.arguments must be valid JSON: {source}"
        ))
    })?;
    let canonical_arguments = serde_json::to_string(&parsed_arguments).map_err(|source| {
        ChatRequestError::invalid_tool_history(format!(
            "tool_call.function.arguments could not be serialized as JSON: {source}"
        ))
    })?;

    Ok(Some(format!(
        "<previous_tool_call id=\"{}\" name=\"{}\">\n{}\n</previous_tool_call>",
        xml_escape_attr(id),
        xml_escape_attr(name),
        canonical_arguments,
    )))
}

fn required_content_text(value: Option<&Value>, path: &str) -> Result<String, ChatRequestError> {
    optional_content_text(value, path, false)?.ok_or_else(|| {
        ChatRequestError::unsupported_content(path, "content is required and must not be null")
    })
}

fn optional_content_text(
    value: Option<&Value>,
    path: &str,
    allow_null: bool,
) -> Result<Option<String>, ChatRequestError> {
    match value {
        Some(Value::String(content)) => Ok(Some(content.clone())),
        Some(Value::Null) if allow_null => Ok(None),
        Some(Value::Null) => Err(ChatRequestError::unsupported_content(
            path,
            "null content is only supported for assistant messages with tool_calls",
        )),
        Some(Value::Array(parts)) => normalize_text_parts(parts, path).map(Some),
        Some(other) => Err(ChatRequestError::unsupported_content(
            path,
            format!(
                "expected a string or text-only content parts array, got {}",
                json_kind(other)
            ),
        )),
        None if allow_null => Ok(None),
        None => Err(ChatRequestError::unsupported_content(
            path,
            "content is required",
        )),
    }
}

fn normalize_text_parts(parts: &[Value], path: &str) -> Result<String, ChatRequestError> {
    if parts.is_empty() {
        return Err(ChatRequestError::unsupported_content(
            path,
            "content parts array must not be empty",
        ));
    }

    let mut text = String::new();
    for (index, part) in parts.iter().enumerate() {
        let object = part.as_object().ok_or_else(|| {
            ChatRequestError::unsupported_content(
                format!("{path}[{index}]"),
                "content parts must be objects",
            )
        })?;
        let kind = required_non_empty_string(object, "type")?;
        if kind != "text" {
            return Err(ChatRequestError::unsupported_content(
                format!("{path}[{index}]"),
                format!("only text content parts are supported, got {kind:?}"),
            ));
        }
        reject_unknown_fields(object, &["type", "text"], "content part")?;
        let part: RawTextContentPart = serde_json::from_value(part.clone()).map_err(|source| {
            ChatRequestError::unsupported_content(
                format!("{path}[{index}]"),
                format!("text content part must match {{type:\"text\", text:string}}: {source}"),
            )
        })?;
        match part.kind {
            TextContentPartType::Text => text.push_str(&part.text),
        }
    }
    Ok(text)
}

fn validate_raw_venice_parameter_types(
    object: &Map<String, Value>,
) -> Result<(), ChatRequestError> {
    if let Some(include_venice_system_prompt) = object.get("include_venice_system_prompt")
        && !include_venice_system_prompt.is_boolean()
    {
        return Err(ChatRequestError::invalid_field(
            "venice_parameters.include_venice_system_prompt",
            format!(
                "expected boolean, got {}",
                json_kind(include_venice_system_prompt)
            ),
        ));
    }
    if let Some(enable_web_search) = object.get("enable_web_search")
        && !(enable_web_search.is_string() || enable_web_search.is_boolean())
    {
        return Err(ChatRequestError::invalid_field(
            "venice_parameters.enable_web_search",
            format!(
                "enable_web_search must be \"off\" or false, got {}",
                json_kind(enable_web_search)
            ),
        ));
    }
    Ok(())
}

fn validate_ignored_client_only_fields(
    object: &Map<String, Value>,
) -> Result<(), ChatRequestError> {
    if let Some(tools) = object.get("tools")
        && !tools.is_array()
    {
        return Err(ChatRequestError::invalid_field(
            "tools",
            "tools must be an array when provided; tools are consumed locally and are not forwarded to Venice E2EE",
        ));
    }
    if let Some(tool_choice) = object.get("tool_choice")
        && !(tool_choice.is_string() || tool_choice.is_object() || tool_choice.is_null())
    {
        return Err(ChatRequestError::invalid_field(
            "tool_choice",
            "tool_choice must be a string, object, or null when provided; tool choice is consumed locally and is not forwarded to Venice E2EE",
        ));
    }
    if let Some(metadata) = object.get("metadata")
        && !(metadata.is_object() || metadata.is_null())
    {
        return Err(ChatRequestError::invalid_field(
            "metadata",
            "metadata must be an object when provided",
        ));
    }
    Ok(())
}

fn non_empty_typed_string<'a>(
    value: &'a str,
    field: &'static str,
) -> Result<&'a str, ChatRequestError> {
    if value.trim().is_empty() {
        return Err(ChatRequestError::invalid_tool_history(format!(
            "{field} must not be empty"
        )));
    }
    Ok(value)
}

fn required_non_empty_string<'a>(
    object: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, ChatRequestError> {
    let value = object
        .get(field)
        .ok_or(ChatRequestError::MissingField { field })?;
    let string = value.as_str().ok_or_else(|| {
        ChatRequestError::invalid_field(field, format!("expected string, got {}", json_kind(value)))
    })?;
    if string.trim().is_empty() {
        return Err(ChatRequestError::invalid_field(field, "must not be empty"));
    }
    Ok(string)
}

fn optional_bool(
    object: &Map<String, Value>,
    field: &'static str,
) -> Result<Option<bool>, ChatRequestError> {
    object
        .get(field)
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                ChatRequestError::invalid_field(
                    field,
                    format!("expected boolean, got {}", json_kind(value)),
                )
            })
        })
        .transpose()
}

fn optional_number(
    object: &Map<String, Value>,
    field: &'static str,
) -> Result<Option<Value>, ChatRequestError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            let number = deserialize_typed_value::<serde_json::Number>(field, value)?;
            Ok(Some(Value::Number(number)))
        }
    }
}

fn optional_u64(
    object: &Map<String, Value>,
    field: &'static str,
) -> Result<Option<u64>, ChatRequestError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => deserialize_typed_value(field, value).map(Some),
    }
}

fn optional_stop(object: &Map<String, Value>) -> Result<Option<StopSequence>, ChatRequestError> {
    match object.get("stop") {
        None | Some(Value::Null) => Ok(None),
        Some(value) => deserialize_typed_value("stop", value).map(Some),
    }
}

fn deserialize_typed_field<T>(field: &'static str, value: &Value) -> Result<T, ChatRequestError>
where
    T: DeserializeOwned,
{
    deserialize_typed_value(field, value)
}

fn deserialize_typed_value<T>(field: &'static str, value: &Value) -> Result<T, ChatRequestError>
where
    T: DeserializeOwned,
{
    serde_json::from_value(value.clone()).map_err(|source| {
        ChatRequestError::invalid_field(
            field,
            format!(
                "expected supported shape, got {}: {source}",
                json_kind(value)
            ),
        )
    })
}

fn deserialize_optional_bool_reject_null<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Bool(value) => Ok(Some(value)),
        other => Err(serde::de::Error::custom(format!(
            "expected boolean, got {}",
            json_kind(&other)
        ))),
    }
}

fn deserialize_optional_web_search_reject_null<'de, D>(
    deserializer: D,
) -> Result<Option<RawVeniceWebSearch>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(value) => Ok(Some(RawVeniceWebSearch::String(value))),
        Value::Bool(value) => Ok(Some(RawVeniceWebSearch::Bool(value))),
        other => Err(serde::de::Error::custom(format!(
            "expected string or boolean, got {}",
            json_kind(&other)
        ))),
    }
}

fn reject_unknown_fields(
    object: &Map<String, Value>,
    allowed: &[&str],
    _context: &str,
) -> Result<(), ChatRequestError> {
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(ChatRequestError::UnsupportedField {
            field: field.clone(),
        });
    }
    Ok(())
}

fn xml_escape_attr(value: &str) -> String {
    let mut escaped = String::new();
    for ch in value.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '"' => escaped.push_str("&quot;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k256::{SecretKey, elliptic_curve::sec1::ToEncodedPoint};
    use serde_json::json;

    fn parse(value: Value) -> ChatCompletionRequest {
        ChatCompletionRequest::parse(&value).expect("request should parse")
    }

    fn model_public_key_hex(secret_key: &SecretKey) -> String {
        let public_key = secret_key.public_key();
        encode_lower_hex(public_key.to_encoded_point(false).as_bytes())
    }

    fn encode_lower_hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }

    #[test]
    fn normalizes_system_user_and_assistant_text_messages() {
        let request = parse(json!({
            "model": "e2ee-test",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": [{"type":"text", "text":"Hello"}]},
                {"role": "assistant", "content": "Hi"}
            ]
        }));

        assert_eq!(
            request.messages,
            vec![
                NormalizedChatMessage::new("system", "You are concise."),
                NormalizedChatMessage::new("user", "Hello"),
                NormalizedChatMessage::new("assistant", "Hi"),
            ]
        );
    }

    #[test]
    fn normalizes_assistant_tool_call_history() {
        let request = parse(json!({
            "model": "e2ee-test",
            "messages": [
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": "{\"query\":\"Venice E2EE\"}"
                        }
                    }]
                }
            ]
        }));

        assert_eq!(
            request.messages[0],
            NormalizedChatMessage::new(
                "assistant",
                "<previous_tool_call id=\"call_abc\" name=\"search_web\">\n{\"query\":\"Venice E2EE\"}\n</previous_tool_call>",
            )
        );
    }

    #[test]
    fn normalizes_tool_result_messages_as_user_context() {
        let request = parse(json!({
            "model": "e2ee-test",
            "messages": [
                {"role": "tool", "tool_call_id": "call_abc", "content": "result text"}
            ]
        }));

        assert_eq!(
            request.messages[0],
            NormalizedChatMessage::new(
                "user",
                "<tool_result id=\"call_abc\">\nresult text\n</tool_result>\n\nUse the tool result above to continue the answer.",
            )
        );
    }

    #[test]
    fn rejects_unsupported_roles_and_content_shapes() {
        let role_error = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"function", "content":"legacy"}]
        }))
        .expect_err("legacy function role should be rejected");
        assert_eq!(role_error.api_error_code(), "unsupported_message_role");

        let content_error = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":[{"type":"image_url", "image_url":{"url":"x"}}]}]
        }))
        .expect_err("image content should be rejected");
        assert_eq!(
            content_error.api_error_code(),
            "unsupported_message_content"
        );

        let assistant_error = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"assistant", "content": null}]
        }))
        .expect_err("assistant null content without tool call should be rejected");
        assert_eq!(assistant_error.api_error_code(), "invalid_request");
    }

    #[test]
    fn rejects_unsupported_top_level_fields_and_unsafe_venice_parameters() {
        let field_error = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "file_ids": ["file_1"]
        }))
        .expect_err("unsupported top-level field should be rejected");
        assert_eq!(field_error.api_error_code(), "unsupported_request_field");

        let web_search_error = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "venice_parameters": {"enable_web_search": "on"}
        }))
        .expect_err("web search should be rejected for E2EE");
        assert_eq!(
            web_search_error.api_error_code(),
            "unsupported_venice_parameter"
        );
    }

    #[test]
    fn rejects_null_or_invalid_typed_subfields_without_silent_option_coercion() {
        let stream_options_null = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "stream_options": null
        }))
        .expect_err("stream_options null should be rejected");
        assert_eq!(stream_options_null.api_error_code(), "invalid_request");

        let include_usage_null = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "stream_options": {"include_usage": null}
        }))
        .expect_err("stream_options.include_usage null should be rejected");
        assert_eq!(include_usage_null.api_error_code(), "invalid_request");

        let venice_params_null = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "venice_parameters": null
        }))
        .expect_err("venice_parameters null should be rejected");
        assert_eq!(venice_params_null.api_error_code(), "invalid_request");

        let invalid_stop = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "stop": ["ok", 42]
        }))
        .expect_err("mixed stop array should be rejected");
        assert_eq!(invalid_stop.api_error_code(), "invalid_request");
    }

    #[test]
    fn constructs_encrypted_request_for_non_streaming_mode() {
        let model_key = SecretKey::random(&mut rand_core::OsRng);
        let model_public_key = model_public_key_hex(&model_key);
        let codec = E2eeCodec::default();
        let request = parse(json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "stream": false,
            "temperature": 0.2,
            "max_tokens": 64,
            "stop": ["END"],
            "venice_parameters": {
                "include_venice_system_prompt": false,
                "enable_web_search": "off"
            }
        }));

        let prepared = request
            .into_venice_e2ee_request(&codec, &model_public_key)
            .expect("request should encrypt");

        assert!(!prepared.client_stream);
        assert!(prepared.upstream.stream);
        assert!(prepared.upstream.stream_options.include_usage);
        assert_eq!(prepared.upstream.temperature, Some(json!(0.2)));
        assert_eq!(prepared.upstream.max_tokens, Some(64));
        assert_eq!(
            prepared.upstream.stop,
            Some(StopSequence::Strings(vec!["END".to_owned()]))
        );
        assert_eq!(
            prepared.upstream.venice_parameters,
            VeniceParameters::default()
        );

        let payload = crate::e2ee::EncryptedPayload::from_hex(&prepared.upstream.messages)
            .expect("messages should be encrypted hex");
        let plaintext = codec
            .decrypt_content(&payload, &model_key)
            .expect("test model key should decrypt messages");
        let messages: Vec<NormalizedChatMessage> = serde_json::from_str(&plaintext)
            .expect("encrypted messages should contain normalized JSON");
        assert_eq!(messages, request.messages);
    }

    #[test]
    fn constructs_encrypted_request_for_streaming_mode_and_usage_option() {
        let model_key = SecretKey::random(&mut rand_core::OsRng);
        let model_public_key = model_public_key_hex(&model_key);
        let codec = E2eeCodec::default();
        let request = parse(json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "stream": true,
            "stream_options": {"include_usage": false}
        }));

        let prepared = request
            .into_venice_e2ee_request(&codec, &model_public_key)
            .expect("request should encrypt");

        assert!(prepared.client_stream);
        assert!(prepared.upstream.stream);
        assert!(!prepared.upstream.stream_options.include_usage);
    }
}
