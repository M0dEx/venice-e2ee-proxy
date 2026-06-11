use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::e2ee::{E2eeCodec, E2eeCodecError};
use crate::util::json_kind;

#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<NormalizedChatMessage>,
    pub stream: bool,
    pub stream_options: OpenAiStreamOptions,
    pub venice_parameters: VeniceParameters,
    pub passthrough: OpenAiPassthroughFields,
    pub tools: Vec<ChatToolDefinition>,
    pub tool_choice: ChatToolChoice,
    pub parallel_tool_calls: Option<bool>,
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
                "parallel_tool_calls",
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
        let tools = parse_tools(object.get("tools"))?;
        validate_tools(&tools)?;
        let tool_choice = parse_tool_choice(object.get("tool_choice"))?;
        validate_tool_choice(&tool_choice)?;
        let parallel_tool_calls = optional_bool(object, "parallel_tool_calls")?;

        Ok(Self {
            model,
            messages,
            stream,
            stream_options,
            venice_parameters,
            passthrough,
            tools,
            tool_choice,
            parallel_tool_calls,
        })
    }

    pub fn to_venice_e2ee_request(
        &self,
        codec: &E2eeCodec,
        model_public_key_hex: &str,
    ) -> Result<PreparedVeniceChatRequest, ChatConstructionError> {
        self.to_venice_e2ee_request_with_messages(codec, model_public_key_hex, &[], &[])
    }

    pub fn to_venice_e2ee_request_with_messages(
        &self,
        codec: &E2eeCodec,
        model_public_key_hex: &str,
        prefix_messages: &[NormalizedChatMessage],
        suffix_messages: &[NormalizedChatMessage],
    ) -> Result<PreparedVeniceChatRequest, ChatConstructionError> {
        let encrypted_messages = prefix_messages
            .iter()
            .chain(self.messages.iter())
            .chain(suffix_messages.iter())
            .map(|message| {
                let content = codec
                    .encrypt_content(&message.content, model_public_key_hex)
                    .map_err(ChatConstructionError::E2ee)?
                    .into_hex();
                Ok(NormalizedChatMessage::new(message.role.clone(), content))
            })
            .collect::<Result<Vec<_>, ChatConstructionError>>()?;

        Ok(PreparedVeniceChatRequest {
            client_stream: self.stream,
            upstream: VeniceE2eeChatRequest {
                model: self.model.clone(),
                messages: encrypted_messages,
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
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChatToolDefinition {
    Function {
        function: ChatToolFunctionDefinition,
    },
}

impl ChatToolDefinition {
    pub fn function(&self) -> &ChatToolFunctionDefinition {
        match self {
            Self::Function { function } => function,
        }
    }

    pub fn name(&self) -> &str {
        &self.function().name
    }

    pub fn parameters_schema(&self) -> Option<&Map<String, Value>> {
        self.function().parameters.as_ref().map(JsonSchema::as_map)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatToolFunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonSchema>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct JsonSchema(Map<String, Value>);

impl JsonSchema {
    pub fn as_map(&self) -> &Map<String, Value> {
        &self.0
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ChatToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function {
        name: String,
    },
}

impl<'de> Deserialize<'de> for ChatToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        ChatToolChoiceWire::deserialize(deserializer).map(Self::from)
    }
}

impl From<ChatToolChoiceWire> for ChatToolChoice {
    fn from(value: ChatToolChoiceWire) -> Self {
        match value {
            ChatToolChoiceWire::Mode(ChatToolChoiceMode::Auto) => Self::Auto,
            ChatToolChoiceWire::Mode(ChatToolChoiceMode::None) => Self::None,
            ChatToolChoiceWire::Mode(ChatToolChoiceMode::Required) => Self::Required,
            ChatToolChoiceWire::Object(ChatToolChoiceObject::Function { function }) => {
                Self::Function {
                    name: function.name,
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
enum ChatToolChoiceWire {
    Mode(ChatToolChoiceMode),
    Object(ChatToolChoiceObject),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChatToolChoiceMode {
    Auto,
    None,
    Required,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ChatToolChoiceObject {
    Function { function: ChatToolChoiceFunction },
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatToolChoiceFunction {
    name: String,
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
        deserialize_typed_value("stream_options", value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VeniceParameters {
    pub enable_e2ee: bool,
    pub include_venice_system_prompt: bool,
    pub enable_web_search: String,
}

impl Default for VeniceParameters {
    fn default() -> Self {
        Self {
            enable_e2ee: true,
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
        let raw: RawVeniceParameters = deserialize_typed_value("venice_parameters", value)?;
        let enable_e2ee = raw.enable_e2ee.unwrap_or(true);
        if !enable_e2ee {
            return Err(ChatRequestError::UnsupportedVeniceParameter {
                field: "venice_parameters.enable_e2ee",
                message: "Venice E2EE must remain enabled for encrypted proxy requests".to_owned(),
            });
        }

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
            enable_e2ee,
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
    pub messages: Vec<NormalizedChatMessage>,
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
    enable_e2ee: Option<bool>,
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
    // Deserialized only to enforce `"type": "text"`; never read afterwards.
    #[serde(rename = "type")]
    _kind: TextContentPartType,
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

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    pub(crate) fn invalid_field(field: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidField {
            field,
            message: message.into(),
        }
    }

    pub(crate) fn unsupported_content(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self::UnsupportedMessageContent {
            path: path.into(),
            message: message.into(),
        }
    }

    pub(crate) fn invalid_tool_history(message: impl Into<String>) -> Self {
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

    let rendered = tool_calls
        .iter()
        .map(render_assistant_tool_call)
        .collect::<Result<Vec<String>, ChatRequestError>>()?;

    Ok(Some(rendered.join("\n")))
}

fn render_assistant_tool_call(
    tool_call: &RawAssistantToolCall,
) -> Result<String, ChatRequestError> {
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

    Ok(format!(
        "<previous_tool_call id=\"{}\" name=\"{}\">\n{}\n</previous_tool_call>",
        xml_escape_attr(id),
        xml_escape_attr(name),
        canonical_arguments,
    ))
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
        let part: RawTextContentPart = serde_json::from_value(part.clone()).map_err(|source| {
            ChatRequestError::unsupported_content(
                format!("{path}[{index}]"),
                format!("text content part must match {{type:\"text\", text:string}}: {source}"),
            )
        })?;
        text.push_str(&part.text);
    }
    Ok(text)
}

fn parse_tools(value: Option<&Value>) -> Result<Vec<ChatToolDefinition>, ChatRequestError> {
    match value {
        None => Ok(Vec::new()),
        Some(value) => deserialize_typed_value("tools", value),
    }
}

fn parse_tool_choice(value: Option<&Value>) -> Result<ChatToolChoice, ChatRequestError> {
    let Some(value) = value else {
        return Ok(ChatToolChoice::default());
    };
    deserialize_typed_value::<Option<ChatToolChoice>>("tool_choice", value)
        .map(|choice| choice.unwrap_or_default())
}

fn validate_tools(tools: &[ChatToolDefinition]) -> Result<(), ChatRequestError> {
    if tools.iter().any(|tool| tool.name().trim().is_empty()) {
        return Err(ChatRequestError::invalid_field(
            "tools",
            "function tool names must not be empty",
        ));
    }
    Ok(())
}

fn validate_tool_choice(tool_choice: &ChatToolChoice) -> Result<(), ChatRequestError> {
    if let ChatToolChoice::Function { name } = tool_choice
        && name.trim().is_empty()
    {
        return Err(ChatRequestError::invalid_field(
            "tool_choice",
            "function tool_choice name must not be empty",
        ));
    }
    Ok(())
}

fn validate_ignored_client_only_fields(
    object: &Map<String, Value>,
) -> Result<(), ChatRequestError> {
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
        hex::encode(public_key.to_encoded_point(false).as_bytes())
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
    fn normalizes_parallel_assistant_tool_call_history() {
        let request = parse(json!({
            "model": "e2ee-test",
            "messages": [
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_one",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": "{\"query\":\"Venice E2EE\"}"
                            }
                        },
                        {
                            "id": "call_two",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"Venice\"}"
                            }
                        }
                    ]
                }
            ]
        }));

        assert_eq!(
            request.messages[0],
            NormalizedChatMessage::new(
                "assistant",
                "<previous_tool_call id=\"call_one\" name=\"search_web\">\n{\"query\":\"Venice E2EE\"}\n</previous_tool_call>\n<previous_tool_call id=\"call_two\" name=\"get_weather\">\n{\"city\":\"Venice\"}\n</previous_tool_call>",
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
    fn parses_tools_into_typed_function_envelopes() {
        let request = parse(json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            }]
        }));

        assert_eq!(request.tools.len(), 1);
        let tool = &request.tools[0];
        let function = tool.function();
        assert_eq!(tool.name(), "search_web");
        assert_eq!(function.description.as_deref(), Some("Search the web"));
        assert_eq!(
            tool.parameters_schema()
                .and_then(|schema| schema.get("required")),
            Some(&json!(["query"]))
        );
        assert_eq!(
            serde_json::to_value(tool).expect("tool should serialize"),
            json!({
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            })
        );
    }

    #[test]
    fn parses_tool_choice_into_typed_shapes() {
        let required = parse(json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "tool_choice": "required"
        }));
        assert_eq!(required.tool_choice, ChatToolChoice::Required);

        let specific = parse(json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "tool_choice": {"type":"function", "function":{"name":"search_web"}}
        }));
        assert_eq!(
            specific.tool_choice,
            ChatToolChoice::Function {
                name: "search_web".to_owned()
            }
        );

        let null_choice = parse(json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "tool_choice": null
        }));
        assert_eq!(null_choice.tool_choice, ChatToolChoice::Auto);
    }

    #[test]
    fn rejects_invalid_tool_and_tool_choice_shapes() {
        for body in [
            json!({
                "model": "e2ee-test",
                "messages": [{"role":"user", "content":"hi"}],
                "tools": ["not an object"]
            }),
            json!({
                "model": "e2ee-test",
                "messages": [{"role":"user", "content":"hi"}],
                "tools": [{"type":"web_search", "function":{"name":"search_web"}}]
            }),
            json!({
                "model": "e2ee-test",
                "messages": [{"role":"user", "content":"hi"}],
                "tools": [{"type":"function", "function":{"name":"search_web", "description": 42}}]
            }),
            json!({
                "model": "e2ee-test",
                "messages": [{"role":"user", "content":"hi"}],
                "tools": [{"type":"function", "function":{"name":"search_web", "parameters": []}}]
            }),
            json!({
                "model": "e2ee-test",
                "messages": [{"role":"user", "content":"hi"}],
                "tools": [{"type":"function", "function":{}}]
            }),
            json!({
                "model": "e2ee-test",
                "messages": [{"role":"user", "content":"hi"}],
                "tools": [{"type":"function", "function":{"name":""}}]
            }),
            json!({
                "model": "e2ee-test",
                "messages": [{"role":"user", "content":"hi"}],
                "tools": [{"type":"function", "function":{"name":"search_web", "extra": true}}]
            }),
            json!({
                "model": "e2ee-test",
                "messages": [{"role":"user", "content":"hi"}],
                "tool_choice": 42
            }),
            json!({
                "model": "e2ee-test",
                "messages": [{"role":"user", "content":"hi"}],
                "tool_choice": "always"
            }),
            json!({
                "model": "e2ee-test",
                "messages": [{"role":"user", "content":"hi"}],
                "tool_choice": {"type":"web_search", "function":{"name":"search_web"}}
            }),
            json!({
                "model": "e2ee-test",
                "messages": [{"role":"user", "content":"hi"}],
                "tool_choice": {"type":"function", "function":{"name":""}}
            }),
        ] {
            let error = ChatCompletionRequest::parse(&body)
                .expect_err("invalid tool shape should be rejected");
            assert_eq!(error.api_error_code(), "invalid_request");
        }
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
    fn serde_layer_rejects_unknown_nested_fields_and_wrong_types() {
        let stream_options_unknown = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "stream_options": {"include_usage": true, "extra": 1}
        }))
        .expect_err("unknown stream_options field should be rejected");
        assert_eq!(stream_options_unknown.api_error_code(), "invalid_request");
        assert!(
            stream_options_unknown
                .to_string()
                .contains("unknown field `extra`"),
            "unexpected message: {stream_options_unknown}"
        );

        let include_usage_string = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "stream_options": {"include_usage": "yes"}
        }))
        .expect_err("non-boolean include_usage should be rejected");
        assert_eq!(include_usage_string.api_error_code(), "invalid_request");
        assert!(
            include_usage_string
                .to_string()
                .contains("expected boolean, got string"),
            "unexpected message: {include_usage_string}"
        );

        let venice_unknown = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "venice_parameters": {"unknown_param": true}
        }))
        .expect_err("unknown venice_parameters field should be rejected");
        assert_eq!(venice_unknown.api_error_code(), "invalid_request");
        assert!(
            venice_unknown
                .to_string()
                .contains("unknown field `unknown_param`"),
            "unexpected message: {venice_unknown}"
        );

        let enable_e2ee_string = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "venice_parameters": {"enable_e2ee": "yes"}
        }))
        .expect_err("non-boolean enable_e2ee should be rejected");
        assert_eq!(enable_e2ee_string.api_error_code(), "invalid_request");
        assert!(
            enable_e2ee_string
                .to_string()
                .contains("expected boolean, got string"),
            "unexpected message: {enable_e2ee_string}"
        );

        let web_search_number = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "venice_parameters": {"enable_web_search": 42}
        }))
        .expect_err("non-string/boolean enable_web_search should be rejected");
        assert_eq!(web_search_number.api_error_code(), "invalid_request");
        assert!(
            web_search_number
                .to_string()
                .contains("expected string or boolean, got number"),
            "unexpected message: {web_search_number}"
        );

        let null_enable_e2ee = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "venice_parameters": {"enable_e2ee": null}
        }))
        .expect_err("null enable_e2ee should be rejected");
        assert_eq!(null_enable_e2ee.api_error_code(), "invalid_request");

        let content_part_unknown = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":[{"type":"text", "text":"hi", "extra":1}]}]
        }))
        .expect_err("unknown content part field should be rejected");
        assert_eq!(
            content_part_unknown.api_error_code(),
            "unsupported_message_content"
        );
        assert!(
            content_part_unknown
                .to_string()
                .contains("unknown field `extra`"),
            "unexpected message: {content_part_unknown}"
        );

        let content_part_non_object = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":["plain string part"]}]
        }))
        .expect_err("non-object content part should be rejected");
        assert_eq!(
            content_part_non_object.api_error_code(),
            "unsupported_message_content"
        );
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
            .to_venice_e2ee_request(&codec, &model_public_key)
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

        assert_eq!(prepared.upstream.messages.len(), request.messages.len());
        assert_eq!(prepared.upstream.messages[0].role, "user");
        assert_ne!(
            prepared.upstream.messages[0].content,
            request.messages[0].content
        );
        let payload =
            crate::e2ee::EncryptedPayload::from_hex(&prepared.upstream.messages[0].content)
                .expect("message content should be encrypted hex");
        let plaintext = codec
            .decrypt_content(&payload, &model_key)
            .expect("test model key should decrypt message content");
        assert_eq!(plaintext, request.messages[0].content);
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
            .to_venice_e2ee_request(&codec, &model_public_key)
            .expect("request should encrypt");

        assert!(prepared.client_stream);
        assert!(prepared.upstream.stream);
        assert!(!prepared.upstream.stream_options.include_usage);
    }

    #[test]
    fn constructs_encrypted_request_with_tool_controller_and_retry_prompt() {
        let model_key = SecretKey::random(&mut rand_core::OsRng);
        let model_public_key = model_public_key_hex(&model_key);
        let codec = E2eeCodec::default();
        let request = parse(json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "tools": [{"type":"function", "function":{"name":"search_web", "parameters":{"type":"object"}}}],
            "tool_choice": "required"
        }));
        let controller = NormalizedChatMessage::new("system", "controller prompt");
        let correction = NormalizedChatMessage::new("system", "retry prompt");

        let prepared = request
            .to_venice_e2ee_request_with_messages(
                &codec,
                &model_public_key,
                std::slice::from_ref(&controller),
                std::slice::from_ref(&correction),
            )
            .expect("request should encrypt");

        assert_eq!(prepared.upstream.messages.len(), 3);
        assert_eq!(prepared.upstream.messages[0].role, "system");
        assert_eq!(prepared.upstream.messages[1].role, "user");
        assert_eq!(prepared.upstream.messages[2].role, "system");

        let decrypted = prepared
            .upstream
            .messages
            .iter()
            .map(|message| {
                let payload = crate::e2ee::EncryptedPayload::from_hex(&message.content)
                    .expect("message content should be encrypted hex");
                codec
                    .decrypt_content(&payload, &model_key)
                    .expect("test model key should decrypt message content")
            })
            .collect::<Vec<_>>();
        assert_eq!(decrypted, vec!["controller prompt", "hi", "retry prompt"]);
        assert!(
            !serde_json::to_value(&prepared.upstream)
                .expect("upstream request should serialize")
                .as_object()
                .expect("upstream request should be object")
                .contains_key("tools")
        );
    }
}
