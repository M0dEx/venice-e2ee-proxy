//! OpenAI-style tool-call emulation.
//!
//! Venice E2EE responses do not expose native function calls, so this module
//! recognizes the v0.1 marker protocol in decrypted assistant text, validates it
//! against the request's OpenAI `tools`, and builds prompt text for the encrypted
//! controller/correction requests.

use std::{collections::HashSet, time::Duration};

use serde_json::{Map, Value};

use crate::{
    config::{ToolMode, ToolsConfig},
    openai::chat::{
        ChatCompletionRequest, ChatRequestError, ChatToolChoice, ChatToolDefinition,
        NormalizedChatMessage,
    },
};

#[derive(Debug, Clone)]
pub struct ToolEmulationContext {
    config: ToolsConfig,
    tools: Vec<ChatToolDefinition>,
    tool_schemas_json: String,
    require_tool_call: bool,
}

impl ToolEmulationContext {
    pub fn from_request(
        config: &ToolsConfig,
        request: &ChatCompletionRequest,
    ) -> Result<Option<Self>, ChatRequestError> {
        if !config.enabled || config.mode == ToolMode::None {
            return Ok(None);
        }
        if matches!(request.tool_choice, ChatToolChoice::None) {
            return Ok(None);
        }
        if request.tools.is_empty() {
            if matches!(
                request.tool_choice,
                ChatToolChoice::Required | ChatToolChoice::Function { .. }
            ) {
                return Err(ChatRequestError::invalid_field(
                    "tool_choice",
                    "tool_choice requires at least one function tool",
                ));
            }
            return Ok(None);
        }
        if request.parallel_tool_calls == Some(true) {
            return Err(ChatRequestError::invalid_field(
                "parallel_tool_calls",
                "parallel tool calls are not supported by the E2EE proxy v0.1 tool emulator",
            ));
        }

        let mut seen_names = HashSet::new();
        for tool in &request.tools {
            if !seen_names.insert(tool.name()) {
                return Err(ChatRequestError::invalid_field(
                    "tools",
                    format!("duplicate function tool name {:?}", tool.name()),
                ));
            }
            if config.validate_json_schema
                && let Some(schema) = tool.parameters_schema()
            {
                validate_schema_shape(schema).map_err(|message| {
                    ChatRequestError::invalid_field(
                        "tools",
                        format!(
                            "tool {:?} has an unsupported or invalid parameters schema: {message}",
                            tool.name()
                        ),
                    )
                })?;
            }
        }

        let (tools, require_tool_call) = match &request.tool_choice {
            ChatToolChoice::Auto => (request.tools.clone(), false),
            ChatToolChoice::Required => (request.tools.clone(), true),
            ChatToolChoice::Function { name } => {
                let selected = request
                    .tools
                    .iter()
                    .find(|tool| tool.name() == name)
                    .cloned()
                    .ok_or_else(|| {
                        ChatRequestError::invalid_field(
                            "tool_choice",
                            format!("requested function tool {name:?} is not present in tools"),
                        )
                    })?;
                (vec![selected], true)
            }
            ChatToolChoice::None => unreachable!("tool_choice none returned above"),
        };

        let tool_schemas_json = serde_json::to_string(&tools).map_err(|source| {
            ChatRequestError::invalid_field(
                "tools",
                format!("tool schemas could not be serialized for the controller prompt: {source}"),
            )
        })?;

        Ok(Some(Self {
            config: config.clone(),
            tools,
            tool_schemas_json,
            require_tool_call,
        }))
    }

    pub fn config(&self) -> &ToolsConfig {
        &self.config
    }

    pub fn max_retries(&self) -> u32 {
        self.config.max_retries
    }

    pub fn marker_timeout(&self) -> Duration {
        Duration::from_millis(self.config.tool_call_marker_timeout_ms)
    }

    pub fn controller_message(&self) -> NormalizedChatMessage {
        let requirement = if self.require_tool_call {
            "You must call exactly one tool. Do not answer the user directly. Output exactly one tool call using this format and nothing else:"
        } else {
            "If a tool is required, do not answer the user directly. Output exactly one tool call using this format and nothing else:"
        };
        let optional_rule = if self.require_tool_call {
            String::new()
        } else {
            format!(
                "\n- If no tool is needed, answer normally and do not use {}.",
                self.config.marker_start
            )
        };

        // Venice E2EE rejects repeated tool-controller `system` prompts when a
        // multi-turn request includes prior assistant/tool history. Keep the
        // controller instruction model-visible as a user message so real
        // OpenAI-style tool conversations can continue across turns.
        NormalizedChatMessage::new(
            "user",
            format!(
                "You have access to tools.\n\n{requirement}\n\n{}\n{}\n{}\n\nRules:\n- TOOL_NAME must exactly match one available tool name.\n- arguments must be valid JSON and must satisfy the tool schema.\n- Call at most one tool.\n- Do not include markdown fences.\n- Do not include explanations.{optional_rule}\n\nAvailable tools:\n{}",
                self.config.marker_start,
                r#"{"name":"TOOL_NAME","arguments":{...}}"#,
                self.config.marker_end,
                self.tool_schemas_json,
            ),
        )
    }

    pub fn correction_message(
        &self,
        validation_error: &str,
        invalid_output: &str,
    ) -> NormalizedChatMessage {
        NormalizedChatMessage::new(
            "system",
            format!(
                "Your previous response attempted a tool call, but it was invalid.\n\nValidation error:\n{validation_error}\n\nInvalid output:\n{invalid_output}\n\nYou must now return exactly one valid tool call and nothing else.\n\nUse this exact format:\n\n{}\n{}\n{}\n\nRules:\n- TOOL_NAME must exactly match one of the available tools.\n- arguments must be a JSON object.\n- arguments must satisfy the tool schema.\n- Do not include markdown fences.\n- Do not include explanations.\n- Do not answer the user directly.\n\nAvailable tools:\n{}",
                self.config.marker_start,
                r#"{"name":"TOOL_NAME","arguments":{...}}"#,
                self.config.marker_end,
                self.tool_schemas_json,
            ),
        )
    }

    pub fn classify_assistant_output(&self, output: &str) -> ToolOutputClassification {
        match self.tool_call_marker_block(output) {
            Ok(Some(marker)) => {
                return match self.validate_marker(marker) {
                    Ok(tool_call) => ToolOutputClassification::ToolCall(tool_call),
                    Err(error) => ToolOutputClassification::InvalidToolCall {
                        error,
                        invalid_output: output.to_owned(),
                    },
                };
            }
            Err(error) => {
                return ToolOutputClassification::InvalidToolCall {
                    error,
                    invalid_output: output.to_owned(),
                };
            }
            Ok(None) => {}
        }

        match scan_initial_marker_prefix(output, &self.config) {
            InitialMarkerScan::ToolCall => match self.validate_marker(output) {
                Ok(tool_call) => ToolOutputClassification::ToolCall(tool_call),
                Err(error) => ToolOutputClassification::InvalidToolCall {
                    error,
                    invalid_output: output.to_owned(),
                },
            },
            InitialMarkerScan::NormalText | InitialMarkerScan::Pending => {
                if self.require_tool_call {
                    ToolOutputClassification::InvalidToolCall {
                        error: ToolCallValidationError::new(
                            "expected the assistant response to include a tool_call marker",
                        ),
                        invalid_output: output.to_owned(),
                    }
                } else {
                    ToolOutputClassification::NormalText
                }
            }
        }
    }

    fn tool_call_marker_block<'a>(
        &self,
        output: &'a str,
    ) -> Result<Option<&'a str>, ToolCallValidationError> {
        let Some(start) = output.find(&self.config.marker_start) else {
            return Ok(None);
        };
        let after_start = &output[start..];
        let Some(end_start) = after_start.find(&self.config.marker_end) else {
            return Err(ToolCallValidationError::new(format!(
                "tool call must end with {}",
                self.config.marker_end
            )));
        };
        let end = start + end_start + self.config.marker_end.len();
        Ok(Some(&output[start..end]))
    }

    pub fn validate_marker(
        &self,
        output: &str,
    ) -> Result<ValidatedToolCall, ToolCallValidationError> {
        if output.len() > self.config.tool_call_max_bytes {
            return Err(ToolCallValidationError::new(format!(
                "tool call marker exceeded max size of {} bytes",
                self.config.tool_call_max_bytes
            )));
        }

        let trimmed = output.trim();
        if !trimmed.starts_with(&self.config.marker_start) {
            return Err(ToolCallValidationError::new(format!(
                "tool call must start with {}",
                self.config.marker_start
            )));
        }
        if !trimmed.ends_with(&self.config.marker_end) {
            return Err(ToolCallValidationError::new(format!(
                "tool call must end with {} and contain no trailing text",
                self.config.marker_end
            )));
        }
        if trimmed.matches(&self.config.marker_start).count() != 1
            || trimmed.matches(&self.config.marker_end).count() != 1
        {
            return Err(ToolCallValidationError::new(
                "expected exactly one tool_call marker",
            ));
        }

        let inner_start = self.config.marker_start.len();
        let inner_end = trimmed.len() - self.config.marker_end.len();
        let inner = trimmed[inner_start..inner_end].trim();
        let value = parse_tool_call_json(inner)?;
        let object = value
            .as_object()
            .ok_or_else(|| ToolCallValidationError::new("tool call JSON must be an object"))?;
        let (name, arguments) = extract_tool_call_name_and_arguments(object)?;
        if name.trim().is_empty() {
            return Err(ToolCallValidationError::new(
                "tool call name must not be empty",
            ));
        }
        let tool = self
            .tools
            .iter()
            .find(|tool| tool.name() == name)
            .ok_or_else(|| ToolCallValidationError::new(format!("unknown tool name {name:?}")))?;
        let arguments = normalize_tool_call_arguments(arguments)?;

        if self.config.validate_json_schema
            && let Some(schema) = tool.parameters_schema()
        {
            validate_value_against_schema(&arguments, schema, "arguments").map_err(|message| {
                ToolCallValidationError::new(format!(
                    "tool call arguments do not satisfy schema: {message}"
                ))
            })?;
        }

        let arguments_json = serde_json::to_string(&arguments).map_err(|source| {
            ToolCallValidationError::new(format!(
                "tool call arguments could not be serialized as JSON: {source}"
            ))
        })?;

        Ok(ValidatedToolCall {
            id: format!("call_{}", uuid::Uuid::new_v4().simple()),
            name: name.to_owned(),
            arguments_json,
        })
    }
}

fn parse_tool_call_json(input: &str) -> Result<Value, ToolCallValidationError> {
    serde_json::from_str(input).or_else(|strict_error| {
        json5::from_str(input).map_err(|json5_error| {
            ToolCallValidationError::new(format!(
                "tool call JSON is invalid: {strict_error}; JSON5 fallback failed: {json5_error}"
            ))
        })
    })
}

fn extract_tool_call_name_and_arguments<'a>(
    object: &'a Map<String, Value>,
) -> Result<(&'a str, &'a Value), ToolCallValidationError> {
    if let Some(function) = object.get("function") {
        let function = function.as_object().ok_or_else(|| {
            ToolCallValidationError::new("tool call function field must be an object")
        })?;
        let name = string_member(function, &["name"])?;
        let arguments = value_member(function, &["arguments", "parameters"])?;
        return Ok((name, arguments));
    }

    let name = string_member(object, &["name", "tool_name", "tool"])?;
    let arguments = value_member(object, &["arguments", "parameters"])?;
    Ok((name, arguments))
}

fn string_member<'a>(
    object: &'a Map<String, Value>,
    names: &[&'static str],
) -> Result<&'a str, ToolCallValidationError> {
    for name in names {
        if let Some(value) = object.get(*name) {
            return value.as_str().ok_or_else(|| {
                ToolCallValidationError::new(format!("tool call {name} field must be a string"))
            });
        }
    }

    Err(ToolCallValidationError::new(format!(
        "tool call is missing {} field",
        names.join(" or ")
    )))
}

fn value_member<'a>(
    object: &'a Map<String, Value>,
    names: &[&'static str],
) -> Result<&'a Value, ToolCallValidationError> {
    for name in names {
        if let Some(value) = object.get(*name) {
            return Ok(value);
        }
    }

    Err(ToolCallValidationError::new(format!(
        "tool call is missing {} field",
        names.join(" or ")
    )))
}

fn normalize_tool_call_arguments(arguments: &Value) -> Result<Value, ToolCallValidationError> {
    match arguments {
        Value::Object(_) => Ok(arguments.clone()),
        Value::String(arguments) => {
            let value: Value = serde_json::from_str(arguments).map_err(|source| {
                ToolCallValidationError::new(format!(
                    "tool call arguments JSON is invalid: {source}"
                ))
            })?;
            if value.is_object() {
                Ok(value)
            } else {
                Err(ToolCallValidationError::new(
                    "tool call arguments JSON string must decode to an object",
                ))
            }
        }
        _ => Err(ToolCallValidationError::new(
            "tool call arguments must be a JSON object or JSON object string",
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolOutputClassification {
    NormalText,
    ToolCall(ValidatedToolCall),
    InvalidToolCall {
        error: ToolCallValidationError,
        invalid_output: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

impl ValidatedToolCall {
    pub fn to_openai_value(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        })
    }

    pub fn to_openai_streaming_value(&self) -> Value {
        serde_json::json!({
            "index": 0,
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallValidationError {
    message: String,
}

impl ToolCallValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for ToolCallValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ToolCallValidationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialMarkerScan {
    Pending,
    NormalText,
    ToolCall,
}

pub fn scan_initial_marker_prefix(output: &str, config: &ToolsConfig) -> InitialMarkerScan {
    if output.is_empty() {
        return InitialMarkerScan::Pending;
    }

    let Some((first_non_ws, _)) = output.char_indices().find(|(_, ch)| !ch.is_whitespace()) else {
        return if output.len() >= config.initial_marker_scan_bytes {
            InitialMarkerScan::NormalText
        } else {
            InitialMarkerScan::Pending
        };
    };

    let prefix = &output[first_non_ws..];
    if prefix.starts_with(&config.marker_start) {
        return InitialMarkerScan::ToolCall;
    }
    if config.marker_start.starts_with(prefix) && output.len() < config.initial_marker_scan_bytes {
        return InitialMarkerScan::Pending;
    }

    InitialMarkerScan::NormalText
}

#[derive(Debug, Clone)]
pub struct ToolCallMarkerBuffer {
    config: ToolsConfig,
    output: String,
    elapsed: Duration,
}

impl ToolCallMarkerBuffer {
    pub fn new(config: &ToolsConfig) -> Self {
        Self {
            config: config.clone(),
            output: String::new(),
            elapsed: Duration::ZERO,
        }
    }

    pub fn push(
        &mut self,
        chunk: &str,
        elapsed: Duration,
    ) -> Result<ToolCallBufferStatus, ToolCallValidationError> {
        self.elapsed = elapsed;
        if self.elapsed > Duration::from_millis(self.config.tool_call_marker_timeout_ms) {
            return Err(ToolCallValidationError::new(format!(
                "tool call marker did not close within {} ms",
                self.config.tool_call_marker_timeout_ms
            )));
        }
        self.output.push_str(chunk);
        if self.output.len() > self.config.tool_call_max_bytes {
            return Err(ToolCallValidationError::new(format!(
                "tool call marker exceeded max size of {} bytes",
                self.config.tool_call_max_bytes
            )));
        }
        if self.output.contains(&self.config.marker_end) {
            return Ok(ToolCallBufferStatus::Complete(self.output.clone()));
        }
        Ok(ToolCallBufferStatus::Pending)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallBufferStatus {
    Pending,
    Complete(String),
}

fn validate_schema_shape(schema: &Map<String, Value>) -> Result<(), String> {
    validate_schema_object_shape(schema, "schema")
}

fn validate_schema_object_shape(object: &Map<String, Value>, path: &str) -> Result<(), String> {
    if let Some(kind) = object.get("type") {
        validate_schema_type_shape(kind, &format!("{path}.type"))?;
    }
    if let Some(required) = object.get("required") {
        let required = required
            .as_array()
            .ok_or_else(|| format!("{path}.required must be an array"))?;
        if required.iter().any(|value| !value.is_string()) {
            return Err(format!("{path}.required must contain only strings"));
        }
    }
    if let Some(properties) = object.get("properties") {
        let properties = properties
            .as_object()
            .ok_or_else(|| format!("{path}.properties must be an object"))?;
        for (name, schema) in properties {
            let schema = schema
                .as_object()
                .ok_or_else(|| format!("{path}.properties.{name} must be an object"))?;
            validate_schema_object_shape(schema, &format!("{path}.properties.{name}"))?;
        }
    }
    if let Some(items) = object.get("items") {
        let items = items
            .as_object()
            .ok_or_else(|| format!("{path}.items must be an object"))?;
        validate_schema_object_shape(items, &format!("{path}.items"))?;
    }
    if let Some(additional) = object.get("additionalProperties") {
        match additional {
            Value::Bool(_) => {}
            Value::Object(additional) => {
                validate_schema_object_shape(additional, &format!("{path}.additionalProperties"))?
            }
            _ => {
                return Err(format!(
                    "{path}.additionalProperties must be a boolean or object"
                ));
            }
        }
    }
    if let Some(enum_values) = object.get("enum")
        && !enum_values.is_array()
    {
        return Err(format!("{path}.enum must be an array"));
    }
    Ok(())
}

fn validate_schema_type_shape(value: &Value, path: &str) -> Result<(), String> {
    match value {
        Value::String(kind) => validate_schema_type_name(kind, path),
        Value::Array(kinds) => {
            if kinds.is_empty() {
                return Err(format!("{path} must not be an empty array"));
            }
            for kind in kinds {
                let kind = kind
                    .as_str()
                    .ok_or_else(|| format!("{path} array must contain only strings"))?;
                validate_schema_type_name(kind, path)?;
            }
            Ok(())
        }
        _ => Err(format!("{path} must be a string or array of strings")),
    }
}

fn validate_schema_type_name(kind: &str, path: &str) -> Result<(), String> {
    match kind {
        "object" | "array" | "string" | "integer" | "number" | "boolean" | "null" => Ok(()),
        other => Err(format!(
            "{path} contains unsupported JSON schema type {other:?}"
        )),
    }
}

fn validate_value_against_schema(
    value: &Value,
    schema: &Map<String, Value>,
    path: &str,
) -> Result<(), String> {
    if let Some(enum_values) = schema.get("enum").and_then(Value::as_array)
        && !enum_values.iter().any(|enum_value| enum_value == value)
    {
        return Err(format!("{path} is not one of the allowed enum values"));
    }

    if let Some(kind) = schema.get("type")
        && !schema_type_matches(value, kind)
    {
        return Err(format!(
            "{path} expected type {}, got {}",
            schema_type_description(kind),
            value_kind(value)
        ));
    }

    if schema_implies_object(schema) {
        let object = value
            .as_object()
            .ok_or_else(|| format!("{path} expected object, got {}", value_kind(value)))?;
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for field in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(field) {
                    return Err(format!("{path}.{field} is required"));
                }
            }
        }
        let properties = schema.get("properties").and_then(Value::as_object);
        if let Some(properties) = properties {
            for (field, property_schema) in properties {
                if let Some(property_value) = object.get(field) {
                    let property_path = format!("{path}.{field}");
                    let property_schema = schema_value_as_object(property_schema, &property_path)?;
                    validate_value_against_schema(property_value, property_schema, &property_path)?;
                }
            }
        }
        if let Some(additional) = schema.get("additionalProperties") {
            match additional {
                Value::Bool(false) => {
                    for field in object.keys() {
                        if properties.is_none_or(|properties| !properties.contains_key(field)) {
                            return Err(format!("{path}.{field} is not allowed by schema"));
                        }
                    }
                }
                Value::Object(additional_schema) => {
                    for (field, additional_value) in object {
                        if properties.is_none_or(|properties| !properties.contains_key(field)) {
                            validate_value_against_schema(
                                additional_value,
                                additional_schema,
                                &format!("{path}.{field}"),
                            )?;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if schema_implies_array(schema) {
        let array = value
            .as_array()
            .ok_or_else(|| format!("{path} expected array, got {}", value_kind(value)))?;
        if let Some(items_schema) = schema.get("items") {
            for (index, item) in array.iter().enumerate() {
                let item_path = format!("{path}[{index}]");
                let items_schema = schema_value_as_object(items_schema, &item_path)?;
                validate_value_against_schema(item, items_schema, &item_path)?;
            }
        }
    }

    Ok(())
}

fn schema_value_as_object<'a>(
    schema: &'a Value,
    path: &str,
) -> Result<&'a Map<String, Value>, String> {
    schema
        .as_object()
        .ok_or_else(|| format!("{path} schema must be an object"))
}

fn schema_implies_object(schema: &Map<String, Value>) -> bool {
    schema
        .get("type")
        .is_some_and(|kind| schema_type_includes(kind, "object"))
        || schema.contains_key("properties")
        || schema.contains_key("required")
        || schema.contains_key("additionalProperties")
}

fn schema_implies_array(schema: &Map<String, Value>) -> bool {
    schema
        .get("type")
        .is_some_and(|kind| schema_type_includes(kind, "array"))
        || schema.contains_key("items")
}

fn schema_type_matches(value: &Value, kind: &Value) -> bool {
    match kind {
        Value::String(kind) => value_matches_schema_type(value, kind),
        Value::Array(kinds) => kinds
            .iter()
            .filter_map(Value::as_str)
            .any(|kind| value_matches_schema_type(value, kind)),
        _ => true,
    }
}

fn schema_type_includes(kind: &Value, expected: &str) -> bool {
    match kind {
        Value::String(kind) => kind == expected,
        Value::Array(kinds) => kinds
            .iter()
            .filter_map(Value::as_str)
            .any(|kind| kind == expected),
        _ => false,
    }
}

fn value_matches_schema_type(value: &Value, kind: &str) -> bool {
    match kind {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn schema_type_description(kind: &Value) -> String {
    match kind {
        Value::String(kind) => kind.clone(),
        Value::Array(kinds) => kinds
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" or "),
        _ => "unknown".to_owned(),
    }
}

fn value_kind(value: &Value) -> &'static str {
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
    use std::time::Duration;

    use serde_json::json;

    use super::*;
    use crate::config::ToolsConfig;

    fn request_with_tool(arguments_schema: Value) -> ChatCompletionRequest {
        ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": arguments_schema
                }
            }]
        }))
        .expect("request should parse")
    }

    fn context_for_request(request: &ChatCompletionRequest) -> ToolEmulationContext {
        ToolEmulationContext::from_request(&ToolsConfig::default(), request)
            .expect("tool context should build")
            .expect("tools should activate")
    }

    #[test]
    fn scans_initial_marker_prefix_until_threshold() {
        let mut config = ToolsConfig {
            initial_marker_scan_bytes: 6,
            ..ToolsConfig::default()
        };

        assert_eq!(
            scan_initial_marker_prefix("   ", &config),
            InitialMarkerScan::Pending
        );
        assert_eq!(
            scan_initial_marker_prefix("<tool", &config),
            InitialMarkerScan::Pending
        );
        assert_eq!(
            scan_initial_marker_prefix("<tool_", &config),
            InitialMarkerScan::NormalText
        );
        assert_eq!(
            scan_initial_marker_prefix(" hello", &config),
            InitialMarkerScan::NormalText
        );

        config.initial_marker_scan_bytes = 16;
        assert_eq!(
            scan_initial_marker_prefix("\n<tool_call>", &config),
            InitialMarkerScan::ToolCall
        );
    }

    #[test]
    fn buffers_complete_marker_and_reports_limits() {
        let config = ToolsConfig {
            tool_call_max_bytes: 64,
            tool_call_marker_timeout_ms: 50,
            ..ToolsConfig::default()
        };
        let mut buffer = ToolCallMarkerBuffer::new(&config);

        assert_eq!(
            buffer
                .push("<tool_call>{", Duration::from_millis(10))
                .expect("partial marker should buffer"),
            ToolCallBufferStatus::Pending
        );
        assert!(matches!(
            buffer
                .push("\"name\":\"x\"}</tool_call>", Duration::from_millis(20))
                .expect("closing marker should complete"),
            ToolCallBufferStatus::Complete(_)
        ));

        let mut too_large = ToolCallMarkerBuffer::new(&config);
        let err = too_large
            .push(
                "<tool_call>012345678901234567890123456789012345678901234567890123456789",
                Duration::ZERO,
            )
            .expect_err("oversized marker should fail");
        assert!(err.message().contains("exceeded max size"));

        let mut timed_out = ToolCallMarkerBuffer::new(&config);
        let err = timed_out
            .push("<tool_call>", Duration::from_millis(51))
            .expect_err("marker timeout should fail");
        assert!(err.message().contains("did not close"));
    }

    #[test]
    fn validates_valid_tool_call_marker() {
        let request = request_with_tool(json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": false
        }));
        let context = context_for_request(&request);

        let classification = context.classify_assistant_output(
            "\n<tool_call>\n{\"name\":\"search_web\",\"arguments\":{\"query\":\"Venice\"}}\n</tool_call>\n",
        );

        let ToolOutputClassification::ToolCall(tool_call) = classification else {
            panic!("expected valid tool call");
        };
        assert_eq!(tool_call.name, "search_web");
        assert_eq!(tool_call.arguments_json, "{\"query\":\"Venice\"}");
    }

    #[test]
    fn validates_common_tool_call_marker_variants() {
        let request = request_with_tool(json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": false
        }));
        let context = context_for_request(&request);

        for marker in [
            r#"<tool_call>{"name":"search_web","parameters":{"query":"Venice"}}</tool_call>"#,
            r#"<tool_call>{"tool_name":"search_web","arguments":{"query":"Venice"}}</tool_call>"#,
            r#"<tool_call>{"function":{"name":"search_web","arguments":{"query":"Venice"}}}</tool_call>"#,
            r#"<tool_call>{"function":{"name":"search_web","parameters":{"query":"Venice"}}}</tool_call>"#,
            r#"<tool_call>{"type":"function","name":"search_web","arguments":"{\"query\":\"Venice\"}"}</tool_call>"#,
            r#"<tool_call>{'name':'search_web','arguments':{'query':'Venice'}}</tool_call>"#,
            r#"<tool_call>{name: 'search_web', arguments: {query: 'Venice'}}</tool_call>"#,
        ] {
            let tool_call = context
                .validate_marker(marker)
                .expect("common marker variant should validate");
            assert_eq!(tool_call.name, "search_web");
            assert_eq!(tool_call.arguments_json, "{\"query\":\"Venice\"}");
        }
    }

    #[test]
    fn rejects_invalid_json_unknown_tool_and_schema_mismatch() {
        let request = request_with_tool(json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": false
        }));
        let context = context_for_request(&request);

        let invalid_json = context
            .validate_marker("<tool_call>{</tool_call>")
            .unwrap_err();
        assert!(invalid_json.message().contains("JSON is invalid"));

        let unknown = context
            .validate_marker(
                "<tool_call>{\"name\":\"unknown\",\"arguments\":{\"query\":\"x\"}}</tool_call>",
            )
            .unwrap_err();
        assert!(unknown.message().contains("unknown tool name"));

        let schema = context
            .validate_marker(
                "<tool_call>{\"name\":\"search_web\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
            )
            .unwrap_err();
        assert!(schema.message().contains("arguments.query is required"));
    }

    #[test]
    fn can_disable_schema_validation_explicitly() {
        let request = request_with_tool(json!({
            "type": "object",
            "required": ["query"]
        }));
        let config = ToolsConfig {
            validate_json_schema: false,
            ..ToolsConfig::default()
        };
        let context = ToolEmulationContext::from_request(&config, &request)
            .expect("tool context should build")
            .expect("tools should activate");

        let tool_call = context
            .validate_marker("<tool_call>{\"name\":\"search_web\",\"arguments\":{}}</tool_call>")
            .expect("schema mismatch should be allowed when validation is disabled");
        assert_eq!(tool_call.arguments_json, "{}");
    }

    #[test]
    fn rejects_multiple_markers_and_non_object_arguments() {
        let request = request_with_tool(json!({"type": "object"}));
        let context = context_for_request(&request);

        let multiple = context
            .validate_marker("<tool_call>{\"name\":\"search_web\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"search_web\",\"arguments\":{}}</tool_call>")
            .unwrap_err();
        assert!(multiple.message().contains("exactly one"));

        let arguments = context
            .validate_marker("<tool_call>{\"name\":\"search_web\",\"arguments\":[]}</tool_call>")
            .unwrap_err();
        assert!(
            arguments
                .message()
                .contains("arguments must be a JSON object")
        );
    }

    #[test]
    fn builds_controller_and_retry_prompts() {
        let request = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "tool_choice": "required",
            "tools": [{"type":"function", "function":{"name":"search_web", "parameters":{"type":"object"}}}]
        }))
        .expect("request should parse");
        let context = context_for_request(&request);

        let controller = context.controller_message();
        assert_eq!(controller.role, "user");
        assert!(
            controller
                .content
                .contains("You must call exactly one tool")
        );
        assert!(controller.content.contains("search_web"));

        let correction = context.correction_message("bad name", "<tool_call>{}</tool_call>");
        assert_eq!(correction.role, "system");
        assert!(correction.content.contains("Validation error:\nbad name"));
        assert!(
            correction
                .content
                .contains("Invalid output:\n<tool_call>{}</tool_call>")
        );
        assert!(
            correction
                .content
                .contains("You must now return exactly one valid tool call")
        );
    }

    #[test]
    fn specific_tool_choice_filters_available_tools() {
        let request = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "tool_choice": {"type":"function", "function":{"name":"search_web"}},
            "tools": [
                {"type":"function", "function":{"name":"search_web", "parameters":{"type":"object"}}},
                {"type":"function", "function":{"name":"other", "parameters":{"type":"object"}}}
            ]
        }))
        .expect("request should parse");
        let context = context_for_request(&request);

        assert!(context.controller_message().content.contains("search_web"));
        assert!(!context.controller_message().content.contains("other"));
    }
}
