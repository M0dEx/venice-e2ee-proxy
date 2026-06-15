//! OpenAI-style tool-call emulation.
//!
//! Venice E2EE responses do not expose native function calls, so this module
//! parses tool calls client-side from decrypted assistant text using a vendored
//! subset of vLLM's Rust tool parser, validates the calls against the request's OpenAI
//! `tools`, and builds prompt text for the encrypted controller/correction
//! requests.
//!
//! Venice E2EE cannot render tools into the server-side chat template (tools
//! arrive encrypted), so the proxy prompt must instruct a model-visible output
//! shape. Use model-family-specific shapes where the bundled vLLM Rust parser
//! has a matching format, and fall back to Hermes JSON otherwise.

use std::{collections::HashSet, time::Duration};

use crate::{
    config::{ToolMode, ToolsConfig},
    openai::chat::{
        ChatCompletionRequest, ChatRequestError, ChatToolChoice, ChatToolDefinition,
        NormalizedChatMessage,
    },
    vllm_tool_parser::{
        Glm47MoeToolParser, HermesToolParser, Qwen3XmlToolParser, Result as ToolParserResult, Tool,
        ToolCallDelta, ToolParseResult, ToolParser,
    },
};
use serde_json::{Map, Value};
use thiserror::Error;
use tracing::warn;

/// Tool-call markers used by the Hermes/Qwen/GLM prompt instructions.
const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";

/// Generates an OpenAI-style tool-call ID.
pub fn generate_tool_call_id() -> String {
    format!("call_{}", uuid::Uuid::new_v4().simple())
}

/// Maximum bytes of invalid assistant output echoed back in a correction
/// prompt; oversized output would otherwise grow each encrypted retry request
/// by the full output size.
const CORRECTION_INVALID_OUTPUT_MAX_BYTES: usize = 4_096;

/// Per-request tool-emulation state derived from config, tools, and tool choice.
#[derive(Debug, Clone)]
pub struct ToolEmulationContext {
    config: ToolsConfig,
    tools: Vec<ChatToolDefinition>,
    tool_schemas_json: String,
    require_tool_call: bool,
    prompt_format: ToolPromptFormat,
}

/// Prompt/parser format selected for the model family handling tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolPromptFormat {
    HermesJson,
    GlmXml,
    QwenXml,
}

impl ToolPromptFormat {
    /// Chooses a prompt/parser format from the requested model id.
    fn for_model(model: &str) -> Self {
        let model = model.to_ascii_lowercase();
        if model.contains("glm") {
            Self::GlmXml
        } else if model.contains("qwen") {
            Self::QwenXml
        } else {
            // Gemma, GPT-OSS, Venice uncensored, and unknown E2EE models have
            // live-tested successfully with the prompt-instructed Hermes JSON
            // format. Their native parser formats either need tokenizer special
            // tokens or are not exposed in the Rust parser crate.
            Self::HermesJson
        }
    }
}

impl ToolEmulationContext {
    /// Builds tool-emulation context for a request, or returns `None` when tools are disabled or unused.
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
            prompt_format: ToolPromptFormat::for_model(&request.model),
        }))
    }

    /// Returns the tool-emulation configuration used by this context.
    pub fn config(&self) -> &ToolsConfig {
        &self.config
    }

    /// Returns the maximum number of correction retries allowed for invalid tool calls.
    pub fn max_retries(&self) -> u32 {
        self.config.max_retries
    }

    /// Returns the maximum time to wait for a non-streamed tool-call marker response.
    pub fn marker_timeout(&self) -> Duration {
        self.config.tool_call_marker_timeout
    }

    /// Creates the preferred tool parser for one assistant turn.
    pub fn create_parser(&self) -> Result<Box<dyn ToolParser>, ToolCallValidationError> {
        self.create_parser_for_format(self.prompt_format)
    }

    /// Creates a parser for a specific prompt format.
    fn create_parser_for_format(
        &self,
        format: ToolPromptFormat,
    ) -> Result<Box<dyn ToolParser>, ToolCallValidationError> {
        let parser = match format {
            ToolPromptFormat::HermesJson => LenientToolParser::create(&[]),
            ToolPromptFormat::GlmXml => Glm47MoeToolParser::create(&self.vllm_tools()),
            ToolPromptFormat::QwenXml => Qwen3XmlToolParser::create(&[]),
        };
        parser.map_err(|error| {
            ToolCallValidationError::new(format!("tool parser could not be created: {error}"))
        })
    }

    /// Converts OpenAI function tools into the vLLM parser tool representation.
    fn vllm_tools(&self) -> Vec<Tool> {
        self.tools
            .iter()
            .map(|tool| {
                let function = tool.function();
                Tool {
                    name: function.name.clone(),
                    description: function.description.clone(),
                    parameters: function
                        .parameters
                        .as_ref()
                        .map(|schema| Value::Object(schema.as_map().clone()))
                        .unwrap_or_else(|| Value::Object(Map::new())),
                    strict: None,
                }
            })
            .collect()
    }

    /// Builds the system/controller prompt message that instructs the model to emit tool calls.
    pub fn controller_message(&self) -> NormalizedChatMessage {
        let requirement = if self.require_tool_call {
            "You must call at least one tool. Do not answer the user directly. Output each tool call using this format and nothing else:"
        } else {
            "If tools are required, do not answer the user directly. Output each tool call using this format and nothing else:"
        };
        let optional_rule = if self.require_tool_call {
            String::new()
        } else {
            format!("\n- If no tool is needed, answer normally and do not use {TOOL_CALL_START}.")
        };

        let content = match self.prompt_format {
            ToolPromptFormat::HermesJson => {
                self.hermes_controller_content(requirement, &optional_rule)
            }
            ToolPromptFormat::GlmXml => {
                self.glm_xml_controller_content(requirement, &optional_rule)
            }
            ToolPromptFormat::QwenXml => {
                self.qwen_xml_controller_content(requirement, &optional_rule)
            }
        };

        // The HTTP layer appends this content to the request's system prompt
        // for tool-emulated requests. Keep the role as a harmless container for
        // callers/tests that inspect the standalone message.
        NormalizedChatMessage::new("user", content)
    }

    /// Builds a correction prompt from the previous validation error and assistant output.
    pub fn correction_message(
        &self,
        validation_error: &str,
        invalid_output: &str,
    ) -> NormalizedChatMessage {
        let invalid_output =
            truncate_at_char_boundary(invalid_output, CORRECTION_INVALID_OUTPUT_MAX_BYTES);
        let content = match self.prompt_format {
            ToolPromptFormat::HermesJson => {
                self.hermes_correction_content(validation_error, &invalid_output)
            }
            ToolPromptFormat::GlmXml => {
                self.glm_xml_correction_content(validation_error, &invalid_output)
            }
            ToolPromptFormat::QwenXml => {
                self.qwen_xml_correction_content(validation_error, &invalid_output)
            }
        };
        NormalizedChatMessage::new("system", content)
    }

    /// Builds Hermes-style JSON tool-call controller instructions.
    fn hermes_controller_content(&self, requirement: &str, optional_rule: &str) -> String {
        format!(
            "You have access to tools.\n\n{requirement}\n\nRequired tool-call format:\n\n{TOOL_CALL_START}\n{}\n{TOOL_CALL_END}\n\nInside each {TOOL_CALL_START} block, output ONLY one valid JSON object with exactly these top-level keys:\n- \"name\": the tool name as a JSON string.\n- \"arguments\": a JSON object containing the tool arguments.\n\nValid single-call example:\n\n{TOOL_CALL_START}\n{}\n{TOOL_CALL_END}\n\nValid multi-call example:\n\n{TOOL_CALL_START}\n{}\n{TOOL_CALL_END}\n{TOOL_CALL_START}\n{}\n{TOOL_CALL_END}\n\nInvalid formats. NEVER use these:\n- {TOOL_CALL_START}TOOL_NAME({{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}}){TOOL_CALL_END}\n- {TOOL_CALL_START}TOOL_NAME{{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}}{TOOL_CALL_END}\n- TOOL_NAME({{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}})\n- {{\"tool\":\"TOOL_NAME\",\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}}\n\nRules:\n- TOOL_NAME must exactly match one available tool name.\n- Always put the tool name in the JSON \"name\" field.\n- Always put tool arguments inside the JSON \"arguments\" object.\n- Do not put arguments directly after the tool name.\n- Do not use function-call syntax like TOOL_NAME(...).\n- arguments must be valid JSON and must satisfy the tool schema.\n- Emit one marker block per tool call.\n- Do not include markdown fences.\n- Do not include explanations.{optional_rule}\n\nAvailable tools:\n{}",
            r#"{"name":"TOOL_NAME","arguments":{...}}"#,
            r#"{"name":"TOOL_NAME","arguments":{"ARGUMENT_NAME":"ARGUMENT_VALUE"}}"#,
            r#"{"name":"TOOL_NAME_1","arguments":{"ARGUMENT_NAME":"ARGUMENT_VALUE"}}"#,
            r#"{"name":"TOOL_NAME_2","arguments":{"ARGUMENT_NAME":"ARGUMENT_VALUE"}}"#,
            self.tool_schemas_json,
        )
    }

    /// Builds Qwen XML-wrapped JSON tool-call controller instructions.
    fn qwen_xml_controller_content(&self, requirement: &str, optional_rule: &str) -> String {
        format!(
            "You have access to tools.\n\n{requirement}\n\nRequired Qwen XML-wrapped JSON tool-call format:\n\n{TOOL_CALL_START}\n{}\n{TOOL_CALL_END}\n\nThere MUST be a newline immediately after {TOOL_CALL_START}. Inside each block, output ONLY one valid JSON object with exactly these top-level keys:\n- \"name\": the tool name as a JSON string.\n- \"arguments\": a JSON object containing the tool arguments.\n\nValid example:\n\n{TOOL_CALL_START}\n{}\n{TOOL_CALL_END}\n\nRules:\n- TOOL_NAME must exactly match one available tool name.\n- Always put the tool name in the JSON \"name\" field.\n- Always put tool arguments inside the JSON \"arguments\" object.\n- Do not use function-call syntax like TOOL_NAME(...).\n- arguments must be valid JSON and must satisfy the tool schema.\n- Emit one marker block per tool call.\n- Do not include markdown fences.\n- Do not include explanations.{optional_rule}\n\nAvailable tools:\n{}",
            r#"{"name":"TOOL_NAME","arguments":{...}}"#,
            r#"{"name":"TOOL_NAME","arguments":{"ARGUMENT_NAME":"ARGUMENT_VALUE"}}"#,
            self.tool_schemas_json,
        )
    }

    /// Builds GLM XML tool-call controller instructions.
    fn glm_xml_controller_content(&self, requirement: &str, optional_rule: &str) -> String {
        format!(
            "You have access to tools.\n\n{requirement}\n\nRequired GLM XML tool-call format:\n\n{TOOL_CALL_START}TOOL_NAME\n<arg_key>ARGUMENT_NAME</arg_key>\n<arg_value>ARGUMENT_VALUE</arg_value>\n{TOOL_CALL_END}\n\nInside each {TOOL_CALL_START} block:\n- Start with the exact tool name as plain text.\n- Then output one <arg_key>/<arg_value> pair for each argument.\n- Put only the raw argument name inside <arg_key>.\n- Put only the raw argument value inside <arg_value>.\n- If an argument value is an object or array, put compact valid JSON inside <arg_value>.\n\nValid single-call example:\n\n{TOOL_CALL_START}TOOL_NAME\n<arg_key>ARGUMENT_NAME</arg_key>\n<arg_value>ARGUMENT_VALUE</arg_value>\n{TOOL_CALL_END}\n\nValid multi-call example:\n\n{TOOL_CALL_START}TOOL_NAME_1\n<arg_key>ARGUMENT_NAME</arg_key>\n<arg_value>ARGUMENT_VALUE</arg_value>\n{TOOL_CALL_END}\n{TOOL_CALL_START}TOOL_NAME_2\n<arg_key>ARGUMENT_NAME</arg_key>\n<arg_value>ARGUMENT_VALUE</arg_value>\n{TOOL_CALL_END}\n\nInvalid formats. NEVER use these:\n- {TOOL_CALL_START}TOOL_NAME({{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}}){TOOL_CALL_END}\n- {TOOL_CALL_START}TOOL_NAME{{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}}{TOOL_CALL_END}\n- {TOOL_CALL_START}{{\"name\":\"TOOL_NAME\",\"arguments\":{{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}}}}{TOOL_CALL_END}\n- TOOL_NAME({{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}})\n\nRules:\n- TOOL_NAME must exactly match one available tool name.\n- Do not output JSON inside {TOOL_CALL_START} except for object/array values inside <arg_value>.\n- Do not use function-call syntax like TOOL_NAME(...).\n- Do not use the Hermes JSON format with \"name\" and \"arguments\" keys.\n- Argument names and values must satisfy the tool schema.\n- Emit one marker block per tool call.\n- Do not include markdown fences.\n- Do not include explanations.{optional_rule}\n\nAvailable tools:\n{}",
            self.tool_schemas_json,
        )
    }

    /// Builds Hermes-style correction instructions after invalid tool-call output.
    fn hermes_correction_content(&self, validation_error: &str, invalid_output: &str) -> String {
        format!(
            "Your previous response attempted a tool call, but it was invalid.\n\nValidation error:\n{validation_error}\n\nInvalid output:\n{invalid_output}\n\nYou must now return only valid tool calls and nothing else.\n\nUse this exact format:\n\n{TOOL_CALL_START}\n{}\n{TOOL_CALL_END}\n\nInside each {TOOL_CALL_START} block, output ONLY one valid JSON object with exactly these top-level keys:\n- \"name\": the tool name as a JSON string.\n- \"arguments\": a JSON object containing the tool arguments.\n\nValid example:\n\n{TOOL_CALL_START}\n{}\n{TOOL_CALL_END}\n\nInvalid formats. NEVER use these:\n- {TOOL_CALL_START}TOOL_NAME({{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}}){TOOL_CALL_END}\n- {TOOL_CALL_START}TOOL_NAME{{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}}{TOOL_CALL_END}\n- TOOL_NAME({{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}})\n\nRules:\n- TOOL_NAME must exactly match one of the available tools.\n- Always put the tool name in the JSON \"name\" field.\n- Always put tool arguments inside the JSON \"arguments\" object.\n- Do not put arguments directly after the tool name.\n- Do not use function-call syntax like TOOL_NAME(...).\n- arguments must be a JSON object.\n- arguments must satisfy the tool schema.\n- Do not include markdown fences.\n- Do not include explanations.\n- Do not answer the user directly.\n\nAvailable tools:\n{}",
            r#"{"name":"TOOL_NAME","arguments":{...}}"#,
            r#"{"name":"TOOL_NAME","arguments":{"ARGUMENT_NAME":"ARGUMENT_VALUE"}}"#,
            self.tool_schemas_json,
        )
    }

    /// Builds Qwen XML-wrapped JSON correction instructions after invalid tool-call output.
    fn qwen_xml_correction_content(&self, validation_error: &str, invalid_output: &str) -> String {
        format!(
            "Your previous response attempted a tool call, but it was invalid.\n\nValidation error:\n{validation_error}\n\nInvalid output:\n{invalid_output}\n\nYou must now return only valid tool calls and nothing else.\n\nUse this exact Qwen XML-wrapped JSON format:\n\n{TOOL_CALL_START}\n{}\n{TOOL_CALL_END}\n\nThere MUST be a newline immediately after {TOOL_CALL_START}. Inside each block, output ONLY one valid JSON object with \"name\" and \"arguments\" top-level keys.\n\nAvailable tools:\n{}",
            r#"{"name":"TOOL_NAME","arguments":{...}}"#, self.tool_schemas_json,
        )
    }

    /// Builds GLM XML correction instructions after invalid tool-call output.
    fn glm_xml_correction_content(&self, validation_error: &str, invalid_output: &str) -> String {
        format!(
            "Your previous response attempted a tool call, but it was invalid.\n\nValidation error:\n{validation_error}\n\nInvalid output:\n{invalid_output}\n\nYou must now return only valid tool calls and nothing else.\n\nUse this exact GLM XML format:\n\n{TOOL_CALL_START}TOOL_NAME\n<arg_key>ARGUMENT_NAME</arg_key>\n<arg_value>ARGUMENT_VALUE</arg_value>\n{TOOL_CALL_END}\n\nInside each {TOOL_CALL_START} block:\n- Start with the exact tool name as plain text.\n- Then output one <arg_key>/<arg_value> pair for each argument.\n- Put only the raw argument name inside <arg_key>.\n- Put only the raw argument value inside <arg_value>.\n\nValid example:\n\n{TOOL_CALL_START}TOOL_NAME\n<arg_key>ARGUMENT_NAME</arg_key>\n<arg_value>ARGUMENT_VALUE</arg_value>\n{TOOL_CALL_END}\n\nInvalid formats. NEVER use these:\n- {TOOL_CALL_START}TOOL_NAME({{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}}){TOOL_CALL_END}\n- {TOOL_CALL_START}TOOL_NAME{{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}}{TOOL_CALL_END}\n- {TOOL_CALL_START}{{\"name\":\"TOOL_NAME\",\"arguments\":{{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}}}}{TOOL_CALL_END}\n- TOOL_NAME({{\"ARGUMENT_NAME\":\"ARGUMENT_VALUE\"}})\n\nRules:\n- TOOL_NAME must exactly match one of the available tools.\n- Do not output JSON inside {TOOL_CALL_START} except for object/array values inside <arg_value>.\n- Do not use function-call syntax like TOOL_NAME(...).\n- Do not use the Hermes JSON format with \"name\" and \"arguments\" keys.\n- Argument names and values must satisfy the tool schema.\n- Do not include markdown fences.\n- Do not include explanations.\n- Do not answer the user directly.\n\nAvailable tools:\n{}",
            self.tool_schemas_json,
        )
    }

    /// Classifies buffered assistant output into normal text, validated tool
    /// calls, or an invalid tool call that feeds the retry/correction loop.
    ///
    /// Mixed text + tool calls classifies as tool calls; the surrounding text
    /// is dropped from the OpenAI message, matching previous behavior.
    pub fn classify_assistant_output(&self, output: &str) -> ToolOutputClassification {
        if output.len() > self.config.tool_call_max_bytes {
            return ToolOutputClassification::InvalidToolCall {
                error: ToolCallValidationError::new(format!(
                    "assistant output exceeded the tool call max size of {} bytes",
                    self.config.tool_call_max_bytes
                )),
                invalid_output: output.to_owned(),
            };
        }

        let result = self.parse_tool_calls(output);

        match result {
            Ok(tool_calls) if tool_calls.is_empty() => {
                if self.require_tool_call {
                    ToolOutputClassification::InvalidToolCall {
                        error: ToolCallValidationError::new(
                            "expected the assistant response to include a tool call",
                        ),
                        invalid_output: output.to_owned(),
                    }
                } else {
                    ToolOutputClassification::NormalText
                }
            }
            Ok(tool_calls) => ToolOutputClassification::ToolCalls(tool_calls),
            Err(error) => ToolOutputClassification::InvalidToolCall {
                error,
                invalid_output: output.to_owned(),
            },
        }
    }

    /// Parses and validates assistant output, using Hermes as a compatibility fallback when needed.
    fn parse_tool_calls(
        &self,
        output: &str,
    ) -> Result<Vec<ValidatedToolCall>, ToolCallValidationError> {
        let result = self.parse_tool_calls_with_format(self.prompt_format, output);
        if self.prompt_format == ToolPromptFormat::HermesJson {
            return result;
        }

        match result {
            Ok(tool_calls) if tool_calls.is_empty() && output.contains(TOOL_CALL_START) => {
                if let Some(fallback_calls) = self.hermes_fallback_tool_calls(output) {
                    Ok(fallback_calls)
                } else {
                    Ok(tool_calls)
                }
            }
            Err(error) => {
                if let Some(fallback_calls) = self.hermes_fallback_tool_calls(output) {
                    Ok(fallback_calls)
                } else {
                    Err(error)
                }
            }
            result => result,
        }
    }

    /// Attempts to parse non-empty Hermes JSON tool calls from model output.
    fn hermes_fallback_tool_calls(&self, output: &str) -> Option<Vec<ValidatedToolCall>> {
        match self.parse_tool_calls_with_format(ToolPromptFormat::HermesJson, output) {
            Ok(tool_calls) if !tool_calls.is_empty() => Some(tool_calls),
            _ => None,
        }
    }

    /// Parses and validates assistant output using a specific prompt/parser format.
    fn parse_tool_calls_with_format(
        &self,
        format: ToolPromptFormat,
        output: &str,
    ) -> Result<Vec<ValidatedToolCall>, ToolCallValidationError> {
        self.create_parser_for_format(format)
            .and_then(|mut parser| {
                parser.parse_complete(output).map_err(|error| {
                    ToolCallValidationError::new(format!("tool call parsing failed: {error}"))
                })
            })
            .and_then(|result| {
                result
                    .calls
                    .iter()
                    .map(|call| self.validate_tool_call(call))
                    .collect::<Result<Vec<_>, _>>()
            })
    }

    /// Validates one coalesced parser tool call against the request's tools.
    fn validate_tool_call(
        &self,
        call: &ToolCallDelta,
    ) -> Result<ValidatedToolCall, ToolCallValidationError> {
        let name = call.name.as_deref().unwrap_or_default();
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

        let arguments: Value = serde_json::from_str(&call.arguments).map_err(|source| {
            ToolCallValidationError::new(format!("tool call arguments JSON is invalid: {source}"))
        })?;
        if !arguments.is_object() {
            return Err(ToolCallValidationError::new(
                "tool call arguments must be a JSON object",
            ));
        }

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
            id: generate_tool_call_id(),
            name: name.to_owned(),
            arguments_json,
        })
    }
}

/// Truncates text to at most `max_bytes` (on a char boundary), marking the cut.
fn truncate_at_char_boundary(text: &str, max_bytes: usize) -> std::borrow::Cow<'_, str> {
    if text.len() <= max_bytes {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    std::borrow::Cow::Owned(format!("{} [output truncated]", &text[..end]))
}

/// Lenient wrapper around the strict Hermes parser, tolerating model
/// deviations observed live against Venice:
/// - A parse that fails to finish (the model or an upstream stop sequence
///   dropped `</tool_call>`; seen with `e2ee-glm-4-7-flash-p`) is retried
///   once with the closing marker appended.
/// - Trailing garbage after a tool call with complete JSON arguments (e.g.
///   `e2ee-glm-5-1` closing a call with a stray `</arg_value>`) drains the
///   rest of the output instead of failing. Input is split before each `<`
///   so such garbage reaches the parser in its own push and cannot take
///   already-parsed deltas down with it.
struct LenientToolParser {
    inner: Box<dyn ToolParser>,
    /// Tracks whether the most recent call's argument JSON is complete, to
    /// distinguish trailing garbage from a truncated call.
    args_scanner: ArgsCompletenessScanner,
    /// Set once trailing garbage was detected; all further input is ignored.
    drained: bool,
}

impl ToolParser for LenientToolParser {
    /// Creates a lenient Hermes parser for the supplied tools.
    fn create(tools: &[Tool]) -> ToolParserResult<Box<dyn ToolParser>> {
        Ok(Box::new(Self {
            inner: HermesToolParser::create(tools)?,
            args_scanner: ArgsCompletenessScanner::default(),
            drained: false,
        }))
    }

    /// Pushes one assistant output chunk through the parser, appending parsed deltas to `output`.
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParseResult) -> ToolParserResult<()> {
        output.append(self.push(chunk)?);
        Ok(())
    }

    /// Pushes one assistant output chunk through the parser and returns parsed deltas.
    fn push(&mut self, chunk: &str) -> ToolParserResult<ToolParseResult> {
        let mut merged = ToolParseResult::default();
        if self.drained {
            return Ok(merged);
        }
        for piece in split_before_tag_starts(chunk) {
            match self.inner.push(piece) {
                Ok(result) => {
                    self.args_scanner.track(&result);
                    merged.normal_text.push_str(&result.normal_text);
                    merged.calls.extend(result.calls);
                }
                Err(error) => {
                    if !self.args_scanner.complete() {
                        return Err(error);
                    }
                    // Some live GLM outputs append a native closing tag after complete
                    // Hermes arguments; keep the parsed call rather than failing on that
                    // incompatible tail.
                    warn!(%error, "ignoring trailing output after a complete tool call");
                    self.drained = true;
                    break;
                }
            }
        }
        Ok(merged)
    }

    /// Finishes parsing and returns any recovered complete tool calls.
    fn finish(&mut self) -> ToolParserResult<ToolParseResult> {
        if self.drained {
            return Ok(ToolParseResult::default());
        }
        let error = match self.inner.finish() {
            Ok(result) => return Ok(result),
            Err(error) => error,
        };
        // Venice sometimes cuts only the closing marker; keep the original
        // parser error when appending that marker still cannot recover a complete call.
        let Ok(mut recovered) = self.inner.push(TOOL_CALL_END) else {
            return Err(error);
        };
        let Ok(finished) = self.inner.finish() else {
            return Err(error);
        };
        recovered.normal_text.push_str(&finished.normal_text);
        recovered.calls.extend(finished.calls);
        Ok(recovered)
    }

    /// Clears parser state and returns uncommitted buffered text.
    fn reset(&mut self) -> String {
        self.args_scanner = ArgsCompletenessScanner::default();
        self.drained = false;
        self.inner.reset()
    }
}

/// Splits text so every `<` starts a new piece, isolating each potential tag
/// (marker, native-format tag, or garbage) in its own parser push.
fn split_before_tag_starts(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    for (index, _) in text.match_indices('<') {
        if index > start {
            pieces.push(&text[start..index]);
        }
        start = index;
    }
    if start < text.len() {
        pieces.push(&text[start..]);
    }
    pieces
}

/// Minimal JSON scanner tracking whether the most recent tool call's
/// argument text forms a complete JSON value (balanced braces/brackets
/// outside strings).
#[derive(Debug, Default)]
struct ArgsCompletenessScanner {
    started: bool,
    depth: u32,
    in_string: bool,
    escaped: bool,
}

impl ArgsCompletenessScanner {
    /// Updates scanner state from parser deltas emitted for tool calls.
    fn track(&mut self, result: &ToolParseResult) {
        for call in &result.calls {
            if call.name.is_some() {
                *self = Self::default();
            }
            self.feed(&call.arguments);
        }
    }

    /// Consumes an argument fragment and updates JSON completeness state.
    fn feed(&mut self, fragment: &str) {
        for ch in fragment.chars() {
            if self.escaped {
                self.escaped = false;
                continue;
            }
            if self.in_string {
                match ch {
                    '\\' => self.escaped = true,
                    '"' => self.in_string = false,
                    _ => {}
                }
                continue;
            }
            match ch {
                '"' => {
                    self.in_string = true;
                    self.started = true;
                }
                '{' | '[' => {
                    self.depth += 1;
                    self.started = true;
                }
                '}' | ']' => self.depth = self.depth.saturating_sub(1),
                ch if !ch.is_whitespace() => self.started = true,
                _ => {}
            }
        }
    }

    /// Returns whether the most recently observed argument text is a complete JSON value.
    fn complete(&self) -> bool {
        self.started && self.depth == 0 && !self.in_string
    }
}

/// Classification of a decrypted assistant response under tool emulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolOutputClassification {
    NormalText,
    ToolCalls(Vec<ValidatedToolCall>),
    InvalidToolCall {
        error: ToolCallValidationError,
        invalid_output: String,
    },
}

/// OpenAI-compatible tool call validated against the request's available tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

impl ValidatedToolCall {
    /// Converts the validated call into an OpenAI `tool_calls` JSON object.
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
}

/// Validation error for parsed assistant tool-call output.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{message}")]
pub struct ToolCallValidationError {
    message: String,
}

impl ToolCallValidationError {
    /// Creates a validation error with a client-facing message.
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the validation error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Validates that a tool parameters schema uses the supported JSON Schema subset.
fn validate_schema_shape(schema: &Map<String, Value>) -> Result<(), String> {
    validate_schema_object_shape(schema, "schema")
}

/// Validates one schema object and nested supported schema objects.
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

/// Validates a JSON Schema `type` value in string or string-array form.
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

/// Validates one supported JSON Schema type name.
fn validate_schema_type_name(kind: &str, path: &str) -> Result<(), String> {
    match kind {
        "object" | "array" | "string" | "integer" | "number" | "boolean" | "null" => Ok(()),
        other => Err(format!(
            "{path} contains unsupported JSON schema type {other:?}"
        )),
    }
}

/// Validates a JSON value against the supported JSON Schema subset.
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

/// Reads a nested schema value as an object for recursive validation.
fn schema_value_as_object<'a>(
    schema: &'a Value,
    path: &str,
) -> Result<&'a Map<String, Value>, String> {
    schema
        .as_object()
        .ok_or_else(|| format!("{path} schema must be an object"))
}

/// Returns whether a schema requires object-specific validation.
fn schema_implies_object(schema: &Map<String, Value>) -> bool {
    schema
        .get("type")
        .is_some_and(|kind| schema_type_includes(kind, "object"))
        || schema.contains_key("properties")
        || schema.contains_key("required")
        || schema.contains_key("additionalProperties")
}

/// Returns whether a schema requires array-specific validation.
fn schema_implies_array(schema: &Map<String, Value>) -> bool {
    schema
        .get("type")
        .is_some_and(|kind| schema_type_includes(kind, "array"))
        || schema.contains_key("items")
}

/// Returns whether a JSON value matches a schema `type` value.
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

/// Returns whether a schema `type` value includes a named type.
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

/// Returns whether a JSON value matches one supported schema type name.
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

/// Formats a schema `type` value for validation error messages.
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

/// Returns a human-readable kind name for a JSON value.
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
    use serde_json::json;

    use super::*;
    use crate::config::ToolsConfig;

    fn request_with_tool(arguments_schema: Value) -> ChatCompletionRequest {
        request_with_tool_for_model("e2ee-test", arguments_schema)
    }

    fn request_with_tool_for_model(model: &str, arguments_schema: Value) -> ChatCompletionRequest {
        ChatCompletionRequest::parse(&json!({
            "model": model,
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
    fn classifies_valid_hermes_tool_call() {
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

        let ToolOutputClassification::ToolCalls(tool_calls) = classification else {
            panic!("expected valid tool call");
        };
        assert_eq!(tool_calls.len(), 1);
        assert!(tool_calls[0].id.starts_with("call_"));
        assert_eq!(tool_calls[0].name, "search_web");
        assert_eq!(tool_calls[0].arguments_json, "{\"query\":\"Venice\"}");
    }

    #[test]
    fn classifies_glm_xml_tool_call_for_glm_models() {
        let request = request_with_tool_for_model(
            "e2ee-glm-5-1",
            json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": false
            }),
        );
        let context = context_for_request(&request);

        let classification = context.classify_assistant_output(
            "<tool_call>search_web\n<arg_key>query</arg_key><arg_value>Venice</arg_value></tool_call>",
        );

        let ToolOutputClassification::ToolCalls(tool_calls) = classification else {
            panic!("expected valid GLM XML tool call");
        };
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "search_web");
        assert_eq!(tool_calls[0].arguments_json, "{\"query\":\"Venice\"}");
        assert!(context.controller_message().content.contains("<arg_key>"));
    }

    #[test]
    fn classifies_qwen_xml_tool_call_for_qwen_models() {
        let request = request_with_tool_for_model(
            "e2ee-qwen3-30b-a3b-p",
            json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": false
            }),
        );
        let context = context_for_request(&request);

        let classification = context.classify_assistant_output(
            "<tool_call>\n{\"name\":\"search_web\",\"arguments\":{\"query\":\"Venice\"}}\n</tool_call>",
        );

        let ToolOutputClassification::ToolCalls(tool_calls) = classification else {
            panic!("expected valid Qwen XML-wrapped JSON tool call");
        };
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "search_web");
        assert_eq!(tool_calls[0].arguments_json, "{\"query\":\"Venice\"}");
        assert!(
            context
                .controller_message()
                .content
                .contains("Qwen XML-wrapped JSON")
        );
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

        // Hermes passes argument text through raw; invalid JSON is caught by
        // our validation layer.
        let ToolOutputClassification::InvalidToolCall { error, .. } = context
            .classify_assistant_output(
                "<tool_call>{\"name\":\"search_web\",\"arguments\":{\"query\":\"x\",}}</tool_call>",
            )
        else {
            panic!("expected invalid JSON to be rejected");
        };
        assert!(error.message().contains("JSON is invalid"));

        let ToolOutputClassification::InvalidToolCall { error, .. } = context
            .classify_assistant_output(
                "<tool_call>{\"name\":\"unknown\",\"arguments\":{\"query\":\"x\"}}</tool_call>",
            )
        else {
            panic!("expected unknown tool to be rejected");
        };
        assert!(error.message().contains("unknown tool name"));

        let ToolOutputClassification::InvalidToolCall { error, .. } = context
            .classify_assistant_output(
                "<tool_call>{\"name\":\"search_web\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
            )
        else {
            panic!("expected schema mismatch to be rejected");
        };
        assert!(error.message().contains("arguments.query is required"));
    }

    #[test]
    fn recovers_tool_call_with_truncated_closing_marker() {
        // Observed live: Venice cuts `</tool_call>` for some models (likely a
        // stop sequence). A complete call missing only the closing marker is
        // recovered leniently.
        let request = request_with_tool(json!({"type": "object"}));
        let context = context_for_request(&request);

        let classification = context.classify_assistant_output(
            "<tool_call>\n{\"name\":\"search_web\",\"arguments\":{\"query\":\"a\"}}\n",
        );

        let ToolOutputClassification::ToolCalls(tool_calls) = classification else {
            panic!("expected truncated closing marker to be recovered, got {classification:?}");
        };
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "search_web");
        assert_eq!(tool_calls[0].arguments_json, "{\"query\":\"a\"}");
    }

    #[test]
    fn ignores_trailing_garbage_after_complete_tool_call() {
        // Exact outputs observed live from `e2ee-glm-5-1`: a Hermes-shaped
        // call "closed" with a stray GLM-native tag.
        let request = request_with_tool_for_model("e2ee-glm-5-1", json!({"type": "object"}));
        let context = context_for_request(&request);

        for output in [
            "<tool_call>{\"name\":\"search_web\",\"arguments\":{\"query\":\"a\"}}</arg_value>",
            "<tool_call>{\"name\":\"search_web\",\"arguments\":{\"query\":\"a\"}}</arg_value></tool_call>",
        ] {
            let classification = context.classify_assistant_output(output);
            let ToolOutputClassification::ToolCalls(tool_calls) = classification else {
                panic!(
                    "expected trailing garbage to be ignored for {output:?}, got {classification:?}"
                );
            };
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].name, "search_web");
            assert_eq!(tool_calls[0].arguments_json, "{\"query\":\"a\"}");
        }
    }

    #[test]
    fn classifies_output_truncated_mid_json_as_invalid_tool_call() {
        let request = request_with_tool(json!({"type": "object"}));
        let context = context_for_request(&request);

        let classification =
            context.classify_assistant_output("<tool_call>{\"name\":\"search_web\",\"argu");

        let ToolOutputClassification::InvalidToolCall { error, .. } = classification else {
            panic!("expected mid-JSON truncation to be invalid, got {classification:?}");
        };
        assert!(error.message().contains("tool call parsing failed"));
    }

    #[test]
    fn classifies_plain_text_and_enforces_required_tool_call() {
        let request = request_with_tool(json!({"type": "object"}));
        let context = context_for_request(&request);
        assert_eq!(
            context.classify_assistant_output("Hello, world!"),
            ToolOutputClassification::NormalText
        );

        let request = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "tool_choice": "required",
            "tools": [{"type":"function", "function":{"name":"search_web", "parameters":{"type":"object"}}}]
        }))
        .expect("request should parse");
        let context = context_for_request(&request);

        let ToolOutputClassification::InvalidToolCall { error, .. } =
            context.classify_assistant_output("Hello, world!")
        else {
            panic!("expected missing required tool call to be invalid");
        };
        assert!(error.message().contains("expected the assistant response"));
    }

    #[test]
    fn classifies_mixed_text_and_tool_call_as_tool_calls() {
        let request = request_with_tool(json!({"type": "object"}));
        let context = context_for_request(&request);

        let classification = context.classify_assistant_output(
            "Let me check.\n<tool_call>{\"name\":\"search_web\",\"arguments\":{\"query\":\"a\"}}</tool_call>",
        );

        let ToolOutputClassification::ToolCalls(tool_calls) = classification else {
            panic!("expected mixed output to classify as tool calls");
        };
        assert_eq!(tool_calls.len(), 1);
    }

    #[test]
    fn classifies_multiple_tool_calls_regardless_of_parallel_tool_calls() {
        let request = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "parallel_tool_calls": false,
            "tools": [{"type":"function", "function":{"name":"search_web", "parameters":{"type":"object"}}}]
        }))
        .expect("request should parse");
        let context = context_for_request(&request);

        // `parallel_tool_calls` is accepted for OpenAI compatibility but
        // ignored; all parsed tool calls are returned.
        let classification = context.classify_assistant_output(
            "<tool_call>{\"name\":\"search_web\",\"arguments\":{\"query\":\"a\"}}</tool_call>\n<tool_call>{\"name\":\"search_web\",\"arguments\":{\"query\":\"b\"}}</tool_call>",
        );
        let ToolOutputClassification::ToolCalls(tool_calls) = classification else {
            panic!("expected two valid tool calls");
        };
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].arguments_json, "{\"query\":\"a\"}");
        assert_eq!(tool_calls[1].arguments_json, "{\"query\":\"b\"}");
        assert_ne!(tool_calls[0].id, tool_calls[1].id);
    }

    #[test]
    fn rejects_oversized_assistant_output() {
        let request = request_with_tool(json!({"type": "object"}));
        let config = ToolsConfig {
            tool_call_max_bytes: 32,
            ..ToolsConfig::default()
        };
        let context = ToolEmulationContext::from_request(&config, &request)
            .expect("tool context should build")
            .expect("tools should activate");

        let ToolOutputClassification::InvalidToolCall { error, .. } =
            context.classify_assistant_output(&"x".repeat(33))
        else {
            panic!("expected oversized output to be invalid");
        };
        assert!(error.message().contains("max size of 32 bytes"));
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

        let classification = context.classify_assistant_output(
            "<tool_call>{\"name\":\"search_web\",\"arguments\":{}}</tool_call>",
        );
        let ToolOutputClassification::ToolCalls(tool_calls) = classification else {
            panic!("schema mismatch should be allowed when validation is disabled");
        };
        assert_eq!(tool_calls[0].arguments_json, "{}");
    }

    #[test]
    fn rejects_non_object_arguments() {
        let request = request_with_tool(json!({"type": "object"}));
        let context = context_for_request(&request);

        // The Hermes parser itself rejects non-object argument payloads, so
        // this surfaces as a parser failure rather than reaching our
        // arguments-must-be-an-object validation.
        let ToolOutputClassification::InvalidToolCall { error, .. } = context
            .classify_assistant_output(
                "<tool_call>{\"name\":\"search_web\",\"arguments\":[]}</tool_call>",
            )
        else {
            panic!("expected non-object arguments to be rejected");
        };
        assert!(error.message().contains("tool call parsing failed"));

        // Our validation layer still rejects non-object arguments that a
        // parser passes through (defense in depth for other families).
        let error = context
            .validate_tool_call(&ToolCallDelta {
                tool_index: 0,
                name: Some("search_web".to_owned()),
                arguments: "[]".to_owned(),
            })
            .unwrap_err();
        assert!(error.message().contains("arguments must be a JSON object"));
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
                .contains("You must call at least one tool")
        );
        assert!(
            controller
                .content
                .contains("Emit one marker block per tool call")
        );
        assert!(controller.content.contains("<tool_call>"));
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
                .contains("You must now return only valid tool calls")
        );

        let optional_request = ChatCompletionRequest::parse(&json!({
            "model": "e2ee-test",
            "messages": [{"role":"user", "content":"hi"}],
            "tools": [{"type":"function", "function":{"name":"search_web", "parameters":{"type":"object"}}}]
        }))
        .expect("request should parse");
        let optional = context_for_request(&optional_request);
        assert!(
            optional
                .controller_message()
                .content
                .contains("If no tool is needed, answer normally")
        );
    }

    #[test]
    fn correction_prompt_truncates_oversized_invalid_output() {
        let request = request_with_tool(json!({"type": "object"}));
        let context = context_for_request(&request);

        let oversized = "x".repeat(CORRECTION_INVALID_OUTPUT_MAX_BYTES + 1);
        let correction = context.correction_message("error", &oversized);
        assert!(correction.content.contains("[output truncated]"));
        assert!(!correction.content.contains(&oversized));

        let short = context.correction_message("error", "<tool_call>{}</tool_call>");
        assert!(!short.content.contains("[output truncated]"));
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
