//! Vendored subset of vLLM's Rust streaming tool parser.
//!
//! Source: https://github.com/vllm-project/vllm/tree/b997071ec493765abbed990c65843ed05e4708a8/rust/src/tool-parser
//! License: Apache-2.0. This module keeps the parser code in-tree so this crate
//! can be published to crates.io without a git dependency.

#![allow(dead_code)]

#[macro_use]
mod error;
mod glm_xml;
mod json;
mod parameters;
mod utils;

use std::collections::{BTreeMap, btree_map};

pub use error::{Result, ToolParserError};
pub use glm_xml::Glm47MoeToolParser;
pub use json::{HermesToolParser, Qwen3XmlToolParser};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One function-style tool made available to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
}

/// One tool-call update emitted while parsing assistant text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// Stable parser-local tool index for this call within one assistant turn.
    pub tool_index: usize,
    /// Function name, present on the first update for one tool call.
    pub name: Option<String>,
    /// Arguments text contributed by this update.
    pub arguments: String,
}

/// Result of advancing tool parsing with one assistant-text input.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolParserOutput {
    /// Plain assistant text that is not part of any tool call.
    pub normal_text: String,
    /// Tool-call updates extracted from this input.
    pub calls: Vec<ToolCallDelta>,
}

/// Compatibility alias for the older vLLM parser API used by this crate.
pub type ToolParseResult = ToolParserOutput;

impl ToolParserOutput {
    /// Append another parser output onto this one.
    pub fn append(&mut self, mut other: Self) {
        self.normal_text.push_str(&other.normal_text);
        self.calls.append(&mut other.calls);
    }

    /// Merge multiple deltas for the same tool call into one complete item.
    pub fn coalesce_calls(mut self) -> Self {
        let mut merged = BTreeMap::<usize, ToolCallDelta>::new();
        let mut order = Vec::new();

        for call in self.calls {
            match merged.entry(call.tool_index) {
                btree_map::Entry::Vacant(entry) => {
                    order.push(call.tool_index);
                    entry.insert(call);
                }
                btree_map::Entry::Occupied(mut entry) => {
                    let existing = entry.get_mut();
                    if existing.name.is_none() {
                        existing.name = call.name;
                    }
                    existing.arguments.push_str(&call.arguments);
                }
            }
        }

        self.calls = order
            .into_iter()
            .filter_map(|tool_index| merged.remove(&tool_index))
            .collect();
        self
    }
}

/// Incremental parser that extracts tool calls from assistant output.
pub trait ToolParser: Send {
    /// Construct a boxed parser instance for one request stream.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static;

    /// Return whether decoded output must preserve tokenizer special tokens.
    fn preserve_special_tokens(&self) -> bool {
        false
    }

    /// Return the parser-provided ID for a tool call by index, if the model emitted one.
    fn tool_call_id(&self, _tool_index: usize) -> Option<&str> {
        None
    }

    /// Feed one decoded text delta into the parser, appending committed output into `output`.
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()>;

    /// Compatibility wrapper for older call sites that expect one output per pushed chunk.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        let mut output = ToolParserOutput::default();
        self.parse_into(chunk, &mut output)?;
        Ok(output)
    }

    /// Flush any buffered partial state at end of stream.
    fn finish(&mut self) -> Result<ToolParserOutput>;

    /// Clear parser state and return currently uncommitted buffered text.
    fn reset(&mut self) -> String;

    /// Parse complete tool calls from final output.
    fn parse_complete(&mut self, text: &str) -> Result<ToolParserOutput> {
        let mut output = self.push(text)?;
        output.append(self.finish()?);
        Ok(output.coalesce_calls())
    }
}
