//! Small shared helpers.

/// Human-readable kind name for a JSON value, used in validation errors.
pub(crate) fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn names_json_kinds() {
        assert_eq!(json_kind(&json!(null)), "null");
        assert_eq!(json_kind(&json!(true)), "boolean");
        assert_eq!(json_kind(&json!(1)), "number");
        assert_eq!(json_kind(&json!("a")), "string");
        assert_eq!(json_kind(&json!([])), "array");
        assert_eq!(json_kind(&json!({})), "object");
    }
}
