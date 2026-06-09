//! Configuration loading and validation.
//!
//! from the configured environment variable.

use std::{
    env::{self, VarError},
    fmt, fs, io,
    path::{Path, PathBuf},
};

use axum::http::HeaderName;
use serde::Deserialize;
use thiserror::Error;

/// Top-level proxy configuration.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ProxyConfig {
    pub server: ServerConfig,
    pub venice: VeniceConfig,
    pub keys: KeysConfig,
    pub session: SessionConfig,
    pub attestation: AttestationConfig,
    pub e2ee: E2eeConfig,
    pub tools: ToolsConfig,
}

impl ProxyConfig {
    /// Loads configuration from a TOML file and validates it.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|source| ConfigError::ReadFile {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml_str(&contents)
    }

    /// Parses TOML configuration, applies defaults, and validates the result.
    pub fn from_toml_str(contents: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(contents).map_err(ConfigError::ParseToml)?;
        config.validate()?;
        Ok(config)
    }

    /// Validates a fully materialized configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_non_empty("server.host", &self.server.host)?;
        validate_http_url("venice.base_url", &self.venice.base_url, false)?;
        validate_env_var_name("venice.api_key_env", &self.venice.api_key_env)?;

        if self.session.idle_ttl_seconds == 0 {
            return Err(ConfigError::invalid(
                "session.idle_ttl_seconds",
                "must be greater than zero",
            ));
        }
        if self.session.max_ttl_seconds == 0 {
            return Err(ConfigError::invalid(
                "session.max_ttl_seconds",
                "must be greater than zero",
            ));
        }
        if self.session.idle_ttl_seconds > self.session.max_ttl_seconds {
            return Err(ConfigError::invalid(
                "session.idle_ttl_seconds",
                "must be less than or equal to session.max_ttl_seconds",
            ));
        }
        if self.session.max_requests == 0 {
            return Err(ConfigError::invalid(
                "session.max_requests",
                "must be greater than zero",
            ));
        }
        validate_header_name("session.headers.preferred", &self.session.headers.preferred)?;
        validate_header_name(
            "session.headers.open_webui",
            &self.session.headers.open_webui,
        )?;

        validate_http_url("attestation.pccs_url", &self.attestation.pccs_url, true)?;
        validate_http_url("attestation.nras_url", &self.attestation.nras_url, false)?;

        validate_non_empty("e2ee.hkdf_info", &self.e2ee.hkdf_info)?;

        validate_non_empty("tools.marker_start", &self.tools.marker_start)?;
        validate_non_empty("tools.marker_end", &self.tools.marker_end)?;
        if self.tools.marker_start == self.tools.marker_end {
            return Err(ConfigError::invalid(
                "tools.marker_end",
                "must differ from tools.marker_start",
            ));
        }
        if self.tools.max_calls_per_turn != 1 {
            return Err(ConfigError::invalid(
                "tools.max_calls_per_turn",
                "must be 1; v0.1 supports exactly one tool call per assistant turn",
            ));
        }
        if self.tools.allow_parallel {
            return Err(ConfigError::invalid(
                "tools.allow_parallel",
                "must be false; v0.1 does not support parallel tool calls",
            ));
        }
        if self.tools.initial_marker_scan_bytes == 0 {
            return Err(ConfigError::invalid(
                "tools.initial_marker_scan_bytes",
                "must be greater than zero",
            ));
        }
        if self.tools.tool_call_max_bytes == 0 {
            return Err(ConfigError::invalid(
                "tools.tool_call_max_bytes",
                "must be greater than zero",
            ));
        }
        if self.tools.tool_call_marker_timeout_ms == 0 {
            return Err(ConfigError::invalid(
                "tools.tool_call_marker_timeout_ms",
                "must be greater than zero",
            ));
        }
        if !self.tools.emit_tool_call_arguments_single_chunk {
            return Err(ConfigError::invalid(
                "tools.emit_tool_call_arguments_single_chunk",
                "must be true; v0.1 emits complete tool-call arguments in one chunk",
            ));
        }

        Ok(())
    }

    /// Looks up the Venice API key from the configured environment variable.
    ///
    /// The secret is returned in a wrapper whose debug representation is
    /// redacted. Callers that need to send the key upstream must explicitly call
    /// [`VeniceApiKey::expose_secret`].
    pub fn venice_api_key_from_env(&self) -> Result<VeniceApiKey, ConfigError> {
        self.venice_api_key_from_env_with(|name| env::var(name))
    }

    /// Testable variant of [`Self::venice_api_key_from_env`].
    pub fn venice_api_key_from_env_with<F>(&self, lookup: F) -> Result<VeniceApiKey, ConfigError>
    where
        F: FnOnce(&str) -> Result<String, VarError>,
    {
        match lookup(&self.venice.api_key_env) {
            Ok(value) if !value.trim().is_empty() => Ok(VeniceApiKey(value)),
            Ok(_) | Err(VarError::NotPresent) => Err(ConfigError::MissingApiKeyEnv {
                env_var: self.venice.api_key_env.clone(),
            }),
            Err(VarError::NotUnicode(_)) => Err(ConfigError::UnreadableApiKeyEnv {
                env_var: self.venice.api_key_env.clone(),
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 11_434,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct VeniceConfig {
    pub base_url: String,
    pub api_key_env: String,
}

impl Default for VeniceConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.venice.ai/api/v1".to_owned(),
            api_key_env: "VENICE_API_KEY".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct KeysConfig {
    pub generate_proxy_instance_key_on_startup: bool,
}

impl Default for KeysConfig {
    fn default() -> Self {
        Self {
            generate_proxy_instance_key_on_startup: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct SessionConfig {
    pub idle_ttl_seconds: u64,
    pub max_ttl_seconds: u64,
    pub max_requests: u64,
    pub fallback_scope: SessionFallbackScope,
    pub headers: SessionHeadersConfig,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            idle_ttl_seconds: 600,
            max_ttl_seconds: 1_800,
            max_requests: 100,
            fallback_scope: SessionFallbackScope::Request,
            headers: SessionHeadersConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct SessionHeadersConfig {
    pub preferred: String,
    pub open_webui: String,
}

impl Default for SessionHeadersConfig {
    fn default() -> Self {
        Self {
            preferred: "X-Venice-Proxy-Session-Id".to_owned(),
            open_webui: "X-OpenWebUI-Chat-Id".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionFallbackScope {
    Agent,
    #[default]
    Request,
    Disabled,
}

impl SessionFallbackScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Request => "request",
            Self::Disabled => "disabled",
        }
    }
}

impl fmt::Display for SessionFallbackScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct AttestationConfig {
    pub mode: AttestationMode,
    pub require_tdx: bool,
    pub require_nvidia: NvidiaRequirement,
    pub allow_debug: bool,
    pub pccs_url: String,
    pub nras_url: String,
}

impl Default for AttestationConfig {
    fn default() -> Self {
        Self {
            mode: AttestationMode::Independent,
            require_tdx: true,
            require_nvidia: NvidiaRequirement::WhenPresent,
            allow_debug: false,
            pccs_url: String::new(),
            nras_url: "https://nras.attestation.nvidia.com/v3/attest/gpu".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttestationMode {
    #[default]
    Independent,
}

impl AttestationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Independent => "independent",
        }
    }
}

impl fmt::Display for AttestationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NvidiaRequirement {
    Required,
    #[default]
    WhenPresent,
    Never,
}

impl NvidiaRequirement {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Required => "required",
            Self::WhenPresent => "when_present",
            Self::Never => "never",
        }
    }
}

impl fmt::Display for NvidiaRequirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct E2eeConfig {
    pub hkdf_info: String,
    pub require_encrypted_response_content: bool,
}

impl Default for E2eeConfig {
    fn default() -> Self {
        Self {
            hkdf_info: "ecdsa_encryption".to_owned(),
            require_encrypted_response_content: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsConfig {
    pub enabled: bool,
    pub mode: ToolMode,
    pub marker_start: String,
    pub marker_end: String,
    pub max_retries: u32,
    pub max_calls_per_turn: u32,
    pub allow_parallel: bool,
    pub initial_marker_scan_bytes: usize,
    pub tool_call_max_bytes: usize,
    pub tool_call_marker_timeout_ms: u64,
    pub validate_json_schema: bool,
    pub emit_tool_call_arguments_single_chunk: bool,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: ToolMode::Emulated,
            marker_start: "<tool_call>".to_owned(),
            marker_end: "</tool_call>".to_owned(),
            max_retries: 2,
            max_calls_per_turn: 1,
            allow_parallel: false,
            initial_marker_scan_bytes: 128,
            tool_call_max_bytes: 65_536,
            tool_call_marker_timeout_ms: 30_000,
            validate_json_schema: true,
            emit_tool_call_arguments_single_chunk: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolMode {
    #[default]
    Emulated,
    None,
}

impl ToolMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Emulated => "emulated",
            Self::None => "none",
        }
    }
}

impl fmt::Display for ToolMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A Venice API key loaded from the configured environment variable.
#[derive(Clone, PartialEq, Eq)]
pub struct VeniceApiKey(String);

impl VeniceApiKey {
    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for VeniceApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("VeniceApiKey([redacted])")
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config {path}: {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to parse config TOML: {0}")]
    ParseToml(#[from] toml::de::Error),
    #[error("invalid config value for {field}: {message}")]
    InvalidValue {
        field: &'static str,
        message: String,
    },
    #[error("Venice API key environment variable {env_var} is not set")]
    MissingApiKeyEnv { env_var: String },
    #[error("Venice API key environment variable {env_var} is not valid Unicode")]
    UnreadableApiKeyEnv { env_var: String },
}

impl ConfigError {
    fn invalid(field: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidValue {
            field,
            message: message.into(),
        }
    }
}

fn validate_non_empty(field: &'static str, value: &str) -> Result<(), ConfigError> {
    if value.trim().is_empty() {
        return Err(ConfigError::invalid(field, "must not be empty"));
    }
    Ok(())
}

fn validate_http_url(
    field: &'static str,
    value: &str,
    allow_empty: bool,
) -> Result<(), ConfigError> {
    let value = value.trim();
    if value.is_empty() {
        if allow_empty {
            return Ok(());
        }
        return Err(ConfigError::invalid(field, "must not be empty"));
    }

    if !(value.starts_with("https://") || value.starts_with("http://")) {
        return Err(ConfigError::invalid(
            field,
            "must start with http:// or https://",
        ));
    }

    Ok(())
}

fn validate_env_var_name(field: &'static str, value: &str) -> Result<(), ConfigError> {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return Err(ConfigError::invalid(field, "must not be empty"));
    };

    if !(first == '_' || first.is_ascii_alphabetic()) {
        return Err(ConfigError::invalid(
            field,
            "must start with an ASCII letter or underscore",
        ));
    }

    if !chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric()) {
        return Err(ConfigError::invalid(
            field,
            "must contain only ASCII letters, digits, and underscores",
        ));
    }

    Ok(())
}

fn validate_header_name(field: &'static str, value: &str) -> Result<(), ConfigError> {
    validate_non_empty(field, value)?;
    HeaderName::from_bytes(value.as_bytes())
        .map_err(|_| ConfigError::invalid(field, "must be a valid HTTP header name"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_spec_draft() {
        let config = ProxyConfig::default();

        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 11_434);
        assert_eq!(config.venice.base_url, "https://api.venice.ai/api/v1");
        assert_eq!(config.venice.api_key_env, "VENICE_API_KEY");
        assert!(config.keys.generate_proxy_instance_key_on_startup);
        assert_eq!(config.session.idle_ttl_seconds, 600);
        assert_eq!(config.session.max_ttl_seconds, 1_800);
        assert_eq!(config.session.max_requests, 100);
        assert_eq!(config.session.fallback_scope, SessionFallbackScope::Request);
        assert_eq!(SessionFallbackScope::Disabled.as_str(), "disabled");
        assert_eq!(
            config.session.headers.preferred,
            "X-Venice-Proxy-Session-Id"
        );
        assert_eq!(config.session.headers.open_webui, "X-OpenWebUI-Chat-Id");
        assert_eq!(config.attestation.mode, AttestationMode::Independent);
        assert!(config.attestation.require_tdx);
        assert_eq!(
            config.attestation.require_nvidia,
            NvidiaRequirement::WhenPresent
        );
        assert_eq!(NvidiaRequirement::Required.as_str(), "required");
        assert_eq!(NvidiaRequirement::Never.as_str(), "never");
        assert!(!config.attestation.allow_debug);
        assert_eq!(config.attestation.pccs_url, "");
        assert_eq!(
            config.attestation.nras_url,
            "https://nras.attestation.nvidia.com/v3/attest/gpu"
        );
        assert_eq!(config.e2ee.hkdf_info, "ecdsa_encryption");
        assert!(config.e2ee.require_encrypted_response_content);
        assert!(config.tools.enabled);
        assert_eq!(config.tools.mode, ToolMode::Emulated);
        assert_eq!(config.tools.marker_start, "<tool_call>");
        assert_eq!(config.tools.marker_end, "</tool_call>");
        assert_eq!(config.tools.max_retries, 2);
        assert_eq!(config.tools.max_calls_per_turn, 1);
        assert!(!config.tools.allow_parallel);
        assert_eq!(config.tools.initial_marker_scan_bytes, 128);
        assert_eq!(config.tools.tool_call_max_bytes, 65_536);
        assert_eq!(config.tools.tool_call_marker_timeout_ms, 30_000);
        assert!(config.tools.validate_json_schema);
        assert!(config.tools.emit_tool_call_arguments_single_chunk);

        config.validate().expect("default config is valid");
    }

    #[test]
    fn toml_config_applies_defaults_for_missing_sections() {
        let config = ProxyConfig::from_toml_str(
            r#"
            [server]
            host = "0.0.0.0"
            port = 8080

            [tools]
            enabled = false
            mode = "none"
            "#,
        )
        .expect("partial config should load with defaults");

        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.venice.api_key_env, "VENICE_API_KEY");
        assert!(!config.tools.enabled);
        assert_eq!(config.tools.mode, ToolMode::None);
        assert_eq!(config.tools.marker_start, "<tool_call>");
    }

    #[test]
    fn validation_rejects_invalid_values() {
        let err = ProxyConfig::from_toml_str(
            r#"
            [venice]
            api_key_env = "not-valid-env-name"
            "#,
        )
        .expect_err("invalid env var name should be rejected");

        assert!(matches!(
            err,
            ConfigError::InvalidValue {
                field: "venice.api_key_env",
                ..
            }
        ));
    }

    #[test]
    fn validation_rejects_unsupported_v0_1_tool_modes() {
        for (field, toml) in [
            ("tools.max_calls_per_turn", "max_calls_per_turn = 2"),
            ("tools.allow_parallel", "allow_parallel = true"),
            (
                "tools.emit_tool_call_arguments_single_chunk",
                "emit_tool_call_arguments_single_chunk = false",
            ),
        ] {
            let err = ProxyConfig::from_toml_str(&format!("[tools]\n{toml}\n"))
                .expect_err("unsupported v0.1 tool mode should be rejected");
            assert!(matches!(
                err,
                ConfigError::InvalidValue { field: actual, .. } if actual == field
            ));
        }
    }

    #[test]
    fn missing_api_key_environment_variable_is_reported() {
        let config = ProxyConfig::default();
        let err = config
            .venice_api_key_from_env_with(|_| Err(VarError::NotPresent))
            .expect_err("missing env var should be reported");

        assert!(matches!(
            err,
            ConfigError::MissingApiKeyEnv { ref env_var } if env_var == "VENICE_API_KEY"
        ));
        assert_eq!(
            err.to_string(),
            "Venice API key environment variable VENICE_API_KEY is not set"
        );
    }

    #[test]
    fn api_key_debug_output_is_redacted() {
        let config = ProxyConfig::default();
        let key = config
            .venice_api_key_from_env_with(|_| Ok("super-secret-test-key".to_owned()))
            .expect("test key should load");

        assert_eq!(key.expose_secret(), "super-secret-test-key");
        assert_eq!(format!("{key:?}"), "VeniceApiKey([redacted])");
    }
}
