//! Attestation fetch, verification policy, and fail-closed checks.
//!
//! This module intentionally does not cache attestation results internally.
//! Attestation/model-key state is tied to the session lifetime, so callers should
//! store a successful [`VerifiedAttestation`] in the session manager only for
//! that session's TTL/request budget. Calling
//! [`AttestationVerifier::verify_model_attestation`] always generates a fresh
//! nonce and fetches fresh Venice evidence.
//!
//! v0.1 deliberately does not implement measurement allowlists for TDX RTMR/MRTD
//! or NVIDIA claims. It verifies the basic Venice attestation envelope, performs
//! local key/address validation, enforces debug-mode policy where evidence exposes
//! it, and exposes strict fail-closed gates for required TDX/NRAS verification.
//! Full DCAP/QVL and NRAS cryptographic verification is not linked; when those
//! verifiers are required by policy, verification fails closed with
//! [`AttestationError::ExternalVerifierUnavailable`].

use std::{fmt, time::SystemTime};

use base64::{Engine as _, engine::general_purpose};
use k256::{PublicKey, elliptic_curve::sec1::ToEncodedPoint};
use rand_core::{OsRng, RngCore};
use serde_json::Value;
use sha2::{Digest as Sha2Digest, Sha256};
use sha3::Keccak256;
use thiserror::Error;

use crate::{
    config::{AttestationConfig, NvidiaRequirement, ProxyConfig},
    util::json_kind,
    venice::{VeniceClient, VeniceClientError},
};

const ATTESTATION_NONCE_BYTES: usize = 32;
const ATTESTATION_NONCE_HEX_CHARS: usize = ATTESTATION_NONCE_BYTES * 2;
const TDX_TEE_TYPE: u32 = 0x81;
const TDX_QUOTE_HEADER_LEN: usize = 48;
const TDX_QUOTE_TEE_TYPE_OFFSET: usize = 4;
const TDX_QUOTE_TEE_TYPE_END: usize = TDX_QUOTE_TEE_TYPE_OFFSET + 4;
const TDX_REPORT_BODY_OFFSET: usize = TDX_QUOTE_HEADER_LEN;
const TDX_REPORT_TD_ATTRIBUTES_OFFSET: usize = TDX_REPORT_BODY_OFFSET + 120;
const TDX_REPORT_TD_ATTRIBUTES_END: usize = TDX_REPORT_TD_ATTRIBUTES_OFFSET + 8;
const TDX_REPORT_DATA_OFFSET: usize = TDX_REPORT_BODY_OFFSET + 520;
const TDX_REPORT_DATA_LEN: usize = 64;
const TDX_REPORT_DATA_END: usize = TDX_REPORT_DATA_OFFSET + TDX_REPORT_DATA_LEN;

#[derive(Clone, Debug)]
pub struct AttestationVerifier {
    policy: AttestationConfig,
    venice_client: VeniceClient,
}

impl AttestationVerifier {
    pub fn from_config(config: &ProxyConfig, venice_client: VeniceClient) -> Self {
        Self::new(config.attestation.clone(), venice_client)
    }

    pub fn new(policy: AttestationConfig, venice_client: VeniceClient) -> Self {
        Self {
            policy,
            venice_client,
        }
    }

    pub fn policy(&self) -> &AttestationConfig {
        &self.policy
    }

    /// Fetches Venice attestation evidence with a fresh nonce and verifies it
    /// according to the configured fail-closed policy.
    pub async fn verify_model_attestation(
        &self,
        model_id: &str,
    ) -> Result<VerifiedAttestation, AttestationError> {
        if model_id.trim().is_empty() {
            return Err(AttestationError::InvalidRequest {
                message: "model id must not be empty".to_owned(),
            });
        }

        let nonce = AttestationNonce::generate();
        let evidence = self
            .venice_client
            .fetch_attestation_evidence(model_id, nonce.as_str())
            .await
            .map_err(AttestationError::Fetch)?;

        self.verify_evidence(model_id, nonce.as_str(), evidence)
    }

    /// Verifies already-fetched evidence. This is public so route/session tests
    /// can exercise policy without a live Venice request.
    pub fn verify_evidence(
        &self,
        requested_model_id: &str,
        client_nonce: &str,
        upstream_response: Value,
    ) -> Result<VerifiedAttestation, AttestationError> {
        verify_attestation_evidence(
            &self.policy,
            requested_model_id,
            client_nonce,
            upstream_response,
        )
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct AttestationNonce(String);

impl AttestationNonce {
    pub fn generate() -> Self {
        let mut bytes = [0_u8; ATTESTATION_NONCE_BYTES];
        OsRng.fill_bytes(&mut bytes);
        Self(hex::encode(bytes))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for AttestationNonce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AttestationNonce").field(&self.0).finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedAttestation {
    pub model_id: String,
    pub model_public_key: String,
    pub signing_address: Option<String>,
    pub tee_provider: Option<String>,
    pub tdx: TdxVerificationSummary,
    pub nvidia: NvidiaVerificationSummary,
    pub verified_at: SystemTime,
    pub attestation_report: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdxVerificationSummary {
    pub present: bool,
    pub verified: bool,
    pub debug: Option<bool>,
    pub tee_type: Option<u32>,
}

impl TdxVerificationSummary {
    fn not_present() -> Self {
        Self {
            present: false,
            verified: false,
            debug: None,
            tee_type: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvidiaVerificationSummary {
    pub present: bool,
    pub verified: NvidiaVerificationStatus,
}

impl NvidiaVerificationSummary {
    fn not_present() -> Self {
        Self {
            present: false,
            verified: NvidiaVerificationStatus::NotPresent,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvidiaVerificationStatus {
    NotPresent,
    IgnoredByPolicy,
    PresentVerifierUnavailable,
}

impl NvidiaVerificationStatus {
    pub fn as_header_value(self) -> &'static str {
        match self {
            Self::NotPresent => "not-present",
            Self::IgnoredByPolicy => "ignored",
            Self::PresentVerifierUnavailable => "verifier-unavailable",
        }
    }
}

#[derive(Debug, Error)]
pub enum AttestationError {
    #[error("invalid attestation request: {message}")]
    InvalidRequest { message: String },
    #[error("TEE attestation fetch failed: {0}")]
    Fetch(#[from] VeniceClientError),
    #[error("TEE attestation response is malformed: {message}")]
    MalformedResponse { message: String },
    #[error("TEE attestation evidence is missing required field {field}")]
    MissingField { field: &'static str },
    #[error("TEE attestation verification failed: {message}")]
    PolicyViolation {
        code: AttestationFailureCode,
        message: String,
    },
    #[error("TEE attestation verifier unavailable: {message}")]
    ExternalVerifierUnavailable {
        verifier: &'static str,
        message: String,
    },
}

impl AttestationError {
    pub fn api_error_type(&self) -> &'static str {
        match self {
            Self::InvalidRequest { .. } => "invalid_request_error",
            Self::ExternalVerifierUnavailable { .. } => "proxy_attestation_verifier_unavailable",
            Self::Fetch(_)
            | Self::MalformedResponse { .. }
            | Self::MissingField { .. }
            | Self::PolicyViolation { .. } => "proxy_attestation_error",
        }
    }

    pub fn api_error_code(&self) -> &'static str {
        match self {
            Self::InvalidRequest { .. } => "invalid_attestation_request",
            Self::Fetch(_) => "attestation_fetch_failed",
            Self::MalformedResponse { .. } => "attestation_malformed_response",
            Self::MissingField { .. } => "attestation_missing_required_field",
            Self::PolicyViolation { code, .. } => code.as_str(),
            Self::ExternalVerifierUnavailable { .. } => "attestation_verifier_unavailable",
        }
    }

    pub fn verifier_unavailable(&self) -> bool {
        matches!(self, Self::ExternalVerifierUnavailable { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttestationFailureCode {
    UpstreamNotVerified,
    NonceMismatch,
    ModelMismatch,
    InvalidSigningKey,
    SigningAddressMismatch,
    DebugModeDetected,
    MissingTdxEvidence,
    InvalidTdxEvidence,
    MissingNvidiaEvidence,
    InvalidNvidiaEvidence,
}

impl AttestationFailureCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UpstreamNotVerified => "attestation_upstream_not_verified",
            Self::NonceMismatch => "attestation_nonce_mismatch",
            Self::ModelMismatch => "attestation_model_mismatch",
            Self::InvalidSigningKey => "attestation_invalid_signing_key",
            Self::SigningAddressMismatch => "attestation_signing_address_mismatch",
            Self::DebugModeDetected => "attestation_debug_mode_detected",
            Self::MissingTdxEvidence => "attestation_missing_tdx_evidence",
            Self::InvalidTdxEvidence => "attestation_invalid_tdx_evidence",
            Self::MissingNvidiaEvidence => "attestation_missing_nvidia_evidence",
            Self::InvalidNvidiaEvidence => "attestation_invalid_nvidia_evidence",
        }
    }
}

fn verify_attestation_evidence(
    policy: &AttestationConfig,
    requested_model_id: &str,
    client_nonce: &str,
    upstream_response: Value,
) -> Result<VerifiedAttestation, AttestationError> {
    validate_nonce_hex(client_nonce)?;

    let evidence = evidence_object(&upstream_response)?;
    let verified = required_bool(evidence, "verified")?;
    if !verified {
        return policy_error(
            AttestationFailureCode::UpstreamNotVerified,
            "Venice did not mark the attestation evidence as verified",
        );
    }

    let nonce = required_string(evidence, "nonce")?;
    if nonce != client_nonce {
        return policy_error(
            AttestationFailureCode::NonceMismatch,
            "attestation nonce does not match the client nonce; evidence may be stale or replayed",
        );
    }

    let model = required_string(evidence, "model")?;
    if model != requested_model_id {
        return policy_error(
            AttestationFailureCode::ModelMismatch,
            format!(
                "attestation model {model:?} does not match requested model {requested_model_id:?}"
            ),
        );
    }

    let signing_key = optional_non_empty_string(evidence, "signing_key")
        .or_else(|| optional_non_empty_string(evidence, "signing_public_key"))
        .ok_or(AttestationError::MissingField {
            field: "signing_key|signing_public_key",
        })?;
    let normalized_signing_key = normalize_public_key_hex(signing_key)?;
    let derived_address = ethereum_address_from_uncompressed_key_hex(&normalized_signing_key)?;
    let signing_address = optional_non_empty_string(evidence, "signing_address")
        .map(normalize_ethereum_address)
        .transpose()?;
    if let Some(signing_address) = &signing_address
        && signing_address != &derived_address
    {
        return policy_error(
            AttestationFailureCode::SigningAddressMismatch,
            format!(
                "signing_address {signing_address} does not match address {derived_address} derived from signing key"
            ),
        );
    }

    if top_level_debug(evidence) == Some(true) && !policy.allow_debug {
        return policy_error(
            AttestationFailureCode::DebugModeDetected,
            "attestation evidence reports debug mode and attestation.allow_debug=false",
        );
    }

    let tdx = evaluate_tdx_policy(
        policy,
        evidence,
        &normalized_signing_key,
        signing_address.as_deref(),
    )?;
    let nvidia = evaluate_nvidia_policy(policy, evidence)?;

    Ok(VerifiedAttestation {
        model_id: requested_model_id.to_owned(),
        model_public_key: normalized_signing_key,
        signing_address,
        tee_provider: optional_non_empty_string(evidence, "tee_provider").map(ToOwned::to_owned),
        tdx,
        nvidia,
        verified_at: SystemTime::now(),
        attestation_report: upstream_response,
    })
}

fn evaluate_tdx_policy(
    policy: &AttestationConfig,
    evidence: &serde_json::Map<String, Value>,
    signing_key: &str,
    signing_address: Option<&str>,
) -> Result<TdxVerificationSummary, AttestationError> {
    let Some(intel_quote) = optional_non_empty_string(evidence, "intel_quote") else {
        return if policy.require_tdx {
            policy_error(
                AttestationFailureCode::MissingTdxEvidence,
                "attestation.require_tdx=true but intel_quote is absent",
            )
        } else {
            Ok(TdxVerificationSummary::not_present())
        };
    };

    let parsed = parse_tdx_quote(intel_quote)?;
    if parsed.tee_type != TDX_TEE_TYPE {
        return policy_error(
            AttestationFailureCode::InvalidTdxEvidence,
            format!(
                "Intel quote teeType 0x{:x} is not TDX teeType 0x{TDX_TEE_TYPE:x}",
                parsed.tee_type
            ),
        );
    }
    if parsed.debug && !policy.allow_debug {
        return policy_error(
            AttestationFailureCode::DebugModeDetected,
            "Intel TDX quote reports debug mode and attestation.allow_debug=false",
        );
    }

    if let Some(reportdata) = optional_non_empty_string(evidence, "tdx_reportdata") {
        verify_reportdata_binding(reportdata, signing_key, signing_address)?;
    }

    if policy.require_tdx {
        let message = if policy.pccs_url.trim().is_empty() {
            "attestation.require_tdx=true requires independent DCAP/QVL quote verification, but no DCAP verifier is linked and attestation.pccs_url is empty".to_owned()
        } else {
            "attestation.require_tdx=true requires independent DCAP/QVL quote verification; PCCS URL is configured but this v0.1 verifier has no DCAP/QVL backend linked".to_owned()
        };
        return Err(AttestationError::ExternalVerifierUnavailable {
            verifier: "tdx-dcap-qvl",
            message,
        });
    }

    Ok(TdxVerificationSummary {
        present: true,
        verified: false,
        debug: Some(parsed.debug),
        tee_type: Some(parsed.tee_type),
    })
}

fn evaluate_nvidia_policy(
    policy: &AttestationConfig,
    evidence: &serde_json::Map<String, Value>,
) -> Result<NvidiaVerificationSummary, AttestationError> {
    let nvidia_payload = evidence
        .get("nvidia_payload")
        .filter(|value| !value.is_null());

    match (policy.require_nvidia, nvidia_payload) {
        (NvidiaRequirement::Required, None) => policy_error(
            AttestationFailureCode::MissingNvidiaEvidence,
            "attestation.require_nvidia=required but nvidia_payload is absent",
        ),
        (NvidiaRequirement::Never, None) => Ok(NvidiaVerificationSummary::not_present()),
        (NvidiaRequirement::Never, Some(_)) => Ok(NvidiaVerificationSummary {
            present: true,
            verified: NvidiaVerificationStatus::IgnoredByPolicy,
        }),
        (_, Some(Value::Object(_))) | (_, Some(Value::String(_))) => {
            Err(AttestationError::ExternalVerifierUnavailable {
                verifier: "nvidia-nras",
                message: "NVIDIA attestation payload is present and policy requires verification, but this v0.1 verifier has no NRAS/local NVIDIA verifier backend linked".to_owned(),
            })
        }
        (_, Some(_)) => policy_error(
            AttestationFailureCode::InvalidNvidiaEvidence,
            "nvidia_payload is present but is not an object or encoded string",
        ),
        (NvidiaRequirement::WhenPresent, None) => Ok(NvidiaVerificationSummary::not_present()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParsedTdxQuote {
    tee_type: u32,
    debug: bool,
}

fn parse_tdx_quote(value: &str) -> Result<ParsedTdxQuote, AttestationError> {
    let bytes = decode_tdx_quote(value)?;

    if bytes.len() < TDX_REPORT_DATA_END {
        return policy_error(
            AttestationFailureCode::InvalidTdxEvidence,
            format!(
                "Intel TDX quote is too short: got {} bytes, need at least {TDX_REPORT_DATA_END}",
                bytes.len()
            ),
        );
    }

    let tee_type = u32::from_le_bytes(
        bytes[TDX_QUOTE_TEE_TYPE_OFFSET..TDX_QUOTE_TEE_TYPE_END]
            .try_into()
            .expect("TDX tee_type slice length is fixed"),
    );
    let td_attributes = u64::from_le_bytes(
        bytes[TDX_REPORT_TD_ATTRIBUTES_OFFSET..TDX_REPORT_TD_ATTRIBUTES_END]
            .try_into()
            .expect("TDX attributes slice length is fixed"),
    );
    let debug = td_attributes & 1 == 1;

    Ok(ParsedTdxQuote { tee_type, debug })
}

fn decode_tdx_quote(value: &str) -> Result<Vec<u8>, AttestationError> {
    let value = value.trim();
    let hex = value.strip_prefix("0x").unwrap_or(value);
    // `hex::decode("")` succeeds with empty bytes, so keep empty input on the base64 path.
    if !hex.is_empty()
        && let Ok(bytes) = hex::decode(hex)
    {
        return Ok(bytes);
    }

    general_purpose::STANDARD
        .decode(value)
        .map_err(|source| AttestationError::PolicyViolation {
            code: AttestationFailureCode::InvalidTdxEvidence,
            message: format!("intel_quote is neither hex nor valid base64: {source}"),
        })
}

fn verify_reportdata_binding(
    reportdata_hex: &str,
    signing_key: &str,
    signing_address: Option<&str>,
) -> Result<(), AttestationError> {
    let reportdata =
        hex::decode(reportdata_hex).map_err(|error| AttestationError::PolicyViolation {
            code: AttestationFailureCode::InvalidTdxEvidence,
            message: format!("tdx_reportdata is not valid hex: {error}"),
        })?;
    if reportdata.len() != TDX_REPORT_DATA_LEN {
        return policy_error(
            AttestationFailureCode::InvalidTdxEvidence,
            format!(
                "tdx_reportdata has {} bytes, expected {TDX_REPORT_DATA_LEN}",
                reportdata.len()
            ),
        );
    }

    let signing_key_bytes =
        hex::decode(signing_key).map_err(|error| AttestationError::PolicyViolation {
            code: AttestationFailureCode::InvalidSigningKey,
            message: format!("normalized signing key is not valid hex: {error}"),
        })?;
    let signing_key_hash = Sha256::digest(&signing_key_bytes);
    if reportdata.starts_with(&signing_key_hash[..]) {
        return Ok(());
    }

    if let Some(signing_address) = signing_address {
        let signing_address_hash = Sha256::digest(signing_address.as_bytes());
        if reportdata.starts_with(&signing_address_hash[..]) {
            return Ok(());
        }
    }

    policy_error(
        AttestationFailureCode::InvalidTdxEvidence,
        "TDX REPORTDATA does not bind the attested signing key or signing address",
    )
}

fn evidence_object(response: &Value) -> Result<&serde_json::Map<String, Value>, AttestationError> {
    if let Value::Object(root) = response {
        if let Some(Value::Object(attestation)) = root.get("attestation") {
            return Ok(attestation);
        }
        return Ok(root);
    }

    Err(AttestationError::MalformedResponse {
        message: format!(
            "expected attestation response object, got {}",
            json_kind(response)
        ),
    })
}

fn required_bool(
    object: &serde_json::Map<String, Value>,
    field: &'static str,
) -> Result<bool, AttestationError> {
    match object.get(field) {
        Some(Value::Bool(value)) => Ok(*value),
        Some(other) => Err(AttestationError::MalformedResponse {
            message: format!("field {field} must be a boolean, got {}", json_kind(other)),
        }),
        None => Err(AttestationError::MissingField { field }),
    }
}

fn required_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, AttestationError> {
    match object.get(field) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(value),
        Some(Value::String(_)) => Err(AttestationError::MalformedResponse {
            message: format!("field {field} must not be empty"),
        }),
        Some(other) => Err(AttestationError::MalformedResponse {
            message: format!("field {field} must be a string, got {}", json_kind(other)),
        }),
        None => Err(AttestationError::MissingField { field }),
    }
}

fn optional_non_empty_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &'static str,
) -> Option<&'a str> {
    match object.get(field) {
        Some(Value::String(value)) if !value.trim().is_empty() => Some(value.as_str()),
        _ => None,
    }
}

fn top_level_debug(object: &serde_json::Map<String, Value>) -> Option<bool> {
    object
        .get("debug")
        .or_else(|| object.get("tdx_debug"))
        .and_then(Value::as_bool)
}

fn normalize_public_key_hex(value: &str) -> Result<String, AttestationError> {
    let value = value.trim().strip_prefix("0x").unwrap_or(value.trim());
    let mut bytes = hex::decode(value).map_err(|error| AttestationError::PolicyViolation {
        code: AttestationFailureCode::InvalidSigningKey,
        message: error.to_string(),
    })?;

    if bytes.len() == 64 {
        let mut uncompressed = Vec::with_capacity(65);
        uncompressed.push(0x04);
        uncompressed.extend_from_slice(&bytes);
        bytes = uncompressed;
    }

    if !matches!(bytes.len(), 33 | 65) {
        return policy_error(
            AttestationFailureCode::InvalidSigningKey,
            format!(
                "signing key must be 33-byte compressed, 64-byte x/y, or 65-byte uncompressed SEC1 public key; got {} bytes",
                bytes.len()
            ),
        );
    }

    let public_key =
        PublicKey::from_sec1_bytes(&bytes).map_err(|_| AttestationError::PolicyViolation {
            code: AttestationFailureCode::InvalidSigningKey,
            message: "signing key is not a valid secp256k1 public key".to_owned(),
        })?;
    Ok(hex::encode(public_key.to_encoded_point(false).as_bytes()))
}

fn ethereum_address_from_uncompressed_key_hex(value: &str) -> Result<String, AttestationError> {
    let bytes = hex::decode(value).map_err(|error| AttestationError::PolicyViolation {
        code: AttestationFailureCode::InvalidSigningKey,
        message: error.to_string(),
    })?;
    if bytes.len() != 65 || bytes.first() != Some(&0x04) {
        return policy_error(
            AttestationFailureCode::InvalidSigningKey,
            "normalized signing key is not an uncompressed 65-byte SEC1 key",
        );
    }

    let hash = Keccak256::digest(&bytes[1..]);
    Ok(format!("0x{}", hex::encode(&hash[12..])))
}

fn normalize_ethereum_address(value: &str) -> Result<String, AttestationError> {
    let value = value.trim();
    let stripped = value.strip_prefix("0x").unwrap_or(value);
    if stripped.len() != 40 || stripped.chars().any(|ch| !ch.is_ascii_hexdigit()) {
        return policy_error(
            AttestationFailureCode::SigningAddressMismatch,
            "signing_address must be a 20-byte Ethereum address encoded as hex",
        );
    }
    Ok(format!("0x{}", stripped.to_ascii_lowercase()))
}

fn validate_nonce_hex(value: &str) -> Result<(), AttestationError> {
    if value.len() != ATTESTATION_NONCE_HEX_CHARS {
        return Err(AttestationError::InvalidRequest {
            message: format!(
                "attestation nonce must be {ATTESTATION_NONCE_HEX_CHARS} hex characters"
            ),
        });
    }
    if value.chars().any(|ch| !ch.is_ascii_hexdigit()) {
        return Err(AttestationError::InvalidRequest {
            message: "attestation nonce must contain only hex characters".to_owned(),
        });
    }
    Ok(())
}

fn policy_error<T>(
    code: AttestationFailureCode,
    message: impl Into<String>,
) -> Result<T, AttestationError> {
    Err(AttestationError::PolicyViolation {
        code,
        message: message.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashMap, net::SocketAddr, time::Duration};

    use axum::{
        Router,
        body::Body,
        extract::Query,
        http::{Response, StatusCode},
        response::IntoResponse,
        routing::get,
    };
    use k256::SecretKey;
    use serde_json::json;
    use tokio::net::TcpListener;

    const MODEL: &str = "e2ee-qwen3-5-122b-a10b";
    const NONCE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn policy_for_basic_success() -> AttestationConfig {
        AttestationConfig {
            require_tdx: false,
            require_nvidia: NvidiaRequirement::WhenPresent,
            ..AttestationConfig::default()
        }
    }

    fn verifier(policy: AttestationConfig) -> AttestationVerifier {
        AttestationVerifier::new(policy, test_venice_client("http://127.0.0.1:1/api/v1"))
    }

    fn test_venice_client(base_url: &str) -> VeniceClient {
        VeniceClient::new(base_url, "test-api-key", Duration::from_secs(1))
            .expect("test Venice client should build")
    }

    fn key_material() -> (String, String) {
        let secret_key = SecretKey::from_slice(&[7_u8; 32]).expect("fixed secret key is valid");
        let public_key = secret_key.public_key();
        let public_key_hex = hex::encode(public_key.to_encoded_point(false).as_bytes());
        let address = ethereum_address_from_uncompressed_key_hex(&public_key_hex)
            .expect("test public key should derive address");
        (public_key_hex, address)
    }

    fn valid_evidence() -> Value {
        let (signing_key, signing_address) = key_material();
        json!({
            "verified": true,
            "nonce": NONCE,
            "model": MODEL,
            "tee_provider": "tdx",
            "signing_key": signing_key,
            "signing_address": signing_address
        })
    }

    #[test]
    fn generated_nonce_is_32_bytes_lower_hex() {
        let nonce = AttestationNonce::generate();

        assert_eq!(nonce.as_str().len(), 64);
        assert!(nonce.as_str().chars().all(|ch| ch.is_ascii_hexdigit()));
        assert!(!nonce.as_str().chars().any(|ch| ch.is_ascii_uppercase()));
    }

    #[test]
    fn valid_basic_evidence_passes_without_optional_hardware_requirements() {
        let result = verifier(policy_for_basic_success())
            .verify_evidence(MODEL, NONCE, valid_evidence())
            .expect("valid basic attestation should pass");

        let (expected_key, expected_address) = key_material();
        assert_eq!(result.model_id, MODEL);
        assert_eq!(result.model_public_key, expected_key);
        assert_eq!(
            result.signing_address.as_deref(),
            Some(expected_address.as_str())
        );
        assert_eq!(result.tee_provider.as_deref(), Some("tdx"));
        assert!(!result.tdx.present);
        assert_eq!(result.nvidia.verified, NvidiaVerificationStatus::NotPresent);
    }

    #[test]
    fn missing_required_fields_fail_closed() {
        let mut evidence = valid_evidence();
        evidence.as_object_mut().unwrap().remove("verified");

        let error = verifier(policy_for_basic_success())
            .verify_evidence(MODEL, NONCE, evidence)
            .expect_err("missing verified field must fail");

        assert!(matches!(
            error,
            AttestationError::MissingField { field: "verified" }
        ));
        assert_eq!(error.api_error_code(), "attestation_missing_required_field");
    }

    #[test]
    fn debug_evidence_fails_when_debug_is_not_allowed() {
        let mut evidence = valid_evidence();
        evidence
            .as_object_mut()
            .unwrap()
            .insert("debug".to_owned(), json!(true));

        let error = verifier(policy_for_basic_success())
            .verify_evidence(MODEL, NONCE, evidence)
            .expect_err("debug attestation must fail");

        assert!(matches!(
            error,
            AttestationError::PolicyViolation {
                code: AttestationFailureCode::DebugModeDetected,
                ..
            }
        ));
    }

    #[test]
    fn tdx_required_mode_fails_on_missing_tdx_evidence() {
        let error = verifier(AttestationConfig {
            require_tdx: true,
            require_nvidia: NvidiaRequirement::Never,
            ..AttestationConfig::default()
        })
        .verify_evidence(MODEL, NONCE, valid_evidence())
        .expect_err("missing required TDX evidence must fail");

        assert!(matches!(
            error,
            AttestationError::PolicyViolation {
                code: AttestationFailureCode::MissingTdxEvidence,
                ..
            }
        ));
    }

    #[test]
    fn tdx_required_mode_fails_on_invalid_tdx_evidence() {
        let mut evidence = valid_evidence();
        evidence
            .as_object_mut()
            .unwrap()
            .insert("intel_quote".to_owned(), json!("not quote encoding"));

        let error = verifier(AttestationConfig {
            require_tdx: true,
            require_nvidia: NvidiaRequirement::Never,
            ..AttestationConfig::default()
        })
        .verify_evidence(MODEL, NONCE, evidence)
        .expect_err("invalid TDX evidence must fail");

        assert!(matches!(
            error,
            AttestationError::PolicyViolation {
                code: AttestationFailureCode::InvalidTdxEvidence,
                ..
            }
        ));
    }

    #[test]
    fn tdx_debug_quote_fails_when_debug_is_not_allowed() {
        let mut evidence = valid_evidence();
        evidence.as_object_mut().unwrap().insert(
            "intel_quote".to_owned(),
            json!(tdx_quote_hex(true, TDX_TEE_TYPE)),
        );

        let error = verifier(AttestationConfig {
            require_tdx: false,
            require_nvidia: NvidiaRequirement::Never,
            allow_debug: false,
            ..AttestationConfig::default()
        })
        .verify_evidence(MODEL, NONCE, evidence)
        .expect_err("debug quote must fail");

        assert!(matches!(
            error,
            AttestationError::PolicyViolation {
                code: AttestationFailureCode::DebugModeDetected,
                ..
            }
        ));
    }

    #[test]
    fn tdx_optional_mode_accepts_legacy_base64_quote_encoding() {
        let mut evidence = valid_evidence();
        evidence.as_object_mut().unwrap().insert(
            "intel_quote".to_owned(),
            json!(tdx_quote_base64(false, TDX_TEE_TYPE)),
        );

        let result = verifier(AttestationConfig {
            require_tdx: false,
            require_nvidia: NvidiaRequirement::Never,
            ..AttestationConfig::default()
        })
        .verify_evidence(MODEL, NONCE, evidence)
        .expect("legacy base64-encoded TDX quote should parse when TDX is optional");

        assert!(result.tdx.present);
        assert_eq!(result.tdx.tee_type, Some(TDX_TEE_TYPE));
    }

    #[test]
    fn tdx_required_mode_fails_closed_when_dcap_verifier_is_unavailable() {
        let mut evidence = valid_evidence();
        evidence.as_object_mut().unwrap().insert(
            "intel_quote".to_owned(),
            json!(tdx_quote_hex(false, TDX_TEE_TYPE)),
        );

        let error = verifier(AttestationConfig {
            require_tdx: true,
            require_nvidia: NvidiaRequirement::Never,
            ..AttestationConfig::default()
        })
        .verify_evidence(MODEL, NONCE, evidence)
        .expect_err("strict TDX should fail without DCAP verifier");

        assert!(matches!(
            error,
            AttestationError::ExternalVerifierUnavailable {
                verifier: "tdx-dcap-qvl",
                ..
            }
        ));
        assert_eq!(error.api_error_code(), "attestation_verifier_unavailable");
    }

    #[test]
    fn nvidia_required_mode_fails_on_missing_nvidia_evidence() {
        let error = verifier(AttestationConfig {
            require_tdx: false,
            require_nvidia: NvidiaRequirement::Required,
            ..AttestationConfig::default()
        })
        .verify_evidence(MODEL, NONCE, valid_evidence())
        .expect_err("missing required NVIDIA evidence must fail");

        assert!(matches!(
            error,
            AttestationError::PolicyViolation {
                code: AttestationFailureCode::MissingNvidiaEvidence,
                ..
            }
        ));
    }

    #[test]
    fn nvidia_required_mode_fails_on_invalid_nvidia_evidence() {
        let mut evidence = valid_evidence();
        evidence
            .as_object_mut()
            .unwrap()
            .insert("nvidia_payload".to_owned(), json!(42));

        let error = verifier(AttestationConfig {
            require_tdx: false,
            require_nvidia: NvidiaRequirement::Required,
            ..AttestationConfig::default()
        })
        .verify_evidence(MODEL, NONCE, evidence)
        .expect_err("invalid NVIDIA evidence must fail");

        assert!(matches!(
            error,
            AttestationError::PolicyViolation {
                code: AttestationFailureCode::InvalidNvidiaEvidence,
                ..
            }
        ));
    }

    #[test]
    fn nvidia_payload_when_present_fails_closed_without_nras_verifier() {
        let mut evidence = valid_evidence();
        evidence
            .as_object_mut()
            .unwrap()
            .insert("nvidia_payload".to_owned(), json!({ "nonce": NONCE }));

        let error = verifier(policy_for_basic_success())
            .verify_evidence(MODEL, NONCE, evidence)
            .expect_err("present NVIDIA evidence must be verified");

        assert!(matches!(
            error,
            AttestationError::ExternalVerifierUnavailable {
                verifier: "nvidia-nras",
                ..
            }
        ));
    }

    #[test]
    fn nonce_mismatch_fails_closed_as_stale_or_replayed_evidence() {
        let mut evidence = valid_evidence();
        evidence.as_object_mut().unwrap().insert(
            "nonce".to_owned(),
            json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        );

        let error = verifier(policy_for_basic_success())
            .verify_evidence(MODEL, NONCE, evidence)
            .expect_err("nonce mismatch must fail");

        assert!(matches!(
            error,
            AttestationError::PolicyViolation {
                code: AttestationFailureCode::NonceMismatch,
                ..
            }
        ));
    }

    #[test]
    fn signing_address_mismatch_fails_closed() {
        let mut evidence = valid_evidence();
        evidence.as_object_mut().unwrap().insert(
            "signing_address".to_owned(),
            json!("0x0000000000000000000000000000000000000000"),
        );

        let error = verifier(policy_for_basic_success())
            .verify_evidence(MODEL, NONCE, evidence)
            .expect_err("address mismatch must fail");

        assert!(matches!(
            error,
            AttestationError::PolicyViolation {
                code: AttestationFailureCode::SigningAddressMismatch,
                ..
            }
        ));
    }

    #[test]
    fn malformed_upstream_response_shape_fails_closed() {
        let error = verifier(policy_for_basic_success())
            .verify_evidence(MODEL, NONCE, json!([]))
            .expect_err("array response must fail");

        assert!(matches!(error, AttestationError::MalformedResponse { .. }));
    }

    #[tokio::test]
    async fn fetches_attestation_with_model_and_nonce_then_verifies() {
        let base_url = spawn_attestation_server(|query| {
            assert_eq!(query.get("model").map(String::as_str), Some(MODEL));
            let nonce = query
                .get("nonce")
                .expect("nonce query parameter should be present");
            assert_eq!(nonce.len(), 64);
            assert!(nonce.chars().all(|ch| ch.is_ascii_hexdigit()));

            let (signing_key, signing_address) = key_material();
            (
                StatusCode::OK,
                serde_json::to_vec(&json!({
                    "verified": true,
                    "nonce": nonce,
                    "model": MODEL,
                    "signing_key": signing_key,
                    "signing_address": signing_address
                }))
                .expect("response should serialize"),
            )
        })
        .await;
        let verifier =
            AttestationVerifier::new(policy_for_basic_success(), test_venice_client(&base_url));

        let result = verifier
            .verify_model_attestation(MODEL)
            .await
            .expect("mock attestation should verify");

        assert_eq!(result.model_id, MODEL);
        assert_eq!(result.model_public_key, key_material().0);
    }

    #[tokio::test]
    async fn malformed_upstream_json_fails_closed() {
        let base_url = spawn_raw_attestation_server(StatusCode::OK, b"{".to_vec()).await;
        let verifier =
            AttestationVerifier::new(policy_for_basic_success(), test_venice_client(&base_url));

        let error = verifier
            .verify_model_attestation(MODEL)
            .await
            .expect_err("malformed upstream JSON must fail");

        assert!(matches!(
            error,
            AttestationError::Fetch(VeniceClientError::MalformedAttestationPayload { .. })
        ));
        assert_eq!(error.api_error_code(), "attestation_fetch_failed");
    }

    #[tokio::test]
    async fn upstream_fetch_errors_fail_closed() {
        let verifier = AttestationVerifier::new(
            policy_for_basic_success(),
            test_venice_client("http://127.0.0.1:1/api/v1"),
        );

        let error = verifier
            .verify_model_attestation(MODEL)
            .await
            .expect_err("connection failure must fail closed");

        assert!(matches!(error, AttestationError::Fetch(_)));
        assert_eq!(error.api_error_code(), "attestation_fetch_failed");
    }

    fn tdx_quote_hex(debug: bool, tee_type: u32) -> String {
        hex::encode(tdx_quote_bytes(debug, tee_type))
    }

    fn tdx_quote_base64(debug: bool, tee_type: u32) -> String {
        general_purpose::STANDARD.encode(tdx_quote_bytes(debug, tee_type))
    }

    fn tdx_quote_bytes(debug: bool, tee_type: u32) -> Vec<u8> {
        let mut bytes = vec![0_u8; TDX_REPORT_DATA_END];
        bytes[TDX_QUOTE_TEE_TYPE_OFFSET..TDX_QUOTE_TEE_TYPE_END]
            .copy_from_slice(&tee_type.to_le_bytes());
        let td_attributes = if debug { 1_u64 } else { 0_u64 };
        bytes[TDX_REPORT_TD_ATTRIBUTES_OFFSET..TDX_REPORT_TD_ATTRIBUTES_END]
            .copy_from_slice(&td_attributes.to_le_bytes());
        bytes
    }

    async fn spawn_attestation_server<F>(handler: F) -> String
    where
        F: Fn(HashMap<String, String>) -> (StatusCode, Vec<u8>) + Clone + Send + Sync + 'static,
    {
        async fn route<F>(
            Query(query): Query<HashMap<String, String>>,
            handler: F,
        ) -> Response<Body>
        where
            F: Fn(HashMap<String, String>) -> (StatusCode, Vec<u8>) + Clone + Send + Sync + 'static,
        {
            let (status, body) = handler(query);
            (status, body).into_response()
        }

        let app = Router::new().route(
            "/api/v1/tee/attestation",
            get({
                let handler = handler.clone();
                move |query| route(query, handler.clone())
            }),
        );
        spawn_router(app).await
    }

    async fn spawn_raw_attestation_server(status: StatusCode, body: Vec<u8>) -> String {
        let app = Router::new().route(
            "/api/v1/tee/attestation",
            get(move || async move { (status, body.clone()) }),
        );
        spawn_router(app).await
    }

    async fn spawn_router(app: Router) -> String {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("test listener should bind");
        let addr: SocketAddr = listener.local_addr().expect("listener should have address");
        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("test server should run");
        });
        format!("http://{addr}/api/v1")
    }
}
