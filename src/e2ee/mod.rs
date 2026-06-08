//! Venice E2EE encryption/decryption codec.
//!
//!
//! ```text
//! ephemeral_public_key[65 bytes] || nonce[12 bytes] || ciphertext_and_gcm_tag
//! ```
//!
//! The packed bytes are serialized as lowercase hex strings in request message
//! content fields and in encrypted Venice streaming response `delta.content`
//! fields.

use std::fmt;

use aes_gcm::{
    Aes256Gcm, Nonce as AesNonce,
    aead::{Aead, KeyInit},
};
use hkdf::Hkdf;
use k256::{
    PublicKey, SecretKey,
    ecdh::{SharedSecret, diffie_hellman},
    elliptic_curve::sec1::ToEncodedPoint,
};
use rand_core::{OsRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use thiserror::Error;
use zeroize::{ZeroizeOnDrop, Zeroizing};

use crate::config::E2eeConfig;

pub const EPHEMERAL_PUBLIC_KEY_LEN: usize = 65;
pub const NONCE_LEN: usize = 12;
pub const AES_256_KEY_LEN: usize = 32;
pub const AES_GCM_TAG_LEN: usize = 16;
pub const PACKED_PREFIX_LEN: usize = EPHEMERAL_PUBLIC_KEY_LEN + NONCE_LEN;
pub const MIN_PACKED_PAYLOAD_LEN: usize = PACKED_PREFIX_LEN + AES_GCM_TAG_LEN;

/// E2EE codec configured with the HKDF info string and fail-closed response policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct E2eeCodec {
    hkdf_info: Vec<u8>,
    require_encrypted_response_content: bool,
}

impl E2eeCodec {
    pub fn from_config(config: &E2eeConfig) -> Result<Self, E2eeCodecError> {
        Self::new(
            config.hkdf_info.as_bytes(),
            config.require_encrypted_response_content,
        )
    }

    pub fn new(
        hkdf_info: impl AsRef<[u8]>,
        require_encrypted_response_content: bool,
    ) -> Result<Self, E2eeCodecError> {
        let hkdf_info = hkdf_info.as_ref();
        if hkdf_info.is_empty() {
            return Err(E2eeCodecError::EmptyHkdfInfo);
        }

        Ok(Self {
            hkdf_info: hkdf_info.to_vec(),
            require_encrypted_response_content,
        })
    }

    pub fn require_encrypted_response_content(&self) -> bool {
        self.require_encrypted_response_content
    }

    /// Derives a Venice AES-256-GCM content key from a local secp256k1 private
    /// key and a peer uncompressed SEC1 public key hex string.
    pub fn derive_content_key(
        &self,
        local_private_key: &SecretKey,
        peer_public_key_hex: &str,
    ) -> Result<ContentEncryptionKey, E2eeCodecError> {
        let peer_public_key = decode_uncompressed_public_key_hex(peer_public_key_hex)?;
        Ok(self.derive_content_key_from_public_key(local_private_key, &peer_public_key))
    }

    /// Encrypts one normalized model-visible text field for a Venice E2EE request.
    pub fn encrypt_content(
        &self,
        plaintext: &str,
        peer_public_key_hex: &str,
    ) -> Result<EncryptedPayload, E2eeCodecError> {
        let peer_public_key = decode_uncompressed_public_key_hex(peer_public_key_hex)?;
        let ephemeral_private_key = SecretKey::random(&mut OsRng);
        let nonce = Nonce::generate();

        self.encrypt_content_with_parts(plaintext, &peer_public_key, ephemeral_private_key, nonce)
    }

    /// Encrypts a serializable normalized payload as canonical JSON bytes before
    /// applying the Venice field codec.
    pub fn encrypt_json_payload<T: Serialize>(
        &self,
        payload: &T,
        peer_public_key_hex: &str,
    ) -> Result<EncryptedPayload, E2eeCodecError> {
        let plaintext = serde_json::to_string(payload).map_err(|source| {
            E2eeCodecError::MalformedEncryptedPayload {
                message: format!("failed to serialize normalized payload: {source}"),
            }
        })?;
        self.encrypt_content(&plaintext, peer_public_key_hex)
    }

    /// Decrypts a packed Venice E2EE hex payload into UTF-8 text.
    pub fn decrypt_content(
        &self,
        payload: &EncryptedPayload,
        recipient_private_key: &SecretKey,
    ) -> Result<String, E2eeCodecError> {
        let packed = PackedEncryptedPayload::unpack(payload)?;
        let peer_public_key =
            PublicKey::from_sec1_bytes(&packed.ephemeral_public_key).map_err(|_| {
                E2eeCodecError::MalformedEncryptedPayload {
                    message: "ephemeral public key is not a valid uncompressed secp256k1 key"
                        .to_owned(),
                }
            })?;
        let key = self.derive_content_key_from_public_key(recipient_private_key, &peer_public_key);
        let cipher = aes256_gcm_from_key(&key)?;
        #[allow(deprecated)]
        let plaintext = Zeroizing::new(
            cipher
                .decrypt(
                    AesNonce::from_slice(&packed.nonce),
                    packed.ciphertext_and_tag.as_slice(),
                )
                .map_err(|_| E2eeCodecError::AuthenticationFailed)?,
        );

        String::from_utf8(plaintext.to_vec()).map_err(|_| E2eeCodecError::InvalidPlaintextUtf8)
    }

    /// Extracts and decrypts `choices[0].delta.content`-style encrypted response
    /// content. Missing content fails closed when configured to require encrypted
    /// response content, and otherwise returns `Ok(None)`.
    pub fn decrypt_response_content(
        &self,
        content: Option<&str>,
        recipient_private_key: &SecretKey,
    ) -> Result<Option<String>, E2eeCodecError> {
        let Some(content) = content else {
            return if self.require_encrypted_response_content {
                Err(E2eeCodecError::MissingEncryptedContent)
            } else {
                Ok(None)
            };
        };

        let payload = EncryptedPayload::from_hex(content)?;
        self.decrypt_content(&payload, recipient_private_key)
            .map(Some)
    }

    /// Converts a JSON string field containing a Venice E2EE payload into the
    /// typed payload helper. Non-string shapes are rejected as unsupported codec
    /// shapes so route code can fail closed before attempting decryption.
    pub fn encrypted_payload_from_json_value(
        &self,
        value: &serde_json::Value,
    ) -> Result<EncryptedPayload, E2eeCodecError> {
        match value {
            serde_json::Value::String(value) => EncryptedPayload::from_hex(value),
            serde_json::Value::Object(object) if object.contains_key("version") => {
                Err(E2eeCodecError::UnsupportedCodecShape {
                    message: "versioned encrypted payload objects are not supported by this codec"
                        .to_owned(),
                })
            }
            other => Err(E2eeCodecError::UnsupportedCodecShape {
                message: format!(
                    "expected encrypted payload string, got {}",
                    json_kind(other)
                ),
            }),
        }
    }

    /// Serializes an encrypted payload into the JSON string shape used by Venice
    /// request/response content fields.
    pub fn encrypted_payload_to_json_value(&self, payload: &EncryptedPayload) -> serde_json::Value {
        serde_json::Value::String(payload.as_hex().to_owned())
    }

    fn encrypt_content_with_parts(
        &self,
        plaintext: &str,
        peer_public_key: &PublicKey,
        ephemeral_private_key: SecretKey,
        nonce: Nonce,
    ) -> Result<EncryptedPayload, E2eeCodecError> {
        let ephemeral_public_key = ephemeral_private_key.public_key();
        let ephemeral_public_key = ephemeral_public_key.to_encoded_point(false);
        let ephemeral_public_key_bytes = ephemeral_public_key.as_bytes();
        debug_assert_eq!(ephemeral_public_key_bytes.len(), EPHEMERAL_PUBLIC_KEY_LEN);

        let key = self.derive_content_key_from_public_key(&ephemeral_private_key, peer_public_key);
        let cipher = aes256_gcm_from_key(&key)?;
        #[allow(deprecated)]
        let ciphertext_and_tag = cipher
            .encrypt(AesNonce::from_slice(nonce.as_bytes()), plaintext.as_bytes())
            .map_err(|_| E2eeCodecError::EncryptionFailed)?;

        let mut packed = Vec::with_capacity(PACKED_PREFIX_LEN + ciphertext_and_tag.len());
        packed.extend_from_slice(ephemeral_public_key_bytes);
        packed.extend_from_slice(nonce.as_bytes());
        packed.extend_from_slice(&ciphertext_and_tag);

        Ok(EncryptedPayload::from_packed_bytes_unchecked(&packed))
    }

    fn derive_content_key_from_public_key(
        &self,
        local_private_key: &SecretKey,
        peer_public_key: &PublicKey,
    ) -> ContentEncryptionKey {
        let shared_secret = diffie_hellman(
            local_private_key.to_nonzero_scalar(),
            peer_public_key.as_affine(),
        );
        derive_aes_key(&shared_secret, &self.hkdf_info)
    }
}

impl Default for E2eeCodec {
    fn default() -> Self {
        Self::from_config(&E2eeConfig::default()).expect("default E2EE config is valid")
    }
}

/// An AES-256-GCM content-encryption key derived from ECDH + HKDF-SHA256.
///
/// Debug output is redacted and the key bytes are zeroized on drop.
pub struct ContentEncryptionKey(Zeroizing<[u8; AES_256_KEY_LEN]>);

impl ContentEncryptionKey {
    fn new(bytes: Zeroizing<[u8; AES_256_KEY_LEN]>) -> Self {
        Self(bytes)
    }

    fn as_slice(&self) -> &[u8] {
        &self.0[..]
    }
}

impl ZeroizeOnDrop for ContentEncryptionKey {}

impl fmt::Debug for ContentEncryptionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ContentEncryptionKey([redacted])")
    }
}

impl PartialEq for ContentEncryptionKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for ContentEncryptionKey {}

fn aes256_gcm_from_key(key: &ContentEncryptionKey) -> Result<Aes256Gcm, E2eeCodecError> {
    // Zeroization note for the resolved RustCrypto stack used here:
    // - `aes-gcm` with `zeroize` wipes its temporary GHASH key during init.
    // - the direct `aes` dependency enables `aes/zeroize`, so AES-256 retained
    //   round keys implement safe zeroizing drop behavior.
    // - the direct `ghash` dependency enables zeroizing of GHASH temporary
    //   conversion buffers.
    // - the direct `polyval` dependency enables zeroizing drops for POLYVAL
    //   backends on targets/configurations where those drops are reachable.
    //
    // Residual concern: in the resolved `polyval` 0.6.x x86/x86_64 autodetect
    // path, the public wrapper uses `ManuallyDrop` and does not expose a safe
    // `Drop`/`ZeroizeOnDrop` implementation. That means `Aes256Gcm` still cannot
    // be proven to zeroize every retained GHASH/POLYVAL byte on this target. We
    // avoid brittle unsafe whole-object wiping and instead keep the cipher scoped
    // to one encrypt/decrypt operation while zeroizing the derived key material
    // and relying on verified safe drop behavior where the crates expose it.
    Aes256Gcm::new_from_slice(key.as_slice()).map_err(|_| {
        E2eeCodecError::MalformedEncryptedPayload {
            message: "derived AES-256-GCM key has invalid length".to_owned(),
        }
    })
}

/// AES-GCM nonce used by the Venice field codec.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Nonce([u8; NONCE_LEN]);

impl Nonce {
    pub fn generate() -> Self {
        let mut bytes = [0_u8; NONCE_LEN];
        OsRng.fill_bytes(&mut bytes);
        Self(bytes)
    }

    pub fn from_bytes(bytes: [u8; NONCE_LEN]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; NONCE_LEN] {
        &self.0
    }
}

impl fmt::Debug for Nonce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Nonce")
            .field(&encode_lower_hex(self.as_bytes()))
            .finish()
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EncryptedPayload(String);

impl EncryptedPayload {
    pub fn from_hex(value: impl Into<String>) -> Result<Self, E2eeCodecError> {
        let value = value.into();
        validate_packed_payload_hex(&value)?;
        Ok(Self(value.to_ascii_lowercase()))
    }

    pub fn as_hex(&self) -> &str {
        &self.0
    }

    pub fn into_hex(self) -> String {
        self.0
    }

    fn from_packed_bytes_unchecked(bytes: &[u8]) -> Self {
        Self(encode_lower_hex(bytes))
    }
}

impl fmt::Debug for EncryptedPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncryptedPayload")
            .field("hex_len", &self.0.len())
            .finish()
    }
}

struct PackedEncryptedPayload {
    ephemeral_public_key: [u8; EPHEMERAL_PUBLIC_KEY_LEN],
    nonce: [u8; NONCE_LEN],
    ciphertext_and_tag: Vec<u8>,
}

impl PackedEncryptedPayload {
    fn unpack(payload: &EncryptedPayload) -> Result<Self, E2eeCodecError> {
        let bytes = decode_hex(payload.as_hex())?;
        if bytes.len() < MIN_PACKED_PAYLOAD_LEN {
            return Err(E2eeCodecError::MalformedEncryptedPayload {
                message: format!(
                    "packed encrypted payload is too short: got {} bytes, need at least {MIN_PACKED_PAYLOAD_LEN}",
                    bytes.len()
                ),
            });
        }

        let mut ephemeral_public_key = [0_u8; EPHEMERAL_PUBLIC_KEY_LEN];
        ephemeral_public_key.copy_from_slice(&bytes[..EPHEMERAL_PUBLIC_KEY_LEN]);
        if ephemeral_public_key[0] != 0x04 {
            return Err(E2eeCodecError::MalformedEncryptedPayload {
                message: "ephemeral public key must be uncompressed SEC1 format".to_owned(),
            });
        }
        PublicKey::from_sec1_bytes(&ephemeral_public_key).map_err(|_| {
            E2eeCodecError::MalformedEncryptedPayload {
                message: "ephemeral public key is not a valid secp256k1 key".to_owned(),
            }
        })?;

        let mut nonce = [0_u8; NONCE_LEN];
        nonce.copy_from_slice(&bytes[EPHEMERAL_PUBLIC_KEY_LEN..PACKED_PREFIX_LEN]);

        Ok(Self {
            ephemeral_public_key,
            nonce,
            ciphertext_and_tag: bytes[PACKED_PREFIX_LEN..].to_vec(),
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum E2eeCodecError {
    #[error("configured E2EE HKDF info must not be empty")]
    EmptyHkdfInfo,
    #[error("encrypted response content is required but missing")]
    MissingEncryptedContent,
    #[error("encrypted payload is malformed: {message}")]
    MalformedEncryptedPayload { message: String },
    #[error("encrypted payload authentication failed")]
    AuthenticationFailed,
    #[error("unsupported E2EE encrypted payload shape: {message}")]
    UnsupportedCodecShape { message: String },
    #[error("invalid E2EE public key: {message}")]
    InvalidPublicKey { message: String },
    #[error("decrypted E2EE payload is not valid UTF-8")]
    InvalidPlaintextUtf8,
    #[error("E2EE encryption failed")]
    EncryptionFailed,
}

fn derive_aes_key(shared_secret: &SharedSecret, hkdf_info: &[u8]) -> ContentEncryptionKey {
    let hkdf = Hkdf::<Sha256>::new(None, shared_secret.raw_secret_bytes());
    let mut output_key = Zeroizing::new([0_u8; AES_256_KEY_LEN]);
    hkdf.expand(hkdf_info, output_key.as_mut_slice())
        .expect("32-byte HKDF-SHA256 output length is always valid");
    ContentEncryptionKey::new(output_key)
}

fn decode_uncompressed_public_key_hex(value: &str) -> Result<PublicKey, E2eeCodecError> {
    let bytes = decode_hex(value).map_err(|error| match error {
        E2eeCodecError::MalformedEncryptedPayload { message } => {
            E2eeCodecError::InvalidPublicKey { message }
        }
        other => other,
    })?;

    if bytes.len() != EPHEMERAL_PUBLIC_KEY_LEN {
        return Err(E2eeCodecError::InvalidPublicKey {
            message: format!(
                "expected {EPHEMERAL_PUBLIC_KEY_LEN} uncompressed SEC1 bytes, got {}",
                bytes.len()
            ),
        });
    }
    if bytes.first() != Some(&0x04) {
        return Err(E2eeCodecError::InvalidPublicKey {
            message: "public key must be uncompressed SEC1 format".to_owned(),
        });
    }

    PublicKey::from_sec1_bytes(&bytes).map_err(|_| E2eeCodecError::InvalidPublicKey {
        message: "public key is not a valid secp256k1 key".to_owned(),
    })
}

fn validate_packed_payload_hex(value: &str) -> Result<(), E2eeCodecError> {
    if value.is_empty() {
        return Err(E2eeCodecError::MalformedEncryptedPayload {
            message: "encrypted payload hex string is empty".to_owned(),
        });
    }
    if !value.len().is_multiple_of(2) {
        return Err(E2eeCodecError::MalformedEncryptedPayload {
            message: "encrypted payload hex string has odd length".to_owned(),
        });
    }
    if value.len() < MIN_PACKED_PAYLOAD_LEN * 2 {
        return Err(E2eeCodecError::MalformedEncryptedPayload {
            message: format!(
                "encrypted payload hex string is too short: got {} chars, need at least {}",
                value.len(),
                MIN_PACKED_PAYLOAD_LEN * 2
            ),
        });
    }
    if let Some((index, ch)) = value.char_indices().find(|(_, ch)| !ch.is_ascii_hexdigit()) {
        return Err(E2eeCodecError::MalformedEncryptedPayload {
            message: format!(
                "encrypted payload hex string contains non-hex character {ch:?} at index {index}"
            ),
        });
    }
    Ok(())
}

fn decode_hex(value: &str) -> Result<Vec<u8>, E2eeCodecError> {
    if !value.len().is_multiple_of(2) {
        return Err(E2eeCodecError::MalformedEncryptedPayload {
            message: "hex string has odd length".to_owned(),
        });
    }

    let mut out = Vec::with_capacity(value.len() / 2);
    let bytes = value.as_bytes();
    for (pair_index, pair) in bytes.chunks_exact(2).enumerate() {
        let high = hex_value(pair[0]).ok_or_else(|| E2eeCodecError::MalformedEncryptedPayload {
            message: format!(
                "hex string contains non-hex character {:?} at index {}",
                pair[0] as char,
                pair_index * 2
            ),
        })?;
        let low = hex_value(pair[1]).ok_or_else(|| E2eeCodecError::MalformedEncryptedPayload {
            message: format!(
                "hex string contains non-hex character {:?} at index {}",
                pair[1] as char,
                pair_index * 2 + 1
            ),
        })?;
        out.push((high << 4) | low);
    }
    Ok(out)
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
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

fn json_kind(value: &serde_json::Value) -> &'static str {
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

    const FIXED_NONCE: [u8; NONCE_LEN] = [
        0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xab,
    ];
    const FIXED_RECIPIENT_PRIVATE_KEY_HEX: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";
    const FIXED_EPHEMERAL_PRIVATE_KEY_HEX: &str =
        "2222222222222222222222222222222222222222222222222222222222222222";
    const DETERMINISTIC_PLAINTEXT: &str = "deterministic Venice E2EE fixture";
    const DETERMINISTIC_CIPHERTEXT_HEX: &str = "04466d7fcae563e5cb09a0d1870bb580344804617879a14949cf22285f1bae3f276728176c3c6431f8eeda4538dc37c865e2784f3a9e77d044f33e407797e1278aa0a1a2a3a4a5a6a7a8a9aaab3b364a6560dc6246955e1379bac6c7a0f453c5b2d9be6eabb00cad9955278b4c401f6793813d7f98ba8f163a5c51b87686";

    fn secret_key_from_hex(value: &str) -> SecretKey {
        let bytes = decode_hex(value).expect("test key hex should decode");
        SecretKey::from_slice(&bytes).expect("test key should be valid")
    }

    fn public_key_hex(secret_key: &SecretKey) -> String {
        encode_lower_hex(secret_key.public_key().to_encoded_point(false).as_bytes())
    }

    #[test]
    fn encrypt_decrypt_round_trip() {
        let codec = E2eeCodec::default();
        let recipient_private_key = SecretKey::random(&mut OsRng);
        let recipient_public_key_hex = public_key_hex(&recipient_private_key);

        let encrypted = codec
            .encrypt_content("hello from local proxy", &recipient_public_key_hex)
            .expect("encryption should succeed");
        let decrypted = codec
            .decrypt_content(&encrypted, &recipient_private_key)
            .expect("decryption should succeed");

        assert_eq!(decrypted, "hello from local proxy");
        assert!(
            encrypted
                .as_hex()
                .chars()
                .all(|ch| !ch.is_ascii_uppercase())
        );
    }

    #[test]
    fn decryption_with_wrong_key_fails_authentication() {
        let codec = E2eeCodec::default();
        let recipient_private_key = SecretKey::random(&mut OsRng);
        let wrong_private_key = SecretKey::random(&mut OsRng);
        let recipient_public_key_hex = public_key_hex(&recipient_private_key);
        let encrypted = codec
            .encrypt_content("secret", &recipient_public_key_hex)
            .expect("encryption should succeed");

        let err = codec
            .decrypt_content(&encrypted, &wrong_private_key)
            .expect_err("wrong key must fail closed");

        assert_eq!(err, E2eeCodecError::AuthenticationFailed);
    }

    #[test]
    fn tampered_ciphertext_fails_authentication() {
        let codec = E2eeCodec::default();
        let recipient_private_key = SecretKey::random(&mut OsRng);
        let recipient_public_key_hex = public_key_hex(&recipient_private_key);
        let encrypted = codec
            .encrypt_content("secret", &recipient_public_key_hex)
            .expect("encryption should succeed");
        let mut packed = decode_hex(encrypted.as_hex()).expect("ciphertext should decode");
        let last = packed.last_mut().expect("ciphertext has tag byte");
        *last ^= 0x01;
        let tampered = EncryptedPayload::from_packed_bytes_unchecked(&packed);

        let err = codec
            .decrypt_content(&tampered, &recipient_private_key)
            .expect_err("tampered ciphertext must fail closed");

        assert_eq!(err, E2eeCodecError::AuthenticationFailed);
    }

    #[test]
    fn malformed_payload_fails_closed() {
        let codec = E2eeCodec::default();
        let recipient_private_key = SecretKey::random(&mut OsRng);

        let err = codec
            .decrypt_response_content(Some("not encrypted"), &recipient_private_key)
            .expect_err("non-hex payload should fail closed");
        assert!(matches!(
            err,
            E2eeCodecError::MalformedEncryptedPayload { .. }
        ));

        let too_short = "04".repeat(EPHEMERAL_PUBLIC_KEY_LEN + NONCE_LEN);
        let err =
            EncryptedPayload::from_hex(too_short).expect_err("short payload should be rejected");
        assert!(matches!(
            err,
            E2eeCodecError::MalformedEncryptedPayload { .. }
        ));
    }

    #[test]
    fn missing_encrypted_response_content_respects_config() {
        let recipient_private_key = SecretKey::random(&mut OsRng);

        let required = E2eeCodec::new("ecdsa_encryption", true).expect("config should be valid");
        let err = required
            .decrypt_response_content(None, &recipient_private_key)
            .expect_err("missing required encrypted content should fail");
        assert_eq!(err, E2eeCodecError::MissingEncryptedContent);

        let optional = E2eeCodec::new("ecdsa_encryption", false).expect("config should be valid");
        let decrypted = optional
            .decrypt_response_content(None, &recipient_private_key)
            .expect("missing optional content should be allowed");
        assert_eq!(decrypted, None);
    }

    #[test]
    fn deterministic_test_vector_with_fixed_nonce_and_ephemeral_key() {
        let codec = E2eeCodec::default();
        let recipient_private_key = secret_key_from_hex(FIXED_RECIPIENT_PRIVATE_KEY_HEX);
        let recipient_public_key = recipient_private_key.public_key();
        let ephemeral_private_key = secret_key_from_hex(FIXED_EPHEMERAL_PRIVATE_KEY_HEX);

        let encrypted = codec
            .encrypt_content_with_parts(
                DETERMINISTIC_PLAINTEXT,
                &recipient_public_key,
                ephemeral_private_key,
                Nonce::from_bytes(FIXED_NONCE),
            )
            .expect("deterministic encryption should succeed");

        assert_eq!(encrypted.as_hex(), DETERMINISTIC_CIPHERTEXT_HEX);
        let decrypted = codec
            .decrypt_content(&encrypted, &recipient_private_key)
            .expect("deterministic fixture should decrypt");
        assert_eq!(decrypted, DETERMINISTIC_PLAINTEXT);
    }

    #[test]
    fn derived_keys_match_from_both_sides_and_debug_is_redacted() {
        fn assert_zeroize_on_drop<T: ZeroizeOnDrop>() {}

        assert_zeroize_on_drop::<SecretKey>();
        assert_zeroize_on_drop::<SharedSecret>();
        assert_zeroize_on_drop::<aes::Aes256>();
        assert_zeroize_on_drop::<ContentEncryptionKey>();

        let codec = E2eeCodec::default();
        let local_private_key = SecretKey::random(&mut OsRng);
        let peer_private_key = SecretKey::random(&mut OsRng);
        let local_public_key_hex = public_key_hex(&local_private_key);
        let peer_public_key_hex = public_key_hex(&peer_private_key);

        let local_key = codec
            .derive_content_key(&local_private_key, &peer_public_key_hex)
            .expect("local derivation should succeed");
        let peer_key = codec
            .derive_content_key(&peer_private_key, &local_public_key_hex)
            .expect("peer derivation should succeed");

        assert_eq!(local_key, peer_key);
        assert_eq!(format!("{local_key:?}"), "ContentEncryptionKey([redacted])");
    }

    #[test]
    fn json_payload_helpers_match_string_shape_and_reject_unsupported_shapes() {
        let codec = E2eeCodec::default();
        let recipient_private_key = SecretKey::random(&mut OsRng);
        let recipient_public_key_hex = public_key_hex(&recipient_private_key);
        let normalized = json!([
            {"role":"system","content":"You are private."},
            {"role":"user","content":"Hello"}
        ]);

        let encrypted = codec
            .encrypt_json_payload(&normalized, &recipient_public_key_hex)
            .expect("JSON payload should encrypt");
        let value = codec.encrypted_payload_to_json_value(&encrypted);
        assert!(value.is_string());
        let parsed = codec
            .encrypted_payload_from_json_value(&value)
            .expect("string payload shape should parse");
        assert_eq!(parsed, encrypted);
        let decrypted = codec
            .decrypt_content(&parsed, &recipient_private_key)
            .expect("payload should decrypt");
        assert_eq!(decrypted, serde_json::to_string(&normalized).unwrap());

        let unsupported = codec
            .encrypted_payload_from_json_value(
                &json!({"version": 1, "ciphertext": encrypted.as_hex()}),
            )
            .expect_err("versioned object shape should be rejected");
        assert!(matches!(
            unsupported,
            E2eeCodecError::UnsupportedCodecShape { .. }
        ));
    }
}
