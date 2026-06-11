//! Startup proxy-instance key management.
//!
//! The proxy generates one secp256k1 keypair per process when configured to do
//! so. E2EE code uses the private key to decrypt Venice response chunks and
//! sends the uncompressed public key hex in Venice E2EE request headers.

use std::{fmt, sync::Arc};

use k256::{SecretKey, elliptic_curve::sec1::ToEncodedPoint};
use rand_core::OsRng;
use zeroize::ZeroizeOnDrop;

use crate::config::KeysConfig;

/// Per-process secp256k1 keypair used by this proxy instance.
#[derive(Clone)]
pub struct ProxyInstanceKey {
    inner: Arc<ProxyInstanceKeyInner>,
}

struct ProxyInstanceKeyInner {
    // Stored for E2EE response decryption while the public key is sent upstream.
    private_key: ProxyInstancePrivateKey,
    public_key_hex: String,
}

struct ProxyInstancePrivateKey(SecretKey);

impl ProxyInstancePrivateKey {
    fn new(private_key: SecretKey) -> Self {
        Self(private_key)
    }

    fn public_key_hex(&self) -> String {
        let public_key = self.0.public_key();
        hex::encode(public_key.to_encoded_point(false).as_bytes())
    }

    fn secret_key(&self) -> &SecretKey {
        &self.0
    }
}

impl ZeroizeOnDrop for ProxyInstancePrivateKey {}

impl fmt::Debug for ProxyInstancePrivateKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[redacted]")
    }
}

impl ProxyInstanceKey {
    /// Generates a new secp256k1 keypair for this process.
    pub fn generate() -> Self {
        let private_key = SecretKey::random(&mut OsRng);
        Self::from_private_key(private_key)
    }

    /// Applies `keys.generate_proxy_instance_key_on_startup`.
    pub fn generate_from_config(config: &KeysConfig) -> Option<Self> {
        config
            .generate_proxy_instance_key_on_startup
            .then(Self::generate)
    }

    fn from_private_key(private_key: SecretKey) -> Self {
        let private_key = ProxyInstancePrivateKey::new(private_key);
        let public_key_hex = private_key.public_key_hex();

        Self {
            inner: Arc::new(ProxyInstanceKeyInner {
                private_key,
                public_key_hex,
            }),
        }
    }

    /// Uncompressed SEC1 public key encoded as lowercase hex.
    ///
    /// Venice expects 65 bytes (`04 || x || y`), represented as 130 hex chars.
    pub fn public_key_hex(&self) -> &str {
        &self.inner.public_key_hex
    }

    #[allow(dead_code)]
    pub(crate) fn private_key(&self) -> &SecretKey {
        self.inner.private_key.secret_key()
    }
}

impl fmt::Debug for ProxyInstanceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProxyInstanceKey")
            .field("private_key", &"[redacted]")
            .field("public_key_hex", &self.public_key_hex())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_uncompressed_public_key_hex_in_venice_format() {
        let key = ProxyInstanceKey::generate();

        assert_eq!(key.public_key_hex().len(), 130);
        assert!(key.public_key_hex().starts_with("04"));
        assert!(key.public_key_hex().chars().all(|c| c.is_ascii_hexdigit()));
        assert!(
            key.public_key_hex()
                .chars()
                .all(|c| !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn respects_startup_key_generation_config() {
        let enabled = KeysConfig {
            generate_proxy_instance_key_on_startup: true,
        };
        let disabled = KeysConfig {
            generate_proxy_instance_key_on_startup: false,
        };

        assert!(ProxyInstanceKey::generate_from_config(&enabled).is_some());
        assert!(ProxyInstanceKey::generate_from_config(&disabled).is_none());
    }

    #[test]
    fn private_key_material_is_zeroized_on_drop() {
        fn assert_zeroize_on_drop<T: ZeroizeOnDrop>() {}

        assert_zeroize_on_drop::<SecretKey>();
        assert_zeroize_on_drop::<ProxyInstancePrivateKey>();

        let key = ProxyInstanceKey::generate();
        let _private_key_ref: &SecretKey = key.private_key();
    }

    #[test]
    fn debug_output_redacts_private_key_material() {
        let key = ProxyInstanceKey::generate();
        let debug = format!("{key:?}");

        assert!(debug.contains("[redacted]"));
        assert!(debug.contains(key.public_key_hex()));
    }
}
