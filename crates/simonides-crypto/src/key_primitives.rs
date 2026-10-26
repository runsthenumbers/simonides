//! Key-custody primitives: Argon2id KDF, ChaCha20-Poly1305 AEAD wrap, HKDF-SHA256.
//!
//! These are pure crypto functions — no I/O, no key persistence. They implement
//! the two-key custody model:
//!   MK = argon2id_derive(passphrase, salt, params)
//!   wrap_pass = aead_wrap(MK, DK, aad)
//!   DK → HKDF → per-purpose subkeys (sqlcipher, vault-AEAD, credentials …)
//!
//! All key material is wrapped in `Zeroizing` and zeroed on drop.

use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    ChaCha20Poly1305, Key, Nonce,
};
use hkdf::Hkdf;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use zeroize::Zeroizing;

use simonides_core::{SimonidesError, SimonidesResult};

/// Argon2id tuning parameters.
///
/// Stored alongside the wrap ciphertext so each
/// enrollment can use independent params — a stronger device can use higher
/// params without breaking existing vaults. Verification reads params from
/// the stored record rather than a global constant.
///
/// Use `PROVISIONAL_DEFAULT` as a conservative starting point; see its own
/// documentation for what "provisional" means here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Argon2Params {
    /// Memory cost in kibibytes (e.g. 19456 = 19 MiB).
    pub m_cost: u32,
    /// Number of iterations.
    pub t_cost: u32,
    /// Degree of parallelism.
    pub p_cost: u32,
}

impl Argon2Params {
    /// OWASP's recommended minimum for Argon2id (19 MiB, 2 passes, 1 lane).
    ///
    /// **Provisional.** These values have not been tuned per device. Stronger
    /// hardware can and should use higher costs; because params are stored with
    /// each enrollment, raising them later does not break existing vaults.
    pub const PROVISIONAL_DEFAULT: Self = Self {
        m_cost: 19456, // 19 MiB
        t_cost: 2,
        p_cost: 1,
    };
}

/// Derive a 32-byte master key from a passphrase and salt using Argon2id.
///
/// Deterministic for fixed (passphrase, salt, params). The salt must be
/// generated fresh per enrollment (16–32 bytes of secure random). Params
/// should be stored alongside the wrap ciphertext.
pub fn argon2id_derive(
    passphrase: &[u8],
    salt: &[u8],
    params: &Argon2Params,
) -> SimonidesResult<Zeroizing<[u8; 32]>> {
    let a2_params =
        argon2::Params::new(params.m_cost, params.t_cost, params.p_cost, Some(32))
            .map_err(|e| SimonidesError::Crypto(format!("argon2 params: {e}")))?;
    let argon2 =
        argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, a2_params);

    let mut key = Zeroizing::new([0u8; 32]);
    argon2
        .hash_password_into(passphrase, salt, key.as_mut())
        .map_err(|e| SimonidesError::Crypto(format!("argon2id derive: {e}")))?;
    Ok(key)
}

/// Wrap a 32-byte data key with ChaCha20-Poly1305.
///
/// On-wire format: `[nonce(12) || ciphertext_with_tag]` — same layout as the
/// simonides vault AEAD. A fresh random 12-byte nonce is generated per call.
///
/// `aad` binds the ciphertext to additional context (e.g. `b"wrap_pass"` or
/// `b"wrap_recovery"`). Pass `b""` when no binding is needed. A wrap produced
/// with one `aad` value is rejected by `aead_unwrap` under any other value.
pub fn aead_wrap(key: &[u8; 32], plaintext: &[u8; 32], aad: &[u8]) -> SimonidesResult<Vec<u8>> {
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key));

    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext_with_tag = cipher
        .encrypt(nonce, Payload { msg: plaintext.as_slice(), aad })
        .map_err(|e| SimonidesError::Crypto(format!("aead wrap: {e}")))?;

    let mut out = Vec::with_capacity(12 + ciphertext_with_tag.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext_with_tag);
    Ok(out)
}

/// Unwrap a ciphertext produced by `aead_wrap`.
///
/// Returns the original 32-byte data key on success. Wrong key, tampered
/// ciphertext, or wrong `aad` → `Err` via AEAD tag rejection. Never returns
/// garbage plaintext.
pub fn aead_unwrap(key: &[u8; 32], wrapped: &[u8], aad: &[u8]) -> SimonidesResult<Zeroizing<[u8; 32]>> {
    if wrapped.len() < 12 {
        return Err(SimonidesError::Crypto(format!(
            "aead unwrap: blob too short ({} bytes, need ≥12 for nonce)",
            wrapped.len()
        )));
    }

    let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
    let nonce = Nonce::from_slice(&wrapped[..12]);
    let ciphertext_with_tag = &wrapped[12..];

    let plaintext = cipher
        .decrypt(nonce, Payload { msg: ciphertext_with_tag, aad })
        .map_err(|_| SimonidesError::Crypto("aead unwrap: AEAD tag failure".into()))?;

    if plaintext.len() != 32 {
        return Err(SimonidesError::Crypto(format!(
            "aead unwrap: decrypted payload is {} bytes, expected 32",
            plaintext.len()
        )));
    }

    let mut out = Zeroizing::new([0u8; 32]);
    out.copy_from_slice(&plaintext);
    Ok(out)
}

/// Derive a 32-byte subkey from input keying material using HKDF-SHA256.
///
/// `domain_sep` must be unique per derived-key purpose so keys from the same
/// IKM are cryptographically independent (e.g. `b"simonides-sqlcipher-v1"`,
/// `b"simonides-vault-aead-v1"`, `b"simonides-credentials-v1"`).
///
/// Domain-separation only matters when multiple subkeys share the same IKM.
/// Per-message or per-enrollment ephemerals with independent IKM are already
/// independent without it (each has its own IKM).
pub fn hkdf_derive(ikm: &[u8; 32], domain_sep: &[u8]) -> SimonidesResult<Zeroizing<[u8; 32]>> {
    let hk = Hkdf::<Sha256>::new(None, ikm.as_slice());
    let mut out = Zeroizing::new([0u8; 32]);
    hk.expand(domain_sep, out.as_mut())
        .map_err(|_| SimonidesError::Internal("hkdf_derive: HKDF expand failed".into()))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrap_key() -> [u8; 32] {
        [0x42u8; 32]
    }

    fn data_key() -> [u8; 32] {
        [0x99u8; 32]
    }

    /// Gate: aead_wrap then aead_unwrap returns the original DK.
    #[test]
    fn aead_wrap_unwrap_roundtrip() {
        let key = wrap_key();
        let dk = data_key();
        let wrapped = aead_wrap(&key, &dk, b"wrap_pass").expect("wrap");
        let recovered = aead_unwrap(&key, &wrapped, b"wrap_pass").expect("unwrap");
        assert_eq!(*recovered, dk);
    }

    /// Gate: wrong key → Err via AEAD tag rejection, no plaintext returned.
    #[test]
    fn aead_unwrap_wrong_key_fails() {
        let key = wrap_key();
        let wrong = [0xFFu8; 32];
        let dk = data_key();
        let wrapped = aead_wrap(&key, &dk, b"wrap_pass").expect("wrap");
        let result = aead_unwrap(&wrong, &wrapped, b"wrap_pass");
        assert!(result.is_err(), "wrong key must return Err, got Ok");
    }

    /// Gate: same IKM, different domain separators → different derived keys.
    #[test]
    fn hkdf_domain_sep_produces_distinct_keys() {
        let ikm = wrap_key();
        let k1 = hkdf_derive(&ikm, b"simonides-sqlcipher-v1").expect("derive sqlcipher");
        let k2 = hkdf_derive(&ikm, b"simonides-vault-aead-v1").expect("derive vault");
        assert_ne!(*k1, *k2, "different domain separators must produce different keys");
    }

    /// Gate: same (passphrase, salt, params) → same MK; different salt → different MK.
    #[test]
    fn argon2id_deterministic() {
        // Low params so the test runs fast on CI; correctness, not strength.
        let params = Argon2Params { m_cost: 64, t_cost: 1, p_cost: 1 };
        let passphrase = b"test-passphrase";
        let salt_a = b"salt-aaaaaaaaaaaa"; // 17 bytes — argon2 min salt is 8
        let salt_b = b"salt-bbbbbbbbbbbb";

        let mk1 = argon2id_derive(passphrase, salt_a, &params).expect("derive 1");
        let mk2 = argon2id_derive(passphrase, salt_a, &params).expect("derive 2");
        assert_eq!(*mk1, *mk2, "same (passphrase, salt, params) must yield same MK");

        let mk3 = argon2id_derive(passphrase, salt_b, &params).expect("derive 3");
        assert_ne!(*mk1, *mk3, "different salt must yield different MK");
    }

    /// Gate: Argon2Params serializes/deserializes correctly (stored-with-wrap path, K3).
    #[test]
    fn argon2_params_roundtrip() {
        let params = Argon2Params { m_cost: 19456, t_cost: 2, p_cost: 1 };
        let json = serde_json::to_string(&params).expect("serialize");
        let back: Argon2Params = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(params, back);
    }

    /// Additional: wrong AAD also causes AEAD rejection.
    #[test]
    fn aead_unwrap_wrong_aad_fails() {
        let key = wrap_key();
        let dk = data_key();
        let wrapped = aead_wrap(&key, &dk, b"wrap_pass").expect("wrap");
        let result = aead_unwrap(&key, &wrapped, b"wrap_recovery");
        assert!(result.is_err(), "wrong AAD must return Err");
    }

    /// Additional: each wrap call uses a fresh nonce (outputs differ even for same inputs).
    #[test]
    fn aead_wrap_nonce_is_fresh_each_call() {
        let key = wrap_key();
        let dk = data_key();
        let w1 = aead_wrap(&key, &dk, b"").expect("wrap 1");
        let w2 = aead_wrap(&key, &dk, b"").expect("wrap 2");
        assert_ne!(w1, w2, "two wraps of the same DK must differ (fresh nonce per call)");
    }

    /// Additional: PROVISIONAL_DEFAULT has expected OWASP values.
    #[test]
    fn provisional_default_values() {
        let p = Argon2Params::PROVISIONAL_DEFAULT;
        assert_eq!(p.m_cost, 19456); // 19 MiB
        assert_eq!(p.t_cost, 2);
        assert_eq!(p.p_cost, 1);
    }
}
