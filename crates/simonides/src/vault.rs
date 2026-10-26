//! Content-addressable byte vault — slice 1b of the palace design.
//!
//! The palace owns the vault (design §1): one system, one API surface. This
//! module holds vault types and free functions; `PalaceStore` in lib.rs owns
//! the single SQLite connection and delegates vault operations here.
//!
//! Layout: `<vault_root>/<aa>/<bb>/<full-hex-sha256>`
//! The path is a pure function of the hash — it is **never stored** in the
//! database. This is the core anti-dangle move at the file layer: the DB
//! never holds a mutable path that can rot.
//!
//! Bytes-before-row write ordering: blobs land on disk and are fsync'd before
//! the `files` row commits. A committed row provably has bytes. A crash between
//! steps leaves at worst an unreferenced blob — harmless, content-addressed,
//! dedup-safe (design §2).
//!
//! Out of scope (deferred): refcount, GC reclaim, drawer↔file linkage,
//! CRDT sync, chunking, encryption-at-rest toggle, dispatcher wiring.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chacha20poly1305::{
    aead::{Aead, Payload},
    ChaCha20Poly1305, Nonce,
};
use rand::RngCore;
use rusqlite::{params, Connection};
use sha2::{Digest, Sha256};

/// Schema DDL for the `files` table. Embedded here for locality; applied by
/// `PalaceStore::init` so it lives in the single unified connection.
pub(crate) const FILES_SCHEMA: &str = r#"
-- Content-addressable file registry.
-- No `path` column: path is derived from content_hash on demand.
-- Layout: <vault_root>/<hash[0..2]>/<hash[2..4]>/<full-hex-sha256>
CREATE TABLE IF NOT EXISTS files (
    content_hash TEXT PRIMARY KEY,
    size         INTEGER NOT NULL,
    mime         TEXT NOT NULL,
    filename     TEXT,
    provenance   TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
"#;

/// Errors from vault operations.
#[derive(Debug, thiserror::Error)]
pub enum VaultError {
    #[error("storage: {0}")]
    Storage(String),

    #[error("blob not found: {hash}")]
    BlobNotFound { hash: String },

    /// AEAD tag failure or malformed ciphertext blob. Always returned as `Err`
    /// (never silently produces garbage plaintext).
    #[error("decryption failed for {hash}: {reason}")]
    Decryption { hash: String, reason: String },

    /// Returned when the post-write read-back detects that a just-committed row
    /// is inconsistent: missing blob, wrong blob, or absent row.
    /// This makes silent corruption structurally impossible (design §3.6).
    #[error(
        "write verification failed for {hash}: \
         row_present={row_present}, file_exists={file_exists}, hash_matches={hash_matches}"
    )]
    WriteVerificationFailed {
        hash: String,
        row_present: bool,
        file_exists: bool,
        hash_matches: bool,
    },
}

pub(crate) type Result<T> = std::result::Result<T, VaultError>;

/// Severity of a `VerifyReport` finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyIssue {
    /// The `files` row exists but no blob is present at the computed path.
    Missing,
    /// A blob is present but its re-hash does not match the stored `content_hash`.
    Corrupted,
}

/// One anomaly found during a `verify()` pass.
#[derive(Debug, Clone)]
pub struct VerifyEntry {
    pub content_hash: String,
    pub issue: VerifyIssue,
}

/// Result of a `verify()` integrity pass.
pub struct VerifyReport {
    /// Total rows inspected.
    pub total: u64,
    /// Rows whose blob is present and re-hashes correctly.
    pub ok: u64,
    /// All anomalies found (empty on a clean vault).
    pub issues: Vec<VerifyEntry>,
}

// ── Free functions operating on the shared PalaceStore connection ─────────────
// Each function takes `conn: &Mutex<Connection>` and acquires the lock only for
// the DB operation, releasing it before any file I/O.  This preserves the lock
// discipline required to avoid deadlock on re-entrant calls within a single
// `put` → `verify_write` → `exists` chain.

// ── AEAD helpers ──────────────────────────────────────────────────────────────

/// Encrypt `plaintext` using ChaCha20-Poly1305.
/// On-disk format: `[nonce(12) || ciphertext_with_tag]`.
/// `aad` (additional authenticated data) binds the ciphertext to its address
/// so a blob moved to a different hash path fails AEAD on decrypt.
fn encrypt_blob(cipher: &ChaCha20Poly1305, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext_with_tag = cipher
        .encrypt(nonce, Payload { msg: plaintext, aad })
        .map_err(|e| VaultError::Storage(format!("encrypt: {e}")))?;
    let mut out = Vec::with_capacity(12 + ciphertext_with_tag.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext_with_tag);
    Ok(out)
}

/// Decrypt a blob written by `encrypt_blob`. Returns `Decryption` error on
/// AEAD tag failure (wrong key, tampered ciphertext, wrong AAD) — never
/// silently produces garbage plaintext.
fn decrypt_blob(
    cipher: &ChaCha20Poly1305,
    encrypted: &[u8],
    aad: &[u8],
    hash: &str,
) -> Result<Vec<u8>> {
    if encrypted.len() < 12 {
        return Err(VaultError::Decryption {
            hash: hash.to_string(),
            reason: format!("blob too short for nonce ({} bytes)", encrypted.len()),
        });
    }
    let nonce = Nonce::from_slice(&encrypted[..12]);
    let ciphertext_with_tag = &encrypted[12..];
    cipher
        .decrypt(nonce, Payload { msg: ciphertext_with_tag, aad })
        .map_err(|e| VaultError::Decryption {
            hash: hash.to_string(),
            reason: format!("AEAD tag failure: {e}"),
        })
}

/// Return `true` if a `files` row exists for `content_hash`.
pub(crate) fn exists(conn: &Mutex<Connection>, content_hash: &str) -> Result<bool> {
    let guard = conn.lock().unwrap();
    let count: i64 = guard
        .query_row(
            "SELECT COUNT(*) FROM files WHERE content_hash = ?1",
            params![content_hash],
            |row| row.get(0),
        )
        .map_err(|e| VaultError::Storage(format!("exists: {e}")))?;
    Ok(count > 0)
}

/// Store bytes and return their SHA-256 hex hash.
///
/// Write protocol (atomic, bytes-before-row):
///
/// 1. Compute SHA-256 of `bytes` (plaintext). The content_hash is always
///    keyed on the plaintext; it is never the hash of ciphertext.
/// 2. Dedup: if the hash has a row AND the blob decrypts correctly to bytes
///    whose hash matches, return early — idempotent no-op, one physical blob.
///    If the row exists but the blob is missing or the AEAD tag fails, fall
///    through and re-write (self-heal) so `put` never returns `Ok` without
///    bytes confirmed on disk.
/// 3. Encrypt `bytes` with ChaCha20-Poly1305 (AAD = content_hash).
/// 4. Write encrypted blob to a temp file (same filesystem) → fsync → atomic
///    rename into the hash-derived path.
/// 5. `INSERT OR IGNORE` the `files` row in a transaction.
/// 6. Read-back verify: row present AND file at computed path AND decrypt+rehash
///    matches. Returns `WriteVerificationFailed` if any check fails.
///
/// A crash between steps 4 and 5 leaves at most an unreferenced blob —
/// harmless (content-addressed, dedup-safe).
pub(crate) fn put(
    conn: &Mutex<Connection>,
    vault_root: &Path,
    cipher: &ChaCha20Poly1305,
    bytes: &[u8],
    mime: &str,
    filename: Option<&str>,
    provenance: &str,
) -> Result<String> {
    // content_hash is always SHA-256 of plaintext (not ciphertext).
    let hex_hash = sha256_hex(bytes);

    // Fast-path dedup: decrypt + re-hash to confirm the blob is correct.
    if exists(conn, &hex_hash)? {
        let target = blob_path(vault_root, &hex_hash);
        let blob_ok = match std::fs::read(&target) {
            Ok(encrypted) => match decrypt_blob(cipher, &encrypted, hex_hash.as_bytes(), &hex_hash) {
                Ok(plaintext) => sha256_hex(&plaintext) == hex_hash,
                Err(_) => false,
            },
            Err(_) => false,
        };
        if blob_ok {
            return Ok(hex_hash);
        }
        // Row present but blob missing or AEAD-corrupt — fall through to re-write (self-heal).
    }

    // Encrypt the plaintext bytes before writing to disk.
    let encrypted_bytes = encrypt_blob(cipher, bytes, hex_hash.as_bytes())?;

    // Ensure the two-level shard directory exists.
    let target = blob_path(vault_root, &hex_hash);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            VaultError::Storage(format!("mkdir shard {}: {e}", parent.display()))
        })?;
    }

    // Write to a temp file on the same filesystem for atomic rename.
    let tmp_dir = vault_root.join("tmp");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| VaultError::Storage(format!("mkdir tmp: {e}")))?;
    let tmp_path = tmp_dir.join(format!("{hex_hash}.tmp"));

    {
        let mut f = std::fs::File::create(&tmp_path).map_err(|e| {
            VaultError::Storage(format!("create tmp {}: {e}", tmp_path.display()))
        })?;
        f.write_all(&encrypted_bytes)
            .map_err(|e| VaultError::Storage(format!("write tmp: {e}")))?;
        f.sync_all()
            .map_err(|e| VaultError::Storage(format!("fsync tmp: {e}")))?;
    }

    // Atomic rename: encrypted bytes land at their permanent address.
    std::fs::rename(&tmp_path, &target).map_err(|e| {
        VaultError::Storage(format!(
            "rename {} → {}: {e}",
            tmp_path.display(),
            target.display()
        ))
    })?;

    // Insert the files row after bytes are durably on disk.
    // `size` records the original plaintext size (not the ciphertext size).
    let now = chrono::Utc::now().to_rfc3339();
    let size = bytes.len() as i64;
    {
        let guard = conn.lock().unwrap();
        // OR IGNORE: idempotent if a concurrent put of identical bytes raced us.
        guard
            .execute(
                "INSERT OR IGNORE INTO files \
                 (content_hash, size, mime, filename, provenance, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![hex_hash, size, mime, filename, provenance, now],
            )
            .map_err(|e| VaultError::Storage(format!("insert files row: {e}")))?;
    }
    // Lock released before verify_write re-acquires via exists().

    verify_write(conn, vault_root, cipher, &hex_hash)?;

    Ok(hex_hash)
}

/// Retrieve the bytes for a content hash. Decrypts the stored ciphertext.
/// Returns `BlobNotFound` if no blob exists at the computed path, or
/// `Decryption` if the AEAD tag fails (wrong key or tampered ciphertext).
pub(crate) fn get(
    cipher: &ChaCha20Poly1305,
    vault_root: &Path,
    content_hash: &str,
) -> Result<Vec<u8>> {
    let path = blob_path(vault_root, content_hash);
    let encrypted = std::fs::read(&path)
        .map_err(|_| VaultError::BlobNotFound { hash: content_hash.to_string() })?;
    decrypt_blob(cipher, &encrypted, content_hash.as_bytes(), content_hash)
}

/// Walk every `files` row, decrypt each blob, re-hash the plaintext, and
/// collect anomalies. AEAD tag failure or plaintext hash mismatch is reported
/// as `VerifyIssue::Corrupted`. Returns a `VerifyReport` with all anomalies.
/// An empty `issues` list means the vault is clean.
pub(crate) fn verify(
    conn: &Mutex<Connection>,
    vault_root: &Path,
    cipher: &ChaCha20Poly1305,
) -> Result<VerifyReport> {
    let hashes: Vec<String> = {
        let guard = conn.lock().unwrap();
        let mut stmt = guard
            .prepare("SELECT content_hash FROM files")
            .map_err(|e| VaultError::Storage(format!("verify prepare: {e}")))?;
        let result: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| VaultError::Storage(format!("verify query: {e}")))?
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| VaultError::Storage(format!("verify collect: {e}")))?;
        result
    };

    let total = hashes.len() as u64;
    let mut issues = Vec::new();

    for hash in &hashes {
        let path = blob_path(vault_root, hash);
        match std::fs::read(&path) {
            Err(_) => {
                issues.push(VerifyEntry { content_hash: hash.clone(), issue: VerifyIssue::Missing });
            }
            Ok(encrypted) => {
                // Decrypt-then-rehash: content_hash is always SHA-256 of plaintext.
                let ok = match decrypt_blob(cipher, &encrypted, hash.as_bytes(), hash) {
                    Ok(plaintext) => sha256_hex(&plaintext) == *hash,
                    Err(_) => false,
                };
                if !ok {
                    issues.push(VerifyEntry {
                        content_hash: hash.clone(),
                        issue: VerifyIssue::Corrupted,
                    });
                }
            }
        }
    }

    Ok(VerifyReport {
        total,
        ok: total - issues.len() as u64,
        issues,
    })
}

/// Read-back verification after a write: row present AND file at computed
/// path AND decrypt+rehash of stored bytes == `hex_hash`.
///
/// All three conditions must hold. The blob is always re-read from disk and
/// decrypted (not trusted from the `put` argument) so this catches any I/O,
/// rename, or encryption anomaly between the write and the DB commit.
pub(crate) fn verify_write(
    conn: &Mutex<Connection>,
    vault_root: &Path,
    cipher: &ChaCha20Poly1305,
    hex_hash: &str,
) -> Result<()> {
    let row_present = exists(conn, hex_hash)?;
    // Lock released before file I/O; re-acquired inside exists() for the row check above.
    let path = blob_path(vault_root, hex_hash);
    let (file_exists, hash_matches) = match std::fs::read(&path) {
        Err(_) => (false, false),
        Ok(encrypted) => {
            let hash_ok = match decrypt_blob(cipher, &encrypted, hex_hash.as_bytes(), hex_hash) {
                Ok(plaintext) => sha256_hex(&plaintext) == hex_hash,
                Err(_) => false,
            };
            (true, hash_ok)
        }
    };
    if row_present && file_exists && hash_matches {
        Ok(())
    } else {
        Err(VaultError::WriteVerificationFailed {
            hash: hex_hash.to_string(),
            row_present,
            file_exists,
            hash_matches,
        })
    }
}

// ── Free helpers ──────────────────────────────────────────────────────────────

/// Compute the hex-encoded SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Derive the on-disk path for a blob from its hash (pure function of the hash).
/// Layout: `<root>/<aa>/<bb>/<full-hex-sha256>`
pub fn blob_path(vault_root: &Path, hex_hash: &str) -> PathBuf {
    vault_root
        .join(&hex_hash[0..2])
        .join(&hex_hash[2..4])
        .join(hex_hash)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PalaceStore;
    use tempfile::TempDir;

    const TEST_KEY: &str = "palace-vault-test-key-fixed-for-all-tests";
    const CORRECT_KEY: &str = "vault-gate-test-key-correct";
    const WRONG_KEY: &str = "vault-gate-test-key-WRONG";

    fn store(tmp: &TempDir) -> PalaceStore {
        PalaceStore::open_in_memory(tmp.path(), TEST_KEY).expect("open vault")
    }

    // ── Acceptance criterion: put→get round-trip ──────────────────────────
    #[test]
    fn put_get_round_trip() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let bytes = b"hello vault";
        let hash = s.vault_put(bytes, "text/plain", Some("hello.txt"), "test").unwrap();
        assert_eq!(s.vault_get(&hash).unwrap(), bytes);
    }

    // ── Acceptance criterion: hash is correct SHA-256 ─────────────────────
    #[test]
    fn hash_is_correct_sha256() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let bytes = b"content for hashing";
        let expected = sha256_hex(bytes);
        let returned = s.vault_put(bytes, "text/plain", None, "test").unwrap();
        assert_eq!(returned, expected);
    }

    // ── Acceptance criterion: path is computed, not stored ────────────────
    // Verifies: (a) no `path` column in the `files` table, and
    //           (b) vault_get() still works from the hash alone.
    #[test]
    fn path_not_stored() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let bytes = b"check path not stored";
        let hash = s.vault_put(bytes, "text/plain", None, "test").unwrap();

        // No `path` column must exist in the schema.
        {
            let guard = s.conn.lock().unwrap();
            let cols: Vec<String> = guard
                .prepare("PRAGMA table_info(files)")
                .unwrap()
                .query_map([], |row| row.get::<_, String>(1))
                .unwrap()
                .collect::<std::result::Result<_, _>>()
                .unwrap();
            assert!(
                !cols.iter().any(|c| c == "path"),
                "files table must NOT have a path column; columns: {cols:?}"
            );
        }

        // vault_get() must resolve from the hash alone.
        assert_eq!(s.vault_get(&hash).unwrap(), bytes.as_ref());
    }

    // ── Acceptance criterion: dedup / idempotent ──────────────────────────
    // Identical bytes stored twice → one physical blob, one DB row, same hash.
    #[test]
    fn dedup_idempotent() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let bytes = b"identical content";

        let h1 = s.vault_put(bytes, "text/plain", None, "first-call").unwrap();
        let h2 = s.vault_put(bytes, "text/plain", None, "second-call").unwrap();
        assert_eq!(h1, h2, "identical bytes must yield the same hash");

        // One physical blob on disk.
        assert!(s.vault_blob_path(&h1).exists(), "blob must exist at computed path");

        // One row in the DB.
        let guard = s.conn.lock().unwrap();
        let count: i64 = guard
            .query_row(
                "SELECT COUNT(*) FROM files WHERE content_hash = ?1",
                params![h1],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "dedup must yield exactly one files row");
    }

    // ── Acceptance criterion: verify() catches a tampered blob ────────────
    #[test]
    fn verify_catches_tampered_blob() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let bytes = b"original content";
        let hash = s.vault_put(bytes, "text/plain", None, "test").unwrap();

        // Overwrite the blob with different bytes.
        std::fs::write(s.vault_blob_path(&hash), b"tampered!").unwrap();

        let report = s.vault_verify().unwrap();
        assert_eq!(report.total, 1);
        assert_eq!(report.ok, 0);
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].content_hash, hash);
        assert_eq!(report.issues[0].issue, VerifyIssue::Corrupted);
    }

    // ── Acceptance criterion: verify() catches a deleted blob ─────────────
    #[test]
    fn verify_catches_deleted_blob() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let bytes = b"will be deleted";
        let hash = s.vault_put(bytes, "text/plain", None, "test").unwrap();

        // Remove the blob but leave the DB row intact.
        std::fs::remove_file(s.vault_blob_path(&hash)).unwrap();

        let report = s.vault_verify().unwrap();
        assert_eq!(report.total, 1);
        assert_eq!(report.ok, 0);
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].content_hash, hash);
        assert_eq!(report.issues[0].issue, VerifyIssue::Missing);
    }

    // ── Acceptance criterion: read-back verify confirms committed record ───
    // After put: row present, file at computed path, vault_get returns plaintext.
    // Note: raw on-disk bytes are ciphertext so sha256(raw) != content_hash;
    // correctness is verified via vault_get (decrypt + compare) instead.
    #[test]
    fn read_back_verify_returns_committed_record() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let bytes = b"verify read-back";
        let hash = s.vault_put(bytes, "text/plain", Some("verify.txt"), "test").unwrap();

        assert!(s.vault_exists(&hash).unwrap(), "row must be present after put");

        let path = s.vault_blob_path(&hash);
        assert!(path.exists(), "blob must exist at computed path");

        // Blob on disk is ciphertext; retrieve via vault_get which decrypts.
        let retrieved = s.vault_get(&hash).unwrap();
        assert_eq!(retrieved, bytes.as_ref(), "vault_get must return original plaintext");
    }

    // ── Acceptance criterion: get on absent hash errors cleanly ───────────
    #[test]
    fn get_absent_hash_errors_cleanly() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let fake = "a".repeat(64);
        let err = s.vault_get(&fake).unwrap_err();
        assert!(
            matches!(err, VaultError::BlobNotFound { .. }),
            "expected BlobNotFound, got: {err}"
        );
    }

    // ── Additional: different bytes → different hashes, both retrievable ───
    #[test]
    fn different_bytes_different_hashes() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let h1 = s.vault_put(b"bytes A", "application/octet-stream", None, "test").unwrap();
        let h2 = s.vault_put(b"bytes B", "application/octet-stream", None, "test").unwrap();
        assert_ne!(h1, h2);
        assert_eq!(s.vault_get(&h1).unwrap(), b"bytes A");
        assert_eq!(s.vault_get(&h2).unwrap(), b"bytes B");
    }

    // ── Additional: verify() returns clean report on an untampered vault ───
    #[test]
    fn verify_clean_vault() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        s.vault_put(b"clean-1", "text/plain", None, "test").unwrap();
        s.vault_put(b"clean-2", "text/plain", None, "test").unwrap();
        let report = s.vault_verify().unwrap();
        assert_eq!(report.total, 2);
        assert_eq!(report.ok, 2);
        assert!(report.issues.is_empty(), "clean vault must have no issues");
    }

    // ── Additional: vault layout uses two-level fan-out ────────────────────
    // The blob must sit at <root>/<aa>/<bb>/<full-hash>, not flat.
    #[test]
    fn vault_layout_two_level_fan_out() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let hash = s.vault_put(b"layout check", "text/plain", None, "test").unwrap();

        let expected = tmp
            .path()
            .join(&hash[0..2])
            .join(&hash[2..4])
            .join(&hash);
        assert!(
            expected.exists(),
            "blob must be at <root>/<aa>/<bb>/<full-hash>, expected: {}",
            expected.display()
        );
        // Confirm vault_blob_path() agrees.
        assert_eq!(s.vault_blob_path(&hash), expected);
    }

    // ── Fix 1a: row + blob present → idempotent, verify clean ────────────
    #[test]
    fn dedup_row_and_blob_present_idempotent() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let bytes = b"row and blob both present";
        let h1 = s.vault_put(bytes, "text/plain", None, "first").unwrap();
        let h2 = s.vault_put(bytes, "text/plain", None, "second").unwrap();
        assert_eq!(h1, h2);
        assert!(s.vault_blob_path(&h1).exists(), "one blob must remain on disk");
        let report = s.vault_verify().unwrap();
        assert!(report.issues.is_empty(), "vault must be clean: {:?}", report.issues);
    }

    // ── Fix 1b: row present, blob deleted → put() self-heals ─────────────
    #[test]
    fn dedup_blob_deleted_is_self_healed() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let bytes = b"blob will be deleted";
        let hash = s.vault_put(bytes, "text/plain", None, "initial").unwrap();

        std::fs::remove_file(s.vault_blob_path(&hash)).unwrap();
        assert!(!s.vault_blob_path(&hash).exists(), "test setup: blob should be absent");

        let healed = s.vault_put(bytes, "text/plain", None, "reheal").unwrap();
        assert_eq!(hash, healed, "hash must be stable");
        assert!(s.vault_blob_path(&hash).exists(), "blob must be restored after self-heal");
        let report = s.vault_verify().unwrap();
        assert!(report.issues.is_empty(), "vault must be clean after self-heal: {:?}", report.issues);
    }

    // ── Fix 1c: row present, blob corrupted → put() restores bytes ───────
    #[test]
    fn dedup_blob_corrupted_is_self_healed() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let bytes = b"blob will be corrupted";
        let hash = s.vault_put(bytes, "text/plain", None, "initial").unwrap();

        std::fs::write(s.vault_blob_path(&hash), b"corrupted garbage").unwrap();

        let healed = s.vault_put(bytes, "text/plain", None, "reheal").unwrap();
        assert_eq!(hash, healed, "hash must be stable");
        assert_eq!(s.vault_get(&hash).unwrap(), bytes, "retrieved bytes must match original after heal");
        let report = s.vault_verify().unwrap();
        assert!(report.issues.is_empty(), "vault must be clean after self-heal: {:?}", report.issues);
    }

    // ── M2 Gate: vault blob AEAD encryption ──────────────────────────────

    /// Gate B-1: raw bytes on disk must NOT match the plaintext.
    /// Proves ciphertext-at-rest (not a passthrough write).
    #[test]
    fn vault_blob_is_ciphertext_at_rest() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let plaintext = b"confidential content that must be encrypted at rest";
        let hash = s.vault_put(plaintext, "text/plain", None, "test").unwrap();

        let raw = std::fs::read(s.vault_blob_path(&hash)).unwrap();
        // Raw on-disk bytes must NOT contain the plaintext.
        assert!(
            !raw.windows(plaintext.len()).any(|w| w == plaintext.as_ref()),
            "blob on disk must not contain plaintext — ciphertext at rest required"
        );
        // And must differ in size (nonce + tag overhead).
        assert_ne!(
            raw.as_slice(),
            plaintext.as_ref(),
            "blob on disk must not equal plaintext bytes"
        );
    }

    /// Gate B-2: vault_get with the wrong key must return Err, not garbage.
    #[test]
    fn vault_wrong_key_fails_decrypt() {
        let tmp = TempDir::new().unwrap();
        // Write with CORRECT_KEY.
        let hash = {
            let s = PalaceStore::open_in_memory(tmp.path(), CORRECT_KEY).unwrap();
            s.vault_put(b"secret that must not be readable with wrong key", "text/plain", None, "test")
                .unwrap()
        };
        // Read with WRONG_KEY — AEAD must reject.
        let wrong_store = PalaceStore::open_in_memory(tmp.path(), WRONG_KEY).unwrap();
        let result = wrong_store.vault_get(&hash);
        assert!(
            result.is_err(),
            "vault_get with wrong key must return Err, got Ok instead"
        );
    }

    /// Gate C: encrypt → decrypt roundtrip is clean; dedup works; tamper
    /// is detected by verify().
    #[test]
    fn vault_encrypted_roundtrip_and_verify() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let plaintext = b"round-trip content that must survive encrypt/decrypt";

        // 1. put → get roundtrip.
        let hash = s.vault_put(plaintext, "text/plain", None, "test").unwrap();
        let retrieved = s.vault_get(&hash).unwrap();
        assert_eq!(retrieved, plaintext.as_ref(), "get must return original plaintext");

        // 2. verify() must be clean.
        let report = s.vault_verify().unwrap();
        assert!(report.issues.is_empty(), "vault must be clean after put: {:?}", report.issues);

        // 3. Second put of identical plaintext dedups to one blob.
        let hash2 = s.vault_put(plaintext, "text/plain", None, "dedup-second").unwrap();
        assert_eq!(hash, hash2, "identical plaintext must dedup to same hash");
        let report2 = s.vault_verify().unwrap();
        assert_eq!(report2.total, 1, "dedup must yield exactly one blob in DB");

        // 4. Tampered ciphertext → verify() must flag it as Corrupted.
        std::fs::write(
            s.vault_blob_path(&hash),
            b"tampered-garbage-bytes-that-will-fail-aead-tag-check",
        )
        .unwrap();
        let report3 = s.vault_verify().unwrap();
        assert_eq!(report3.issues.len(), 1, "tampered blob must be flagged by verify");
        assert_eq!(report3.issues[0].issue, VerifyIssue::Corrupted);
    }

    // ── Fix 2: WriteVerificationFailed branch is exercised ───────────────
    // Fault-injection: insert a row then delete the blob so verify_write fires
    // the error path (row_present=true, file_exists=false, hash_matches=false).
    #[test]
    fn verify_write_fails_when_blob_absent_after_row_inserted() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp);
        let bytes = b"fault injection target";
        let hash = s.vault_put(bytes, "text/plain", None, "test").unwrap();

        // Delete the blob to produce the row-present / blob-missing state.
        std::fs::remove_file(s.vault_blob_path(&hash)).unwrap();

        let err = verify_write(&s.conn, &s.vault_root, &s.vault_cipher, &hash).unwrap_err();
        match err {
            VaultError::WriteVerificationFailed {
                hash: ref h,
                row_present,
                file_exists,
                hash_matches,
            } => {
                assert_eq!(h, &hash);
                assert!(row_present, "row must still be present in DB");
                assert!(!file_exists, "file must be absent");
                assert!(!hash_matches, "hash_matches must be false when file is absent");
            }
            other => panic!("expected WriteVerificationFailed, got: {other}"),
        }
    }
}
