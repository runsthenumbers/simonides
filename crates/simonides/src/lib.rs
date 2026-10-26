//! `simonides` â€” the durable system-of-record for all palace content.
//!
//! Slices 1a + 1b + M3b: drawer persistence, atomic verifiable writes,
//! tombstone semantics, the reindex-pending health floor, the content-
//! addressable byte vault, lexical (BM25) retrieval, and vector retrieval
//! (BGE-small-en-v1.5 embeddings stored as BLOBs in the sqlcipher DB).
//!
//! Encryption invariant: ALL private data â€” drawer content, chunk text,
//! embedding vectors â€” lives inside the sqlcipher DB.  No plaintext sidecars,
//! no lancedb directories, no external vector store.

pub mod retrieval;
pub mod rerank;
pub mod vault;
pub mod kg;
pub mod sync_impl;
/// P4: closets â€” a disposable, locally-derived projection over drawers.
/// Deliberately NOT part of the CRDT sync surface; see the module docs.
pub mod closets;
pub mod autotunnel;
/// The membrane — structural L1/L2 separation via two keyed databases.
/// An outer-tier session is cryptographically incapable of reading an
/// inner-tier drawer: the private file is simply not attached, so the rows are
/// absent rather than filtered. See the module docs for why an envelope over
/// L2 content was rejected (metadata leak — `SEC-CANON` §7).
pub mod membrane;
pub(crate) mod embedder;
pub use embedder::run_embed_worker;

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicU64;

use chacha20poly1305::{aead::KeyInit, ChaCha20Poly1305, Key};
use rusqlite::{params, Connection};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use zeroize::Zeroize;

use simonides_core::sync::{SubscriptionId, WriteSink};
use simonides_sync::tunnel_merge::TunnelRecord;

// Bump when a migration is needed.
const CURRENT_SCHEMA_VERSION: &str = "5";

/// M6 CRDT tables â€” inside the sqlcipher DB (never plaintext sidecars).
/// Applied in `init()` after `KG_SCHEMA`. All DDL uses `IF NOT EXISTS`.
const CRDT_SCHEMA: &str = r#"
-- Automerge document bytes per drawer. Stored in the sqlcipher DB so they
-- are encrypted at rest â€” never written to a filesystem file.
CREATE TABLE IF NOT EXISTS crdt_docs (
    drawer_id  TEXT PRIMARY KEY REFERENCES drawers(id) ON DELETE CASCADE,
    doc_bytes  BLOB NOT NULL,
    updated_at TEXT NOT NULL
);

-- OR-set of tunnels (directed edges). from+to+relation is the triple key.
CREATE TABLE IF NOT EXISTS tunnels (
    id              TEXT PRIMARY KEY,
    from_drawer_id  TEXT NOT NULL,
    to_drawer_id    TEXT NOT NULL,
    relation        TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 1.0,
    hlc_wall        INTEGER NOT NULL DEFAULT 0,
    hlc_logical     INTEGER NOT NULL DEFAULT 0,
    hlc_node_id     BLOB NOT NULL,
    tombstoned      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(from_drawer_id, to_drawer_id, relation)
);
CREATE INDEX IF NOT EXISTS tunnels_from_idx ON tunnels(from_drawer_id);
CREATE INDEX IF NOT EXISTS tunnels_to_idx   ON tunnels(to_drawer_id);
"#;

/// Lexical retrieval schema â€” chunks + FTS5 + DML triggers.
/// Applied in `init()` after the drawers and vault schemas. All DDL uses
/// `IF NOT EXISTS` so this is safe to run on every re-open.
const CHUNKS_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    drawer_id    TEXT NOT NULL REFERENCES drawers(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    char_offset  INTEGER NOT NULL,
    char_length  INTEGER NOT NULL,
    -- BLOB of packed little-endian f32 values; NULL until rebuild_indexes() runs.
    embedding    BLOB,
    UNIQUE(drawer_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_drawer_idx ON chunks(drawer_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content_rowid=rowid,
    tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_fts_ins AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_del AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_upd AFTER UPDATE ON chunks BEGIN
    UPDATE chunks_fts SET content = new.content WHERE rowid = new.rowid;
END;
"#;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Drawers are the primary content unit in the palace.
--
-- Identity contract (design Â§3a):
--   `id` is a UUID v7 assigned once at creation and never mutated.
--   UUID v7 encodes the creation timestamp in its high bits, so:
--     ORDER BY id DESC  â‰¡  newest-first
--   `wing` and `room` are metadata columns â€” a move updates them without
--   touching id. Derived layers (BM25, vectors, KG) reference `id` only,
--   so a rename/move never dangles a reference.
--
-- Tombstone contract (design Â§3c):
--   Rows are NEVER physically deleted. `state` transitions liveâ†’tombstoned;
--   that is a terminal, irreversible change. A reference to a tombstoned
--   drawer resolves to "superseded", never to "missing". The distinction is
--   always diagnosable via state.
--
-- Reindex contract (design requirement #7):
--   `reindex_pending = 1` means derived indexes for this row are stale or
--   absent. `health()` surfaces the count; search results may be incomplete
--   until `rebuild_indexes()` clears all pending flags. On tombstone,
--   `reindex_pending` is cleared â€” tombstoned rows are excluded from search
--   and need no indexing.
--
-- Addressing (M1):
--   `seq` is a per-(wing,room) monotonic integer, 1-indexed, assigned at
--   insert inside the write transaction. Canonical handle: `wing/room#seq`.
--   UUID v7 prefix resolution is secondary (low leading-bit entropy).
--   `name` is an optional human-readable label enabling `wing/room:name`
--   lookup via resolve().
CREATE TABLE IF NOT EXISTS drawers (
    id               TEXT PRIMARY KEY,
    wing             TEXT NOT NULL,
    room             TEXT NOT NULL,
    content          TEXT NOT NULL,
    provenance       TEXT NOT NULL,
    state            TEXT NOT NULL DEFAULT 'live',
    reindex_pending  INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL,
    seq              INTEGER,
    name             TEXT
);

-- UUID v7 ids sort chronologically; this index serves ORDER BY id DESC.
CREATE INDEX IF NOT EXISTS drawers_id_desc ON drawers(id DESC);
-- Location lookup (list by wing/room, state filter).
CREATE INDEX IF NOT EXISTS drawers_wing_room_state ON drawers(wing, room, state);
-- Partial index on stale rows â€” the hot path for reindex queue scans.
CREATE INDEX IF NOT EXISTS drawers_reindex ON drawers(id) WHERE reindex_pending = 1;
-- Seq handle lookup: O(log n) exact lookup by canonical handle.
CREATE INDEX IF NOT EXISTS drawers_wing_room_seq ON drawers(wing, room, seq);
-- Named lookup; partial â€” only rows with a non-NULL name are indexed.
CREATE INDEX IF NOT EXISTS drawers_name ON drawers(name) WHERE name IS NOT NULL;
"#;

// Column list for all SELECT queries that feed row_to_drawer.
// Positions: 0=id 1=wing 2=room 3=content 4=provenance 5=state
//            6=reindex_pending 7=created_at 8=seq 9=name
const COLS: &str =
    "id, wing, room, content, provenance, state, reindex_pending, created_at, \
     COALESCE(seq, 0) AS seq, name";

/// Tunnel column list in the exact order `sync_impl::row_to_tunnel` decodes.
/// Positions: 0=id 1=from 2=to 3=relation 4=confidence
///            5=hlc_wall 6=hlc_logical 7=hlc_node_id 8=tombstoned
const TUNNEL_COLS: &str =
    "id, from_drawer_id, to_drawer_id, relation, confidence, \
     hlc_wall, hlc_logical, hlc_node_id, tombstoned";

/// Hard ceiling on graph-walk depth for `follow_tunnels` and `traverse`. The
/// caller's requested depth is clamped to this, so even a pathological input
/// (huge depth on a dense/cyclic graph) can never run away â€” dumb-model-safe.
/// The per-walk `visited` set independently guarantees cycle-termination.
const MAX_TRAVERSAL_DEPTH: usize = 64;

/// P5 reserved system meta keys. These rows are seeded on open and owned by the
/// migration/embedding layers; [`PalaceStore::set_meta`] refuses to write them so
/// no caller can corrupt migration/embedding state through the public meta API.
const RESERVED_META_KEYS: &[&str] = &["schema_version", "embedding_model_version"];

/// Diary convention: an agent's journal lives in wing `wing_<agent>`, room
/// `personal-diary`. Entries are ordinary drawers, so they are durable and sync
/// like everything else; `diary_write` is a thin convenience over `insert`.
///
/// Read and write must use the same constant. If they ever disagree with the
/// stored data, `diary_read` returns an empty list rather than an error: a wrong
/// answer, not a failure, and an easy one to miss.
const DIARY_ROOM: &str = "personal-diary";

/// Errors from `PalaceStore` operations.
#[derive(Debug, thiserror::Error)]
pub enum PalaceError {
    #[error("storage: {0}")]
    Storage(String),

    #[error("not found: {id}")]
    NotFound { id: String },

    /// The Â§3.6 mis-route structural fix. Returned when a write's read-back
    /// finds the record at a different wing/room than the caller intended.
    /// This makes silent mis-routing structurally impossible: success requires
    /// a confirmed location, not just a successful INSERT/UPDATE.
    #[error(
        "write verification failed: record {id} landed at \
         wing={actual_wing}/room={actual_room}, \
         expected wing={expected_wing}/room={expected_room}"
    )]
    WriteVerificationFailed {
        id: String,
        expected_wing: String,
        expected_room: String,
        actual_wing: String,
        actual_room: String,
    },

    /// P5 reserved-key guard. Returned by [`PalaceStore::set_meta`] when a caller
    /// tries to write one of the reserved system keys (`schema_version`,
    /// `embedding_model_version`). Those rows are owned by the migration and
    /// embedding layers; a caller clobbering `schema_version` would corrupt
    /// migration state. The write is refused LOUDLY â€” never silently dropped and
    /// never applied â€” so a dumb model cannot footgun the version table.
    #[error(
        "meta key '{key}' is a reserved system key and cannot be set through \
         set_meta â€” it is owned by the migration/embedding layer"
    )]
    ReservedMetaKey { key: String },

    /// A configuration file (for example a tier policy) could not be read or is
    /// invalid. Raised at load time, so a bad file stops startup instead of
    /// silently routing data to the wrong tier.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

type Result<T> = std::result::Result<T, PalaceError>;

/// Lifecycle state of a drawer. Transition is one-way: `Live` â†’ `Tombstoned`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrawerState {
    Live,
    /// The drawer has been retired. The row is never deleted; `get()` returns
    /// `Some(Tombstoned)`. References resolve to "superseded," not "missing."
    Tombstoned,
}

/// A single drawer record.
#[derive(Debug, Clone)]
pub struct Drawer {
    /// Stable UUID v7 assigned at creation. Never mutated. Move-invariant:
    /// changing wing/room leaves this field unchanged.
    pub id: String,
    /// Wing label (metadata â€” updated on move, `id` is unchanged).
    pub wing: String,
    /// Room label (metadata â€” updated on move, `id` is unchanged).
    pub room: String,
    /// Verbatim content text.
    pub content: String,
    /// Origin descriptor â€” who or what created this drawer.
    pub provenance: String,
    /// Lifecycle state.
    pub state: DrawerState,
    /// `true` when derived indexes (BM25, vector ANN) are stale or absent.
    /// Cleared on tombstone (tombstoned rows are excluded from search).
    pub reindex_pending: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Per-(wing,room) monotonic counter, 1-indexed. Canonical handle form:
    /// `wing/room#seq`. Rooms are independent â€” room A and room B both start
    /// at 1. Assigned atomically in the insert transaction.
    pub seq: i64,
    /// Optional human-readable label. Enables `wing/room:name` lookup via
    /// `resolve()`. Set via `insert_named()`; `None` for unnamed drawers.
    pub name: Option<String>,
}

/// A typed, directed edge between two drawers â€” the public projection of the
/// CRDT-mergeable [`TunnelRecord`]. The HLC that stamps the edge for merge is an
/// internal sync concern and is deliberately not exposed here; consumers see the
/// stable identity, endpoints, relation, confidence, and liveness.
#[derive(Debug, Clone, PartialEq)]
pub struct Tunnel {
    /// Stable UUID v7 assigned when the edge is first created. Reused on an
    /// idempotent re-create of the same `(from, to, relation)` triple.
    pub id: String,
    /// Source drawer id (full UUID, always resolved from the caller's reference).
    pub from_drawer_id: String,
    /// Destination drawer id (full UUID).
    pub to_drawer_id: String,
    /// Free-form relation label; the third element of the triple key.
    pub relation: String,
    /// Edge confidence in `[0.0, 1.0]`. Defaults to `1.0` when unspecified.
    pub confidence: f32,
    /// `true` once retired via `tombstone_tunnel`. Tombstoned edges are excluded
    /// from `list_tunnels`, `find_tunnels*`, `follow_tunnels`, and `traverse`,
    /// but the row is never physically deleted (tombstone-not-delete discipline).
    pub tombstoned: bool,
}

impl From<TunnelRecord> for Tunnel {
    fn from(t: TunnelRecord) -> Self {
        Tunnel {
            id: t.id,
            from_drawer_id: t.from_drawer_id,
            to_drawer_id: t.to_drawer_id,
            relation: t.relation,
            confidence: t.confidence,
            tombstoned: t.tombstoned,
        }
    }
}

/// Outcome of `resolve()`. Always loud â€” never a bare not-found.
#[derive(Debug)]
pub enum ResolveOutcome {
    Found(Drawer),
    /// The reference matched more than one drawer; caller must be more specific.
    Ambiguous {
        query: String,
        candidates: Vec<Drawer>,
    },
    /// No drawer matched the reference. `nearest` contains recently-inserted
    /// live drawers to help orient the caller; never a silent empty result.
    NotFound {
        query: String,
        nearest: Vec<Drawer>,
    },
}

/// Snapshot of store health for monitoring and diagnostics.
pub struct HealthReport {
    pub total: u64,
    pub live: u64,
    pub tombstoned: u64,
    /// Live rows whose derived indexes are stale or absent.
    /// Non-zero means search results may be incomplete.
    pub reindex_pending: u64,
    /// Number of entities in the KG (0 until rebuild_indexes runs).
    pub kg_entities: u64,
    /// Number of triples in the KG (0 until rebuild_indexes runs).
    pub kg_triples: u64,
    /// P4: closets in the local projection (0 until `rebuild_closets` runs).
    pub closets: u64,
    /// P4: rooms whose closet projection no longer matches their drawers. A
    /// stale projection is never silently served as current â€” each of these is
    /// one `rebuild_closets` away from correct.
    pub closets_stale: u64,
    /// PH-core: chunks whose embedding is present, vs total chunks. The vector
    /// index lives IN the chunks table (no separate file), so this invariant is a
    /// COUNTED FACT, not a flag â€” `chunks_total - chunks_embedded` is exactly how
    /// many chunks are invisible to vector search right now. Populated on EVERY
    /// call, healthy or not: a number you can only read when it is wrong is not a
    /// monitor.
    pub chunks_total: u64,
    pub chunks_embedded: u64,
    /// Model id that produced the stored embeddings (`meta.embedding_model_version`).
    /// A skew from the running embedder means the embeddings are stale by definition.
    pub embedding_model_version: String,
}

impl std::fmt::Display for HealthReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let index_status = if self.reindex_pending > 0 {
            format!("{} pending / results may be incomplete", self.reindex_pending)
        } else {
            "indexes current".to_string()
        };
        write!(
            f,
            "total={} live={} tombstoned={} reindex={} kg_entities={} kg_triples={} \
             closets={} closets_stale={} chunks={} embedded={} model={}",
            self.total, self.live, self.tombstoned, index_status,
            self.kg_entities, self.kg_triples,
            self.closets, self.closets_stale,
            self.chunks_total, self.chunks_embedded, self.embedding_model_version,
        )?;
        if self.closets_stale > 0 {
            write!(
                f,
                " (closet projection stale in {} room(s) / run rebuild_closets)",
                self.closets_stale
            )?;
        }
        if self.chunks_embedded < self.chunks_total {
            write!(
                f,
                " (vector recall PARTIAL: {} of {} chunks invisible to vector search / run rebuild_indexes)",
                self.chunks_total - self.chunks_embedded,
                self.chunks_total
            )?;
        }
        Ok(())
    }
}

/// What a `rebuild_indexes` pass actually did â€” makes the re-derive win
/// observable. `embedded` = chunks sent to the model this pass; `reused` = chunks
/// whose stored blob was kept without re-embedding. A pass that silently
/// re-embeds everything (the pre-PH behaviour) shows `reused == 0`: the
/// regression is visible in the report, not hidden.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RebuildReport {
    pub embedded: usize,
    pub reused: usize,
    pub drawers: usize,
}

/// What a `rebuild_kg` pass produced â€” entities/triples built from drawer content,
/// no embedding. Makes the extract-only half measurable and idempotent.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct KgBuildReport {
    pub drawers_scanned: usize,
    pub entities: usize,
    pub triples: usize,
}

/// The durable palace record store. Owns the drawer catalog, file vault, and
/// vector index (all in a single sqlcipher connection). Thread-safe via
/// internal Mutexes. Encryption is mandatory.
pub struct PalaceStore {
    pub(crate) conn: Mutex<Connection>,
    pub(crate) vault_root: PathBuf,
    pub(crate) vault_cipher: ChaCha20Poly1305,
    /// Lazily-initialised BGE-small-en-v1.5 ONNX session.  `None` until the
    /// first call to `embed_texts()`.  Serialised by Mutex so the OnceLock
    /// pattern is not needed (embedding is never on the hot path).
    embedder: Mutex<Option<embedder::Embedder>>,
    /// M4: disposable in-memory KG projection. `None` until `rebuild_indexes()`
    /// runs.  KG search leg degrades gracefully when absent.
    pub(crate) kg_projection: Mutex<Option<kg::KgProjection>>,
    /// M6: write-event subscribers (id, sink). Protected by Mutex.
    pub(crate) write_sinks: Mutex<Vec<(SubscriptionId, Arc<dyn WriteSink>)>>,
    /// M6: monotonic counter for subscription IDs.
    pub(crate) next_sub_id: AtomicU64,
    /// Which tier each `(wing, room)` is stamped with on insert. Fail-safe (every
    /// drawer `L3`) until a policy is set; see [`membrane::TierPolicy`].
    pub(crate) tier_policy: Mutex<membrane::TierPolicy>,
    /// MEMBRANE: where the private (L2) database lives, plus its session policy.
    /// `None` until `configure_membrane()` — a store that was never configured
    /// simply has no inner tier, which is correct for the many call sites
    /// (tests, one-shot tools) that only ever touch a single database.
    pub(crate) membrane_cfg: Mutex<Option<(PathBuf, membrane::MembraneConfig)>>,
    /// Record of the live crossing. `None` == closed.
    ///
    /// This is bookkeeping for expiry, **not** the security boundary. Whether
    /// L2 rows are reachable is decided by sqlite's attach list, which
    /// `membrane_is_open()` interrogates directly. If this field and sqlite
    /// ever disagree, sqlite is right — that asymmetry is deliberate, because a
    /// bool we maintain is exactly the kind of thing a bug can flip.
    pub(crate) membrane_session: Mutex<Option<membrane::MembraneSession>>,
    /// Injectable time source so the 60-minute idle policy is testable without
    /// waiting 60 minutes.
    pub(crate) clock: Arc<dyn membrane::Clock>,
}

impl PalaceStore {
    /// Open the store at `db_path`, creating directories and applying the
    /// schema on first open. Vault blobs are stored under `vault_root`.
    /// WAL mode is enabled; FK enforcement is ON for this connection.
    ///
    /// `key` is passed directly to sqlcipher's PRAGMA key (must be the first
    /// statement â€” before WAL, FK, or schema DDL). The caller derives key
    /// material via simonides-crypto and passes it in; simonides never derives.
    /// Opening with the wrong key returns `Err` immediately (eager rejection).
    pub fn open(db_path: &Path, vault_root: &Path, key: &str) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                PalaceError::Storage(format!("mkdir {}: {e}", parent.display()))
            })?;
        }
        std::fs::create_dir_all(vault_root).map_err(|e| {
            PalaceError::Storage(format!("mkdir vault_root {}: {e}", vault_root.display()))
        })?;
        let conn = Connection::open(db_path)
            .map_err(|e| PalaceError::Storage(format!("open {}: {e}", db_path.display())))?;
        Self::init(conn, vault_root.to_path_buf(), key)
    }

    /// Open an in-memory store. Vault blobs are still written to the
    /// filesystem at `vault_root`. Intended for tests; the in-memory DB is
    /// lost on drop.
    pub fn open_in_memory(vault_root: &Path, key: &str) -> Result<Self> {
        std::fs::create_dir_all(vault_root).map_err(|e| {
            PalaceError::Storage(format!("mkdir vault_root {}: {e}", vault_root.display()))
        })?;
        let conn = Connection::open_in_memory()
            .map_err(|e| PalaceError::Storage(format!("open in_memory: {e}")))?;
        Self::init(conn, vault_root.to_path_buf(), key)
    }

    fn init(conn: Connection, vault_root: PathBuf, key: &str) -> Result<Self> {
        // sqlcipher PRAGMA key MUST be the absolute first statement â€” before
        // WAL, FK pragma, and all schema DDL. Without this ordering sqlcipher
        // silently no-ops and writes plaintext. The header gate test catches
        // any regression where this ordering is broken.
        conn.pragma_update(None, "key", key)
            .map_err(|e| PalaceError::Storage(format!("sqlcipher key: {e}")))?;

        // WAL persists on disk; set before schema DDL.
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| PalaceError::Storage(format!("WAL pragma: {e}")))?;
        // FK enforcement is per-connection; enable it for the connection lifetime.
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|e| PalaceError::Storage(format!("FK pragma: {e}")))?;

        conn.execute_batch(SCHEMA)
            .map_err(|e| PalaceError::Storage(format!("schema: {e}")))?;
        conn.execute_batch(vault::FILES_SCHEMA)
            .map_err(|e| PalaceError::Storage(format!("vault schema: {e}")))?;
        conn.execute_batch(CHUNKS_SCHEMA)
            .map_err(|e| PalaceError::Storage(format!("chunks schema: {e}")))?;

        // Seed meta rows on first open; IGNORE conflict on re-open.
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?1)",
            params![CURRENT_SCHEMA_VERSION],
        )
        .map_err(|e| PalaceError::Storage(format!("seed schema_version: {e}")))?;
        // embedding_model_version is empty until the embedding layer is configured.
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('embedding_model_version', '')",
            [],
        )
        .map_err(|e| PalaceError::Storage(format!("seed embedding_model_version: {e}")))?;

        // Apply any pending migrations (no-op for fresh databases at current version).
        Self::migrate(&conn)?;

        // M4: KG tables (additive DDL, IF NOT EXISTS â€” no migration bump needed).
        conn.execute_batch(kg::KG_SCHEMA)
            .map_err(|e| PalaceError::Storage(format!("kg schema: {e}")))?;

        // M6: CRDT tables (crdt_docs + tunnels â€” always inside sqlcipher DB).
        conn.execute_batch(CRDT_SCHEMA)
            .map_err(|e| PalaceError::Storage(format!("crdt schema: {e}")))?;

        // P4: closet projection tables. Inside the same sqlcipher DB â€” the
        // projection is private-derived data, so it is encrypted at rest for
        // free and never written to a plaintext sidecar. Additive DDL
        // (IF NOT EXISTS), so no schema-version bump is needed.
        conn.execute_batch(closets::CLOSETS_SCHEMA)
            .map_err(|e| PalaceError::Storage(format!("closets schema: {e}")))?;

        // Derive a 32-byte vault cipher key from the passed key material.
        // SHA-256 is used as a format conversion (string â†’ 32 bytes); the
        // caller is expected to pass already-derived key material.
        let mut vault_key_bytes: [u8; 32] = Sha256::digest(key.as_bytes()).into();
        let vault_cipher = ChaCha20Poly1305::new(Key::from_slice(&vault_key_bytes));
        vault_key_bytes.zeroize();

        // Build KG projection if triples already exist (re-open of an existing DB).
        let initial_projection = kg::KgProjection::build(&conn)
            .ok()
            .filter(|p| !p.node_map.is_empty());

        Ok(Self {
            conn: Mutex::new(conn),
            vault_root,
            vault_cipher,
            embedder: Mutex::new(None),
            kg_projection: Mutex::new(initial_projection),
            write_sinks: Mutex::new(Vec::new()),
            next_sub_id: AtomicU64::new(0),
            tier_policy: Mutex::new(membrane::TierPolicy::fail_safe()),
            // Membrane starts unconfigured and closed. A store must be told
            // where its private half lives before a crossing is possible.
            membrane_cfg: Mutex::new(None),
            membrane_session: Mutex::new(None),
            clock: Arc::new(membrane::SystemClock),
        })
    }

    // ── Membrane ────────────────────────────────────────────────────────────

    /// Set the tier policy this store stamps on new drawers. A store that is never
    /// given one stamps everything `L3` (fail-safe).
    pub fn set_tier_policy(&self, policy: membrane::TierPolicy) {
        *self.tier_policy.lock().unwrap() = policy;
    }

    /// The tier policy currently in force.
    pub fn tier_policy(&self) -> membrane::TierPolicy {
        self.tier_policy.lock().unwrap().clone()
    }

    /// Point this store at its private (L2) half and set the session policy.
    ///
    /// Additive: existing call sites that never call this keep exactly their
    /// current behaviour — one database, no inner tier. Nothing about the open
    /// tier changes when a membrane is configured; it only becomes *possible*
    /// to cross.
    pub fn configure_membrane(&self, priv_path: PathBuf, cfg: membrane::MembraneConfig) {
        *self.membrane_cfg.lock().unwrap() = Some((priv_path, cfg));
    }

    /// Swap the time source. Tests only — production uses `SystemClock`.
    #[cfg(test)]
    pub(crate) fn set_clock(&mut self, clock: Arc<dyn membrane::Clock>) {
        self.clock = clock;
    }

    /// Path of the keyslot record that sits beside the private database.
    fn keyslots_path(priv_path: &Path) -> PathBuf {
        let mut p = priv_path.to_path_buf();
        let name = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        p.set_file_name(format!("{name}.keyslots.json"));
        p
    }

    /// Is the private database attached *right now*?
    ///
    /// Asks sqlite, then applies the expiry policy — and if the session has
    /// expired, **detaches before returning false**. Expiry that only reports
    /// would be a lie: the rows would still be readable by anything that
    /// queried without asking first.
    pub fn membrane_is_open(&self) -> bool {
        let attached = {
            let guard = self.conn.lock().unwrap();
            membrane::is_attached(&guard)
        };
        if !attached {
            *self.membrane_session.lock().unwrap() = None;
            return false;
        }
        let expired = {
            let cfg = self.membrane_cfg.lock().unwrap();
            let sess = self.membrane_session.lock().unwrap();
            match (cfg.as_ref(), sess.as_ref()) {
                (Some((_, cfg)), Some(s)) => s.is_expired(cfg, self.clock.now_secs()),
                // Attached with no session record: treat as expired and close.
                // An attachment nobody is tracking is not a session, it is a leak.
                (_, None) => true,
                _ => false,
            }
        };
        if expired {
            let _ = self.membrane_close();
            return false;
        }
        true
    }

    /// Register activity, renewing the idle clock when policy allows.
    pub fn membrane_touch(&self) {
        let renew = self
            .membrane_cfg
            .lock()
            .unwrap()
            .as_ref()
            .map(|(_, c)| c.renew_on_activity)
            .unwrap_or(false);
        if !renew {
            return;
        }
        if let Some(s) = self.membrane_session.lock().unwrap().as_mut() {
            s.last_activity_secs = self.clock.now_secs();
        }
    }

    /// Cross the membrane using a Data Key that the caller already holds.
    ///
    /// Separate from the passphrase path so tests and the splitter can attach
    /// without a KDF round, and so the passphrase never has to exist in a
    /// signature that a caller might log.
    pub fn membrane_open_with_dk(&self, dk: &[u8; 32]) -> Result<()> {
        let (priv_path, _) = self
            .membrane_cfg
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| PalaceError::Storage("membrane not configured".into()))?;
        {
            let guard = self.conn.lock().unwrap();
            if membrane::is_attached(&guard) {
                return Ok(()); // already open; opening twice is not an error
            }
            membrane::attach_private(&guard, &priv_path, dk)?;
        }
        let now = self.clock.now_secs();
        *self.membrane_session.lock().unwrap() = Some(membrane::MembraneSession {
            opened_at_secs: now,
            last_activity_secs: now,
            priv_path,
        });
        Ok(())
    }

    /// Cross the membrane with the passphrase.
    ///
    /// **The passphrase is a `&[u8]` the caller reads from a local console and
    /// drops immediately.** It must never arrive over MCP: that
    /// would put it in a JSON-RPC frame, through every relay on the way, past
    /// their logging. The tool call is a *request to prompt*, not a carrier of the
    /// secret — which also means no remote caller can trigger a crossing remotely,
    /// even holding the passphrase. That falls out of the design rather than
    /// being a rule anyone has to remember.
    pub fn membrane_open(&self, passphrase: &[u8]) -> Result<()> {
        let (priv_path, _) = self
            .membrane_cfg
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| PalaceError::Storage("membrane not configured".into()))?;
        let slots = membrane::MembraneKeyslots::read_from(&Self::keyslots_path(&priv_path))?;
        let dk = slots.unwrap_with_passphrase(passphrase)?;
        self.membrane_open_with_dk(&dk)
    }

    /// Cross using the recovery code instead of the passphrase.
    pub fn membrane_open_with_recovery(&self, recovery_code: &[u8]) -> Result<()> {
        let (priv_path, _) = self
            .membrane_cfg
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| PalaceError::Storage("membrane not configured".into()))?;
        let slots = membrane::MembraneKeyslots::read_from(&Self::keyslots_path(&priv_path))?;
        let dk = slots.unwrap_with_recovery(recovery_code)?;
        self.membrane_open_with_dk(&dk)
    }

    /// Lock the membrane: detach the private database and forget the session.
    ///
    /// Idempotent — closing a closed membrane succeeds. Both lock paths
    /// (explicit close, idle expiry) land here, so there is one teardown to get
    /// right and one to test.
    pub fn membrane_close(&self) -> Result<()> {
        {
            let guard = self.conn.lock().unwrap();
            if membrane::is_attached(&guard) {
                membrane::detach_private(&guard)?;
            }
        }
        *self.membrane_session.lock().unwrap() = None;
        Ok(())
    }

    // â”€â”€ Schema migration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Apply pending migrations to `conn`. Called from `init()` before the
    /// connection is wrapped in the Mutex.
    fn migrate(conn: &Connection) -> Result<()> {
        let ver: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .map_err(|e| PalaceError::Storage(format!("migrate: read schema_version: {e}")))?;

        if ver.as_str() >= CURRENT_SCHEMA_VERSION {
            return Ok(());
        }

        // v1 â†’ v2: add `seq` and `name` columns, backfill seq.
        if ver.as_str() < "2" {
            let has_seq: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM pragma_table_info('drawers') WHERE name = 'seq'",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .map(|c| c > 0)
                .map_err(|e| PalaceError::Storage(format!("migrate: check seq col: {e}")))?;

            if !has_seq {
                conn.execute_batch("ALTER TABLE drawers ADD COLUMN seq INTEGER")
                    .map_err(|e| PalaceError::Storage(format!("migrate: add seq: {e}")))?;
                // Backfill: deterministic ordering by created_at ASC, then id ASC for ties.
                // Produces a correct 1-indexed monotonic seq per (wing, room).
                conn.execute_batch(
                    "UPDATE drawers SET seq = (
                        SELECT COUNT(*) FROM drawers d2
                        WHERE d2.wing  = drawers.wing
                          AND d2.room  = drawers.room
                          AND (d2.created_at < drawers.created_at
                               OR (d2.created_at = drawers.created_at
                                   AND d2.id < drawers.id))
                    ) + 1",
                )
                .map_err(|e| PalaceError::Storage(format!("migrate: backfill seq: {e}")))?;
            }

            let has_name: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM pragma_table_info('drawers') WHERE name = 'name'",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .map(|c| c > 0)
                .map_err(|e| PalaceError::Storage(format!("migrate: check name col: {e}")))?;

            if !has_name {
                conn.execute_batch("ALTER TABLE drawers ADD COLUMN name TEXT")
                    .map_err(|e| PalaceError::Storage(format!("migrate: add name: {e}")))?;
            }

            conn.execute(
                "UPDATE meta SET value = '2' WHERE key = 'schema_version'",
                [],
            )
            .map_err(|e| PalaceError::Storage(format!("migrate: bump version: {e}")))?;
        }

        // v2 â†’ v3: add `embedding BLOB` column to `chunks`.
        if ver.as_str() < "3" {
            let has_embedding: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM pragma_table_info('chunks') WHERE name = 'embedding'",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .map(|c| c > 0)
                .map_err(|e| PalaceError::Storage(format!("migrate: check embedding col: {e}")))?;

            if !has_embedding {
                conn.execute_batch("ALTER TABLE chunks ADD COLUMN embedding BLOB")
                    .map_err(|e| PalaceError::Storage(format!("migrate: add embedding: {e}")))?;
            }

            conn.execute(
                "UPDATE meta SET value = '3' WHERE key = 'schema_version'",
                [],
            )
            .map_err(|e| PalaceError::Storage(format!("migrate: bump to v3: {e}")))?;
        }

        // v3 â†’ v4: add M6 CRDT fields to entities (properties, hlc_wall, hlc_logical, hlc_node_id).
        // Fresh DBs get these columns via the updated KG_SCHEMA DDL; this migration
        // only runs against existing v3 DBs that predate the M6 column additions.
        if ver.as_str() < "4" {
            let has_entities: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='entities'",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .map(|c| c > 0)
                .map_err(|e| PalaceError::Storage(format!("migrate v3â†’v4: check entities: {e}")))?;

            if has_entities {
                let has_properties: bool = conn
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('entities') WHERE name = 'properties'",
                        [],
                        |r| r.get::<_, i64>(0),
                    )
                    .map(|c| c > 0)
                    .map_err(|e| PalaceError::Storage(format!("migrate v3â†’v4: check properties: {e}")))?;

                if !has_properties {
                    conn.execute_batch(
                        "ALTER TABLE entities ADD COLUMN properties TEXT NOT NULL DEFAULT '{}';\
                         ALTER TABLE entities ADD COLUMN hlc_wall INTEGER NOT NULL DEFAULT 0;\
                         ALTER TABLE entities ADD COLUMN hlc_logical INTEGER NOT NULL DEFAULT 0;\
                         ALTER TABLE entities ADD COLUMN hlc_node_id BLOB;",
                    )
                    .map_err(|e| PalaceError::Storage(format!("migrate v3â†’v4: alter entities: {e}")))?;
                }
            }

            conn.execute(
                "UPDATE meta SET value = '4' WHERE key = 'schema_version'",
                [],
            )
            .map_err(|e| PalaceError::Storage(format!("migrate: bump to v4: {e}")))?;
        }

        // v4 â†’ v5: widen the triples fact-key UNIQUE index from
        // (subject_id, predicate, object_id) to include valid_from, so each
        // validity WINDOW is its own row (bitemporal PK fix, brief PK 2026-07-15).
        // Under the old narrow index a re-assertion after invalidation reused the
        // closed row's slot and write_triple's INSERT OR REPLACE ate the earlier
        // window â€” silent temporal-history loss. The wide index lets two windows
        // coexist as two rows.
        //
        // The index is rebuilt on a table that already holds rows. The old key is
        // STRICTLY STRICTER than the new one (any set of rows unique on (s,p,o) is
        // trivially unique on (s,p,o,valid_from)), so the rebuild can never fail on
        // existing data â€” every triple survives with its window intact. Guarded on
        // the triples table existing and the CREATE is IF NOT EXISTS, so this is
        // idempotent and safe on re-open. On a FRESH DB this whole block is skipped
        // (schema_version is seeded at CURRENT) and KG_SCHEMA creates the wide index
        // directly. This migration writes schema_version via the internal UPDATE
        // path, NOT set_meta â€” set_meta refuses the reserved key (P5 guard).
        if ver.as_str() < "5" {
            let has_triples: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='triples'",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .map(|c| c > 0)
                .map_err(|e| PalaceError::Storage(format!("migrate v4â†’v5: check triples: {e}")))?;

            if has_triples {
                conn.execute_batch(
                    "DROP INDEX IF EXISTS triples_fact_key;\
                     CREATE UNIQUE INDEX IF NOT EXISTS triples_fact_key \
                         ON triples(subject_id, predicate, object_id, valid_from);",
                )
                .map_err(|e| {
                    PalaceError::Storage(format!("migrate v4â†’v5: rebuild triples_fact_key: {e}"))
                })?;
            }

            conn.execute(
                "UPDATE meta SET value = '5' WHERE key = 'schema_version'",
                [],
            )
            .map_err(|e| PalaceError::Storage(format!("migrate: bump to v5: {e}")))?;
        }

        Ok(())
    }

    // â”€â”€ Drawer operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Insert a new drawer without a name. See `insert_named` for the named
    /// form. Assigns a UUID v7 id and a per-(wing,room) monotonic `seq` inside
    /// a single write transaction; returns the committed drawer after read-back
    /// verification.
    pub fn insert(
        &self,
        wing: &str,
        room: &str,
        content: &str,
        provenance: &str,
    ) -> Result<Drawer> {
        self.insert_impl(wing, room, content, provenance, None)
    }

    /// Insert a drawer with an optional human-readable `name` label, enabling
    /// `wing/room:name` resolution via `resolve()`.
    pub fn insert_named(
        &self,
        wing: &str,
        room: &str,
        content: &str,
        provenance: &str,
        name: &str,
    ) -> Result<Drawer> {
        self.insert_impl(wing, room, content, provenance, Some(name))
    }

    fn insert_impl(
        &self,
        wing: &str,
        room: &str,
        content: &str,
        provenance: &str,
        name: Option<&str>,
    ) -> Result<Drawer> {
        // Stamp the tier at creation, from the same policy the tiered layer routes
        // by. Idempotent: a provenance that already carries a tag (for example from
        // an importer) is left alone. Deciding the tier here, once, is what keeps
        // untagged drawers from accumulating.
        let provenance_owned = self.tier_policy.lock().unwrap().tag_provenance(provenance, wing, room);
        let provenance: &str = &provenance_owned;

        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        {
            let mut guard = self.conn.lock().unwrap();
            // Explicit transaction: seq assignment + INSERT are atomic.
            // Single connection + Mutex means no other writer can race us, but the
            // explicit transaction makes the read-modify-write visible and correct.
            let tx = guard
                .transaction()
                .map_err(|e| PalaceError::Storage(format!("insert tx begin: {e}")))?;
            let seq: i64 = tx
                .query_row(
                    "SELECT COALESCE(MAX(seq), 0) + 1 \
                     FROM drawers WHERE wing = ?1 AND room = ?2",
                    params![wing, room],
                    |row| row.get(0),
                )
                .map_err(|e| PalaceError::Storage(format!("insert seq assign: {e}")))?;
            tx.execute(
                "INSERT INTO drawers \
                 (id, wing, room, content, provenance, state, reindex_pending, \
                  created_at, seq, name) \
                 VALUES (?1, ?2, ?3, ?4, ?5, 'live', 1, ?6, ?7, ?8)",
                params![id, wing, room, content, provenance, now, seq, name],
            )
            .map_err(|e| PalaceError::Storage(format!("insert: {e}")))?;
            // Chunk synchronously in the same transaction â€” lexical index is
            // always current; a written drawer is immediately keyword-findable.
            retrieval::chunk_in_tx(&tx, &id, content, retrieval::DEFAULT_CHUNK_SIZE)?;
            tx.commit()
                .map_err(|e| PalaceError::Storage(format!("insert tx commit: {e}")))?;
        }
        // Keep the KG live per-write: extract this drawer's entities/triples inline
        // (cheap, model-free) so a saved note is immediately in the graph and a
        // full-corpus rebuild_kg is never needed for new content. Best-effort -- the
        // drawer is already committed; a derived-projection failure must not fail the
        // write.
        if let Err(e) = self.kg_extract_drawer(&id, content) {
            tracing::warn!("insert: inline kg extract failed for {id}: {e}");
        }
        // Lock released before read-back; verify re-acquires via get().
        self.verify_and_return(&id, wing, room)
    }

    /// Fetch a drawer by id. Returns `Some` for both `Live` and `Tombstoned`
    /// rows; returns `None` only when the id is entirely absent from the store.
    pub fn get(&self, id: &str) -> Result<Option<Drawer>> {
        let guard = self.conn.lock().unwrap();
        let sql = format!("SELECT {COLS} FROM drawers WHERE id = ?1");
        let mut stmt = guard
            .prepare(&sql)
            .map_err(|e| PalaceError::Storage(format!("get prepare: {e}")))?;
        let mut rows = stmt
            .query_map(params![id], row_to_drawer)
            .map_err(|e| PalaceError::Storage(format!("get query: {e}")))?;
        match rows.next() {
            Some(r) => r
                .map_err(|e| PalaceError::Storage(e.to_string()))?
                .map(Some),
            None => Ok(None),
        }
    }

    /// Update the `wing` and `room` of a live drawer. Returns the drawer after
    /// read-back verification. Returns `NotFound` if absent or tombstoned.
    ///
    /// **Real CRDT op** (parity with [`update_drawer`]): the relocation is folded
    /// into the drawer's Automerge doc in `crdt_docs` within the same transaction,
    /// so it advances the drawer's causal history and survives
    /// `produce_delta` â†’ `apply_remote_delta`. A bare SQL UPDATE would leave the
    /// AM doc stale, so the move would not be recorded as a real op.
    pub fn move_drawer(&self, id: &str, new_wing: &str, new_room: &str) -> Result<Drawer> {
        {
            let mut guard = self.conn.lock().unwrap();
            let tx = guard
                .transaction()
                .map_err(|e| PalaceError::Storage(format!("move_drawer tx begin: {e}")))?;
            let n = tx
                .execute(
                    "UPDATE drawers SET wing = ?1, room = ?2 \
                     WHERE id = ?3 AND state = 'live'",
                    params![new_wing, new_room, id],
                )
                .map_err(|e| PalaceError::Storage(format!("move: {e}")))?;
            if n == 0 {
                // Absent or tombstoned â€” nothing moved. tx rolls back on drop.
                return Err(PalaceError::NotFound { id: id.to_string() });
            }
            move_crdt_location_in_tx(&tx, id, new_wing, new_room)?;
            tx.commit()
                .map_err(|e| PalaceError::Storage(format!("move_drawer tx commit: {e}")))?;
        }
        // Lock released before read-back.
        self.verify_and_return(id, new_wing, new_room)
    }

    /// Edit a live drawer's content in place. Ported from the predecessor store's
    /// `update_drawer`, extended with the two palace invariants the predecessor store
    /// lacks:
    ///   1. **Reindex floor** â€” sets `reindex_pending = 1` and regenerates the
    ///      lexical chunks so BM25/FTS reflect the new content immediately, and
    ///      `health()` surfaces the vector index as stale until `rebuild_indexes()`.
    ///   2. **Real CRDT op** â€” folds the change into the drawer's Automerge doc
    ///      in `crdt_docs` (never a plaintext sidecar), so the edit survives
    ///      `produce_delta` â†’ `apply_remote_delta`. A bare SQL UPDATE would leave
    ///      the AM doc stale and the edit would be lost when a peer merges.
    ///
    /// Only live drawers are editable: an absent id or a tombstoned drawer
    /// returns `NotFound` (mirrors `move_drawer`). Returns the drawer after
    /// read-back. Everything below runs in one transaction so a failure rolls
    /// back the content, chunk, and CRDT writes together.
    pub fn update_drawer(&self, id: &str, new_content: &str) -> Result<Drawer> {
        {
            let mut guard = self.conn.lock().unwrap();
            let tx = guard
                .transaction()
                .map_err(|e| PalaceError::Storage(format!("update_drawer tx begin: {e}")))?;
            let n = tx
                .execute(
                    "UPDATE drawers SET content = ?1, reindex_pending = 1 \
                     WHERE id = ?2 AND state = 'live'",
                    params![new_content, id],
                )
                .map_err(|e| PalaceError::Storage(format!("update_drawer: {e}")))?;
            if n == 0 {
                // Absent or tombstoned â€” nothing edited. tx rolls back on drop.
                return Err(PalaceError::NotFound { id: id.to_string() });
            }
            // Regenerate lexical chunks (chunk_in_tx deletes existing chunks
            // first â€” regeneration semantics), so the OLD content leaves the
            // FTS index and the NEW content becomes keyword-findable at once.
            retrieval::chunk_in_tx(&tx, id, new_content, retrieval::DEFAULT_CHUNK_SIZE)?;
            // Real CRDT op: fold the edit into the drawer's Automerge document.
            update_crdt_content_in_tx(&tx, id, new_content)?;
            tx.commit()
                .map_err(|e| PalaceError::Storage(format!("update_drawer tx commit: {e}")))?;
        }
        // Lock released before read-back.
        self.get(id)?.ok_or_else(|| {
            PalaceError::Storage(format!("read-back: {id} not found immediately after update"))
        })
    }

    /// Alias for [`update_drawer`], provided for naming parity with
    /// the predecessor store's `update_content`. Identical semantics.
    pub fn update_content(&self, id: &str, new_content: &str) -> Result<Drawer> {
        self.update_drawer(id, new_content)
    }

    /// Retire a drawer. Transitions `state` from `Live` to `Tombstoned` and
    /// clears `reindex_pending`. The row is NEVER physically deleted.
    ///
    /// **Real CRDT op** (parity with [`update_drawer`]): the retirement is folded
    /// into the drawer's Automerge doc in `crdt_docs` within the same transaction,
    /// so it advances the drawer's causal history and survives
    /// `produce_delta` â†’ `apply_remote_delta`. A bare SQL state flip would leave
    /// the AM doc stale, so the tombstone would not be recorded as a real op.
    pub fn tombstone(&self, id: &str) -> Result<()> {
        let mut guard = self.conn.lock().unwrap();
        let tx = guard
            .transaction()
            .map_err(|e| PalaceError::Storage(format!("tombstone tx begin: {e}")))?;
        let n = tx
            .execute(
                "UPDATE drawers SET state = 'tombstoned', reindex_pending = 0 WHERE id = ?1",
                params![id],
            )
            .map_err(|e| PalaceError::Storage(format!("tombstone: {e}")))?;
        if n == 0 {
            // Absent id â€” nothing retired. tx rolls back on drop.
            return Err(PalaceError::NotFound { id: id.to_string() });
        }
        // Tombstoned drawers are excluded from search â€” remove their chunks.
        tx.execute("DELETE FROM chunks WHERE drawer_id = ?1", params![id])
            .map_err(|e| PalaceError::Storage(format!("tombstone chunk delete: {e}")))?;
        tombstone_crdt_in_tx(&tx, id)?;
        tx.commit()
            .map_err(|e| PalaceError::Storage(format!("tombstone tx commit: {e}")))?;
        Ok(())
    }

    /// List live drawers newest-first (by UUID v7 id descending). Tombstoned
    /// rows are excluded.
    pub fn list(&self, limit: usize) -> Result<Vec<Drawer>> {
        let guard = self.conn.lock().unwrap();
        let sql = format!(
            "SELECT {COLS} FROM drawers WHERE state = 'live' ORDER BY id DESC LIMIT ?1"
        );
        let mut stmt = guard
            .prepare(&sql)
            .map_err(|e| PalaceError::Storage(format!("list prepare: {e}")))?;
        let rows = stmt
            .query_map(params![limit as i64], row_to_drawer)
            .map_err(|e| PalaceError::Storage(format!("list query: {e}")))?;
        let mut out = Vec::new();
        for r in rows {
            let inner = r.map_err(|e| PalaceError::Storage(e.to_string()))?;
            out.push(inner?);
        }
        Ok(out)
    }

    /// Enumerate the distinct wing labels that currently hold at least one live
    /// drawer, sorted ascending. Ported from the predecessor store's `list_wings`, adapted
    /// to the palace's flat model where wing/room are metadata columns on
    /// `drawers` rather than separate tables. Live-only, consistent with `list()`
    /// and `search()`: a wing whose only drawers are tombstoned does not appear.
    /// Deterministic whole-corpus pagination â€” the migration-verify instrument.
    /// Unlike `list(limit)` (newest-first, limit-only, cannot walk past `limit`),
    /// this pages EVERY live drawer in a stable `id ASC` order with limit+offset,
    /// so a verifier can enumerate the corpus and prove each drawer appears exactly
    /// once (none dropped, none duplicated). Optional wing/room filter; `None`
    /// matches all. MIG-3's per-drawer count-parity rides on this.
    pub fn list_paged(
        &self,
        wing: Option<&str>,
        room: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Drawer>> {
        let guard = self.conn.lock().unwrap();
        let sql = format!(
            "SELECT {COLS} FROM drawers \
             WHERE state = 'live' \
             AND (?1 IS NULL OR wing = ?1) \
             AND (?2 IS NULL OR room = ?2) \
             ORDER BY id ASC LIMIT ?3 OFFSET ?4"
        );
        let mut stmt = guard
            .prepare(&sql)
            .map_err(|e| PalaceError::Storage(format!("list_paged prepare: {e}")))?;
        let rows = stmt
            .query_map(params![wing, room, limit as i64, offset as i64], row_to_drawer)
            .map_err(|e| PalaceError::Storage(format!("list_paged query: {e}")))?;
        let mut out = Vec::new();
        for r in rows {
            let inner = r.map_err(|e| PalaceError::Storage(e.to_string()))?;
            out.push(inner?);
        }
        Ok(out)
    }

    /// Count live drawers matching an optional wing/room filter. Mirrors
    /// `list_paged`'s WHERE so a caller can report a real `total` for pagination:
    /// a nonexistent wing returns 0, a populated wing its true count -- never a
    /// silent 0 that conflates "empty" with "couldn't answer" (the list_drawers
    /// contract: a client paginating by `total` must get a real one).
    pub fn count_live(&self, wing: Option<&str>, room: Option<&str>) -> Result<usize> {
        let guard = self.conn.lock().unwrap();
        let n: i64 = guard
            .query_row(
                "SELECT COUNT(*) FROM drawers \
                 WHERE state = 'live' \
                 AND (?1 IS NULL OR wing = ?1) \
                 AND (?2 IS NULL OR room = ?2)",
                params![wing, room],
                |r| r.get(0),
            )
            .map_err(|e| PalaceError::Storage(format!("count_live: {e}")))?;
        Ok(n as usize)
    }

    pub fn list_wings(&self) -> Result<Vec<String>> {
        let guard = self.conn.lock().unwrap();
        let mut stmt = guard
            .prepare(
                "SELECT DISTINCT wing FROM drawers WHERE state = 'live' ORDER BY wing ASC",
            )
            .map_err(|e| PalaceError::Storage(format!("list_wings prepare: {e}")))?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(|e| PalaceError::Storage(format!("list_wings query: {e}")))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| PalaceError::Storage(e.to_string()))?);
        }
        Ok(out)
    }

    /// Enumerate the distinct room labels within `wing` that hold at least one
    /// live drawer, sorted ascending. Live-only, mirroring `list_wings`. An
    /// unknown wing returns an empty vec, never an error.
    pub fn list_rooms(&self, wing: &str) -> Result<Vec<String>> {
        let guard = self.conn.lock().unwrap();
        let mut stmt = guard
            .prepare(
                "SELECT DISTINCT room FROM drawers \
                 WHERE wing = ?1 AND state = 'live' ORDER BY room ASC",
            )
            .map_err(|e| PalaceError::Storage(format!("list_rooms prepare: {e}")))?;
        let rows = stmt
            .query_map(params![wing], |r| r.get::<_, String>(0))
            .map_err(|e| PalaceError::Storage(format!("list_rooms query: {e}")))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| PalaceError::Storage(e.to_string()))?);
        }
        Ok(out)
    }

    // â”€â”€ Meta (key-value config / introspection) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Read a value from the `meta` key-value table. Returns `Ok(None)` for a key
    /// that has never been set â€” an absent key is a valid state, not an error.
    ///
    /// Both user keys and the reserved system keys (`schema_version`,
    /// `embedding_model_version`) are readable through this method; only writes to
    /// the reserved keys are refused (see [`set_meta`](Self::set_meta)).
    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let guard = self.conn.lock().unwrap();
        match guard.query_row(
            "SELECT value FROM meta WHERE key = ?1",
            params![key],
            |r| r.get::<_, String>(0),
        ) {
            Ok(value) => Ok(Some(value)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(PalaceError::Storage(format!("get_meta: {e}"))),
        }
    }

    /// Write a user value into the `meta` key-value table (upsert).
    ///
    /// **Reserved-key guard (P5):** the reserved system keys `schema_version` and
    /// `embedding_model_version` are owned by the migration/embedding layers and
    /// are write-protected through this API. Attempting to set either returns
    /// [`PalaceError::ReservedMetaKey`] LOUDLY and writes nothing, so a caller
    /// cannot corrupt migration state by clobbering the version rows. The guard
    /// runs before any DB touch â€” the reserved value is never even momentarily
    /// overwritten.
    ///
    /// `meta` is for config/versions, never secrets (credential discipline).
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        if RESERVED_META_KEYS.contains(&key) {
            return Err(PalaceError::ReservedMetaKey { key: key.to_string() });
        }
        let guard = self.conn.lock().unwrap();
        guard
            .execute(
                "INSERT INTO meta (key, value) VALUES (?1, ?2) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![key, value],
            )
            .map_err(|e| PalaceError::Storage(format!("set_meta: {e}")))?;
        Ok(())
    }

    // â”€â”€ Diary (per-agent journal, thin convenience over drawers) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Append a diary entry for `agent`. Writes an ordinary drawer into wing
    /// `wing_<agent>`, room `diary`, **via the normal insert path** â€” so the entry
    /// is durable AND CRDT-syncing for free (brief Â§3): it rides the drawer sync
    /// path with no new merge rule.
    ///
    /// Topic and timestamp are stamped as drawer metadata, readable back cleanly:
    /// - `topic` (when `Some`) is stored in the drawer's `name` field.
    /// - the ISO-8601 (RFC 3339) creation timestamp is the drawer's `created_at`,
    ///   set by the insert path.
    ///
    /// The verbatim `entry` is the drawer `content` â€” content in, content out, no
    /// prefix parsing for the reader.
    pub fn diary_write(&self, agent: &str, entry: &str, topic: Option<&str>) -> Result<Drawer> {
        let wing = diary_wing(agent);
        let provenance = format!("diary:{agent}");
        match topic {
            Some(t) => self.insert_named(&wing, DIARY_ROOM, entry, &provenance, t),
            None => self.insert(&wing, DIARY_ROOM, entry, &provenance),
        }
    }

    /// Read the last `last_n` diary entries for `agent`, **NEWEST-FIRST**.
    ///
    /// Ordering is load-bearing (brief Â§4): this is the exact class of the
    /// 2026-05-30 `list_drawers` bug â€” a "check my recent entries" call that
    /// silently returned OLDEST-first, so callers saw stale entries and reported
    /// "nothing new." The ordering is fixed HERE, in the engine, under a gate
    /// (`diary_read_is_newest_first`), so no MCP wrapper can reintroduce it.
    /// UUID v7 ids sort chronologically, so `ORDER BY id DESC` is newest-first.
    ///
    /// `last_n` is the caller's explicit bound, never a silent truncation of a
    /// larger true set. An agent with no entries yields `Ok(vec![])` â€” an empty
    /// journal is a valid state, not an error (unlike an unknown KG entity).
    pub fn diary_read(&self, agent: &str, last_n: usize) -> Result<Vec<Drawer>> {
        let wing = diary_wing(agent);
        let guard = self.conn.lock().unwrap();
        let sql = format!(
            "SELECT {COLS} FROM drawers \
             WHERE wing = ?1 AND room = ?2 AND state = 'live' \
             ORDER BY id DESC LIMIT ?3"
        );
        let mut stmt = guard
            .prepare(&sql)
            .map_err(|e| PalaceError::Storage(format!("diary_read prepare: {e}")))?;
        let rows = stmt
            .query_map(params![wing, DIARY_ROOM, last_n as i64], row_to_drawer)
            .map_err(|e| PalaceError::Storage(format!("diary_read query: {e}")))?;
        let mut out = Vec::new();
        for r in rows {
            let inner = r.map_err(|e| PalaceError::Storage(e.to_string()))?;
            out.push(inner?);
        }
        Ok(out)
    }

    // â”€â”€ Tunnels (typed directed edges) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Create a typed directed edge `from --relation--> to`.
    ///
    /// Both endpoints are resolved through [`resolve`](Self::resolve), so a
    /// short/prefix id, a `wing/room#seq` handle, or a `wing/room:name` all work
    /// â€” never exact-id-only. If either endpoint does not resolve to an existing
    /// drawer the call fails **loudly** (`NotFound` / ambiguous `Storage` error);
    /// no silent dangling edges are ever written (dumb-safe invariant).
    ///
    /// Idempotent on the `UNIQUE(from, to, relation)` triple: a repeat create of
    /// the same triple reuses the existing edge's id (so there is always exactly
    /// one edge per triple) and refreshes its confidence/HLC. Re-creating a
    /// tombstoned edge revives it.
    ///
    /// The write goes through the existing CRDT path (`write_tunnel`, HLC-stamped
    /// via `hlc_now`), so the edge is a real mergeable op that survives
    /// `produce_delta` â†’ `apply_remote_delta` rather than a bare local SQL row.
    pub fn create_tunnel(
        &self,
        from: &str,
        to: &str,
        relation: &str,
        confidence: Option<f32>,
    ) -> Result<Tunnel> {
        // Resolve + validate BEFORE opening the write transaction: a bad endpoint
        // is a loud error, never a dangling edge.
        let from_id = self.resolve_endpoint(from)?;
        let to_id = self.resolve_endpoint(to)?;

        let record = {
            let mut guard = self.conn.lock().unwrap();
            let tx = guard
                .transaction()
                .map_err(|e| PalaceError::Storage(format!("create_tunnel tx begin: {e}")))?;
            // Idempotency: reuse the existing row's id for this triple so the
            // write updates the one edge in place rather than spawning a
            // duplicate. `write_tunnel` is INSERT OR REPLACE keyed on `id`.
            let existing = sync_impl::query_tunnel(&tx, &from_id, &to_id, relation)?;
            let id = existing
                .map(|t| t.id)
                .unwrap_or_else(|| Uuid::now_v7().to_string());
            let record = TunnelRecord {
                id,
                from_drawer_id: from_id,
                to_drawer_id: to_id,
                relation: relation.to_string(),
                confidence: confidence.unwrap_or(1.0),
                hlc: sync_impl::hlc_now(),
                tombstoned: false,
            };
            sync_impl::write_tunnel(&tx, &record)?;
            tx.commit()
                .map_err(|e| PalaceError::Storage(format!("create_tunnel tx commit: {e}")))?;
            record
        };
        Ok(Tunnel::from(record))
    }

    /// List every live edge touching `drawer_ref`, in **both** directions
    /// (`from = id OR to = id`). Tombstoned edges are excluded. The reference is
    /// resolved (prefix-tolerant); an unresolvable reference is a loud error.
    pub fn list_tunnels(&self, drawer_ref: &str) -> Result<Vec<Tunnel>> {
        let id = self.resolve_endpoint(drawer_ref)?;
        self.collect_tunnels(
            "(from_drawer_id = ?1 OR to_drawer_id = ?1) AND tombstoned = 0",
            params![id],
        )
    }

    /// Return every live edge touching `id`, in **both** directions
    /// (`from = id OR to = id`), by **EXACT id only** — no reference resolution,
    /// no prefix tolerance, no enumerate. This is the cross-tier walk's edge
    /// primitive: the tier-aware BFS in `simonides-mcp` calls it against every
    /// readable store for the same id, so — unlike [`list_tunnels`] — it must NOT
    /// route through `resolve_endpoint`, which requires the drawer resident in
    /// *this* store and would error in exactly the cross-tier store that holds
    /// the edge but not the drawer. Exact-id-only is also the tier-oracle guard:
    /// a prefix or list-all here would let a caller probe for ids resident in a
    /// tier it cannot read. Absence contributes an empty vec, never an error.
    pub fn tunnels_referencing(&self, id: &str) -> Result<Vec<Tunnel>> {
        self.collect_tunnels(
            "(from_drawer_id = ?1 OR to_drawer_id = ?1) AND tombstoned = 0",
            params![id],
        )
    }

    /// Find every live edge with the given `relation`, across the whole store.
    pub fn find_tunnels(&self, relation: &str) -> Result<Vec<Tunnel>> {
        self.collect_tunnels(
            "relation = ?1 AND tombstoned = 0",
            params![relation],
        )
    }

    /// Find every live edge with the given `relation` originating at
    /// `from_ref` (outgoing only). The reference is resolved (prefix-tolerant);
    /// an unresolvable reference is a loud error.
    pub fn find_tunnels_from(&self, from_ref: &str, relation: &str) -> Result<Vec<Tunnel>> {
        let from_id = self.resolve_endpoint(from_ref)?;
        self.collect_tunnels(
            "from_drawer_id = ?1 AND relation = ?2 AND tombstoned = 0",
            params![from_id, relation],
        )
    }

    /// Follow `relation` edges **outward** from `drawer_ref`, breadth-first, and
    /// return the drawers reached (deduped, in discovery order). Directed:
    /// hops go `from --relation--> to`. Tombstoned edges are never traversed.
    ///
    /// Bounded twice over: the walk stops at `depth` hops (itself clamped to
    /// [`MAX_TRAVERSAL_DEPTH`]), and a `visited` set guarantees termination even
    /// when the graph contains a cycle. `depth == 0` returns no drawers.
    pub fn follow_tunnels(
        &self,
        drawer_ref: &str,
        relation: &str,
        depth: usize,
    ) -> Result<Vec<Drawer>> {
        let start = self.resolve_endpoint(drawer_ref)?;
        let cap = depth.min(MAX_TRAVERSAL_DEPTH);

        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start.clone());
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((start, 0));
        let mut reached: Vec<String> = Vec::new();

        while let Some((node, d)) = queue.pop_front() {
            if d >= cap {
                continue;
            }
            let edges = self.collect_tunnels(
                "from_drawer_id = ?1 AND relation = ?2 AND tombstoned = 0",
                params![node, relation],
            )?;
            for edge in edges {
                if visited.insert(edge.to_drawer_id.clone()) {
                    reached.push(edge.to_drawer_id.clone());
                    queue.push_back((edge.to_drawer_id, d + 1));
                }
            }
        }
        self.load_drawers(&reached)
    }

    /// Breadth-first graph walk from `drawer_ref` across **all** live edges
    /// (both directions, any relation), returning the reachable drawers (deduped,
    /// in discovery order, excluding the start). Same double bound as
    /// `follow_tunnels`: `max_depth` clamped to [`MAX_TRAVERSAL_DEPTH`] plus a
    /// `visited` set for cycle-safety.
    pub fn traverse(&self, drawer_ref: &str, max_depth: usize) -> Result<Vec<Drawer>> {
        let start = self.resolve_endpoint(drawer_ref)?;
        let cap = max_depth.min(MAX_TRAVERSAL_DEPTH);

        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start.clone());
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((start, 0));
        let mut reached: Vec<String> = Vec::new();

        while let Some((node, d)) = queue.pop_front() {
            if d >= cap {
                continue;
            }
            let edges = self.collect_tunnels(
                "(from_drawer_id = ?1 OR to_drawer_id = ?1) AND tombstoned = 0",
                params![node],
            )?;
            for edge in edges {
                // Undirected step: the neighbour is whichever endpoint is not us.
                let neighbour = if edge.from_drawer_id == node {
                    edge.to_drawer_id
                } else {
                    edge.from_drawer_id
                };
                if visited.insert(neighbour.clone()) {
                    reached.push(neighbour.clone());
                    queue.push_back((neighbour, d + 1));
                }
            }
        }
        self.load_drawers(&reached)
    }

    /// Retire an edge by its tunnel id. Sets `tombstoned = 1`; the row is never
    /// physically deleted (matches the drawer tombstone discipline). Written
    /// through the CRDT path with a fresh HLC so the tombstone propagates on
    /// sync. An unknown tunnel id is a loud `NotFound`.
    pub fn tombstone_tunnel(&self, id: &str) -> Result<()> {
        let mut guard = self.conn.lock().unwrap();
        let tx = guard
            .transaction()
            .map_err(|e| PalaceError::Storage(format!("tombstone_tunnel tx begin: {e}")))?;
        // Load the existing edge so from/to/relation/confidence are preserved
        // through the tombstone write. Missing -> loud NotFound.
        let mut record = {
            let sql = format!("SELECT {TUNNEL_COLS} FROM tunnels WHERE id = ?1 LIMIT 1");
            let mut stmt = tx
                .prepare(&sql)
                .map_err(|e| PalaceError::Storage(format!("tombstone_tunnel prepare: {e}")))?;
            let mut rows = stmt
                .query_map(params![id], sync_impl::row_to_tunnel)
                .map_err(|e| PalaceError::Storage(format!("tombstone_tunnel query: {e}")))?;
            match rows.next() {
                Some(r) => r.map_err(|e| PalaceError::Storage(e.to_string()))?,
                None => return Err(PalaceError::NotFound { id: id.to_string() }),
            }
        };
        record.tombstoned = true;
        record.hlc = sync_impl::hlc_now();
        sync_impl::write_tunnel(&tx, &record)?;
        tx.commit()
            .map_err(|e| PalaceError::Storage(format!("tombstone_tunnel tx commit: {e}")))?;
        Ok(())
    }

    /// Resolve a drawer reference to its full id, failing loudly. `Found` â†’
    /// the id; `Ambiguous` â†’ a descriptive `Storage` error; `NotFound` â†’
    /// `NotFound`. Shared by every tunnel entry point so all lookups route
    /// through `resolve()` (prefix-tolerant), never exact-id-only.
    fn resolve_endpoint(&self, reference: &str) -> Result<String> {
        match self.resolve(reference)? {
            ResolveOutcome::Found(d) => Ok(d.id),
            ResolveOutcome::Ambiguous { candidates, .. } => Err(PalaceError::Storage(format!(
                "ambiguous drawer reference '{reference}': matched {} drawers â€” be more specific",
                candidates.len()
            ))),
            ResolveOutcome::NotFound { .. } => {
                Err(PalaceError::NotFound { id: reference.to_string() })
            }
        }
    }

    /// Run a tunnel query with the given `WHERE` clause + params and decode the
    /// rows into public [`Tunnel`]s. Ordered by `id` for a stable result.
    fn collect_tunnels(
        &self,
        where_clause: &str,
        params: impl rusqlite::Params,
    ) -> Result<Vec<Tunnel>> {
        let guard = self.conn.lock().unwrap();
        let sql =
            format!("SELECT {TUNNEL_COLS} FROM tunnels WHERE {where_clause} ORDER BY id");
        let mut stmt = guard
            .prepare(&sql)
            .map_err(|e| PalaceError::Storage(format!("collect_tunnels prepare: {e}")))?;
        let rows = stmt
            .query_map(params, sync_impl::row_to_tunnel)
            .map_err(|e| PalaceError::Storage(format!("collect_tunnels query: {e}")))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(Tunnel::from(r.map_err(|e| PalaceError::Storage(e.to_string()))?));
        }
        Ok(out)
    }

    /// Load drawers by id in the given order, skipping any that are absent
    /// (defensive â€” a live edge should always point at a real drawer, but a
    /// dangling id must never panic the walk).
    fn load_drawers(&self, ids: &[String]) -> Result<Vec<Drawer>> {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(d) = self.get(id)? {
                out.push(d);
            }
        }
        Ok(out)
    }

    /// Return a health snapshot.
    pub fn health(&self) -> Result<HealthReport> {
        // Computed before the guard is taken: `stale_closet_rooms` acquires the
        // connection lock itself, and this Mutex is not reentrant.
        let closets_stale = self.stale_closet_rooms()?.len() as u64;

        let guard = self.conn.lock().unwrap();
        let total: i64 = guard
            .query_row("SELECT COUNT(*) FROM drawers", [], |row| row.get(0))
            .map_err(|e| PalaceError::Storage(format!("health total: {e}")))?;
        let live: i64 = guard
            .query_row(
                "SELECT COUNT(*) FROM drawers WHERE state = 'live'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| PalaceError::Storage(format!("health live: {e}")))?;
        let tombstoned: i64 = guard
            .query_row(
                "SELECT COUNT(*) FROM drawers WHERE state = 'tombstoned'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| PalaceError::Storage(format!("health tombstoned: {e}")))?;
        let reindex_pending: i64 = guard
            .query_row(
                "SELECT COUNT(*) FROM drawers WHERE reindex_pending = 1",
                [],
                |row| row.get(0),
            )
            .map_err(|e| PalaceError::Storage(format!("health reindex_pending: {e}")))?;
        let kg_entities: i64 = guard
            .query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))
            .map_err(|e| PalaceError::Storage(format!("health kg_entities: {e}")))?;
        let kg_triples: i64 = guard
            .query_row("SELECT COUNT(*) FROM triples", [], |row| row.get(0))
            .map_err(|e| PalaceError::Storage(format!("health kg_triples: {e}")))?;
        let closets: i64 = guard
            .query_row("SELECT COUNT(*) FROM closets", [], |row| row.get(0))
            .map_err(|e| PalaceError::Storage(format!("health closets: {e}")))?;
        let chunks_total: i64 = guard
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .map_err(|e| PalaceError::Storage(format!("health chunks_total: {e}")))?;
        let chunks_embedded: i64 = guard
            .query_row(
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .map_err(|e| PalaceError::Storage(format!("health chunks_embedded: {e}")))?;
        let embedding_model_version: String = guard
            .query_row(
                "SELECT value FROM meta WHERE key = 'embedding_model_version'",
                [],
                |row| row.get(0),
            )
            .unwrap_or_default();
        Ok(HealthReport {
            total: total as u64,
            live: live as u64,
            tombstoned: tombstoned as u64,
            reindex_pending: reindex_pending as u64,
            kg_entities: kg_entities as u64,
            kg_triples: kg_triples as u64,
            closets: closets as u64,
            closets_stale,
            chunks_total: chunks_total as u64,
            chunks_embedded: chunks_embedded as u64,
            embedding_model_version,
        })
    }

    /// Embed one or more texts using the lazily-initialised BGE-small-en-v1.5
    /// session.  Callers should treat `Err` as "embedder unavailable" and fall
    /// back gracefully (e.g. BM25-only search).
    pub(crate) fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut guard = self.embedder.lock().unwrap();
        if guard.is_none() {
            *guard = Some(embedder::Embedder::try_new()?);
        }
        guard.as_mut().unwrap().embed(texts)
    }

    /// Embed a batch of chunk texts. The server binary sets SIMONIDES_EMBED_WORKER
    /// to its own path at startup, so each batch is embedded in a
    /// CHILD process that exits afterward -- bounded, reclaimable memory, the only
    /// path that survives the ONNX runtime's process-global retention. When unset
    /// (tests, small in-process use) it falls back to the in-process embedder.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match std::env::var("SIMONIDES_EMBED_WORKER") {
            Ok(exe) if !exe.is_empty() => embedder::embed_via_subprocess(&exe, texts),
            _ => self.embed_texts(texts),
        }
    }

    /// Embed all `reindex_pending` live drawers and store the resulting vectors
    /// as BLOBs in `chunks.embedding`.  Clears `reindex_pending` on each drawer
    /// after its chunks are embedded and stored atomically.
    ///
    /// Lexical chunking is synchronous (happens in `insert()`); this method
    /// handles the async/background vector half.  Calling it when no drawers
    /// are pending is a fast no-op.
    pub fn rebuild_vectors(&self) -> Result<RebuildReport> {
        let mut report = RebuildReport::default();

        // Batch size = chunks embedded per CHILD process (see the embedder worker).
        // Each batch is embedded in a subprocess that exits afterward, so peak RSS
        // is bounded by one batch regardless of corpus size -- tunable per device
        // via SIMONIDES_EMBED_BATCH (default 2000; lower it on a small machine).
        let batch_size: usize = std::env::var("SIMONIDES_EMBED_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(2000);

        // If the stored embedding model differs from the running one, every stored
        // vector is stale -> re-embed all. Otherwise only chunks with NO blob are
        // embedded; chunks that already carry one are reused untouched.
        let model_version_stale: bool = {
            let guard = self.conn.lock().unwrap();
            let stored: String = guard
                .query_row(
                    "SELECT value FROM meta WHERE key = 'embedding_model_version'",
                    [],
                    |r| r.get(0),
                )
                .unwrap_or_default();
            stored != embedder::MODEL_ID
        };

        // Pending live drawers + their total chunk count (report.reused = total - embedded).
        let (pending_drawers, total_pending_chunks): (usize, usize) = {
            let guard = self.conn.lock().unwrap();
            let d = guard
                .query_row(
                    "SELECT COUNT(*) FROM drawers WHERE reindex_pending = 1 AND state = 'live'",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .unwrap_or(0) as usize;
            let c = guard
                .query_row(
                    "SELECT COUNT(*) FROM chunks c JOIN drawers d ON d.id = c.drawer_id \
                     WHERE d.reindex_pending = 1 AND d.state = 'live'",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .unwrap_or(0) as usize;
            (d, c)
        };
        report.drawers = pending_drawers;
        if pending_drawers == 0 {
            return Ok(report);
        }

        // Make sure the model is on disk before any worker starts. A worker that
        // finds it missing downloads it itself, and two processes fetching the
        // same file at once (this one for a search, a worker for a rebuild) fail.
        // Loading it here goes through the in-process init lock, which is shared
        // with search, so afterwards every worker finds a complete cache.
        if std::env::var_os("SIMONIDES_EMBED_WORKER").is_some() {
            self.embed_texts(&["warm-up"])?;
        }
        // Model change: null the stored vectors of pending drawers so the
        // null-driven loop below re-embeds them all.
        if model_version_stale {
            let guard = self.conn.lock().unwrap();
            guard
                .execute(
                    "UPDATE chunks SET embedding = NULL WHERE drawer_id IN \
                     (SELECT id FROM drawers WHERE reindex_pending = 1 AND state = 'live')",
                    [],
                )
                .map_err(|e| PalaceError::Storage(format!("rebuild null stale: {e}")))?;
        }

        // Null-driven, streamed, memory-bounded loop. Each iteration pulls at most
        // `batch_size` not-yet-embedded chunks, embeds them in a fresh child process
        // (which exits and frees all retained native memory), and stores the blobs.
        // Resumable: a crash mid-run leaves embedded chunks non-null, so re-running
        // simply continues with what remains.
        loop {
            let batch: Vec<(String, String)> = {
                let guard = self.conn.lock().unwrap();
                let mut stmt = guard
                    .prepare(
                        "SELECT c.id, c.content FROM chunks c \
                         JOIN drawers d ON d.id = c.drawer_id \
                         WHERE d.reindex_pending = 1 AND d.state = 'live' \
                         AND c.embedding IS NULL \
                         LIMIT ?1",
                    )
                    .map_err(|e| PalaceError::Storage(format!("rebuild batch prepare: {e}")))?;
                let rows = stmt
                    .query_map(params![batch_size as i64], |r| {
                        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                    })
                    .map_err(|e| PalaceError::Storage(format!("rebuild batch query: {e}")))?;
                let mut out = Vec::new();
                for r in rows {
                    out.push(r.map_err(|e| PalaceError::Storage(e.to_string()))?);
                }
                out
            };
            if batch.is_empty() {
                break;
            }

            let texts: Vec<&str> = batch.iter().map(|(_, c)| c.as_str()).collect();
            let vecs = self.embed_batch(&texts)?;
            if vecs.len() != batch.len() {
                return Err(PalaceError::Storage(format!(
                    "embed returned {} vectors for {} chunks",
                    vecs.len(),
                    batch.len()
                )));
            }

            {
                let mut guard = self.conn.lock().unwrap();
                let tx = guard
                    .transaction()
                    .map_err(|e| PalaceError::Storage(format!("rebuild tx begin: {e}")))?;
                for ((id, _), v) in batch.iter().zip(vecs.iter()) {
                    let blob = embedder::vec_to_blob(v);
                    tx.execute(
                        "UPDATE chunks SET embedding = ?1 WHERE id = ?2",
                        params![blob, id],
                    )
                    .map_err(|e| PalaceError::Storage(format!("rebuild store blob: {e}")))?;
                }
                tx.commit()
                    .map_err(|e| PalaceError::Storage(format!("rebuild tx commit: {e}")))?;
            }
            report.embedded += batch.len();
        }

        // Clear the pending flag for every drawer that now has no un-embedded chunk
        // -- counts the FACT (no null embeddings), not a flag set blindly.
        {
            let guard = self.conn.lock().unwrap();
            guard
                .execute(
                    "UPDATE drawers SET reindex_pending = 0 \
                     WHERE state = 'live' AND reindex_pending = 1 \
                     AND NOT EXISTS (SELECT 1 FROM chunks \
                        WHERE chunks.drawer_id = drawers.id AND chunks.embedding IS NULL)",
                    [],
                )
                .map_err(|e| PalaceError::Storage(format!("rebuild clear pending: {e}")))?;
        }

        report.reused = total_pending_chunks.saturating_sub(report.embedded);

        // Record which model produced these embeddings.
        {
            let guard = self.conn.lock().unwrap();
            guard
                .execute(
                    "UPDATE meta SET value = ?1 WHERE key = 'embedding_model_version'",
                    params![embedder::MODEL_ID],
                )
                .map_err(|e| {
                    PalaceError::Storage(format!("rebuild set embedding_model_version: {e}"))
                })?;
        }

        Ok(report)
    }

    /// Extract-only KG build over EVERY live drawer (independent of reindex_pending,
    /// which drives embedding). Idempotent: kg_upsert_entity dedups by
    /// canonical_name COLLATE NOCASE, kg_add_triple dedups by fact key, so
    /// re-running yields the same graph. Does NOT embed. This is the "graph" half
    /// that the flat-vs-graph A/B isolates from the vectors-only "flat" half.
    pub fn rebuild_kg(&self) -> Result<KgBuildReport> {
        let live: Vec<(String, String)> = {
            let guard = self.conn.lock().unwrap();
            let mut stmt = guard
                .prepare("SELECT id, content FROM drawers WHERE state = 'live'")
                .map_err(|e| PalaceError::Storage(format!("rebuild_kg list: {e}")))?;
            let rows = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
                .map_err(|e| PalaceError::Storage(format!("rebuild_kg query: {e}")))?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(|e| PalaceError::Storage(e.to_string()))?);
            }
            out
        };
        let mut report = KgBuildReport::default();
        for (drawer_id, content) in &live {
            report.drawers_scanned += 1;
            if let Err(e) = self.kg_extract_drawer(drawer_id, content) {
                tracing::warn!("rebuild_kg: extraction failed for {drawer_id}: {e}");
            }
        }
        if let Err(e) = self.kg_rebuild_projection() {
            tracing::warn!("rebuild_kg: projection rebuild failed: {e}");
        }
        {
            let guard = self.conn.lock().unwrap();
            report.entities = guard
                .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get::<_, i64>(0))
                .unwrap_or(0) as usize;
            report.triples = guard
                .query_row("SELECT COUNT(*) FROM triples", [], |r| r.get::<_, i64>(0))
                .unwrap_or(0) as usize;
        }
        Ok(report)
    }

    /// Backward-compatible combined pass: vectors THEN KG. Existing callers keep
    /// their embed+extract behaviour; the split lets the A/B isolate the halves.
    pub fn rebuild_indexes(&self) -> Result<RebuildReport> {
        let report = self.rebuild_vectors()?;
        self.rebuild_kg()?;
        Ok(report)
    }

    // â”€â”€ Resolve â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Resolve a reference to a drawer. Never returns a silent not-found.
    ///
    /// Accepted reference forms (tried in order):
    /// 1. **Full UUID** â€” 36-char hex-with-hyphens â†’ exact id lookup.
    /// 2. **Handle** â€” `wing/room#N` (N is the per-room seq) â†’ seq lookup.
    /// 3. **Named** â€” `wing/room:name` â†’ name column lookup.
    /// 4. **ID prefix** â€” any shorter hex-and-hyphen string â†’ LIKE prefix scan.
    ///
    /// Returns `Found`, `Ambiguous{candidates}`, or `NotFound{nearest}`.
    pub fn resolve(&self, reference: &str) -> Result<ResolveOutcome> {
        let query = reference.to_string();

        if is_full_uuid(reference) {
            return match self.get(reference)? {
                Some(d) => Ok(ResolveOutcome::Found(d)),
                None => Ok(ResolveOutcome::NotFound {
                    query,
                    nearest: self.recent_live(5)?,
                }),
            };
        }

        if let Some(hash_pos) = reference.find('#') {
            let location = &reference[..hash_pos];
            let seq_str = &reference[hash_pos + 1..];
            if let Ok(seq) = seq_str.parse::<i64>() {
                if let Some(slash_pos) = location.find('/') {
                    let wing = &location[..slash_pos];
                    let room = &location[slash_pos + 1..];
                    return self.resolve_handle(&query, wing, room, seq);
                }
            }
            // Malformed handle â€” fall through to prefix search.
        }

        if let Some(colon_pos) = reference.find(':') {
            let location = &reference[..colon_pos];
            let name = &reference[colon_pos + 1..];
            if let Some(slash_pos) = location.find('/') {
                let wing = &location[..slash_pos];
                let room = &location[slash_pos + 1..];
                return self.resolve_named_form(&query, wing, room, name);
            }
            // Malformed named form â€” fall through to prefix search.
        }

        // Default: UUID prefix scan.
        self.resolve_prefix(&query, reference)
    }

    fn resolve_handle(
        &self,
        query: &str,
        wing: &str,
        room: &str,
        seq: i64,
    ) -> Result<ResolveOutcome> {
        // Nested scope: guard + stmt drop before the fallback re-acquires the lock.
        let drawers = {
            let guard = self.conn.lock().unwrap();
            let sql = format!(
                "SELECT {COLS} FROM drawers WHERE wing = ?1 AND room = ?2 AND seq = ?3"
            );
            let mut stmt = guard
                .prepare(&sql)
                .map_err(|e| PalaceError::Storage(format!("resolve_handle prepare: {e}")))?;
            let rows = stmt
                .query_map(params![wing, room, seq], row_to_drawer)
                .map_err(|e| PalaceError::Storage(format!("resolve_handle query: {e}")))?;
            collect_rows(rows)?
        };
        match drawers.len() {
            0 => Ok(ResolveOutcome::NotFound {
                query: query.to_string(),
                nearest: self.room_nearest(wing, room, seq, 5)?,
            }),
            1 => Ok(ResolveOutcome::Found(drawers.into_iter().next().unwrap())),
            _ => Ok(ResolveOutcome::Ambiguous {
                query: query.to_string(),
                candidates: drawers,
            }),
        }
    }

    fn resolve_named_form(
        &self,
        query: &str,
        wing: &str,
        room: &str,
        name: &str,
    ) -> Result<ResolveOutcome> {
        let drawers = {
            let guard = self.conn.lock().unwrap();
            let sql = format!(
                "SELECT {COLS} FROM drawers WHERE wing = ?1 AND room = ?2 AND name = ?3"
            );
            let mut stmt = guard
                .prepare(&sql)
                .map_err(|e| PalaceError::Storage(format!("resolve_named prepare: {e}")))?;
            let rows = stmt
                .query_map(params![wing, room, name], row_to_drawer)
                .map_err(|e| PalaceError::Storage(format!("resolve_named query: {e}")))?;
            collect_rows(rows)?
        };
        match drawers.len() {
            0 => Ok(ResolveOutcome::NotFound {
                query: query.to_string(),
                nearest: self.room_live(wing, room, 5)?,
            }),
            1 => Ok(ResolveOutcome::Found(drawers.into_iter().next().unwrap())),
            _ => Ok(ResolveOutcome::Ambiguous {
                query: query.to_string(),
                candidates: drawers,
            }),
        }
    }

    fn resolve_prefix(&self, query: &str, prefix: &str) -> Result<ResolveOutcome> {
        let drawers = {
            let guard = self.conn.lock().unwrap();
            let pattern = format!("{prefix}%");
            let sql = format!("SELECT {COLS} FROM drawers WHERE id LIKE ?1 ORDER BY id");
            let mut stmt = guard
                .prepare(&sql)
                .map_err(|e| PalaceError::Storage(format!("resolve_prefix prepare: {e}")))?;
            let rows = stmt
                .query_map(params![pattern], row_to_drawer)
                .map_err(|e| PalaceError::Storage(format!("resolve_prefix query: {e}")))?;
            collect_rows(rows)?
        };
        match drawers.len() {
            0 => Ok(ResolveOutcome::NotFound {
                query: query.to_string(),
                nearest: self.recent_live(5)?,
            }),
            1 => Ok(ResolveOutcome::Found(drawers.into_iter().next().unwrap())),
            _ => Ok(ResolveOutcome::Ambiguous {
                query: query.to_string(),
                candidates: drawers,
            }),
        }
    }

    /// Up to `limit` most-recently-inserted live drawers (by id DESC).
    fn recent_live(&self, limit: usize) -> Result<Vec<Drawer>> {
        let guard = self.conn.lock().unwrap();
        let sql = format!(
            "SELECT {COLS} FROM drawers WHERE state = 'live' ORDER BY id DESC LIMIT ?1"
        );
        let mut stmt = guard
            .prepare(&sql)
            .map_err(|e| PalaceError::Storage(format!("recent_live prepare: {e}")))?;
        let rows = stmt
            .query_map(params![limit as i64], row_to_drawer)
            .map_err(|e| PalaceError::Storage(format!("recent_live query: {e}")))?;
        collect_rows(rows)
    }

    /// Live drawers in a specific room, ordered by seq DESC.
    /// Falls back to `recent_live` when the room has no live drawers.
    fn room_live(&self, wing: &str, room: &str, limit: usize) -> Result<Vec<Drawer>> {
        let results = {
            let guard = self.conn.lock().unwrap();
            let sql = format!(
                "SELECT {COLS} FROM drawers \
                 WHERE wing = ?1 AND room = ?2 AND state = 'live' \
                 ORDER BY seq DESC LIMIT ?3"
            );
            let mut stmt = guard
                .prepare(&sql)
                .map_err(|e| PalaceError::Storage(format!("room_live prepare: {e}")))?;
            let rows = stmt
                .query_map(params![wing, room, limit as i64], row_to_drawer)
                .map_err(|e| PalaceError::Storage(format!("room_live query: {e}")))?;
            collect_rows(rows)?
        };
        if results.is_empty() {
            self.recent_live(limit)
        } else {
            Ok(results)
        }
    }

    /// Live drawers in a room ordered by proximity to `target_seq`.
    /// Falls back to `recent_live` when the room has no live drawers.
    fn room_nearest(&self, wing: &str, room: &str, target_seq: i64, limit: usize) -> Result<Vec<Drawer>> {
        let results = {
            let guard = self.conn.lock().unwrap();
            let sql = format!(
                "SELECT {COLS} FROM drawers \
                 WHERE wing = ?1 AND room = ?2 AND state = 'live' AND seq IS NOT NULL \
                 ORDER BY ABS(seq - ?3) ASC LIMIT ?4"
            );
            let mut stmt = guard
                .prepare(&sql)
                .map_err(|e| PalaceError::Storage(format!("room_nearest prepare: {e}")))?;
            let rows = stmt
                .query_map(params![wing, room, target_seq, limit as i64], row_to_drawer)
                .map_err(|e| PalaceError::Storage(format!("room_nearest query: {e}")))?;
            collect_rows(rows)?
        };
        if results.is_empty() {
            self.recent_live(limit)
        } else {
            Ok(results)
        }
    }

    fn verify_and_return(
        &self,
        id: &str,
        expected_wing: &str,
        expected_room: &str,
    ) -> Result<Drawer> {
        let drawer = self.get(id)?.ok_or_else(|| {
            PalaceError::Storage(format!("read-back: {id} not found immediately after write"))
        })?;
        if drawer.wing != expected_wing || drawer.room != expected_room {
            return Err(PalaceError::WriteVerificationFailed {
                id: id.to_string(),
                expected_wing: expected_wing.to_string(),
                expected_room: expected_room.to_string(),
                actual_wing: drawer.wing,
                actual_room: drawer.room,
            });
        }
        Ok(drawer)
    }

    // â”€â”€ Vault operations (delegate to vault free-fns on the shared conn) â”€â”€â”€â”€â”€â”€

    /// Store bytes and return their SHA-256 hex hash. Idempotent â€” identical
    /// bytes stored twice result in one physical blob. Blobs are encrypted at
    /// rest with ChaCha20-Poly1305 using the key supplied to `open()`.
    pub fn vault_put(
        &self,
        bytes: &[u8],
        mime: &str,
        filename: Option<&str>,
        provenance: &str,
    ) -> std::result::Result<String, vault::VaultError> {
        vault::put(&self.conn, &self.vault_root, &self.vault_cipher, bytes, mime, filename, provenance)
    }

    /// Retrieve the bytes for a content hash. Returns `BlobNotFound` if
    /// no blob exists at the computed path. Decrypts the stored ciphertext.
    pub fn vault_get(
        &self,
        content_hash: &str,
    ) -> std::result::Result<Vec<u8>, vault::VaultError> {
        vault::get(&self.vault_cipher, &self.vault_root, content_hash)
    }

    /// Return `true` if a `files` row exists for `content_hash`.
    pub fn vault_exists(
        &self,
        content_hash: &str,
    ) -> std::result::Result<bool, vault::VaultError> {
        vault::exists(&self.conn, content_hash)
    }

    /// Walk every `files` row, decrypt each blob, re-hash the plaintext, and
    /// collect anomalies. AEAD tag failure or plaintext hash mismatch is
    /// reported as `VerifyIssue::Corrupted`.
    pub fn vault_verify(
        &self,
    ) -> std::result::Result<vault::VerifyReport, vault::VaultError> {
        vault::verify(&self.conn, &self.vault_root, &self.vault_cipher)
    }

    /// Compute the on-disk path for a blob from its hex hash.
    pub fn vault_blob_path(&self, hex_hash: &str) -> PathBuf {
        vault::blob_path(&self.vault_root, hex_hash)
    }
}

/// P5: the wing that holds an agent's diary. One place so `diary_write` and
/// `diary_read` agree on the location and no wrapper re-derives it inconsistently.
fn diary_wing(agent: &str) -> String {
    format!("wing_{agent}")
}

fn row_to_drawer(row: &rusqlite::Row<'_>) -> rusqlite::Result<Result<Drawer>> {
    let id: String = row.get(0)?;
    let wing: String = row.get(1)?;
    let room: String = row.get(2)?;
    let content: String = row.get(3)?;
    let provenance: String = row.get(4)?;
    let state_str: String = row.get(5)?;
    let reindex_raw: i32 = row.get(6)?;
    let created_at_str: String = row.get(7)?;
    // Columns 8 and 9 are COALESCE(seq,0) and name â€” added in schema v2.
    let seq: i64 = row.get(8)?;
    let name: Option<String> = row.get(9)?;

    let result = (|| -> Result<Drawer> {
        let state = match state_str.as_str() {
            "live" => DrawerState::Live,
            "tombstoned" => DrawerState::Tombstoned,
            other => return Err(PalaceError::Storage(format!("unknown state: {other}"))),
        };
        let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
            .map_err(|e| PalaceError::Storage(format!("created_at parse: {e}")))?
            .with_timezone(&chrono::Utc);
        Ok(Drawer {
            id,
            wing,
            room,
            content,
            provenance,
            state,
            reindex_pending: reindex_raw != 0,
            created_at,
            seq,
            name,
        })
    })();
    Ok(result)
}

/// Collect `query_map` rows into a `Vec<Drawer>`, propagating both rusqlite
/// errors and inner `PalaceError`s.
fn collect_rows(
    rows: impl Iterator<Item = rusqlite::Result<Result<Drawer>>>,
) -> Result<Vec<Drawer>> {
    let mut out = Vec::new();
    for r in rows {
        let inner = r.map_err(|e| PalaceError::Storage(e.to_string()))?;
        out.push(inner?);
    }
    Ok(out)
}

/// Fold a content edit into the drawer's Automerge document inside the write
/// transaction. This is what makes `update_drawer` a real CRDT op rather than a
/// bare SQL UPDATE: the merged content that a peer resolves on
/// `apply_remote_delta` is read from `crdt_docs` (the AM doc's `content` key),
/// not from the drawers row â€” so if the AM doc is left stale the edit is lost.
///
/// Loads the drawer's existing AM doc when present (repeated edits and any prior
/// sync history stay on one causal chain); otherwise bootstraps a fresh doc.
/// The `content` scalar-string key at ROOT matches the shape simonides-sync's
/// `merge_am_docs_with_content` reads. Doc bytes live in the sqlcipher DB â€”
/// never a plaintext sidecar.
fn update_crdt_content_in_tx(
    tx: &rusqlite::Transaction<'_>,
    drawer_id: &str,
    new_content: &str,
) -> Result<()> {
    use automerge::transaction::Transactable;
    fold_into_crdt_doc_in_tx(tx, drawer_id, "update_drawer", |doc| {
        doc.put(automerge::ROOT, "content", new_content)
            .map_err(|e| PalaceError::Storage(format!("update_drawer crdt put: {e}")))
    })
}

/// Fold a move (wing/room relocation) into the drawer's Automerge document
/// inside the write transaction. Same rationale as [`update_crdt_content_in_tx`]:
/// without this a `move_drawer` is a bare SQL UPDATE that never advances the
/// drawer's CRDT causal history, so the relocation isn't recorded as a real op
/// and can't propagate deterministically the way P1's edit and P2's tunnels do.
/// The `wing`/`room` scalar-string keys at ROOT sit alongside the `content` key;
/// the sync-crate merge sources wing/room from the SQL columns (which this tx
/// also updates), so folding them here keeps the AM doc a faithful, complete
/// record of every mutation. Doc bytes stay inside the sqlcipher DB.
fn move_crdt_location_in_tx(
    tx: &rusqlite::Transaction<'_>,
    drawer_id: &str,
    new_wing: &str,
    new_room: &str,
) -> Result<()> {
    use automerge::transaction::Transactable;
    fold_into_crdt_doc_in_tx(tx, drawer_id, "move_drawer", |doc| {
        doc.put(automerge::ROOT, "wing", new_wing)
            .map_err(|e| PalaceError::Storage(format!("move_drawer crdt put wing: {e}")))?;
        doc.put(automerge::ROOT, "room", new_room)
            .map_err(|e| PalaceError::Storage(format!("move_drawer crdt put room: {e}")))
    })
}

/// Fold a tombstone (retirement) into the drawer's Automerge document inside the
/// write transaction. Same rationale as [`update_crdt_content_in_tx`]: a bare SQL
/// state flip never advances the CRDT doc, so the retirement isn't a real op and
/// can't propagate deterministically. Sets a `tombstoned = true` boolean scalar
/// at ROOT. The row itself is never deleted (tombstone is a state transition);
/// the AM doc bytes stay inside the sqlcipher DB.
fn tombstone_crdt_in_tx(tx: &rusqlite::Transaction<'_>, drawer_id: &str) -> Result<()> {
    use automerge::transaction::Transactable;
    fold_into_crdt_doc_in_tx(tx, drawer_id, "tombstone", |doc| {
        doc.put(automerge::ROOT, "tombstoned", true)
            .map_err(|e| PalaceError::Storage(format!("tombstone crdt put: {e}")))
    })
}

/// Shared spine for every drawer mutation that must be a real CRDT op (content
/// edit, move, tombstone): load the drawer's existing Automerge doc (so repeated
/// mutations and any prior sync history stay on one causal chain) or bootstrap a
/// fresh one, apply `mutate`, then persist the new bytes back to `crdt_docs`
/// (inside the sqlcipher DB â€” never a plaintext sidecar). `label` names the
/// caller for error messages.
fn fold_into_crdt_doc_in_tx(
    tx: &rusqlite::Transaction<'_>,
    drawer_id: &str,
    label: &str,
    mutate: impl FnOnce(&mut automerge::AutoCommit) -> Result<()>,
) -> Result<()> {
    use automerge::AutoCommit;

    let existing: rusqlite::Result<Vec<u8>> = tx.query_row(
        "SELECT doc_bytes FROM crdt_docs WHERE drawer_id = ?1",
        params![drawer_id],
        |r| r.get(0),
    );
    let mut doc = match existing {
        Ok(bytes) if !bytes.is_empty() => AutoCommit::load(&bytes)
            .map_err(|e| PalaceError::Storage(format!("{label} crdt load: {e}")))?,
        Ok(_) => AutoCommit::new(),
        Err(rusqlite::Error::QueryReturnedNoRows) => AutoCommit::new(),
        Err(e) => return Err(PalaceError::Storage(format!("{label} crdt read: {e}"))),
    };

    mutate(&mut doc)?;

    let bytes = doc.save();
    let now = chrono::Utc::now().to_rfc3339();
    tx.execute(
        "INSERT INTO crdt_docs (drawer_id, doc_bytes, updated_at) VALUES (?1, ?2, ?3) \
         ON CONFLICT(drawer_id) DO UPDATE SET \
             doc_bytes = excluded.doc_bytes, updated_at = excluded.updated_at",
        params![drawer_id, bytes, now],
    )
    .map_err(|e| PalaceError::Storage(format!("{label} crdt upsert: {e}")))?;
    Ok(())
}

/// Returns `true` iff `s` is a well-formed UUID string (36 chars, hyphens at
/// positions 8, 13, 18, 23, all other chars lowercase hex).
fn is_full_uuid(s: &str) -> bool {
    if s.len() != 36 {
        return false;
    }
    s.chars().enumerate().all(|(i, c)| match i {
        8 | 13 | 18 | 23 => c == '-',
        _ => c.is_ascii_hexdigit(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// The well-known SQLite3 file-format magic. A sqlcipher-encrypted file
    /// must NOT start with this sequence.
    const SQLITE_MAGIC: &[u8; 16] = b"SQLite format 3\0";

    const TEST_KEY: &str = "palace-test-fixed-key-for-all-tests";
    const CORRECT_KEY: &str = "palace-test-sqlcipher-key-correct";
    const WRONG_KEY: &str = "palace-test-sqlcipher-key-WRONG";

    fn store() -> (PalaceStore, TempDir) {
        let tmp = TempDir::new().expect("tempdir");
        let s = PalaceStore::open_in_memory(tmp.path(), TEST_KEY).expect("open in-memory store");
        (s, tmp)
    }

    /// Test-only backdoor: inserts a row directly into the DB, bypassing the
    /// verifiable-write layer. Used only to construct wrong-target scenarios
    /// that test the verification catches the mismatch.
    fn insert_raw(store: &PalaceStore, id: &str, wing: &str, room: &str, content: &str) {
        let guard = store.conn.lock().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        guard
            .execute(
                "INSERT INTO drawers \
                 (id, wing, room, content, provenance, state, reindex_pending, created_at) \
                 VALUES (?1, ?2, ?3, ?4, 'test-raw', 'live', 1, ?5)",
                params![id, wing, room, content, now],
            )
            .expect("insert_raw");
    }

    // â”€â”€ Tier A gate: unified store has drawers + files + meta in one DB â”€â”€â”€
    #[test]
    fn unified_store_schema() {
        let (s, _tmp) = store();
        let guard = s.conn.lock().unwrap();

        let tables: Vec<String> = guard
            .prepare(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            )
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();

        assert!(tables.contains(&"drawers".to_string()), "drawers table must exist; tables: {tables:?}");
        assert!(tables.contains(&"files".to_string()), "files table must exist; tables: {tables:?}");
        assert!(tables.contains(&"meta".to_string()), "meta table must exist; tables: {tables:?}");

        // Both meta version rows must be present.
        let schema_ver: String = guard
            .query_row("SELECT value FROM meta WHERE key = 'schema_version'", [], |r| r.get(0))
            .expect("schema_version must exist");
        let embed_ver: String = guard
            .query_row("SELECT value FROM meta WHERE key = 'embedding_model_version'", [], |r| r.get(0))
            .expect("embedding_model_version must exist");
        assert_eq!(schema_ver, "5");
        assert_eq!(embed_ver, "");
    }

    #[test]
    fn open_creates_meta_rows() {
        let (s, _tmp) = store();
        let guard = s.conn.lock().unwrap();
        let schema_ver: String = guard
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .expect("schema_version row must exist");
        let embed_ver: String = guard
            .query_row(
                "SELECT value FROM meta WHERE key = 'embedding_model_version'",
                [],
                |row| row.get(0),
            )
            .expect("embedding_model_version row must exist");
        assert_eq!(schema_ver, "5");
        assert_eq!(embed_ver, "");
    }

    #[test]
    fn insert_round_trip() {
        let (s, _tmp) = store();
        let d = s.insert("testwing", "inbox", "hello world", "agent").expect("insert");
        assert_eq!(d.wing, "testwing");
        assert_eq!(d.room, "inbox");
        assert_eq!(d.content, "hello world");
        // Insert stamps the tier onto provenance: the caller's text is kept as a
        // prefix and the tag is appended. This store was given no tier policy, so
        // it is fail-safe and the unmapped wing goes to the most restrictive tier.
        assert_eq!(d.provenance, "agent|tier=L3");
        assert_eq!(
            membrane::tier_from_provenance(&d.provenance),
            Some("L3"),
            "every inserted drawer must carry a tier — untagged rows halt the splitter"
        );
        assert_eq!(d.state, DrawerState::Live);
        assert!(d.reindex_pending, "new rows must start with reindex_pending=true");

        let fetched = s.get(&d.id).expect("get").expect("some");
        assert_eq!(fetched.id, d.id);
        assert_eq!(fetched.content, "hello world");
        assert_eq!(fetched.provenance, d.provenance, "the tag persists through storage");
    }

    // ── Membrane, end-to-end through the store API ──────────────────────────

    /// Build a two-tier pair: an open store and a private database holding one
    /// L2 drawer. Returns (open_store, tmpdir, priv_path, dk).
    fn two_tier_store() -> (PalaceStore, tempfile::TempDir, PathBuf, [u8; 32]) {
        let tmp = tempfile::tempdir().expect("tmp");
        let mut dk = [0u8; 32];
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut dk);

        let priv_path = tmp.path().join("palace_private.db");
        {
            let p = PalaceStore::open(
                &priv_path,
                &tmp.path().join("vault_priv"),
                membrane::raw_key_literal(&dk).as_str(),
            )
            .expect("private store");
            p.insert("private", "notes", "PRIVATE MATERIAL", "seed").expect("seed L2");
        }

        let open_store = PalaceStore::open(
            &tmp.path().join("palace.db"),
            &tmp.path().join("vault_open"),
            "open-tier-key",
        )
        .expect("open store");
        open_store.insert("public", "notes", "open content", "seed").expect("seed L1");

        (open_store, tmp, priv_path, dk)
    }

    /// Count L2 drawers visible through the attached private schema. `Err` when
    /// the schema is not attached — which is the whole point.
    fn priv_count(s: &PalaceStore) -> std::result::Result<i64, rusqlite::Error> {
        let g = s.conn.lock().unwrap();
        g.query_row("SELECT COUNT(*) FROM priv.drawers", [], |r| r.get(0))
    }

    #[test]
    fn store_membrane_open_read_close() {
        let (s, _tmp, priv_path, dk) = two_tier_store();
        s.configure_membrane(priv_path, membrane::MembraneConfig::default());

        assert!(!s.membrane_is_open(), "starts closed");
        assert!(priv_count(&s).is_err(), "closed: private schema unresolvable");

        s.membrane_open_with_dk(&dk).expect("cross");
        assert!(s.membrane_is_open());
        assert_eq!(priv_count(&s).expect("open: readable"), 1);

        s.membrane_close().expect("close");
        assert!(!s.membrane_is_open());
        assert!(priv_count(&s).is_err(), "closed again: absent, not filtered");
    }

    /// Closing a closed membrane is not an error, and opening twice is not
    /// either. Both matter because the idle timer and an explicit close can
    /// race, and a crossing that errors on a double-open invites callers to
    /// track state themselves — which is the bool we deliberately don't trust.
    #[test]
    fn store_membrane_open_and_close_are_idempotent() {
        let (s, _tmp, priv_path, dk) = two_tier_store();
        s.configure_membrane(priv_path, membrane::MembraneConfig::default());

        s.membrane_close().expect("close when never open");
        s.membrane_open_with_dk(&dk).expect("open");
        s.membrane_open_with_dk(&dk).expect("open twice");
        assert_eq!(priv_count(&s).expect("still readable"), 1);
        s.membrane_close().expect("close");
        s.membrane_close().expect("close twice");
        assert!(!s.membrane_is_open());
    }

    /// **Idle expiry actually DETACHES — it does not merely report `false`.**
    ///
    /// An expiry that only reports would be a lie: anything querying without
    /// first asking `membrane_is_open()` would still read L2. Verified with an
    /// injected clock so the 60-minute policy is exercised in microseconds.
    #[test]
    fn store_membrane_idle_expiry_detaches_not_just_reports() {
        let (mut s, _tmp, priv_path, dk) = two_tier_store();
        let clock = Arc::new(membrane::TestClock::new(10_000));
        s.set_clock(clock.clone());
        s.configure_membrane(priv_path, membrane::MembraneConfig::default()); // 60 min idle

        s.membrane_open_with_dk(&dk).expect("cross");
        assert_eq!(priv_count(&s).expect("readable while open"), 1);

        clock.advance(std::time::Duration::from_secs(59 * 60));
        assert!(s.membrane_is_open(), "59 min idle: still open");

        clock.advance(std::time::Duration::from_secs(2 * 60)); // 61 min
        assert!(!s.membrane_is_open(), "61 min idle: must auto-lock");
        assert!(
            priv_count(&s).is_err(),
            "expiry must DETACH — a query that never called is_open() must still find nothing"
        );
    }

    /// Activity renews the idle clock, so a working session does not lock
    /// under the default policy.
    #[test]
    fn store_membrane_touch_renews_idle_window() {
        let (mut s, _tmp, priv_path, dk) = two_tier_store();
        let clock = Arc::new(membrane::TestClock::new(0));
        s.set_clock(clock.clone());
        s.configure_membrane(priv_path, membrane::MembraneConfig::default());
        s.membrane_open_with_dk(&dk).expect("cross");

        for _ in 0..10 {
            clock.advance(std::time::Duration::from_secs(30 * 60)); // 30 min
            s.membrane_touch();
            assert!(s.membrane_is_open(), "touched every 30 min: must stay open");
        }
        clock.advance(std::time::Duration::from_secs(61 * 60));
        assert!(!s.membrane_is_open(), "then goes quiet for 61 min: locks");
    }

    /// The passphrase path, end to end: seal keyslots beside the private db,
    /// open with the passphrase, open with the recovery code, reject a wrong
    /// passphrase without attaching anything.
    #[test]
    fn store_membrane_passphrase_and_recovery_paths() {
        let (s, _tmp, priv_path, dk) = two_tier_store();
        let params = simonides_crypto::key_primitives::Argon2Params { m_cost: 8, t_cost: 1, p_cost: 1 };
        let slots = membrane::MembraneKeyslots::seal(&dk, b"open-sesame", b"REC-CODE-XYZZY", &params)
            .expect("seal");
        let kp = {
            let mut p = priv_path.clone();
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            p.set_file_name(format!("{n}.keyslots.json"));
            p
        };
        slots.write_to(&kp).expect("write keyslots");
        s.configure_membrane(priv_path, membrane::MembraneConfig::default());

        assert!(s.membrane_open(b"wrong").is_err(), "wrong passphrase must fail");
        assert!(!s.membrane_is_open(), "failed crossing attaches NOTHING");

        s.membrane_open(b"open-sesame").expect("passphrase opens");
        assert_eq!(priv_count(&s).expect("readable"), 1);
        s.membrane_close().unwrap();

        s.membrane_open_with_recovery(b"REC-CODE-XYZZY").expect("recovery opens");
        assert_eq!(priv_count(&s).expect("readable"), 1);
    }

    /// **Diaries land in the room the taxonomy actually uses.**
    ///
    /// Every other diary test writes and reads through DIARY_ROOM, so they all
    /// agree with each other no matter what it is set to — they cannot detect
    /// that the constant disagrees with the STORED taxonomy.
    ///
    /// So this asserts against the room name literally, the way the migration
    /// wrote it and the way `list_drawers` finds it. A test that shares the
    /// bug's own assumption is not a test of that assumption.
    #[test]
    fn diary_lands_in_the_room_the_taxonomy_actually_uses() {
        let (s, _tmp) = store();
        s.set_tier_policy(
            membrane::TierPolicy::from_toml_str("[[rule]]\nroom = \"personal-diary\"\ntier = \"L2\"\n").unwrap(),
        );
        let d = s.diary_write("ada", "an entry", None).expect("write");
        assert_eq!(d.room, "personal-diary", "diary must use the migrated room name");
        assert_eq!(d.wing, "wing_ada");

        // And the round trip has to work through the LITERAL room, not the const.
        let listed = s
            .list_paged(Some("wing_ada"), Some("personal-diary"), 10, 0)
            .expect("list");
        assert_eq!(listed.len(), 1, "list_drawers must see what diary_write wrote");
        let read = s.diary_read("ada", 10).expect("read");
        assert_eq!(read.len(), 1, "diary_read must see what diary_write wrote");
        assert_eq!(read[0].id, listed[0].id, "both paths must find the SAME drawer");

        // diary_write is its own write path, so it must stamp the tier too.
        assert_eq!(membrane::tier_from_provenance(&d.provenance), Some("L2"));
    }
    /// Every write path stamps a tier from the store's policy. An untagged drawer
    /// is one whose tier nobody decided, so a second write path that skipped the
    /// stamp would be a second hole.
    #[test]
    fn every_insert_path_stamps_a_tier() {
        let (s, _tmp) = store();
        s.set_tier_policy(
            membrane::TierPolicy::from_toml_str(
                "[[rule]]\nroom = \"diary\"\ntier = \"L2\"\n\n[[rule]]\nwing = \"public\"\ntier = \"L1\"\n\n[[rule]]\nwing = \"private\"\ntier = \"L3\"\n",
            )
            .unwrap(),
        );

        let a = s.insert("public", "notes", "x", "agent").expect("insert");
        assert_eq!(membrane::tier_from_provenance(&a.provenance), Some("L1"));

        let b = s.insert("private", "health", "x", "agent").expect("insert");
        assert_eq!(membrane::tier_from_provenance(&b.provenance), Some("L3"));

        // insert_named is a separate public entry point - it must not be a hole.
        let c = s
            .insert_named("agent_ada", "diary", "x", "agent", "label")
            .expect("insert_named");
        assert_eq!(
            membrane::tier_from_provenance(&c.provenance),
            Some("L2"),
            "insert_named must tag too - a second write path is a second hole"
        );

        // A pre-tagged provenance gets exactly one marker, and it is the policy's:
        // a caller cannot choose its own tier label.
        let d = s.insert("public", "handoffs", "x", "src|tier=L2").expect("insert");
        assert_eq!(d.provenance, "src|tier=L1", "the policy tier replaces a caller-supplied one");
    }

    // â”€â”€ Acceptance criterion: move-invariance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #[test]
    fn move_invariance() {
        let (s, _tmp) = store();
        let d = s.insert("wing_a", "room_x", "some content", "test").expect("insert");
        let original_id = d.id.clone();

        let moved = s.move_drawer(&d.id, "wing_b", "room_y").expect("move");

        assert_eq!(moved.id, original_id, "id must not change on move");
        assert_eq!(moved.wing, "wing_b");
        assert_eq!(moved.room, "room_y");

        let fetched = s.get(&original_id).expect("get").expect("some");
        assert_eq!(fetched.id, original_id);
        assert_eq!(fetched.wing, "wing_b");
        assert_eq!(fetched.room, "room_y");
    }

    // â”€â”€ Acceptance criterion: move â‰  tombstone â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #[test]
    fn move_not_tombstone_distinguishable() {
        let (s, _tmp) = store();
        let live = s.insert("w", "r", "alpha", "test").expect("insert live");
        let retiring = s.insert("w", "r", "beta", "test").expect("insert retiring");

        s.tombstone(&retiring.id).expect("tombstone");
        let moved = s.move_drawer(&live.id, "w2", "r2").expect("move live");

        let tombstoned = s
            .get(&retiring.id)
            .expect("get")
            .expect("tombstoned row must resolve to Some, never None");
        assert_eq!(tombstoned.state, DrawerState::Tombstoned);

        let refetched_moved = s.get(&moved.id).expect("get").expect("some");
        assert_eq!(refetched_moved.state, DrawerState::Live);
        assert_eq!(refetched_moved.wing, "w2");

        assert_ne!(tombstoned.state, refetched_moved.state);
    }

    // â”€â”€ Acceptance criterion: tombstone never row-deletes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #[test]
    fn tombstone_never_row_deletes() {
        let (s, _tmp) = store();
        let d = s.insert("w", "r", "keep me forever", "test").expect("insert");
        s.tombstone(&d.id).expect("tombstone");

        let fetched = s
            .get(&d.id)
            .expect("get")
            .expect("row must exist after tombstone â€” never deleted");
        assert_eq!(fetched.state, DrawerState::Tombstoned);

        let h = s.health().expect("health");
        assert_eq!(h.tombstoned, 1);
        assert_eq!(h.total, 1, "total includes tombstoned rows");

        let live = s.list(100).expect("list");
        assert!(live.iter().all(|dr| dr.id != d.id), "tombstoned must not appear in list()");
    }

    // â”€â”€ Acceptance criterion: atomic verifiable write â€” normal path â”€â”€â”€â”€â”€â”€â”€
    #[test]
    fn atomic_write_verification_normal_path() {
        let (s, _tmp) = store();
        let d = s
            .insert("palace", "cedar", "verified content", "agent")
            .expect("insert must succeed and pass location verification");
        assert_eq!(d.wing, "palace");
        assert_eq!(d.room, "cedar");
    }

    // â”€â”€ Acceptance criterion: atomic verifiable write â€” caught wrong-target â”€
    #[test]
    fn atomic_write_verification_catches_wrong_target() {
        let (s, _tmp) = store();
        let id = Uuid::now_v7().to_string();

        insert_raw(&s, &id, "wrong_wing", "wrong_room", "misdirected content");

        let err = s
            .verify_and_return(&id, "intended_wing", "intended_room")
            .expect_err("verification must fail when actual location != intended location");

        match err {
            PalaceError::WriteVerificationFailed {
                ref actual_wing,
                ref actual_room,
                ref expected_wing,
                ref expected_room,
                ..
            } => {
                assert_eq!(actual_wing, "wrong_wing");
                assert_eq!(actual_room, "wrong_room");
                assert_eq!(expected_wing, "intended_wing");
                assert_eq!(expected_room, "intended_room");
            }
            other => panic!("expected WriteVerificationFailed, got: {other}"),
        }
    }

    // â”€â”€ Acceptance criterion: health() reports reindex_pending â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #[test]
    fn rebuild_regenerates_blob_after_content_edit() {
        // Regression lock. Reuse-not-re-embed holds by
        // construction only because `update` replaces a drawer's chunks with
        // NULL-embedding rows. Nothing was RED if a future "update in place"
        // refactor left a stale blob on changed content. This is that red.
        let (s, _tmp) = store();
        let d = s
            .insert("w", "r", "the original content about apples", "test")
            .expect("insert");
        s.rebuild_indexes().expect("first rebuild embeds original");

        let read_blob = |store: &PalaceStore| -> Vec<u8> {
            let g = store.conn.lock().unwrap();
            g.query_row(
                "SELECT embedding FROM chunks WHERE drawer_id = ?1 LIMIT 1",
                params![d.id],
                |r| r.get::<_, Option<Vec<u8>>>(0),
            )
            .expect("read blob")
            .expect("chunk must be embedded")
        };
        let before = read_blob(&s);

        s.update_content(&d.id, "entirely different content about spacecraft")
            .expect("edit content");
        let report = s.rebuild_indexes().expect("rebuild re-embeds the new content");
        assert!(
            report.embedded >= 1,
            "changed content must be RE-EMBEDDED, not reused: {report:?}"
        );

        let after = read_blob(&s);
        assert_ne!(
            before, after,
            "the embedding blob must regenerate for edited content â€” reuse must \
             NEVER serve the stale vector"
        );
    }

    #[test]
    fn rebuild_kg_builds_graph_without_embedding() {
        // Extract-only: regex extractor, no model. Builds the KG; embeds nothing.
        let (s, _tmp) = store();
        s.insert("w", "r", "Mira met Hanna in Dili to discuss Cedar.", "test").expect("insert");
        s.insert("w", "r", "Dili is where Mira lives now.", "test").expect("insert");
        let rep = s.rebuild_kg().expect("rebuild_kg");
        assert!(rep.drawers_scanned >= 2, "scanned all live drawers: {rep:?}");
        let stats = s.kg_stats().expect("stats");
        assert!(stats.entities >= 1, "KG must be populated by extract-only: {stats:?}");
        let h = s.health().expect("health");
        assert_eq!(h.chunks_embedded, 0, "rebuild_kg must NOT embed â€” chunks stay NULL: {h}");
    }

    #[test]
    fn rebuild_vectors_embeds_without_building_kg() {
        let (s, _tmp) = store();
        s.insert("w", "r", "Mira met Hanna in Dili.", "test").expect("insert");
        // Inline extraction seeds the KG on insert; capture that baseline and prove
        // rebuild_vectors (the vectors half) leaves it UNCHANGED.
        let before = s.kg_stats().expect("stats before");
        let rep = s.rebuild_vectors().expect("rebuild_vectors");
        assert!(rep.embedded >= 1, "vectors half must embed: {rep:?}");
        let after = s.kg_stats().expect("stats after");
        assert_eq!(after.entities, before.entities, "rebuild_vectors must not CHANGE the KG: {after:?}");
        assert_eq!(after.triples, before.triples, "rebuild_vectors adds no triples: {after:?}");
    }

    #[test]
    fn insert_extracts_kg_inline() {
        let (s, _tmp) = store();
        // No rebuild_kg call anywhere: the KG must be live purely from the insert.
        s.insert("w", "r", "Mira works on Cedar in Dili.", "test").expect("insert");
        let stats = s.kg_stats().expect("kg_stats");
        assert!(stats.entities >= 2, "insert must extract entities inline (Mira/Cedar/Dili): {stats:?}");
        assert!(stats.triples >= 1, "insert must extract the works_on triple inline: {stats:?}");
    }

    #[test]
    fn rebuild_indexes_still_does_both() {
        let (s, _tmp) = store();
        s.insert("w", "r", "Mira met Hanna in Dili.", "test").expect("insert");
        s.rebuild_indexes().expect("rebuild_indexes");
        let h = s.health().expect("health");
        assert!(h.chunks_embedded >= 1, "combined pass builds vectors");
        let stats = s.kg_stats().expect("stats");
        assert!(stats.entities >= 1, "combined pass builds KG");
    }

    #[test]
    fn rebuild_indexes_reuses_stored_embeddings() {
        let (s, _tmp) = store();
        s.insert("w", "r", "alpha content", "test").expect("insert");
        s.insert("w", "r", "bravo content", "test").expect("insert");

        // Reach the post-embed state HERMETICALLY (no real model): a blob for every
        // chunk + the current model version recorded. Then re-flag for reindex.
        {
            let guard = s.conn.lock().unwrap();
            guard
                .execute("UPDATE chunks SET embedding = X'00000000'", [])
                .expect("dummy blobs");
            guard
                .execute(
                    "UPDATE meta SET value = ?1 WHERE key = 'embedding_model_version'",
                    params![embedder::MODEL_ID],
                )
                .expect("record current model version");
            guard
                .execute("UPDATE drawers SET reindex_pending = 1", [])
                .expect("re-flag");
        }

        let report = s.rebuild_indexes().expect("rebuild");
        assert_eq!(
            report.embedded, 0,
            "already-embedded chunks under the current model must NOT be re-embedded \
             (the model is never called) â€” seconds, not hours: {report:?}"
        );
        assert!(report.reused >= 2, "every stored blob must be reused: {report:?}");

        // Reuse never rewrites the blobs.
        let blobs: Vec<Vec<u8>> = {
            let guard = s.conn.lock().unwrap();
            let mut st = guard.prepare("SELECT embedding FROM chunks").unwrap();
            let rows = st.query_map([], |r| r.get::<_, Vec<u8>>(0)).unwrap();
            rows.map(|r| r.unwrap()).collect()
        };
        assert!(
            blobs.iter().all(|b| b == &[0u8, 0, 0, 0]),
            "reused blobs must be byte-untouched"
        );
    }

    #[test]
    fn rebuild_indexes_re_embeds_on_model_version_change() {
        // Uses the real embedder, like the existing rebuild_indexes tests.
        let (s, _tmp) = store();
        s.insert("w", "r", "some content to embed", "test").expect("insert");
        let first = s.rebuild_indexes().expect("first rebuild");
        assert!(first.embedded >= 1, "first pass embeds: {first:?}");

        // No content change + same model -> reuse everything.
        {
            let guard = s.conn.lock().unwrap();
            guard.execute("UPDATE drawers SET reindex_pending = 1", []).expect("re-flag");
        }
        let noop = s.rebuild_indexes().expect("noop rebuild");
        assert_eq!(noop.embedded, 0, "unchanged content + same model reuses all: {noop:?}");

        // Model version changes -> every embedding stale -> re-embed all.
        {
            let guard = s.conn.lock().unwrap();
            guard
                .execute("UPDATE meta SET value = 'OLD-MODEL' WHERE key = 'embedding_model_version'", [])
                .expect("downgrade version");
            guard.execute("UPDATE drawers SET reindex_pending = 1", []).expect("re-flag");
        }
        let after = s.rebuild_indexes().expect("rebuild after version change");
        assert!(after.embedded >= 1, "a model-version change must re-embed: {after:?}");
    }

    #[test]
    fn health_counts_embedding_coverage_as_a_fact_not_the_flag() {
        let (s, _tmp) = store();
        s.insert("w", "r", "alpha", "test").expect("insert");
        s.insert("w", "r", "bravo", "test").expect("insert");
        s.insert("w", "r", "charlie", "test").expect("insert");

        let h = s.health().expect("health");
        assert!(h.chunks_total >= 3, "one chunk per short drawer: {h}");
        assert_eq!(h.chunks_embedded, 0, "nothing embedded yet");
        assert!(h.to_string().contains("embedded=0"), "coverage in Display: {h}");
        assert!(h.to_string().contains("PARTIAL"), "degraded store must say so: {h}");

        // THE POINT: clear the flag on every drawer WITHOUT embedding anything.
        // A flag-based monitor would now report healthy. The counted fact must not.
        {
            let guard = s.conn.lock().unwrap();
            guard
                .execute("UPDATE drawers SET reindex_pending = 0", [])
                .expect("force flag clean");
        }
        let h2 = s.health().expect("health after flag cleared");
        assert_eq!(h2.reindex_pending, 0, "the flag now claims clean");
        assert_eq!(h2.chunks_embedded, 0, "but nothing is actually embedded");
        assert!(
            h2.to_string().contains("PARTIAL"),
            "coverage shortfall must survive a cleared flag â€” count the fact, not the flag: {h2}"
        );

        // Whole case: mark every chunk embedded. The monitor reports whole AND the
        // numbers are still present on the healthy path (inverse of the live palace
        // bug, where the number vanished exactly when it was fine).
        {
            let guard = s.conn.lock().unwrap();
            guard
                .execute("UPDATE chunks SET embedding = X'00'", [])
                .expect("mark all chunks embedded");
        }
        let h3 = s.health().expect("health when whole");
        assert_eq!(h3.chunks_total, h3.chunks_embedded, "fully covered");
        assert!(h3.to_string().contains("embedded="), "numbers present when healthy: {h3}");
        assert!(!h3.to_string().contains("PARTIAL"), "a whole store must not cry partial: {h3}");
    }

    #[test]
    fn list_paged_walks_the_whole_corpus_each_drawer_once() {
        let (s, _tmp) = store();
        let n = 25usize;
        let mut inserted = std::collections::HashSet::new();
        for i in 0..n {
            let d = s.insert("w", "r", &format!("drawer number {i}"), "test").expect("insert");
            inserted.insert(d.id);
        }

        let page = 7usize;
        let mut seen: Vec<String> = Vec::new();
        let mut offset = 0usize;
        loop {
            let batch = s.list_paged(None, None, page, offset).expect("page");
            if batch.is_empty() {
                break;
            }
            for d in &batch {
                seen.push(d.id.clone());
            }
            offset += page;
            assert!(offset <= n + 2 * page, "pagination must terminate");
        }
        assert_eq!(seen.len(), n, "every drawer seen exactly once â€” none dropped, none duplicated");
        let seen_set: std::collections::HashSet<_> = seen.into_iter().collect();
        assert_eq!(seen_set.len(), n, "no duplicate across page boundaries");
        assert_eq!(seen_set, inserted, "the paged set equals the inserted set");

        s.insert("other", "x", "elsewhere", "test").expect("insert other");
        let filtered = s.list_paged(Some("w"), Some("r"), 100, 0).expect("filtered");
        assert_eq!(filtered.len(), n, "wing/room filter scopes enumeration");
        let just_other = s.list_paged(Some("other"), None, 100, 0).expect("other");
        assert_eq!(just_other.len(), 1, "wing filter alone works");
    }

    #[test]
    fn health_reports_reindex_pending() {
        let (s, _tmp) = store();
        s.insert("w", "r", "a", "test").expect("insert");
        s.insert("w", "r", "b", "test").expect("insert");
        s.insert("w", "r", "c", "test").expect("insert");

        let h = s.health().expect("health");
        assert_eq!(h.total, 3);
        assert_eq!(h.live, 3);
        assert_eq!(h.tombstoned, 0);
        assert_eq!(h.reindex_pending, 3, "all new rows must start pending reindex");

        {
            let guard = s.conn.lock().unwrap();
            guard
                .execute(
                    "UPDATE drawers SET reindex_pending = 0 \
                     WHERE id = (SELECT id FROM drawers LIMIT 1)",
                    [],
                )
                .expect("clear one pending row");
        }
        let h2 = s.health().expect("health after partial clear");
        assert_eq!(h2.reindex_pending, 2);

        let d = s.insert("w", "r", "to be tombstoned", "test").expect("insert");
        s.tombstone(&d.id).expect("tombstone");
        let h3 = s.health().expect("health after tombstone");
        assert_eq!(h3.reindex_pending, 2, "tombstone must clear reindex_pending");
        assert_eq!(h3.tombstoned, 1);
    }

    // â”€â”€ Acceptance criterion: newest-first list ordering via v7 id â”€â”€â”€â”€â”€â”€â”€â”€
    #[test]
    fn newest_first_ordering() {
        let (s, _tmp) = store();
        let a = s.insert("w", "r", "first", "test").expect("a");
        let b = s.insert("w", "r", "second", "test").expect("b");
        let c = s.insert("w", "r", "third", "test").expect("c");

        let list = s.list(10).expect("list");
        assert_eq!(list.len(), 3);

        assert!(
            list[0].id >= list[1].id && list[1].id >= list[2].id,
            "list must be ordered by id DESC (newest first)"
        );

        assert_eq!(list[0].id, c.id, "newest (c) must be first");
        assert_eq!(list[1].id, b.id);
        assert_eq!(list[2].id, a.id, "oldest (a) must be last");
    }

    #[test]
    fn list_excludes_tombstoned() {
        let (s, _tmp) = store();
        let live = s.insert("w", "r", "live content", "test").expect("insert");
        let dead = s.insert("w", "r", "dead content", "test").expect("insert");
        s.tombstone(&dead.id).expect("tombstone");

        let list = s.list(10).expect("list");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, live.id);
    }

    #[test]
    fn move_tombstoned_drawer_is_rejected() {
        let (s, _tmp) = store();
        let d = s.insert("w", "r", "hello", "test").expect("insert");
        s.tombstone(&d.id).expect("tombstone");

        let err = s
            .move_drawer(&d.id, "new_wing", "new_room")
            .expect_err("moving a tombstoned drawer must fail");
        assert!(
            matches!(err, PalaceError::NotFound { .. }),
            "expected NotFound, got: {err}"
        );
    }

    #[test]
    fn rebuild_indexes_succeeds() {
        let (s, _tmp) = store();
        s.insert("w", "r", "cedar mesh privacy network", "test").expect("insert");
        s.rebuild_indexes().expect("rebuild_indexes must succeed (embeds pending drawers)");
    }

    #[test]
    fn get_absent_id_returns_none() {
        let (s, _tmp) = store();
        let result = s.get("00000000-0000-0000-0000-000000000000").expect("get");
        assert!(result.is_none());
    }

    #[test]
    fn tombstone_absent_id_returns_not_found() {
        let (s, _tmp) = store();
        let err = s
            .tombstone("00000000-0000-0000-0000-000000000000")
            .expect_err("tombstone of absent id must fail");
        assert!(matches!(err, PalaceError::NotFound { .. }));
    }

    #[test]
    fn health_display_surfaces_pending_message() {
        let (s, _tmp) = store();
        s.insert("w", "r", "test", "test").expect("insert");
        let h = s.health().expect("health");
        let display = h.to_string();
        assert!(display.contains("pending"), "display must mention 'pending' when reindex_pending > 0");
    }

    #[test]
    fn open_on_disk_is_idempotent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("palace.db");
        let vault_root = dir.path().join("vault");
        // First open: creates schema
        let _s1 = PalaceStore::open(&db_path, &vault_root, TEST_KEY).expect("first open");
        drop(_s1);
        // Second open with same key: must not fail (IF NOT EXISTS guards schema reapplication)
        let s2 = PalaceStore::open(&db_path, &vault_root, TEST_KEY).expect("second open");
        let h = s2.health().expect("health on re-open");
        assert_eq!(h.total, 0);
    }

    /// PK bitemporal migration gate (brief PK, 2026-07-15): open a v4-shaped store
    /// holding real triples, re-open so the v4â†’v5 migration runs, and assert every
    /// triple survives with its window intact under the widened fact-key index.
    #[test]
    fn migration_v4_to_v5_preserves_existing_triples() {
        use chrono::{TimeZone, Utc};
        use simonides_core::memory::KgDirection;

        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("palace.db");
        let vault_root = dir.path().join("vault");
        let t0 = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let t1 = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();

        // 1. Open at the current schema and lay down triples with real windows.
        {
            let s = PalaceStore::open(&db_path, &vault_root, TEST_KEY).expect("first open");
            s.kg_add_triple("Mira", "lives_in", "Dili", Some(t0), None, None)
                .expect("open fact");
            s.kg_add_triple("Sarah", "works_on", "ProjectX", Some(t0), None, None)
                .expect("fact to close");
            s.kg_invalidate("Sarah", "works_on", "ProjectX", t1)
                .expect("close its window â†’ [t0, t1)");
            s.kg_add_triple("Hanna", "married_to", "Mira", None, None, None)
                .expect("windowless fact");

            // 2. Downgrade the on-disk DB to look like a v4 store: the narrow
            //    (subject, predicate, object) fact-key index + schema_version = '4'.
            //    A real v4 store could only ever hold one row per (s,p,o), which is
            //    exactly what we have, so the narrow UNIQUE index builds cleanly.
            let guard = s.conn.lock().unwrap();
            guard
                .execute_batch(
                    "DROP INDEX IF EXISTS triples_fact_key;\
                     CREATE UNIQUE INDEX triples_fact_key \
                         ON triples(subject_id, predicate, object_id);",
                )
                .expect("install narrow v4 index");
            guard
                .execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'", [])
                .expect("downgrade schema_version to 4");
            drop(guard);
            drop(s);
        }

        // 3. Re-open: init() runs the v4â†’v5 migration.
        let s = PalaceStore::open(&db_path, &vault_root, TEST_KEY).expect("re-open migrates v4â†’v5");

        // schema_version is bumped.
        assert_eq!(
            s.get_meta("schema_version").expect("read version").as_deref(),
            Some("5"),
            "re-open must migrate schema_version to 5"
        );

        // The fact-key index is now the widened 4-column (â€¦, valid_from) form.
        {
            let guard = s.conn.lock().unwrap();
            let cols: i64 = guard
                .query_row(
                    "SELECT COUNT(*) FROM pragma_index_info('triples_fact_key')",
                    [],
                    |r| r.get(0),
                )
                .expect("index_info");
            assert_eq!(
                cols, 4,
                "triples_fact_key must be the widened 4-column index after migration"
            );
        }

        // Every triple survived with its window intact.
        let stats = s.kg_stats().expect("stats");
        assert_eq!(stats.triples, 3, "all three triples must survive the migration");
        assert_eq!(stats.open_triples, 2, "the two open facts survive open");
        assert_eq!(stats.invalidated_triples, 1, "the closed fact keeps its closed window");

        let sarah = s.kg_timeline("Sarah").expect("sarah timeline");
        assert_eq!(sarah.len(), 1);
        assert_eq!(sarah[0].valid_from, Some(t0), "closed window's valid_from intact");
        assert_eq!(sarah[0].valid_to, Some(t1), "closed window's valid_to intact");

        let mira = s
            .kg_query_entity("Mira", None, KgDirection::Both)
            .expect("mira now-view");
        assert_eq!(mira.len(), 2, "Mira's two open facts are visible post-migration");

        // Proof the wide index is live: a re-assertion after invalidation now opens
        // a SECOND window instead of overwriting â€” the behaviour the migration enables.
        let t2 = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
        s.kg_add_triple("Sarah", "works_on", "ProjectX", Some(t2), None, None)
            .expect("re-assert on the migrated store");
        assert_eq!(
            s.kg_timeline("Sarah").expect("sarah timeline 2").len(),
            2,
            "post-migration, re-assertion opens a second window (wide index active)"
        );
    }

    // â”€â”€ Tier A: seq assignment â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn seq_assigned_on_insert() {
        let (s, _tmp) = store();
        let d = s.insert("testwing", "inbox", "first", "test").expect("insert");
        assert_eq!(d.seq, 1, "first insert in a room must get seq = 1");
    }

    #[test]
    fn seq_monotonic_per_room() {
        let (s, _tmp) = store();
        let a = s.insert("testwing", "inbox", "a", "test").expect("a");
        let b = s.insert("testwing", "inbox", "b", "test").expect("b");
        let c = s.insert("testwing", "inbox", "c", "test").expect("c");
        assert_eq!(a.seq, 1);
        assert_eq!(b.seq, 2);
        assert_eq!(c.seq, 3, "seq must be strictly monotonic within the same room");
    }

    #[test]
    fn seq_independent_across_rooms() {
        let (s, _tmp) = store();
        // Two inserts in room A.
        let a1 = s.insert("testwing", "room_a", "a1", "test").expect("a1");
        let a2 = s.insert("testwing", "room_a", "a2", "test").expect("a2");
        // One insert in room B: should start at 1, independent of room A.
        let b1 = s.insert("testwing", "room_b", "b1", "test").expect("b1");

        assert_eq!(a1.seq, 1);
        assert_eq!(a2.seq, 2);
        assert_eq!(b1.seq, 1, "each room has its own independent seq counter");
    }

    #[test]
    fn seq_unique_per_room() {
        let (s, _tmp) = store();
        let n = 10usize;
        let drawers: Vec<_> = (0..n)
            .map(|i| s.insert("w", "r", &format!("content {i}"), "test").expect("insert"))
            .collect();

        let seqs: Vec<i64> = drawers.iter().map(|d| d.seq).collect();
        let mut sorted = seqs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), n, "seq values must be unique within a room; got: {seqs:?}");
    }

    // â”€â”€ Tier B: resolve() â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn resolve_full_uuid_found() {
        let (s, _tmp) = store();
        let d = s.insert("testwing", "inbox", "hello", "test").expect("insert");
        match s.resolve(&d.id).expect("resolve") {
            ResolveOutcome::Found(found) => assert_eq!(found.id, d.id),
            other => panic!("expected Found, got: {other:?}"),
        }
    }

    #[test]
    fn resolve_full_uuid_miss_not_found_with_nearest() {
        let (s, _tmp) = store();
        // Insert a live drawer so nearest is non-empty.
        s.insert("w", "r", "anchor", "test").expect("insert");
        let fake_id = "00000000-0000-7000-8000-000000000000";
        match s.resolve(fake_id).expect("resolve") {
            ResolveOutcome::NotFound { nearest, .. } => {
                assert!(!nearest.is_empty(), "nearest must be non-empty");
            }
            other => panic!("expected NotFound, got: {other:?}"),
        }
    }

    #[test]
    fn resolve_prefix_unambiguous_found() {
        let (s, _tmp) = store();
        let d = s.insert("testwing", "inbox", "unique", "test").expect("insert");
        // 8-char prefix of a UUID v7 is the timestamp high bits; with only one
        // drawer in the store this prefix is unambiguous.
        let prefix = &d.id[..8];
        match s.resolve(prefix).expect("resolve") {
            ResolveOutcome::Found(found) => assert_eq!(found.id, d.id),
            other => panic!("expected Found, got: {other:?}"),
        }
    }

    #[test]
    fn resolve_prefix_ambiguous() {
        let (s, _tmp) = store();
        // Insert two drawers and construct a shared prefix by using the same id
        // prefix â€” we simulate ambiguity by searching for a very short prefix
        // that matches both.  The simplest approach: use a raw insert with
        // ids that share a common prefix.
        let guard = s.conn.lock().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        guard
            .execute(
                "INSERT INTO drawers \
                 (id, wing, room, content, provenance, state, reindex_pending, created_at, seq, name) \
                 VALUES ('aaaaaaaa-0000-7000-8000-000000000001', 'w', 'r', 'c1', 't', 'live', 1, ?1, 1, NULL)",
                params![now],
            )
            .expect("insert raw 1");
        guard
            .execute(
                "INSERT INTO drawers \
                 (id, wing, room, content, provenance, state, reindex_pending, created_at, seq, name) \
                 VALUES ('aaaaaaaa-0000-7000-8000-000000000002', 'w', 'r', 'c2', 't', 'live', 1, ?1, 2, NULL)",
                params![now],
            )
            .expect("insert raw 2");
        drop(guard);

        match s.resolve("aaaaaaaa").expect("resolve") {
            ResolveOutcome::Ambiguous { candidates, .. } => {
                assert_eq!(candidates.len(), 2, "both prefix-matching drawers must be candidates");
            }
            other => panic!("expected Ambiguous, got: {other:?}"),
        }
    }

    #[test]
    fn resolve_handle_found() {
        let (s, _tmp) = store();
        let d = s.insert("testwing", "inbox", "handle test", "test").expect("insert");
        assert_eq!(d.seq, 1);
        match s.resolve("testwing/inbox#1").expect("resolve") {
            ResolveOutcome::Found(found) => assert_eq!(found.id, d.id),
            other => panic!("expected Found, got: {other:?}"),
        }
    }

    #[test]
    fn resolve_handle_miss_not_found_with_nearest() {
        let (s, _tmp) = store();
        s.insert("testwing", "inbox", "seq1", "test").expect("insert");
        s.insert("testwing", "inbox", "seq2", "test").expect("insert");
        // seq 99 doesn't exist; nearest should be the closest existing seqs.
        match s.resolve("testwing/inbox#99").expect("resolve") {
            ResolveOutcome::NotFound { nearest, .. } => {
                assert!(!nearest.is_empty(), "nearest must be non-empty on handle miss");
            }
            other => panic!("expected NotFound, got: {other:?}"),
        }
    }

    #[test]
    fn resolve_named_found() {
        let (s, _tmp) = store();
        let d = s
            .insert_named("testwing", "handoffs", "content", "agent", "m5-result")
            .expect("insert_named");
        match s.resolve("testwing/handoffs:m5-result").expect("resolve") {
            ResolveOutcome::Found(found) => {
                assert_eq!(found.id, d.id);
                assert_eq!(found.name.as_deref(), Some("m5-result"));
            }
            other => panic!("expected Found, got: {other:?}"),
        }
    }

    #[test]
    fn resolve_named_miss_not_found_with_nearest() {
        let (s, _tmp) = store();
        s.insert_named("w", "r", "x", "t", "alpha").expect("insert");
        match s.resolve("w/r:no-such-name").expect("resolve") {
            ResolveOutcome::NotFound { nearest, .. } => {
                assert!(!nearest.is_empty(), "nearest must be non-empty on name miss");
            }
            other => panic!("expected NotFound, got: {other:?}"),
        }
    }

    #[test]
    fn resolve_8char_prefix_resolves_or_loud() {
        let (s, _tmp) = store();
        let d = s.insert("w", "r", "content", "test").expect("insert");
        let prefix8 = &d.id[..8];
        // Must not silently return nothing â€” either Found (unique) or Ambiguous.
        match s.resolve(prefix8).expect("resolve") {
            ResolveOutcome::Found(_) | ResolveOutcome::Ambiguous { .. } => {}
            ResolveOutcome::NotFound { .. } => {
                // Acceptable only if the prefix genuinely matches nothing (shouldn't happen
                // since the drawer we just inserted shares this prefix).
                panic!("resolve of an 8-char prefix of an existing id must not be NotFound");
            }
        }
    }

    // â”€â”€ Tier C: membrane â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn resolve_is_local_store_only() {
        // Structural: resolve() has no parameters for external endpoints and
        // uses only the local SQLite connection.  This test confirms the API
        // surface carries no outbound capability: we call resolve() and it
        // returns without any network operation.
        let (s, _tmp) = store();
        let d = s.insert("w", "r", "membrane", "test").expect("insert");
        // If this call returns at all without an external connection, the membrane
        // is intact â€” resolve() is local-only by construction.
        let _ = s.resolve(&d.id).expect("resolve must complete locally");
    }

    // â”€â”€ M2 Gate: DB sqlcipher encryption â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Gate A-1: raw bytes of the written .db file must NOT start with the
    /// well-known SQLite3 magic. sqlcipher-encrypted files have an opaque
    /// header â€” if the magic appears the PRAGMA key was silently ignored.
    #[test]
    fn premium_db_header_is_not_sqlite_magic() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let db_path = dir.path().join("palace.db");
        let vault_root = dir.path().join("vault");
        {
            let s = PalaceStore::open(&db_path, &vault_root, CORRECT_KEY)
                .expect("open with correct key");
            s.insert("test", "room", "content that must be encrypted at rest", "test")
                .expect("insert");
            // Drop to flush WAL to disk.
        }
        assert!(db_path.exists(), "palace.db must exist after write");
        let raw = std::fs::read(&db_path).expect("read raw bytes");
        assert!(raw.len() >= 16, "db file too short ({} bytes)", raw.len());
        let header: &[u8; 16] = raw[..16].try_into().expect("16-byte slice");
        assert_ne!(
            header,
            SQLITE_MAGIC,
            "DB header matches SQLite magic â€” sqlcipher PRAGMA key was ignored (plaintext at rest)"
        );
    }

    // â”€â”€ M3b Gate E: embeddings in encrypted DB, no plaintext vector dir â”€â”€â”€â”€â”€â”€

    /// Verifies the encryption-gate extension: after rebuild_indexes(), embedding
    /// vectors live only inside the sqlcipher DB â€” no lancedb/vector directory
    /// is created outside it, and the DB itself is ciphertext (not plaintext).
    #[test]
    fn embeddings_in_encrypted_db_no_plaintext_vector_dir() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let db_path = dir.path().join("palace.db");
        let vault_root = dir.path().join("vault");

        let s = PalaceStore::open(&db_path, &vault_root, "gate-e-encrypt-test-key-m3b")
            .expect("open");
        s.insert(
            "w",
            "r",
            "private content with personal embedding data about cedar privacy",
            "test",
        )
        .expect("insert");
        s.rebuild_indexes().expect("rebuild_indexes");

        // (i) No vector/lancedb/index directory created alongside the DB.
        let children: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(
            !children.iter().any(|n| {
                n.contains("lance")
                    || n.contains("vector")
                    || n.contains("index")
                    || n.contains(".hnswlib")
            }),
            "no vector index directory must be created outside the encrypted DB; found: {children:?}"
        );

        // (ii) DB header is sqlcipher-encrypted (not plain SQLite magic).
        let raw = std::fs::read(&db_path).expect("read db bytes");
        assert!(raw.len() >= 16, "db file too short");
        let header: &[u8; 16] = raw[..16].try_into().unwrap();
        assert_ne!(
            header,
            SQLITE_MAGIC,
            "DB header matches SQLite magic â€” embeddings written to plaintext DB"
        );

        // (iii) Embedding blobs exist inside the keyed DB (proves in-DB storage).
        let count: i64 = {
            let guard = s.conn.lock().unwrap();
            guard
                .query_row(
                    "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL",
                    [],
                    |r| r.get(0),
                )
                .expect("count embeddings")
        };
        assert!(
            count > 0,
            "at least one embedding blob must be stored in the encrypted DB after rebuild_indexes"
        );
    }

    /// P4 gate (brief Â§7): the closet projection is private-DERIVED data, so it
    /// inherits M2 encryption-at-rest â€” it lives in the sqlcipher DB and writes
    /// no plaintext sidecar. This is the recurring plaintext-sidecar trap
    /// (the vault-blob hole in M2, the embeddings hole in M3b); a projection is
    /// exactly the kind of "it's only derived" data that invites one.
    #[test]
    fn closets_in_encrypted_db_no_plaintext() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let db_path = dir.path().join("palace.db");
        let vault_root = dir.path().join("vault");

        let s = PalaceStore::open(&db_path, &vault_root, "gate-p4-closet-encryption-key")
            .expect("open");
        // A distinctive entity that would be trivially greppable in the clear.
        for content in [
            "Zylotrix and Mira met about the private acquisition.",
            "Mira briefed Zylotrix on the confidential terms.",
        ] {
            s.insert("w", "r", content, "test").expect("insert");
        }
        let report = s.rebuild_closets("w", "r").expect("rebuild_closets");
        assert!(report.closets_written > 0, "the fixture must write closets");

        // (i) The closets are really in the keyed DB.
        let count: i64 = {
            let guard = s.conn.lock().unwrap();
            guard
                .query_row("SELECT COUNT(*) FROM closets", [], |r| r.get(0))
                .expect("count closets")
        };
        assert!(count > 0, "closets must be stored in the encrypted DB");

        // (ii) No closet sidecar file/dir appeared next to the DB.
        let children: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(
            !children.iter().any(|n| n.contains("closet")),
            "no closet sidecar may be written outside the encrypted DB; found: {children:?}"
        );

        // (iii) The DB header is sqlcipher-encrypted, not plain SQLite.
        let raw = std::fs::read(&db_path).expect("read db bytes");
        assert!(raw.len() >= 16, "db file too short");
        let header: &[u8; 16] = raw[..16].try_into().unwrap();
        assert_ne!(header, SQLITE_MAGIC, "closets written to a plaintext DB");

        // (iv) The strongest form: the closet label appears NOWHERE in plaintext
        // on disk â€” not in the DB, not in the WAL, not in any sidecar.
        let needle = b"Zylotrix";
        for entry in std::fs::read_dir(dir.path()).unwrap().filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap_or_default();
            assert!(
                !bytes.windows(needle.len()).any(|w| w == needle),
                "closet label found in plaintext at {} â€” private-derived data must be \
                 encrypted at rest",
                path.display()
            );
        }
    }

    /// Gate A-2: opening an encrypted DB with the wrong key must return Err
    /// immediately (eager rejection â€” the key check is at open time, not deferred).
    #[test]
    fn wrong_key_is_rejected() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let db_path = dir.path().join("palace.db");
        let vault_root = dir.path().join("vault");
        {
            let s = PalaceStore::open(&db_path, &vault_root, CORRECT_KEY)
                .expect("open with correct key");
            s.insert("test", "room", "payload written with correct key", "test")
                .expect("insert");
        }
        let result = PalaceStore::open(&db_path, &vault_root, WRONG_KEY);
        assert!(
            result.is_err(),
            "opening encrypted palace DB with wrong key must return Err (eager rejection)"
        );
    }

    // â”€â”€ M6 gate tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Gate 1: Automerge state in the encrypted store â€” no plaintext sidecar.
    /// Verifies that `crdt_docs` lives inside the sqlcipher DB and that the
    /// on-disk file is NOT plain SQLite (which would mean plaintext spill).
    #[test]
    fn automerge_state_in_encrypted_store_no_plaintext() {
        use simonides_core::sync::SyncableStore;
        use simonides_sync::delta::{DrawerDeltaItem, PalaceDelta, encode_delta};
        use simonides_core::sync::{Hlc, SyncDelta};

        let dir = tempfile::tempdir().expect("tmpdir");
        let db_path = dir.path().join("palace.db");
        let vault_root = dir.path().join("vault");
        let store = PalaceStore::open(&db_path, &vault_root, CORRECT_KEY).expect("open");

        // Build a trivial AM doc via simonides_sync helpers (using automerge directly).
        let mut doc = automerge::AutoCommit::new();
        {
            use automerge::transaction::Transactable;
            doc.put(automerge::ROOT, "content", "gate test content").unwrap();
        }
        let am_bytes = doc.save();

        let hlc = Hlc { wall_time: 1_000_000, logical: 0, node_id: [1u8; 16] };
        let delta = PalaceDelta {
            drawers: vec![DrawerDeltaItem {
                drawer_id: uuid::Uuid::now_v7().to_string(),
                content: "gate test content".to_string(),
                automerge_bytes: am_bytes.clone(),
                hlc,
                wing: "test".to_string(),
                room: "crdt".to_string(),
                provenance: "gate-test".to_string(),
                tombstoned: false,
                name: None,
            }],
            palace_records: vec![],
            tunnels: vec![],
            triples: vec![],
            entities: vec![],
        };

        let payload = encode_delta(&delta).expect("encode");
        let sync_delta = SyncDelta { payload, source_hlc: hlc };
        store.apply_remote_delta(sync_delta).expect("apply_remote_delta");

        // Assert 1: crdt_docs row exists inside the keyed connection.
        let doc_count: i64 = {
            let g = store.conn.lock().unwrap();
            g.query_row("SELECT COUNT(*) FROM crdt_docs", [], |r| r.get(0))
                .expect("count crdt_docs")
        };
        assert!(doc_count > 0, "crdt_docs must have a row after apply_remote_delta");

        // Assert 2: raw DB bytes are NOT plain SQLite (proves sqlcipher is active).
        let raw = std::fs::read(&db_path).expect("read db");
        assert!(raw.len() >= 16);
        let header: [u8; 16] = raw[..16].try_into().unwrap();
        let sqlite_magic = b"SQLite format 3\0";
        assert_ne!(&header, sqlite_magic, "DB must not be plaintext SQLite");
    }

    /// Gate 2: dispatch_delta routes to the per-table merge functions.
    /// Calls dispatch_delta directly and verifies the write-set shape.
    #[test]
    fn merge_routes_to_per_table_fns() {
        use simonides_sync::delta::{dispatch_delta, DrawerDeltaItem, LocalState, PalaceDelta};
        use simonides_core::sync::{ApplyOutcome, Hlc};

        let hlc = Hlc { wall_time: 2_000_000, logical: 0, node_id: [2u8; 16] };
        let delta = PalaceDelta {
            drawers: vec![DrawerDeltaItem {
                drawer_id: "test-drawer-id".to_string(),
                content: "routed content".to_string(),
                automerge_bytes: vec![],
                hlc,
                wing: "test".to_string(),
                room: "route".to_string(),
                provenance: "gate-test".to_string(),
                tombstoned: false,
                name: None,
            }],
            palace_records: vec![],
            tunnels: vec![],
            triples: vec![],
            entities: vec![],
        };

        let local = LocalState::default();
        let result = dispatch_delta(&delta, &local).expect("dispatch_delta");

        // A new drawer (no local match) produces a clean outcome and one drawer write.
        assert!(
            matches!(result.outcome, ApplyOutcome::Clean),
            "expected Clean outcome for net-new drawer"
        );
        assert_eq!(result.drawer_writes.len(), 1, "one drawer write expected");
        assert_eq!(result.drawer_writes[0].drawer_id, "test-drawer-id");
        assert_eq!(result.triple_writes.len(), 0);
        assert_eq!(result.entity_writes.len(), 0);
        assert_eq!(result.tunnel_writes.len(), 0);
    }

    /// Gate 3: merge then reindex â€” reindex_pending flag raised, health() surfaces staleness.
    /// Note: does NOT call rebuild_indexes() (which requires the full embedder); only
    /// verifies that the flag is raised and health() reports it.
    #[test]
    fn merge_then_reindex_no_silent_stale() {
        use simonides_core::sync::{Hlc, SyncDelta, SyncableStore};
        use simonides_sync::delta::{DrawerDeltaItem, PalaceDelta, encode_delta};

        let (store, _tmp) = store();

        let hlc = Hlc { wall_time: 3_000_000, logical: 0, node_id: [3u8; 16] };
        let drawer_id = uuid::Uuid::now_v7().to_string();
        let delta = PalaceDelta {
            drawers: vec![DrawerDeltaItem {
                drawer_id: drawer_id.clone(),
                content: "reindex gate content".to_string(),
                automerge_bytes: vec![],
                hlc,
                wing: "test".to_string(),
                room: "reindex".to_string(),
                provenance: "gate-test".to_string(),
                tombstoned: false,
                name: None,
            }],
            palace_records: vec![],
            tunnels: vec![],
            triples: vec![],
            entities: vec![],
        };
        let payload = encode_delta(&delta).expect("encode");
        store.apply_remote_delta(SyncDelta { payload, source_hlc: hlc }).expect("apply");

        // reindex_pending must be 1 for the merged drawer.
        let pending: i64 = {
            let g = store.conn.lock().unwrap();
            g.query_row(
                "SELECT reindex_pending FROM drawers WHERE id = ?1",
                rusqlite::params![drawer_id],
                |r| r.get(0),
            )
            .expect("read reindex_pending")
        };
        assert_eq!(pending, 1, "merged drawer must have reindex_pending = 1");

        // health() must report > 0 pending.
        let h = store.health().expect("health");
        assert!(h.reindex_pending > 0, "health() must surface reindex_pending > 0 after merge");
    }

    /// Gate 4: concurrent delta apply converges â€” two stores exchanging deltas
    /// reach the same final state for a shared entity.
    #[test]
    fn concurrent_delta_apply_converges() {
        use simonides_core::sync::{Hlc, PeerClock, SyncableStore};

        // Two in-memory stores, each seeded with the same entity UUID so
        // merge_entities() finds a local match and applies the LWW merge.
        let dir_a = tempfile::tempdir().expect("dir_a");
        let dir_b = tempfile::tempdir().expect("dir_b");
        let store_a = PalaceStore::open_in_memory(dir_a.path(), CORRECT_KEY).expect("store_a");
        let store_b = PalaceStore::open_in_memory(dir_b.path(), CORRECT_KEY).expect("store_b");

        let entity_id = uuid::Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();

        // Seed both stores with the same entity at wall_time=1000.
        for s in [&store_a, &store_b] {
            let g = s.conn.lock().unwrap();
            g.execute(
                "INSERT INTO entities \
                 (id, canonical_name, entity_type, source_drawer_id, \
                  created_at, updated_at, properties, hlc_wall, hlc_logical, hlc_node_id) \
                 VALUES (?1, 'ConvergeEntity', 'person', NULL, ?2, ?2, '{}', 1000, 0, ZEROBLOB(16))",
                rusqlite::params![entity_id, now],
            )
            .expect("seed entity");
        }

        // Store A: produce a snapshot, apply it to B.
        let peer_clock = PeerClock {
            peer_node_id: [0u8; 16],
            last_seen_hlc: Hlc { wall_time: 0, logical: 0, node_id: [0u8; 16] },
        };
        let delta_a = store_a.produce_delta(&peer_clock).expect("produce_delta A");
        store_b.apply_remote_delta(delta_a).expect("apply Aâ†’B");

        // Store B: produce a snapshot, apply it to A.
        let delta_b = store_b.produce_delta(&peer_clock).expect("produce_delta B");
        store_a.apply_remote_delta(delta_b).expect("apply Bâ†’A");

        // Both stores must have exactly one entity with the same canonical_name.
        let count_a: i64 = {
            let g = store_a.conn.lock().unwrap();
            g.query_row("SELECT COUNT(*) FROM entities WHERE canonical_name = 'ConvergeEntity'", [], |r| r.get(0))
                .expect("count A")
        };
        let count_b: i64 = {
            let g = store_b.conn.lock().unwrap();
            g.query_row("SELECT COUNT(*) FROM entities WHERE canonical_name = 'ConvergeEntity'", [], |r| r.get(0))
                .expect("count B")
        };
        assert_eq!(count_a, 1, "store A must have exactly one ConvergeEntity after sync");
        assert_eq!(count_b, 1, "store B must have exactly one ConvergeEntity after sync");
    }

    // â”€â”€ P1 gates: in-place edit + wing/room enumeration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// P1 gate: an in-place edit mutates the content, re-stales the derived
    /// indexes (`reindex_pending = 1`), regenerates the lexical index so search
    /// finds the NEW content, and drops the OLD content from that index.
    ///
    /// Hermetic by construction (crate rule: no environment-coupled tests). The
    /// lexical index is regenerated synchronously inside `update_drawer`, so this
    /// asserts the reindex without `rebuild_indexes()` â€” which would couple the
    /// test to the BGE ONNX embedder download. The vector half of the reindex is
    /// covered by the dedicated `rebuild_indexes_*` tests. `search()` degrades to
    /// BM25 when the embedder is unavailable, so the NEW-content assertion holds
    /// regardless of vector state.
    #[test]
    fn update_drawer_mutates_and_reindexes() {
        use crate::retrieval::SearchResult;

        let (s, _tmp) = store();
        let d = s
            .insert("w", "r", "alpha_marker_term cedar original body", "test")
            .expect("insert");

        // Simulate a completed reindex so the edit's re-staling is observable
        // (clearing the flag directly avoids coupling to the vector embedder).
        {
            let g = s.conn.lock().unwrap();
            g.execute(
                "UPDATE drawers SET reindex_pending = 0 WHERE id = ?1",
                params![d.id],
            )
            .expect("clear reindex_pending");
        }
        assert!(
            !s.get(&d.id).unwrap().unwrap().reindex_pending,
            "precondition: reindex_pending cleared before the edit"
        );

        let updated = s
            .update_drawer(&d.id, "beta_marker_term cedar revised body")
            .expect("update_drawer");
        assert_eq!(updated.content, "beta_marker_term cedar revised body");
        assert!(updated.reindex_pending, "edit must re-stale the vector index");

        let refetched = s.get(&d.id).unwrap().unwrap();
        assert_eq!(refetched.content, "beta_marker_term cedar revised body");
        assert!(
            refetched.reindex_pending,
            "persisted row must show reindex_pending after the edit"
        );

        // NEW content is findable â€” the lexical index was regenerated by the edit.
        match s.search("beta_marker_term", 10).expect("search new term") {
            SearchResult::Hits(hits) => assert!(
                hits.iter().any(|h| h.id == d.id),
                "edited drawer must be findable by its NEW content"
            ),
            SearchResult::Empty { query } => {
                panic!("expected Hits for new term; got Empty {{ query: {query:?} }}")
            }
        }

        // OLD content is gone from the lexical index.
        let old_bm25 = s
            .search_bm25_ranked("alpha_marker_term", 10)
            .expect("bm25 old term");
        assert!(
            !old_bm25.iter().any(|(id, _)| id == &d.id),
            "OLD content must be gone from the lexical index after the edit"
        );
    }

    /// P1 gate: the edit is a real CRDT op â€” it updates the drawer's Automerge
    /// doc in `crdt_docs` â€” so it survives the sync round-trip
    /// `produce_delta` â†’ `apply_remote_delta` into a peer store. A bare SQL
    /// UPDATE would write no AM doc (assertion 1) and, having no doc to carry the
    /// change, would not propagate the content to the peer (assertion 2).
    #[test]
    fn update_drawer_is_crdt_consistent() {
        use automerge::{AutoCommit, ReadDoc, ScalarValue, Value};
        use simonides_core::sync::{Hlc, PeerClock, SyncableStore};

        let (a, _ta) = store();
        let d = a.insert("w", "r", "crdt_old_alpha", "test").expect("insert");
        a.update_drawer(&d.id, "crdt_new_beta").expect("update_drawer");

        // (1) Source of truth is a real CRDT op: A's Automerge doc for the drawer
        //     decodes to the NEW content. A bare SQL UPDATE writes no crdt_docs
        //     row, so this query_row would find nothing and the test would fail.
        let doc_bytes: Vec<u8> = {
            let g = a.conn.lock().unwrap();
            g.query_row(
                "SELECT doc_bytes FROM crdt_docs WHERE drawer_id = ?1",
                params![d.id],
                |r| r.get(0),
            )
            .expect("update_drawer must persist an Automerge doc (real CRDT op)")
        };
        let doc = AutoCommit::load(&doc_bytes).expect("load AM doc");
        let content = match doc.get(automerge::ROOT, "content").expect("get content key") {
            Some((Value::Scalar(sv), _)) => match sv.as_ref() {
                ScalarValue::Str(s) => s.to_string(),
                other => panic!("content scalar is not a string: {other:?}"),
            },
            other => panic!("AM doc is missing the content key: {other:?}"),
        };
        assert_eq!(content, "crdt_new_beta", "CRDT doc must carry the edited content");

        // (2) The edit survives produce_delta â†’ apply_remote_delta into a fresh
        //     peer store (which has no competing doc, so the merge is deterministic).
        let clock = PeerClock {
            peer_node_id: [0u8; 16],
            last_seen_hlc: Hlc { wall_time: 0, logical: 0, node_id: [0u8; 16] },
        };
        let delta = a.produce_delta(&clock).expect("produce_delta");
        let (b, _tb) = store();
        b.apply_remote_delta(delta).expect("apply_remote_delta");

        let got = b
            .get(&d.id)
            .expect("get on peer")
            .expect("drawer must be present on peer after sync");
        assert_eq!(
            got.content, "crdt_new_beta",
            "edit must survive the CRDT sync round-trip"
        );
    }

    /// PC gate: a move is a real CRDT op â€” it folds the new wing/room into the
    /// drawer's Automerge doc in `crdt_docs` â€” so it survives the sync round-trip
    /// `produce_delta` â†’ `apply_remote_delta` into a fresh peer. A bare SQL UPDATE
    /// would write no AM doc (assertion 1) and leave the move off the drawer's
    /// causal history. Hermetic: BM25/lexical path only, no embedder download.
    #[test]
    fn move_drawer_is_crdt_consistent() {
        use automerge::{AutoCommit, ReadDoc, ScalarValue, Value};
        use simonides_core::sync::{Hlc, PeerClock, SyncableStore};

        let (a, _ta) = store();
        let d = a.insert("old_wing", "old_room", "move_marker body", "test").expect("insert");
        let moved = a
            .move_drawer(&d.id, "new_wing", "new_room")
            .expect("move_drawer");
        assert_eq!(moved.wing, "new_wing");
        assert_eq!(moved.room, "new_room");

        // (1) The move is a real CRDT op: A's Automerge doc for the drawer decodes
        //     to the NEW wing/room. A bare SQL UPDATE writes no crdt_docs row, so
        //     this query_row would find nothing and the test would fail.
        let doc_bytes: Vec<u8> = {
            let g = a.conn.lock().unwrap();
            g.query_row(
                "SELECT doc_bytes FROM crdt_docs WHERE drawer_id = ?1",
                params![d.id],
                |r| r.get(0),
            )
            .expect("move_drawer must persist an Automerge doc (real CRDT op)")
        };
        let doc = AutoCommit::load(&doc_bytes).expect("load AM doc");
        let read_key = |key: &str| -> String {
            match doc.get(automerge::ROOT, key).expect("get key") {
                Some((Value::Scalar(sv), _)) => match sv.as_ref() {
                    ScalarValue::Str(s) => s.to_string(),
                    other => panic!("{key} scalar is not a string: {other:?}"),
                },
                other => panic!("AM doc is missing the {key} key: {other:?}"),
            }
        };
        assert_eq!(read_key("wing"), "new_wing", "CRDT doc must carry the moved wing");
        assert_eq!(read_key("room"), "new_room", "CRDT doc must carry the moved room");

        // (2) The move survives produce_delta â†’ apply_remote_delta into a fresh
        //     peer store.
        let clock = PeerClock {
            peer_node_id: [0u8; 16],
            last_seen_hlc: Hlc { wall_time: 0, logical: 0, node_id: [0u8; 16] },
        };
        let delta = a.produce_delta(&clock).expect("produce_delta");
        let (b, _tb) = store();
        b.apply_remote_delta(delta).expect("apply_remote_delta");

        let got = b
            .get(&d.id)
            .expect("get on peer")
            .expect("drawer must be present on peer after sync");
        assert_eq!(got.wing, "new_wing", "move must survive the CRDT sync round-trip");
        assert_eq!(got.room, "new_room", "move must survive the CRDT sync round-trip");
        assert_eq!(got.state, DrawerState::Live, "moved drawer stays live on the peer");
    }

    /// PC gate: a tombstone is a real CRDT op â€” it folds a `tombstoned = true`
    /// flag into the drawer's Automerge doc in `crdt_docs` â€” so the retirement
    /// survives the sync round-trip `produce_delta` â†’ `apply_remote_delta` into a
    /// fresh peer. A bare SQL state flip would write no AM doc (assertion 1) and
    /// leave the tombstone off the drawer's causal history. Hermetic: no embedder.
    #[test]
    fn tombstone_is_crdt_consistent() {
        use automerge::{AutoCommit, ReadDoc, ScalarValue, Value};
        use simonides_core::sync::{Hlc, PeerClock, SyncableStore};

        let (a, _ta) = store();
        let d = a.insert("w", "r", "tomb_marker body", "test").expect("insert");
        a.tombstone(&d.id).expect("tombstone");

        // (1) The tombstone is a real CRDT op: A's Automerge doc for the drawer
        //     decodes with tombstoned = true. A bare SQL state flip writes no
        //     crdt_docs row, so this query_row would find nothing and the test
        //     would fail.
        let doc_bytes: Vec<u8> = {
            let g = a.conn.lock().unwrap();
            g.query_row(
                "SELECT doc_bytes FROM crdt_docs WHERE drawer_id = ?1",
                params![d.id],
                |r| r.get(0),
            )
            .expect("tombstone must persist an Automerge doc (real CRDT op)")
        };
        let doc = AutoCommit::load(&doc_bytes).expect("load AM doc");
        match doc.get(automerge::ROOT, "tombstoned").expect("get tombstoned key") {
            Some((Value::Scalar(sv), _)) => match sv.as_ref() {
                ScalarValue::Boolean(b) => {
                    assert!(*b, "CRDT doc must carry tombstoned = true")
                }
                other => panic!("tombstoned scalar is not a bool: {other:?}"),
            },
            other => panic!("AM doc is missing the tombstoned key: {other:?}"),
        }

        // (2) The tombstone survives produce_delta â†’ apply_remote_delta into a
        //     fresh peer store: the drawer is present but retired.
        let clock = PeerClock {
            peer_node_id: [0u8; 16],
            last_seen_hlc: Hlc { wall_time: 0, logical: 0, node_id: [0u8; 16] },
        };
        let delta = a.produce_delta(&clock).expect("produce_delta");
        let (b, _tb) = store();
        b.apply_remote_delta(delta).expect("apply_remote_delta");

        let got = b
            .get(&d.id)
            .expect("get on peer")
            .expect("drawer row must still exist on peer (tombstone never deletes)");
        assert_eq!(
            got.state,
            DrawerState::Tombstoned,
            "tombstone must survive the CRDT sync round-trip"
        );
        // Retired drawers are excluded from the live listing on the peer.
        let live = b.list(100).expect("list on peer");
        assert!(
            !live.iter().any(|dr| dr.id == d.id),
            "tombstoned drawer must be excluded from the peer's live list"
        );
    }

    /// P1 gate: wing/room enumeration returns the distinct live labels, sorted,
    /// scoped correctly, and excludes tombstoned-only locations.
    #[test]
    fn list_wings_rooms_enumerate() {
        let (s, _tmp) = store();
        s.insert("alpha", "one", "a1", "test").expect("insert alpha/one");
        s.insert("alpha", "two", "a2", "test").expect("insert alpha/two");
        s.insert("beta", "one", "b1", "test").expect("insert beta/one");
        let dead = s.insert("gamma", "solo", "g1", "test").expect("insert gamma/solo");
        s.tombstone(&dead.id).expect("tombstone gamma");

        // Wings: distinct, sorted ascending, tombstoned-only 'gamma' excluded.
        let wings = s.list_wings().expect("list_wings");
        assert_eq!(wings, vec!["alpha".to_string(), "beta".to_string()]);

        // Rooms are scoped to the wing and sorted ascending.
        assert_eq!(
            s.list_rooms("alpha").expect("list_rooms alpha"),
            vec!["one".to_string(), "two".to_string()]
        );
        assert_eq!(
            s.list_rooms("beta").expect("list_rooms beta"),
            vec!["one".to_string()]
        );

        // A tombstoned-only wing enumerates no rooms.
        assert!(
            s.list_rooms("gamma").expect("list_rooms gamma").is_empty(),
            "tombstoned-only wing must enumerate no live rooms"
        );
        // An unknown wing returns empty, never an error.
        assert!(
            s.list_rooms("nonexistent").expect("list_rooms unknown").is_empty(),
            "unknown wing must return an empty vec"
        );
    }

    // â”€â”€ P2 gates: tunnels API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// P2 gate: creating the same `(from, to, relation)` triple twice yields a
    /// single edge (idempotent upsert on the UNIQUE triple), and the edge's id
    /// is stable across the repeat create.
    #[test]
    fn create_tunnel_idempotent() {
        let (s, _tmp) = store();
        let a = s.insert("w", "r", "node a", "test").expect("insert a");
        let b = s.insert("w", "r", "node b", "test").expect("insert b");

        let t1 = s
            .create_tunnel(&a.id, &b.id, "relates_to", Some(0.9))
            .expect("create tunnel 1");
        let t2 = s
            .create_tunnel(&a.id, &b.id, "relates_to", Some(0.5))
            .expect("create tunnel 2 (same triple)");

        assert_eq!(t1.id, t2.id, "repeat create of the same triple must reuse the id");

        // Exactly one edge exists for the triple, both directions considered.
        let edges = s.list_tunnels(&a.id).expect("list_tunnels a");
        assert_eq!(edges.len(), 1, "same triple twice must yield exactly one edge");
        assert_eq!(edges[0].confidence, 0.5, "repeat create refreshes confidence (upsert)");
    }

    /// P2 gate: `list_tunnels` returns edges in BOTH directions â€” an edge where
    /// the drawer is the source and an edge where it is the destination.
    #[test]
    fn list_tunnels_both_directions() {
        let (s, _tmp) = store();
        let hub = s.insert("w", "r", "hub", "test").expect("insert hub");
        let out = s.insert("w", "r", "outgoing target", "test").expect("insert out");
        let inc = s.insert("w", "r", "incoming source", "test").expect("insert inc");

        s.create_tunnel(&hub.id, &out.id, "points_to", None).expect("outgoing edge");
        s.create_tunnel(&inc.id, &hub.id, "points_to", None).expect("incoming edge");

        let edges = s.list_tunnels(&hub.id).expect("list_tunnels hub");
        assert_eq!(edges.len(), 2, "hub must see both its outgoing and incoming edges");
        assert!(
            edges.iter().any(|e| e.from_drawer_id == hub.id && e.to_drawer_id == out.id),
            "outgoing edge must be listed"
        );
        assert!(
            edges.iter().any(|e| e.from_drawer_id == inc.id && e.to_drawer_id == hub.id),
            "incoming edge must be listed"
        );
    }

    /// Cross-tier primitive gate: `tunnels_referencing` is EXACT-ID-ONLY and
    /// tolerates absence. Unlike `list_tunnels` it never resolves a prefix (that
    /// would be a tier oracle) and never errors on a non-resident id — the
    /// cross-tier BFS calls it against stores that hold the edge but not the
    /// drawer, and against stores that hold neither.
    #[test]
    fn tunnels_referencing_is_exact_id_only_and_tolerates_absence() {
        let (s, _tmp) = store();
        let hub = s.insert("w", "r", "hub", "test").expect("insert hub");
        let out = s.insert("w", "r", "outgoing target", "test").expect("insert out");
        let inc = s.insert("w", "r", "incoming source", "test").expect("insert inc");
        s.create_tunnel(&hub.id, &out.id, "points_to", None).expect("outgoing edge");
        s.create_tunnel(&inc.id, &hub.id, "points_to", None).expect("incoming edge");

        // Exact id: both directions, the same pair list_tunnels resolves to.
        let edges = s.tunnels_referencing(&hub.id).expect("exact id");
        assert_eq!(edges.len(), 2, "exact id must see both incoming and outgoing edges");

        // A prefix of a real id must match NOTHING here (list_tunnels would
        // resolve it) — exact-id-only is the tier-oracle guard.
        let prefix = &hub.id[..8];
        let by_prefix = s.tunnels_referencing(prefix).expect("prefix is not an error");
        assert!(by_prefix.is_empty(), "a prefix must match nothing: exact-id-only");

        // A well-formed but non-resident id contributes empty, never an error.
        let absent = s
            .tunnels_referencing("00000000-0000-0000-0000-000000000000")
            .expect("absence must be Ok(empty), not Err");
        assert!(absent.is_empty(), "no edges for an id this store never saw");
    }

    /// P2 gate: `follow_tunnels` is depth-capped and cycle-safe â€” it terminates
    /// and respects the depth bound even when the graph contains a cycle.
    #[test]
    fn follow_tunnels_depth_capped() {
        let (s, _tmp) = store();
        let a = s.insert("w", "r", "cycle a", "test").expect("insert a");
        let b = s.insert("w", "r", "cycle b", "test").expect("insert b");
        let c = s.insert("w", "r", "cycle c", "test").expect("insert c");

        // A --next--> B --next--> C --next--> A  (a 3-cycle).
        s.create_tunnel(&a.id, &b.id, "next", None).expect("a->b");
        s.create_tunnel(&b.id, &c.id, "next", None).expect("b->c");
        s.create_tunnel(&c.id, &a.id, "next", None).expect("c->a");

        // depth 1 from A reaches only B.
        let d1 = s.follow_tunnels(&a.id, "next", 1).expect("follow depth 1");
        assert_eq!(d1.len(), 1, "depth 1 reaches exactly one hop");
        assert_eq!(d1[0].id, b.id);

        // depth 2 from A reaches B then C.
        let d2 = s.follow_tunnels(&a.id, "next", 2).expect("follow depth 2");
        assert_eq!(d2.len(), 2, "depth 2 reaches two hops");

        // A huge depth on a cycle must still terminate and, because the start is
        // in `visited`, never re-yield A â€” it reaches only B and C (the two other
        // distinct nodes). This is the runaway/cycle-safety assertion.
        let deep = s.follow_tunnels(&a.id, "next", 1000).expect("follow deep on a cycle");
        assert_eq!(
            deep.len(),
            2,
            "a cycle must not run away: only the two distinct reachable nodes are returned"
        );
        assert!(!deep.iter().any(|d| d.id == a.id), "start must not be re-yielded on a cycle");

        // depth 0 reaches nothing.
        assert!(
            s.follow_tunnels(&a.id, "next", 0).expect("follow depth 0").is_empty(),
            "depth 0 must return no drawers"
        );
    }

    /// P2 gate: a created tunnel is a real CRDT op â€” it survives
    /// `produce_delta` â†’ `apply_remote_delta` into a fresh peer store.
    #[test]
    fn tunnel_survives_crdt_roundtrip() {
        use simonides_core::sync::{Hlc, PeerClock, SyncableStore};

        let (a, _ta) = store();
        let d1 = a.insert("w", "r", "roundtrip src", "test").expect("insert src");
        let d2 = a.insert("w", "r", "roundtrip dst", "test").expect("insert dst");
        let created = a
            .create_tunnel(&d1.id, &d2.id, "linked", Some(0.8))
            .expect("create tunnel");

        let clock = PeerClock {
            peer_node_id: [0u8; 16],
            last_seen_hlc: Hlc { wall_time: 0, logical: 0, node_id: [0u8; 16] },
        };
        let delta = a.produce_delta(&clock).expect("produce_delta");
        let (b, _tb) = store();
        b.apply_remote_delta(delta).expect("apply_remote_delta");

        // The peer must carry the edge (both its drawers synced too, so
        // list_tunnels resolves the endpoint).
        let edges = b.list_tunnels(&d1.id).expect("list_tunnels on peer");
        assert_eq!(edges.len(), 1, "edge must survive the sync round-trip");
        assert_eq!(edges[0].id, created.id, "same edge id after merge");
        assert_eq!(edges[0].from_drawer_id, d1.id);
        assert_eq!(edges[0].to_drawer_id, d2.id);
        assert_eq!(edges[0].relation, "linked");
    }

    /// P2 gate: `create_tunnel` rejects a missing endpoint LOUDLY â€” it never
    /// writes a silent dangling edge.
    #[test]
    fn create_tunnel_rejects_missing_endpoint() {
        let (s, _tmp) = store();
        let real = s.insert("w", "r", "real node", "test").expect("insert real");
        let ghost = "00000000-0000-7000-8000-000000000000"; // well-formed but absent

        // Missing `to`.
        let err = s
            .create_tunnel(&real.id, ghost, "rel", None)
            .expect_err("must reject a missing destination");
        assert!(matches!(err, PalaceError::NotFound { .. }), "missing endpoint -> NotFound");

        // Missing `from`.
        let err = s
            .create_tunnel(ghost, &real.id, "rel", None)
            .expect_err("must reject a missing source");
        assert!(matches!(err, PalaceError::NotFound { .. }), "missing endpoint -> NotFound");

        // Nothing was written â€” the real node has no dangling edges.
        assert!(
            s.list_tunnels(&real.id).expect("list_tunnels real").is_empty(),
            "a rejected create must leave no dangling edge"
        );
    }

    /// P2 gate: tunnel endpoint lookups are prefix-tolerant â€” a short id prefix
    /// resolves through `resolve()`, both when creating and when listing.
    ///
    /// Uses controlled ids (not `insert`'s UUID v7) so a genuinely SHORT prefix
    /// is unambiguous: two v7 ids created microseconds apart share their leading
    /// timestamp hex, which would make an 8-char prefix ambiguous and flake. The
    /// controlled ids share no prefix, so a 4-char prefix is decisive.
    #[test]
    fn tunnel_lookup_prefix_tolerant() {
        let (s, _tmp) = store();
        let a_id = "aaaaaaaa-0000-7000-8000-000000000001";
        let b_id = "bbbbbbbb-0000-7000-8000-000000000002";
        insert_raw(&s, a_id, "w", "r", "prefix node a");
        insert_raw(&s, b_id, "w", "r", "prefix node b");

        // Create using short prefixes for both endpoints.
        let t = s
            .create_tunnel("aaaa", "bbbb", "prefixed", None)
            .expect("create via short prefixes");
        assert_eq!(t.from_drawer_id, a_id, "prefix must resolve to the full source id");
        assert_eq!(t.to_drawer_id, b_id, "prefix must resolve to the full dest id");

        // List using a prefix, too.
        let edges = s.list_tunnels("aaaa").expect("list via prefix");
        assert_eq!(edges.len(), 1, "prefix lookup must find the edge");
        assert_eq!(edges[0].id, t.id);
    }

    /// P2 gate: a tombstoned tunnel is excluded from both `list_tunnels` and
    /// `follow_tunnels`, but the row is never physically deleted.
    #[test]
    fn tombstoned_tunnel_excluded_from_list_and_follow() {
        let (s, _tmp) = store();
        let a = s.insert("w", "r", "ts a", "test").expect("insert a");
        let b = s.insert("w", "r", "ts b", "test").expect("insert b");

        let t = s.create_tunnel(&a.id, &b.id, "gone", None).expect("create tunnel");
        // Present before tombstone.
        assert_eq!(s.list_tunnels(&a.id).expect("list before").len(), 1);
        assert_eq!(s.follow_tunnels(&a.id, "gone", 1).expect("follow before").len(), 1);

        s.tombstone_tunnel(&t.id).expect("tombstone tunnel");

        // Excluded from list and follow after tombstone.
        assert!(
            s.list_tunnels(&a.id).expect("list after").is_empty(),
            "tombstoned edge must be excluded from list_tunnels"
        );
        assert!(
            s.follow_tunnels(&a.id, "gone", 1).expect("follow after").is_empty(),
            "tombstoned edge must not be traversed by follow_tunnels"
        );

        // Row is NOT physically deleted â€” it still exists, flagged tombstoned.
        // Scope the guard so it drops before the tombstone_tunnel call below
        // (the store's Mutex is non-reentrant â€” holding it here while calling a
        // method that re-locks would deadlock on the same thread).
        let (count, flagged): (i64, i64) = {
            let guard = s.conn.lock().unwrap();
            guard
                .query_row(
                    "SELECT COUNT(*), COALESCE(MAX(tombstoned), 0) FROM tunnels WHERE id = ?1",
                    params![t.id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .expect("query tombstoned row")
        };
        assert_eq!(count, 1, "tombstoned tunnel row must still exist (never hard-deleted)");
        assert_eq!(flagged, 1, "the surviving row must be flagged tombstoned");

        // Tombstoning an unknown tunnel id is a loud NotFound.
        let err = s
            .tombstone_tunnel("no-such-tunnel-id")
            .expect_err("unknown id must error");
        assert!(matches!(err, PalaceError::NotFound { .. }));
    }

    // â”€â”€ P5 gates: meta + diary parity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// P5 gate (brief Â§6, the Â§4 ordering-bug class): `diary_read` defaults
    /// NEWEST-FIRST. Write â‰¥3 entries; `diary_read(agent, 2)` returns the newest
    /// two, newest first. This is the exact class of the 2026-05-30 `list_drawers`
    /// bug (a "recent entries" read that silently returned oldest-first); the
    /// ordering is pinned here, in the engine, so no wrapper can reintroduce it.
    /// Ordering asserted by matching the returned drawer ids (same discipline as
    /// `newest_first_ordering`), which is monotone under UUID v7.
    #[test]
    fn diary_read_is_newest_first() {
        let (s, _tmp) = store();
        let e1 = s.diary_write("scribe", "entry one", Some("t1")).expect("write 1");
        let e2 = s.diary_write("scribe", "entry two", Some("t2")).expect("write 2");
        let e3 = s.diary_write("scribe", "entry three", None).expect("write 3");

        let recent = s.diary_read("scribe", 2).expect("diary_read");
        assert_eq!(recent.len(), 2, "last_n=2 must bound the result to two entries");

        // Newest first: e3 then e2. e1 is beyond the last_n=2 bound.
        assert_eq!(recent[0].id, e3.id, "newest entry must be first");
        assert_eq!(recent[0].content, "entry three");
        assert_eq!(recent[1].id, e2.id, "second-newest must be second");
        assert_eq!(recent[1].content, "entry two");
        assert!(
            recent[0].id >= recent[1].id,
            "diary_read must be ordered by id DESC (newest-first)"
        );
        assert!(
            recent.iter().all(|d| d.id != e1.id),
            "the oldest entry must be excluded by the last_n=2 bound"
        );
    }

    /// P5 gate (brief Â§6, proves Â§3): a diary entry is an ordinary drawer, so it
    /// syncs for free. A `diary_write` on store A survives
    /// `produce_delta` â†’ `apply_remote_delta` onto a FRESH store B and is readable
    /// via `diary_read` on B. This is the deliberate INVERSE of P4's no-CRDT
    /// closet gate â€” diary rides the drawer CRDT path with no new sync work.
    #[test]
    fn diary_entry_is_a_durable_syncing_drawer() {
        use simonides_core::sync::{Hlc, PeerClock, SyncableStore};

        let (store_a, _ta) = store();
        let (store_b, _tb) = store();

        let written = store_a
            .diary_write("ada", "a durable journal line", Some("handoff"))
            .expect("diary_write on A");

        let clock = PeerClock {
            peer_node_id: [0u8; 16],
            last_seen_hlc: Hlc { wall_time: 0, logical: 0, node_id: [0u8; 16] },
        };
        let delta = store_a.produce_delta(&clock).expect("produce_delta A");
        store_b.apply_remote_delta(delta).expect("apply Aâ†’B onto fresh B");

        let on_b = store_b.diary_read("ada", 10).expect("diary_read on B");
        assert_eq!(on_b.len(), 1, "the diary entry must have arrived on fresh B via sync");
        assert_eq!(on_b[0].id, written.id, "same drawer id after sync");
        assert_eq!(on_b[0].content, "a durable journal line", "content survived sync");
        assert_eq!(
            on_b[0].name.as_deref(),
            Some("handoff"),
            "topic (stamped in name) survived sync"
        );
    }

    /// P5 gate (brief Â§6): `set_meta` refuses the reserved system keys LOUDLY and
    /// leaves the real value untouched. A caller clobbering `schema_version` would
    /// corrupt migration state; the guard makes that structurally impossible.
    #[test]
    fn set_meta_rejects_reserved_system_keys() {
        let (s, _tmp) = store();

        // The real schema_version is seeded to "5" on open (migrations).
        let before = s.get_meta("schema_version").expect("get schema_version");
        assert_eq!(before.as_deref(), Some("5"), "precondition: schema_version seeded to 5");

        // Attempting to set it is a loud, typed error.
        let err = s
            .set_meta("schema_version", "99")
            .expect_err("reserved key must be refused");
        assert!(
            matches!(err, PalaceError::ReservedMetaKey { ref key } if key == "schema_version"),
            "must be a ReservedMetaKey error naming the key, got: {err:?}"
        );

        // And nothing was written â€” the real value is untouched.
        let after = s.get_meta("schema_version").expect("get schema_version again");
        assert_eq!(after.as_deref(), Some("5"), "schema_version must be untouched after refusal");

        // The embedding model version is the other reserved key.
        let err2 = s
            .set_meta("embedding_model_version", "tampered")
            .expect_err("embedding_model_version must be refused");
        assert!(matches!(err2, PalaceError::ReservedMetaKey { .. }));
    }

    /// P5 gate (brief Â§6): a user meta key round-trips through set/get, an update
    /// overwrites, and an absent key reads back as `Ok(None)` (not an error).
    #[test]
    fn get_set_meta_round_trip() {
        let (s, _tmp) = store();

        // Absent key â†’ Ok(None).
        assert_eq!(
            s.get_meta("no_such_key").expect("get absent"),
            None,
            "an unset key must read back as Ok(None)"
        );

        // Set then get.
        s.set_meta("ui_theme", "dark").expect("set user key");
        assert_eq!(s.get_meta("ui_theme").expect("get user key").as_deref(), Some("dark"));

        // Upsert overwrites.
        s.set_meta("ui_theme", "light").expect("update user key");
        assert_eq!(s.get_meta("ui_theme").expect("get user key again").as_deref(), Some("light"));
    }

    /// P5 gate (brief Â§6): an agent with no diary entries yields `Ok(vec![])`, not
    /// an error. An empty journal is a valid state (unlike an unknown KG entity).
    #[test]
    fn diary_read_empty_agent_is_ok_empty() {
        let (s, _tmp) = store();
        let entries = s.diary_read("never_journaled", 5).expect("empty diary must be Ok");
        assert!(entries.is_empty(), "no entries must yield an empty vec, not an error");
    }
}
