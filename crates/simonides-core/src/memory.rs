//! Memory store types.
//!
//! The memory store has two wings: standard (full extraction pipeline,
//! embeddings, summaries) and premium (literal text only, encrypted at rest
//! with a higher-friction unlock). See spec section 5.3.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MemoryWing {
    #[default]
    Standard,
    Premium,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemorySource {
    #[default]
    Conversation,
    Document,
    Extraction,
    Manual,
    Skill,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryEntry {
    pub id: String,
    pub wing: MemoryWing,
    pub source: MemorySource,
    pub content: String,
    pub source_ref: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryStats {
    pub standard_entries: u64,
    pub premium_entries: u64,
    pub disk_bytes: u64,
    pub embeddings_count: u64,
    pub last_consolidation_at: Option<chrono::DateTime<chrono::Utc>>,
    // Palace-aware counts. Filled in when the palace storage layer is active;
    // zero on a legacy flat-entries-only store. Additive: existing callers that
    // construct MemoryStats { ... } literally still compile because all new
    // fields have Default impls (u64 → 0).
    #[serde(default)]
    pub palace_wings: u64,
    #[serde(default)]
    pub palace_rooms: u64,
    #[serde(default)]
    pub palace_drawers: u64,
    #[serde(default)]
    pub kg_entities: u64,
    #[serde(default)]
    pub kg_triples: u64,
    #[serde(default)]
    pub by_source: HashMap<String, u64>,
}

// ============================================================================
// Palace structure types (adapted from the MemPalace data model).
// All additive; existing memory types above are not changed.
// ============================================================================

/// A wing of the palace — an entity-partition (a person, a project, a domain).
/// Lives in exactly one tier (standard or premium). The tier determines which
/// database file backs the wing; all rooms / closets / drawers within the wing
/// inherit the same tier by construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalaceWing {
    pub id: String,
    pub name: String,
    pub tier: MemoryWing,
    pub description: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// A room within a wing — topic-or-time grouping. Default usage is one room
/// per day or per session, but topical rooms (e.g. "fundraising") are also
/// supported.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Room {
    pub id: String,
    pub wing_id: String,
    pub name: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// A closet within a room — topical grouping of drawers. Optional; drawers
/// can sit directly under a room with no closet.
///
/// `content` and `source_file`
/// are the indexing-layer fields. `content` carries the packed pointer-
/// document body in the local MemPalace convention
/// (`topic|entities|→drawer_id_a,drawer_id_b` lines, greedily packed to
/// ~1500 chars per closet). `source_file` is the file the closet was
/// extracted from. Both are optional; non-indexing closets leave them
/// None and are perfectly usable as plain topical groupings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Closet {
    pub id: String,
    pub room_id: String,
    pub label: String,
    pub summary: Option<String>,
    pub content: Option<String>,
    pub source_file: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// A drawer — the canonical unit of content. Verbatim, never summarized.
/// Drawers carry full palace addressing (wing/room/closet) plus the
/// existing flat-entry provenance (source, source_ref).
///
/// `source_file` and
/// `chunk_index` are the indexing-layer fields. `source_file` is the
/// full path of the file this drawer was mined from; `chunk_index` is
/// the 0-based position within the source file's chunk sequence. The
/// pair lets the indexing-layer drawer-grep neighbor expansion find the
/// best-matching chunk and its ±radius siblings. Both optional;
/// drawers created via `add_drawer` leave them None.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drawer {
    pub id: String,
    pub wing_id: String,
    pub room_id: String,
    pub closet_id: Option<String>,
    pub source: MemorySource,
    pub content: String,
    pub source_ref: Option<String>,
    pub added_by: Option<String>,
    pub source_file: Option<String>,
    pub chunk_index: Option<i64>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// A tunnel — typed directional edge between two drawers. May cross wings.
/// Relations are free-form strings ("supersedes", "refines", "contradicts",
/// "depends_on", ...). v1 does not constrain the relation vocabulary; that's
/// a v1.x design call when usage patterns emerge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tunnel {
    pub id: String,
    pub from_drawer_id: String,
    pub to_drawer_id: String,
    pub relation: String,
    pub confidence: Option<f32>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

// ============================================================================
// Knowledge graph types (adapted from MemPalace).
// KG lives in STANDARD TIER ONLY. Premium content never produces KG facts.
// Temporal validity follows the MemPalace knowledge-graph model.
// ============================================================================

/// A node in the knowledge graph — a person, project, tool, concept, or
/// other entity. Identified by `id` (UUID) but addressable by `(name, type)`
/// via the table's UNIQUE constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgEntity {
    pub id: String,
    pub name: String,
    /// e.g. "person", "project", "tool", "concept". Free-form string in v1.
    pub entity_type: Option<String>,
    /// Free-form JSON for entity-specific attributes (e.g. email, GitHub
    /// handle, employee number). Stored as `serde_json::Value` so callers
    /// can attach arbitrary structured data without schema churn.
    pub properties: serde_json::Value,
}

/// A directed, typed, time-windowed edge between entities (or between an
/// entity and a literal value in the `object` slot). Temporal validity:
/// the triple is "true" for `as_of` if
/// `(valid_from IS NULL OR valid_from <= as_of) AND (valid_to IS NULL OR valid_to >= as_of)`.
///
/// Invalidation is bitemporal: `kg_invalidate()` sets `valid_to`, never
/// DELETEs the row. The triple stops appearing in current queries but
/// historical `as_of` queries still see it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgTriple {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub valid_from: Option<chrono::DateTime<chrono::Utc>>,
    pub valid_to: Option<chrono::DateTime<chrono::Utc>>,
    pub confidence: Option<f32>,
    /// Provenance: which closet the triple was extracted from. Stored as a
    /// raw TEXT id (no FK enforcement) because closets may live in either
    /// the standard or premium database and cross-database FKs aren't a
    /// thing in sqlite. By convention, only standard-tier closet IDs
    /// should appear here — premium content never produces KG facts.
    pub source_closet: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Direction filter for `kg_query_entity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KgDirection {
    /// Triples where the entity is the subject (outgoing edges).
    Outgoing,
    /// Triples where the entity is the object (incoming edges).
    Incoming,
    /// Both directions, deduped and ordered by `created_at`.
    Both,
}

// ============================================================================
// Retrieval-stack types: corpus profiling and personalization signals.
// All additive; existing types above are not changed.
// ============================================================================

/// Detected corpus kind, drives the auto-tune parameters.
/// `Mixed` is the conservative default when no clear profile dominates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CorpusProfileKind {
    /// Long-form, technical, dense vocabulary (academic prose, documentation,
    /// research notes). Tuned BM25 b=0.75, larger chunk_size, larger closets.
    ResearchDoc,
    /// Short-form, casual, high vocabulary repetition (chat exports, voice
    /// utterances captured as text). Tuned BM25 b=0.4, smaller chunks/closets.
    ChatExport,
    /// Voice-transcribed input with disfluencies. Similar to ChatExport but
    /// reserved for explicit downstream tuning of disfluency-tolerant scoring.
    Voice,
    /// Default when the profile detector doesn't see a clean signal.
    Mixed,
}

impl CorpusProfileKind {
    /// Convert to the on-disk TEXT representation matching the SQL CHECK constraint.
    pub fn as_str(&self) -> &'static str {
        match self {
            CorpusProfileKind::ResearchDoc => "research-doc",
            CorpusProfileKind::ChatExport => "chat-export",
            CorpusProfileKind::Voice => "voice",
            CorpusProfileKind::Mixed => "mixed",
        }
    }

    /// Parse from the on-disk TEXT representation. Returns None on unknown values
    /// rather than panicking so legacy / forward-compat data doesn't crash reads.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "research-doc" => Some(CorpusProfileKind::ResearchDoc),
            "chat-export" => Some(CorpusProfileKind::ChatExport),
            "voice" => Some(CorpusProfileKind::Voice),
            "mixed" => Some(CorpusProfileKind::Mixed),
            _ => None,
        }
    }
}

/// A snapshot of the corpus characteristics + the tuned retrieval parameters
/// derived from them. One row per profile-shift detection; latest row by
/// `snapshot_at` is the current profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusProfile {
    pub id: String,
    pub kind: CorpusProfileKind,
    /// Average drawer content length in characters at the snapshot time.
    pub avg_content_len: i64,
    /// Number of unique terms in the indexed corpus.
    pub vocab_size: i64,
    /// Total terms / unique terms; higher = more repetition (chat-style).
    pub vocab_repetition_ratio: f32,
    /// ISO 639-1 code if the language detector returned high-confidence output.
    pub primary_language: Option<String>,
    pub snapshot_at: chrono::DateTime<chrono::Utc>,
    pub drawer_count_at_snapshot: i64,
    // ---- Auto-tuned retrieval parameters ----
    /// BM25 length-normalization parameter. Higher b = stronger length penalty.
    pub bm25_b: f32,
    /// Target chunk size (chars) for entity-extraction and embedding windows.
    pub chunk_size: i64,
    /// Maximum chars per closet pointer document (canonical mempalace limit).
    pub closet_char_limit: i64,
}

/// The kind of engagement the user expressed against a query result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum EngagementSignal {
    /// User opened the drawer for full reading.
    Opened,
    /// User copied content from the drawer.
    Copied,
    /// User explicitly dismissed the result (told assistant it was wrong).
    Dismissed,
    /// User clicked a tunnel from the drawer to navigate elsewhere.
    Followed,
}

impl EngagementSignal {
    pub fn as_str(&self) -> &'static str {
        match self {
            EngagementSignal::Opened => "Opened",
            EngagementSignal::Copied => "Copied",
            EngagementSignal::Dismissed => "Dismissed",
            EngagementSignal::Followed => "Followed",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Opened" => Some(EngagementSignal::Opened),
            "Copied" => Some(EngagementSignal::Copied),
            "Dismissed" => Some(EngagementSignal::Dismissed),
            "Followed" => Some(EngagementSignal::Followed),
            _ => None,
        }
    }

    /// Per-signal weight used by the personalization-bias heuristic.
    /// Negative weight on `Dismissed` actively de-prioritizes similar results.
    pub fn weight(&self) -> f32 {
        match self {
            EngagementSignal::Opened => 1.0,
            EngagementSignal::Copied => 1.5,
            EngagementSignal::Followed => 1.2,
            EngagementSignal::Dismissed => -0.8,
        }
    }
}

/// A single engagement event recorded against a query result. Never synced
/// (simonides-sync explicitly excludes the feedback_signals table); never telemetered.
/// In Über Private mode `record_engagement()` is a no-op and stage 4 of the
/// retrieval pipeline does not read these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSignal {
    pub id: String,
    /// Correlates with the QueryResponse.query_id that produced the result.
    pub query_id: String,
    pub drawer_id: String,
    pub signal: EngagementSignal,
    pub recorded_at: chrono::DateTime<chrono::Utc>,
    /// Always the local-device user; reserved for future multi-account separation.
    pub user_id_local: String,
}

// ============================================================================
// Public retrieval-pipeline types: the five-stage pipeline's
// QueryRequest/QueryResponse surface for callers.
// ============================================================================

/// Top-level retrieval request from a caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// User's raw query text. Preserved unchanged through the pipeline; the reranker
    /// scores against this, not against the KG-expanded version.
    pub query: String,
    /// Optional recent assistant turns for disambiguation. Currently a stub —
    /// downstream stages may use this in future iterations.
    #[serde(default)]
    pub conversation_context: Option<Vec<String>>,
    /// Explicit filters narrowing the candidate pool.
    #[serde(default)]
    pub filters: QueryFilters,
    /// KG-expansion depth (locked-decision #19 default = 1; max clamped to 3).
    /// `None` uses the default.
    #[serde(default)]
    pub kg_expansion_depth: Option<u8>,
    /// Whether to run the cross-encoder reranker stage. Default true; can be
    /// disabled for low-latency cases that prefer the hybrid-only output.
    #[serde(default = "default_true")]
    pub rerank: bool,
    /// Whether to apply feedback-signal personalization. Default true; force-disabled
    /// at the storage layer when Über Private mode is active.
    #[serde(default = "default_true")]
    pub personalize: bool,
    /// How many results to return after assembly.
    #[serde(default)]
    pub max_results: Option<usize>,
    /// Deprecated: no semantics under v3 single-tier. Ignored. Kept for
    /// signature-compat with v0.2-era callers; will be removed when those callers
    /// are updated.
    #[serde(default)]
    #[allow(dead_code)]
    pub include_premium_legacy: bool,
}

fn default_true() -> bool { true }

/// Explicit filters applied during retrieval. All optional; absence = no filter
/// on that dimension.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryFilters {
    pub wing_ids: Option<Vec<String>>,
    pub room_ids: Option<Vec<String>>,
    pub closet_ids: Option<Vec<String>>,
    pub source_types: Option<Vec<MemorySource>>,
    pub date_range: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
}

/// One retrieval result with full provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedDrawer {
    pub drawer: Drawer,
    pub score: f32,
    pub score_breakdown: ScoreBreakdown,
    pub tunnels: Vec<Tunnel>,
    pub closet_context: Option<String>,
    pub has_divergent_siblings: bool,
}

/// Per-stage score contribution. Populated as the pipeline runs; zero for stages
/// that didn't execute (rerank disabled, personalize disabled, etc.).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub bm25: f32,
    pub vector: f32,
    pub hybrid: f32,
    pub rerank: f32,
    pub personalization: f32,
    pub final_score: f32,
}

/// Per-stage timing data, populated when debug-telemetry is enabled.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageTimings {
    pub kg_expansion_ms: f64,
    pub hybrid_retrieval_ms: f64,
    pub rerank_ms: f64,
    pub personalization_ms: f64,
    pub assembly_ms: f64,
}

/// Top-level retrieval response returned to the caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub results: Vec<RankedDrawer>,
    /// Entities that the KG-expansion stage matched in the query text.
    pub kg_entities_matched: Vec<String>,
    /// Conflict events surfaced from sync that should be presented alongside results.
    /// Empty in the default case where sync is up-to-date and no divergence detected.
    #[serde(default)]
    pub conflict_events: Vec<serde_json::Value>,
    /// Unique id correlating this response with future feedback signals.
    pub query_id: String,
    /// Optional per-stage timings when debug-telemetry is on.
    #[serde(default)]
    pub stage_timings: Option<StageTimings>,
}

/// Intermediate type passed between pipeline stages. Stage 4 (personalization)
/// produces these; Stage 5 (assembly) consumes them and hydrates into RankedDrawer.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub drawer_id: String,
    pub score_breakdown: ScoreBreakdown,
    pub final_score: f32,
}
