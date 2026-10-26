//! Knowledge-graph layer â€” M4.
//!
//! ## What lives here
//!
//! - `KG_SCHEMA` â€” DDL for `entities` + `triples` tables (keyed DB; never plaintext).
//! - `KgProjection` â€” disposable petgraph DiGraph rebuilt from `triples` by
//!   `rebuild_indexes()`.  `None` until first build; KG search leg degrades
//!   gracefully when projection is absent.
//! - `extract_entities` / `extract_triples` â€” rule-based, deterministic
//!   extraction from drawer text.  No ML, no external calls, inspectable.
//!   Bar = "useful + honest," not "complete."
//! - KG CRUD methods on `PalaceStore` (upsert entity, add triple, list names).
//!
//! ## Schema note (M6 contract)
//!
//! The `triples` table mirrors `simonides_sync::kg_merge::TripleRecord` field-for-
//! field so `merge_triples()` drops straight onto it in M6 with no migration.
//! HLC is split into three columns (wall/logical/node_id) because SQLite has no
//! composite value type; the merge layer reassembles the `Hlc` struct on load.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
use rusqlite::params;
use uuid::Uuid;

use simonides_core::memory::{KgDirection, KgEntity, KgTriple};
use simonides_sync::kg_merge::TripleRecord;

use crate::sync_impl;
use crate::{PalaceError, PalaceStore};

type Result<T> = std::result::Result<T, PalaceError>;

// â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Max depth-1 neighbors per matched entity in KG expansion.
/// Caps fan-out; named constant so it is visible and tunable without hunting
/// magic numbers.
pub const MAX_KG_EXPANSION_NEIGHBORS: usize = 10;

/// Maximum total expansion terms added across all matched entities.
pub const MAX_KG_EXPANSION_TERMS: usize = 30;

// â”€â”€ Schema â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// DDL for the KG tables.  Run after core schema + migration in `init()`.
/// All `IF NOT EXISTS` â†’ safe to re-run on every open.
pub(crate) const KG_SCHEMA: &str = r#"
-- KG entities: real-world things mentioned in drawers.
-- source_drawer_id carries provenance (dumb-proof lock: every entity links
-- to the drawer that first named it).
CREATE TABLE IF NOT EXISTS entities (
    id               TEXT PRIMARY KEY,
    canonical_name   TEXT NOT NULL,
    entity_type      TEXT,
    source_drawer_id TEXT REFERENCES drawers(id),
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    -- M6: CRDT merge fields (migration adds these to existing DBs).
    properties       TEXT NOT NULL DEFAULT '{}',
    hlc_wall         INTEGER NOT NULL DEFAULT 0,
    hlc_logical      INTEGER NOT NULL DEFAULT 0,
    hlc_node_id      BLOB
);
-- Case-insensitive unique name: "Mira" and "mira" are the same entity.
CREATE UNIQUE INDEX IF NOT EXISTS entities_name_uq
    ON entities(canonical_name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS entities_name_idx
    ON entities(canonical_name COLLATE NOCASE);

-- KG triples: mirrors TripleRecord field-for-field (M6 contract).
-- Tombstone-not-delete: facts are superseded via valid_to/replaces/replaced_by,
-- never physically removed.
CREATE TABLE IF NOT EXISTS triples (
    id               TEXT PRIMARY KEY,
    subject_id       TEXT NOT NULL REFERENCES entities(id),
    predicate        TEXT NOT NULL,
    object_id        TEXT NOT NULL REFERENCES entities(id),
    valid_from       TEXT,      -- RFC3339 | NULL  (TripleRecord.valid_from)
    valid_to         TEXT,      -- RFC3339 | NULL  (TripleRecord.valid_to)
    confidence       REAL NOT NULL DEFAULT 1.0,
    hlc_wall         INTEGER NOT NULL DEFAULT 0,   -- Hlc.wall_time  (u64)
    hlc_logical      INTEGER NOT NULL DEFAULT 0,   -- Hlc.logical    (u32)
    hlc_node_id      BLOB NOT NULL,                -- Hlc.node_id    ([u8;16])
    replaces         TEXT,                          -- Option<String>
    replaced_by      TEXT,                          -- Option<String>
    source_drawer_id TEXT REFERENCES drawers(id)
);
-- Bitemporal fact key: (subject, predicate, object, valid_from). Each validity
-- WINDOW is its own row â€” re-asserting a fact AFTER it was invalidated opens a
-- new window (new valid_from) instead of overwriting the earlier one. Widened
-- from the old (subject, predicate, object) key by migration v4â†’v5; on a fresh
-- DB this DDL creates the wide index directly. (The old narrow key made
-- write_triple's INSERT OR REPLACE eat the earlier window â€” the PK data-loss bug.)
CREATE UNIQUE INDEX IF NOT EXISTS triples_fact_key
    ON triples(subject_id, predicate, object_id, valid_from);
CREATE INDEX IF NOT EXISTS triples_subject_idx   ON triples(subject_id);
CREATE INDEX IF NOT EXISTS triples_object_idx    ON triples(object_id);
CREATE INDEX IF NOT EXISTS triples_predicate_idx ON triples(predicate);
"#;

// â”€â”€ In-memory petgraph projection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Disposable in-memory projection of the `triples` table as a directed graph.
/// Rebuilt by `rebuild_indexes()`.  If `None`, search() KG leg is skipped.
pub struct KgProjection {
    pub graph: DiGraph<String, String>,
    pub node_map: HashMap<String, petgraph::graph::NodeIndex>,
}

impl KgProjection {
    /// Build the projection from the current state of the `entities` and
    /// `triples` tables.  Only currently-valid triples (valid_to IS NULL or in
    /// the future) are included.
    pub fn build(conn: &rusqlite::Connection) -> Result<Self> {
        let mut graph: DiGraph<String, String> = DiGraph::new();
        let mut node_map: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();

        // Load all entity id â†’ canonical_name pairs first.
        let mut entity_names: HashMap<String, String> = HashMap::new();
        {
            let mut stmt = conn
                .prepare("SELECT id, canonical_name FROM entities")
                .map_err(|e| PalaceError::Storage(format!("kg proj entities: {e}")))?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })
                .map_err(|e| PalaceError::Storage(format!("kg proj entity rows: {e}")))?;
            for r in rows {
                let (id, name) = r.map_err(|e| PalaceError::Storage(e.to_string()))?;
                entity_names.insert(id, name);
            }
        }

        // Add one graph node per entity.
        for name in entity_names.values() {
            let idx = graph.add_node(name.clone());
            node_map.insert(name.clone(), idx);
        }

        // Add edges for currently-valid triples.
        {
            let now = chrono::Utc::now().to_rfc3339();
            let mut stmt = conn
                .prepare(
                    "SELECT subject_id, predicate, object_id FROM triples \
                     WHERE valid_to IS NULL OR valid_to > ?1",
                )
                .map_err(|e| PalaceError::Storage(format!("kg proj triples: {e}")))?;
            let rows = stmt
                .query_map(params![now], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                })
                .map_err(|e| PalaceError::Storage(format!("kg proj triple rows: {e}")))?;
            for r in rows {
                let (s_id, pred, o_id) =
                    r.map_err(|e| PalaceError::Storage(e.to_string()))?;
                if let (Some(s_name), Some(o_name)) =
                    (entity_names.get(&s_id), entity_names.get(&o_id))
                {
                    if let (Some(&si), Some(&oi)) =
                        (node_map.get(s_name), node_map.get(o_name))
                    {
                        graph.add_edge(si, oi, pred);
                    }
                }
            }
        }

        Ok(KgProjection { graph, node_map })
    }

    /// Depth-1 expansion: find entities matching tokens in `query`, return
    /// (entity_name, weight) pairs for direct graph neighbors.
    ///
    /// Fan-out is capped at `MAX_KG_EXPANSION_TERMS` total across all matched
    /// entities.  Both outgoing and incoming edges are traversed (undirected
    /// traversal for depth-1).
    pub fn expand(
        &self,
        query: &str,
        entity_names_in_db: &[String],
    ) -> Vec<(String, f32)> {
        let tokens = tokenize_lower(query);

        // Find graph nodes whose name appears as a token in the query.
        let matched: Vec<petgraph::graph::NodeIndex> = entity_names_in_db
            .iter()
            .filter(|e| tokens.contains(&e.to_lowercase()))
            .filter_map(|e| self.node_map.get(e))
            .copied()
            .collect();

        if matched.is_empty() {
            return Vec::new();
        }

        let mut expansion: Vec<(String, f32)> = Vec::new();
        // Visited set includes the matched entities themselves so they don't
        // appear in their own expansion (matches the predecessor store's visited-set
        // discipline in query_expansion.rs).
        let mut seen: std::collections::HashSet<petgraph::graph::NodeIndex> =
            matched.iter().copied().collect();

        'outer: for &start in &matched {
            // Outgoing edges (subject â†’ object direction).
            for edge in self.graph.edges(start) {
                let neighbor = edge.target();
                if seen.contains(&neighbor) {
                    continue;
                }
                seen.insert(neighbor);
                expansion.push((self.graph[neighbor].clone(), predicate_weight(edge.weight())));
                if expansion.len() >= MAX_KG_EXPANSION_TERMS {
                    break 'outer;
                }
            }
            // Incoming edges (object â†’ subject direction, undirected traversal).
            for edge in self
                .graph
                .edges_directed(start, petgraph::Direction::Incoming)
            {
                let neighbor = edge.source();
                if seen.contains(&neighbor) {
                    continue;
                }
                seen.insert(neighbor);
                expansion.push((self.graph[neighbor].clone(), predicate_weight(edge.weight())));
                if expansion.len() >= MAX_KG_EXPANSION_TERMS {
                    break 'outer;
                }
            }
        }

        expansion
    }
}

/// Predicate weight map â€” mirrors `default_predicate_weights()` in the predecessor store
/// so the same relationship-strength assumptions hold across both crates.
fn predicate_weight(predicate: &str) -> f32 {
    match predicate {
        "works_on" => 1.0,
        "lives_in" => 0.9,
        "daughter_of" | "son_of" | "married_to" => 0.95,
        "loves" => 0.7,
        "mentioned_in" => 0.4,
        "similar_to" => 0.3,
        _ => 0.5,
    }
}

/// Split query into lowercase tokens for entity matching.
fn tokenize_lower(query: &str) -> Vec<String> {
    query
        .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

// â”€â”€ Rule-based extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Title-cased words that are NOT proper-noun entity candidates.
const STOP_WORDS: &[&str] = &[
    "The", "A", "An", "In", "On", "At", "To", "For", "Of", "And", "Or", "But",
    "Is", "Are", "Was", "Were", "Has", "Have", "Had", "This", "That", "These",
    "Those", "I", "We", "You", "He", "She", "It", "They", "My", "Our", "Your",
    "His", "Her", "Its", "Their", "From", "With", "By", "Be", "As", "Up",
    "About", "If", "No", "Not", "So", "Can", "Do", "All", "Each", "Just",
    "New", "Old", "One", "Two", "Also", "Then", "Than", "When", "What",
    "Which", "Who", "How", "Where", "Why", "Will", "Could", "Should", "Would",
    "Been", "More", "Most", "Very", "Some", "Any", "Every", "Both",
];

/// Rule-based entity extraction: finds title-cased tokens (â‰¥ 2 chars) not in the
/// stop-word list.  Deterministic, inspectable.  Bar = "useful + honest."
///
/// Returns deduplicated canonical names preserving first-seen casing.
pub fn extract_entities(text: &str) -> Vec<String> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut entities: Vec<String> = Vec::new();

    for token in text.split(|c: char| {
        c.is_whitespace()
            || c == ','
            || c == '.'
            || c == '!'
            || c == '?'
            || c == ';'
            || c == ':'
            || c == '"'
            || c == '\''
            || c == '('
            || c == ')'
            || c == '['
            || c == ']'
    }) {
        if token.len() < 2 {
            continue;
        }
        let mut chars = token.chars();
        let first = match chars.next() {
            Some(c) => c,
            None => continue,
        };
        if !first.is_uppercase() {
            continue;
        }
        if STOP_WORDS.contains(&token) {
            continue;
        }
        // Deduplicate case-insensitively; keep first-seen casing.
        let lower = token.to_lowercase();
        if seen.insert(lower) {
            entities.push(token.to_string());
        }
    }
    entities
}

/// Rule-based triple extraction.  Scans `text` for known predicate-keyword
/// patterns between pairs of entities found in the same passage.
///
/// Patterns (ordered most-specific first for greedy matching):
///   "Mira works on Cedar"  â†’ (Mira, works_on, Cedar)
///   "Hanna lives in Dili"  â†’ (Hanna, lives_in, Dili)
///
/// Returns `(subject, predicate, object)` tuples; no duplicates within a call.
pub fn extract_triples(
    text: &str,
    entities: &[String],
) -> Vec<(String, String, String)> {
    if entities.len() < 2 {
        return Vec::new();
    }

    // (keyword tokens, predicate).  Two-word phrases are matched as adjacent
    // tokens, not as a raw substring.
    const PATTERNS: &[(&str, &str)] = &[
        ("works on", "works_on"),
        ("worked on", "works_on"),
        ("lives in", "lives_in"),
        ("lived in", "lives_in"),
        ("married to", "married_to"),
        ("daughter of", "daughter_of"),
        ("son of", "son_of"),
        ("similar to", "similar_to"),
        ("mentioned in", "mentioned_in"),
        ("loves", "loves"),
        ("created", "created"),
        ("founded", "founded"),
    ];

    // Occurrence-driven extraction.  The previous implementation paired every
    // entity with every other -- O(k^2) in the entities found in ONE drawer --
    // and substring-tested each pair against the full text.  On real corpus
    // documents (hundreds of capitalized tokens) that pegged a core for minutes
    // per drawer and never converged.  A triple can only exist where a predicate
    // keyword sits directly BETWEEN two entity tokens (`<entity> <kw..> <entity>`),
    // so tokenize once and inspect exactly the token before and after each
    // keyword occurrence.  O(P * tokens) per drawer, no pairing.
    //
    // Tokenize identically to `extract_entities` (same whitespace+punctuation
    // split) so entity tokens line up -- e.g. a trailing "." on the object must
    // not defeat the match, and "cedar" must NOT match inside "cedarfast" (a
    // false positive the old substring test produced).
    let split = |c: char| {
        c.is_whitespace()
            || matches!(
                c,
                ',' | '.' | '!' | '?' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']'
            )
    };
    let toks_lower: Vec<String> = text
        .split(split)
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect();
    if toks_lower.len() < 3 {
        return Vec::new();
    }

    // lowercased entity -> first-seen original casing, for output.
    let mut canon: std::collections::HashMap<String, String> =
        std::collections::HashMap::with_capacity(entities.len());
    for e in entities {
        canon.entry(e.to_lowercase()).or_insert_with(|| e.clone());
    }

    let mut triples: Vec<(String, String, String)> = Vec::new();
    let mut seen: std::collections::HashSet<(String, String, String)> =
        std::collections::HashSet::new();

    for &(keyword, predicate) in PATTERNS {
        let kw: Vec<&str> = keyword.split(' ').collect();
        let klen = kw.len();
        if toks_lower.len() < klen + 2 {
            continue;
        }
        // A keyword at `start` needs one token on each side, so start in
        // 1 ..= len-klen-1.
        for start in 1..=(toks_lower.len() - klen - 1) {
            if (0..klen).all(|o| toks_lower[start + o] == kw[o]) {
                let subj_l = &toks_lower[start - 1];
                let obj_l = &toks_lower[start + klen];
                if subj_l == obj_l {
                    continue;
                }
                if let (Some(subj), Some(obj)) = (canon.get(subj_l), canon.get(obj_l)) {
                    let t = (subj.clone(), predicate.to_string(), obj.clone());
                    if seen.insert(t.clone()) {
                        triples.push(t);
                    }
                }
            }
        }
    }
    triples
}

// â”€â”€ KG CRUD on PalaceStore â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl PalaceStore {
    /// Upsert an entity by canonical name (case-insensitive dedup).
    /// Returns the entity's UUID id.
    pub fn kg_upsert_entity(
        &self,
        canonical_name: &str,
        entity_type: Option<&str>,
        source_drawer_id: Option<&str>,
    ) -> Result<String> {
        let now = chrono::Utc::now().to_rfc3339();
        let guard = self.conn.lock().unwrap();

        // Check for existing entity (case-insensitive).
        let existing: Option<String> = guard
            .query_row(
                "SELECT id FROM entities WHERE canonical_name = ?1 COLLATE NOCASE",
                params![canonical_name],
                |r| r.get(0),
            )
            .ok();

        if let Some(id) = existing {
            guard
                .execute(
                    "UPDATE entities SET updated_at = ?1 WHERE id = ?2",
                    params![now, id],
                )
                .map_err(|e| {
                    PalaceError::Storage(format!("kg_upsert_entity update: {e}"))
                })?;
            return Ok(id);
        }

        let id = Uuid::now_v7().to_string();
        guard
            .execute(
                "INSERT INTO entities \
                 (id, canonical_name, entity_type, source_drawer_id, created_at, updated_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![id, canonical_name, entity_type, source_drawer_id, now, now],
            )
            .map_err(|e| {
                PalaceError::Storage(format!("kg_upsert_entity insert: {e}"))
            })?;
        Ok(id)
    }

    /// Add an extraction-derived triple by entity UUIDs.  Tombstone-not-delete:
    /// if the same (subject, predicate, object) already exists (UNIQUE INDEX),
    /// the insert is skipped (Scenario A coalesce â€” same fact twice is one fact)
    /// and the existing id is returned.
    ///
    /// This is the *extraction* entry point (`kg_extract_drawer`), which carries
    /// drawer provenance in `source_drawer_id`.  The user-facing assertion path
    /// is [`kg_add_triple`](Self::kg_add_triple), which takes entity NAMES and
    /// carries a validity window.  Both stamp a real HLC so the row is a
    /// mergeable `TripleRecord`, never an HLC-zero row the merge layer can't
    /// order.
    pub fn kg_add_triple_by_ids(
        &self,
        subject_id: &str,
        predicate: &str,
        object_id: &str,
        source_drawer_id: Option<&str>,
    ) -> Result<String> {
        let guard = self.conn.lock().unwrap();

        // Check if fact-key already exists.
        let existing: Option<String> = guard
            .query_row(
                "SELECT id FROM triples \
                 WHERE subject_id = ?1 AND predicate = ?2 AND object_id = ?3",
                params![subject_id, predicate, object_id],
                |r| r.get(0),
            )
            .ok();

        if let Some(id) = existing {
            return Ok(id);
        }

        let id = Uuid::now_v7().to_string();
        let hlc = sync_impl::hlc_now();
        let node_bytes: Vec<u8> = hlc.node_id.to_vec();
        guard
            .execute(
                "INSERT OR IGNORE INTO triples \
                 (id, subject_id, predicate, object_id, valid_from, valid_to, \
                  confidence, hlc_wall, hlc_logical, hlc_node_id, \
                  replaces, replaced_by, source_drawer_id) \
                 VALUES (?1, ?2, ?3, ?4, NULL, NULL, 1.0, ?5, ?6, ?7, NULL, NULL, ?8)",
                params![
                    id,
                    subject_id,
                    predicate,
                    object_id,
                    hlc.wall_time as i64,
                    hlc.logical as i64,
                    node_bytes,
                    source_drawer_id,
                ],
            )
            .map_err(|e| PalaceError::Storage(format!("kg_add_triple_by_ids: {e}")))?;
        Ok(id)
    }

    /// List all entity canonical names (for query-token matching in `expand()`).
    pub fn kg_list_entity_names(&self) -> Result<Vec<String>> {
        let guard = self.conn.lock().unwrap();
        let mut stmt = guard
            .prepare("SELECT canonical_name FROM entities ORDER BY canonical_name")
            .map_err(|e| {
                PalaceError::Storage(format!("kg_list_entity_names prepare: {e}"))
            })?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(|e| {
                PalaceError::Storage(format!("kg_list_entity_names query: {e}"))
            })?;
        let mut names = Vec::new();
        for r in rows {
            names.push(r.map_err(|e| PalaceError::Storage(e.to_string()))?);
        }
        Ok(names)
    }

    /// Rule-based extraction pass for one drawer.  Upserts found entities and
    /// triples.  Called from `rebuild_indexes()` â€” never from `insert()`.
    pub(crate) fn kg_extract_drawer(&self, drawer_id: &str, content: &str) -> Result<()> {
        let entity_names = extract_entities(content);
        if entity_names.is_empty() {
            return Ok(());
        }

        let mut entity_ids: HashMap<String, String> = HashMap::new();
        for name in &entity_names {
            let id = self.kg_upsert_entity(name, None, Some(drawer_id))?;
            entity_ids.insert(name.clone(), id);
        }

        for (subj, pred, obj) in extract_triples(content, &entity_names) {
            if let (Some(s_id), Some(o_id)) =
                (entity_ids.get(&subj), entity_ids.get(&obj))
            {
                let s_id = s_id.clone();
                let o_id = o_id.clone();
                self.kg_add_triple_by_ids(&s_id, &pred, &o_id, Some(drawer_id))?;
            }
        }
        Ok(())
    }

    /// Rebuild the in-memory petgraph projection from the current `triples` table.
    /// Called at the end of `rebuild_indexes()`.
    pub(crate) fn kg_rebuild_projection(&self) -> Result<()> {
        let projection = {
            let guard = self.conn.lock().unwrap();
            KgProjection::build(&guard)?
        };
        *self.kg_projection.lock().unwrap() = Some(projection);
        Ok(())
    }
}

// â”€â”€ P3: KG query + write surface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Ported from the predecessor store onto the storage already in
// this crate.  Two shape differences are load-bearing and worth stating once:
//
// 1. **Names in, names out; ids underneath.**  the predecessor store stored raw name
//    strings in `kg_triples`.  Palace stores entity UUIDs with an FK into
//    `entities`.  The query surface still speaks NAMES (that is the API the
//    assistant drives), so every read joins through `entities` and every write
//    upserts the entity first.
//
// 2. **Writes are CRDT writes (brief Â§4).**  `kg_add_triple` / `kg_invalidate`
//    build a `kg_merge::TripleRecord`, stamp a fresh HLC, and persist through
//    `sync_impl::write_triple` â€” the same seam `apply_remote_delta` uses.  A
//    bare `INSERT`/`UPDATE` here would be the PC hazard again: a row that never
//    leaves the device.  Reads are pure.  Every write refreshes the projection.

/// Triple read joined to `entities` so subject/object come back as canonical
/// names.  Column order matches [`row_to_kg_triple`].
const KG_TRIPLE_SELECT: &str = "SELECT t.id, se.canonical_name, t.predicate, oe.canonical_name, \
     t.valid_from, t.valid_to, t.confidence, t.source_drawer_id \
     FROM triples t \
     JOIN entities se ON se.id = t.subject_id \
     JOIN entities oe ON oe.id = t.object_id";

/// Temporal-validity window, exclusive on the upper bound (brief Â§3): a triple
/// invalidated at `t1` is visible for `as_of < t1` and invisible at `as_of >= t1`.
const KG_VALID_AT: &str = "(t.valid_from IS NULL OR t.valid_from <= ?2) \
     AND (t.valid_to IS NULL OR t.valid_to > ?2)";

/// Aggregate KG counts.  Every field is a full `COUNT(*)` â€” no sampling, no cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KgStats {
    pub entities: u64,
    pub triples: u64,
    /// Triples whose validity window is still open (`valid_to IS NULL`).
    pub open_triples: u64,
    /// Triples closed by `kg_invalidate` (`valid_to` set).  Still present â€”
    /// invalidation never deletes the row.
    pub invalidated_triples: u64,
}

/// Creation instant of a UUIDv7 id: the 48-bit big-endian millisecond prefix.
/// `triples` mirrors `TripleRecord` (M6 contract), which has no `created_at`
/// column â€” so the v7 id IS the creation clock.  Decoded here rather than
/// adding a column that would diverge from the record shape.  Non-v7 ids
/// (possible only for a hand-seeded row) return `None`; the caller falls back.
fn created_at_of(id: &str) -> Option<DateTime<Utc>> {
    let uuid = Uuid::parse_str(id).ok()?;
    if uuid.get_version_num() != 7 {
        return None;
    }
    let b = uuid.as_bytes();
    let ms = ((b[0] as u64) << 40)
        | ((b[1] as u64) << 32)
        | ((b[2] as u64) << 24)
        | ((b[3] as u64) << 16)
        | ((b[4] as u64) << 8)
        | (b[5] as u64);
    DateTime::<Utc>::from_timestamp_millis(ms as i64)
}

fn parse_ts(s: &str) -> Option<DateTime<Utc>> {
    s.parse::<DateTime<Utc>>().ok()
}

fn row_to_kg_triple(row: &rusqlite::Row<'_>) -> rusqlite::Result<KgTriple> {
    let id: String = row.get(0)?;
    let valid_from = row.get::<_, Option<String>>(4)?.as_deref().and_then(parse_ts);
    let valid_to = row.get::<_, Option<String>>(5)?.as_deref().and_then(parse_ts);
    let created_at = created_at_of(&id)
        .or(valid_from)
        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).expect("epoch is a valid instant"));
    Ok(KgTriple {
        id,
        subject: row.get(1)?,
        predicate: row.get(2)?,
        object: row.get(3)?,
        valid_from,
        valid_to,
        confidence: Some(row.get::<_, f64>(6)? as f32),
        source_closet: row.get(7)?,
        created_at,
    })
}

/// Entity id for `name` (case-insensitive), or `None` if the KG has never heard
/// of it.  Callers that need loudness use [`require_entity_id`].
fn entity_id_of(conn: &rusqlite::Connection, name: &str) -> Result<Option<String>> {
    conn.query_row(
        "SELECT id FROM entities WHERE canonical_name = ?1 COLLATE NOCASE",
        params![name],
        |r| r.get::<_, String>(0),
    )
    .map(Some)
    .or_else(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => Ok(None),
        other => Err(PalaceError::Storage(format!("entity_id_of: {other}"))),
    })
}

/// Entity id for `name`, or a LOUD `NotFound` (brief Â§5: "entity unknown to the
/// KG" must never be collapsed into the same empty `Vec` that means "entity
/// known, but nothing matched the window/direction").
fn require_entity_id(conn: &rusqlite::Connection, name: &str) -> Result<String> {
    entity_id_of(conn, name)?.ok_or_else(|| PalaceError::NotFound {
        id: format!("kg entity '{name}' (unknown to the knowledge graph)"),
    })
}

fn collect_triples(
    conn: &rusqlite::Connection,
    sql: &str,
    params: impl rusqlite::Params,
) -> Result<Vec<KgTriple>> {
    let mut stmt = conn
        .prepare(sql)
        .map_err(|e| PalaceError::Storage(format!("kg triple query prepare: {e}")))?;
    let rows = stmt
        .query_map(params, row_to_kg_triple)
        .map_err(|e| PalaceError::Storage(format!("kg triple query: {e}")))?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| PalaceError::Storage(e.to_string()))?);
    }
    Ok(out)
}

impl PalaceStore {
    /// Triples touching `entity` that are valid at `as_of` (defaults to **now**),
    /// filtered by `direction`.
    ///
    /// Time-travel (brief Â§3): a fact valid over `[t0, t1)` â€” i.e. asserted at
    /// `t0`, invalidated at `t1` â€” is returned for `as_of âˆˆ [t0, t1)` and NOT for
    /// `as_of >= t1`.  The row is still there either way; only the window moves.
    ///
    /// **Unknown entity is a loud error, not an empty list.**  `Ok(vec![])` means
    /// "this entity exists in the KG and nothing matched your window/direction";
    /// `Err(NotFound)` means "the KG has never heard of this entity."  Collapsing
    /// the two would let a typo read as "no relationships."
    ///
    /// No result cap â€” every matching triple is returned (brief Â§5: no silent
    /// truncation).
    pub fn kg_query_entity(
        &self,
        entity: &str,
        as_of: Option<DateTime<Utc>>,
        direction: KgDirection,
    ) -> Result<Vec<KgTriple>> {
        let as_of = as_of.unwrap_or_else(Utc::now).to_rfc3339();
        let guard = self.conn.lock().unwrap();
        let entity_id = require_entity_id(&guard, entity)?;

        let side = match direction {
            KgDirection::Outgoing => "t.subject_id = ?1",
            KgDirection::Incoming => "t.object_id = ?1",
            KgDirection::Both => "(t.subject_id = ?1 OR t.object_id = ?1)",
        };
        // Newest-first, mirroring the predecessor store's `ORDER BY created_at DESC`;
        // UUIDv7 ids are time-ordered, so id ordering IS creation ordering.
        let sql = format!("{KG_TRIPLE_SELECT} WHERE {side} AND {KG_VALID_AT} ORDER BY t.id DESC");
        collect_triples(&guard, &sql, params![entity_id, as_of])
    }

    /// Full history of `entity` â€” every triple that ever touched it, **including
    /// invalidated ones**, oldest first.  This is the "life story" view; use
    /// [`kg_query_entity`](Self::kg_query_entity) for the currently-true view.
    ///
    /// Unknown entity is a loud `NotFound` (same rule as `kg_query_entity`).
    pub fn kg_timeline(&self, entity: &str) -> Result<Vec<KgTriple>> {
        let guard = self.conn.lock().unwrap();
        let entity_id = require_entity_id(&guard, entity)?;
        let sql = format!(
            "{KG_TRIPLE_SELECT} \
             WHERE t.subject_id = ?1 OR t.object_id = ?1 \
             ORDER BY t.id ASC"
        );
        collect_triples(&guard, &sql, params![entity_id])
    }

    /// Assert a fact.  Entities are upserted by name (case-insensitive), so
    /// `kg_add_triple("Mira", "lives_in", "Dili", ..)` works without a separate
    /// entity-creation step.
    ///
    /// **This is a CRDT write (brief Â§4).**  It builds a `TripleRecord`, stamps a
    /// fresh HLC, and persists through the same `write_triple` seam that
    /// `apply_remote_delta` uses â€” so the fact survives `produce_delta` â†’
    /// `apply_remote_delta` onto a peer instead of dying on this device.
    ///
    /// **Bitemporal-safe on the fact key `(subject, predicate, object)`:**
    /// - Re-asserting a fact that is **currently open** updates that one window in
    ///   place (new confidence, new HLC) â€” idempotent, no duplicate row. This is
    ///   the `kg_merge` Scenario-A coalesce rule the merge layer applies when two
    ///   devices assert the same still-true fact.
    /// - Re-asserting a fact whose last window was **invalidated** opens a NEW
    ///   window (fresh id, `replaces` â†’ the prior window) â€” the earlier closed
    ///   window `[t0, t1)` is preserved, never overwritten. Both windows are then
    ///   visible in [`kg_timeline`](Self::kg_timeline) and each is returned by
    ///   [`kg_query_entity`](Self::kg_query_entity) at an `as_of` inside it.
    #[allow(clippy::too_many_arguments)]
    pub fn kg_add_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        valid_from: Option<DateTime<Utc>>,
        valid_to: Option<DateTime<Utc>>,
        confidence: Option<f32>,
    ) -> Result<KgTriple> {
        if subject.trim().is_empty() || predicate.trim().is_empty() || object.trim().is_empty() {
            return Err(PalaceError::Storage(
                "kg_add_triple: subject, predicate and object must all be non-empty".to_string(),
            ));
        }
        if let (Some(from), Some(to)) = (valid_from, valid_to) {
            if to < from {
                return Err(PalaceError::Storage(format!(
                    "kg_add_triple: valid_to ({to}) precedes valid_from ({from})"
                )));
            }
        }

        // Upsert both endpoints BEFORE opening the write transaction (each takes
        // the conn lock itself), so the triple's FKs always resolve.
        let subject_id = self.kg_upsert_entity(subject, None, None)?;
        let object_id = self.kg_upsert_entity(object, None, None)?;

        let id = {
            let mut guard = self.conn.lock().unwrap();
            let tx = guard
                .transaction()
                .map_err(|e| PalaceError::Storage(format!("kg_add_triple tx begin: {e}")))?;

            // Open/closed branch (PK bitemporal fix). Re-asserting a fact that is
            // currently TRUE continues the same window: reuse the OPEN row's id so
            // write_triple (INSERT OR REPLACE on id) lands in place â€” idempotent,
            // one row. Re-asserting a fact whose last window was CLOSED (invalidated)
            // opens a NEW window with a fresh id, so the earlier [t0,t1) window is
            // preserved rather than overwritten. `replaces` links the new window to
            // its predecessor when one exists (one extra read; the field already
            // exists on TripleRecord).
            let (id, replaces) = match sync_impl::query_current_triple(
                &tx, &subject_id, predicate, &object_id,
            )? {
                Some(open) => (open.id, open.replaces),
                None => {
                    let replaces = sync_impl::query_latest_triple(
                        &tx, &subject_id, predicate, &object_id,
                    )?
                    .map(|prev| prev.id);
                    (Uuid::now_v7().to_string(), replaces)
                }
            };

            let record = TripleRecord {
                id: id.clone(),
                subject_id,
                predicate: predicate.to_string(),
                object_id,
                valid_from,
                valid_to,
                confidence: confidence.unwrap_or(1.0),
                hlc: sync_impl::hlc_now(),
                replaces,
                replaced_by: None,
            };
            sync_impl::write_triple(&tx, &record)?;
            tx.commit()
                .map_err(|e| PalaceError::Storage(format!("kg_add_triple tx commit: {e}")))?;
            id
        };

        // The petgraph projection is derived from `triples` â€” refresh it so the
        // write is visible to KG expansion immediately (no silent staleness).
        self.kg_rebuild_projection()?;

        self.kg_get_triple(&id)?.ok_or_else(|| {
            PalaceError::Storage(format!("kg_add_triple: triple {id} vanished after commit"))
        })
    }

    /// Close the validity window on a currently-open fact: sets `valid_to = ended`.
    ///
    /// **Never deletes the row** (brief Â§3, tombstone-not-delete).  After this,
    /// `kg_query_entity` at `as_of >= ended` no longer returns the fact, but
    /// `kg_query_entity` at an earlier `as_of` still does, and `kg_timeline`
    /// always does.
    ///
    /// **This is a CRDT write (brief Â§4)** â€” same `TripleRecord` + fresh HLC +
    /// `write_triple` seam as `kg_add_triple`, so the invalidation propagates.
    ///
    /// Loud on every miss (brief Â§5): unknown entity, unknown fact, and
    /// already-invalidated fact each return a distinct error rather than a
    /// silent zero-rows-updated.
    pub fn kg_invalidate(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        ended: DateTime<Utc>,
    ) -> Result<()> {
        let (subject_id, object_id) = {
            let guard = self.conn.lock().unwrap();
            (
                require_entity_id(&guard, subject)?,
                require_entity_id(&guard, object)?,
            )
        };

        {
            let mut guard = self.conn.lock().unwrap();
            let tx = guard
                .transaction()
                .map_err(|e| PalaceError::Storage(format!("kg_invalidate tx begin: {e}")))?;

            // Close the currently-OPEN window (deterministic â€” never an arbitrary
            // row once a fact key holds several windows). No open window is loud in
            // two distinguishable ways: a closed window still on file means the fact
            // was already invalidated; no row at all means the fact is unknown.
            let mut record = match sync_impl::query_current_triple(
                &tx, &subject_id, predicate, &object_id,
            )? {
                Some(open) => open,
                None => {
                    return match sync_impl::query_latest_triple(
                        &tx, &subject_id, predicate, &object_id,
                    )? {
                        Some(closed) => Err(PalaceError::Storage(format!(
                            "kg_invalidate: ({subject}, {predicate}, {object}) has no open window \
                             â€” it was already invalidated{} â€” nothing to close",
                            closed
                                .valid_to
                                .map(|t| format!(" at {t}"))
                                .unwrap_or_default()
                        ))),
                        None => Err(PalaceError::NotFound {
                            id: format!("kg triple ({subject}, {predicate}, {object})"),
                        }),
                    };
                }
            };

            record.valid_to = Some(ended);
            record.hlc = sync_impl::hlc_now();
            sync_impl::write_triple(&tx, &record)?;
            tx.commit()
                .map_err(|e| PalaceError::Storage(format!("kg_invalidate tx commit: {e}")))?;
        }

        self.kg_rebuild_projection()?;
        Ok(())
    }

    /// One triple by id, joined out to names.  `None` if absent.
    pub fn kg_get_triple(&self, id: &str) -> Result<Option<KgTriple>> {
        let guard = self.conn.lock().unwrap();
        let sql = format!("{KG_TRIPLE_SELECT} WHERE t.id = ?1");
        Ok(collect_triples(&guard, &sql, params![id])?.into_iter().next())
    }

    /// Every entity, optionally filtered by `entity_type`, name-ordered.
    /// Uncapped â€” the full set, always.
    pub fn kg_list_entities(&self, type_filter: Option<&str>) -> Result<Vec<KgEntity>> {
        let guard = self.conn.lock().unwrap();
        let sql = match type_filter {
            Some(_) => "SELECT id, canonical_name, entity_type, COALESCE(properties, '{}') \
                        FROM entities WHERE entity_type = ?1 ORDER BY canonical_name ASC",
            None => "SELECT id, canonical_name, entity_type, COALESCE(properties, '{}') \
                     FROM entities ORDER BY canonical_name ASC",
        };
        let mut stmt = guard
            .prepare(sql)
            .map_err(|e| PalaceError::Storage(format!("kg_list_entities prepare: {e}")))?;

        let map_row = |row: &rusqlite::Row<'_>| -> rusqlite::Result<KgEntity> {
            let props: String = row.get(3)?;
            Ok(KgEntity {
                id: row.get(0)?,
                name: row.get(1)?,
                entity_type: row.get(2)?,
                properties: serde_json::from_str(&props)
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
            })
        };

        let rows = match type_filter {
            Some(t) => stmt.query_map(params![t], map_row),
            None => stmt.query_map([], map_row),
        }
        .map_err(|e| PalaceError::Storage(format!("kg_list_entities query: {e}")))?;

        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| PalaceError::Storage(e.to_string()))?);
        }
        Ok(out)
    }

    /// Aggregate KG counts.  Splits open vs invalidated triples so a caller can
    /// see that invalidated facts are still *there* (tombstone-not-delete), not
    /// gone.
    pub fn kg_stats(&self) -> Result<KgStats> {
        let guard = self.conn.lock().unwrap();
        let count = |sql: &str| -> Result<u64> {
            guard
                .query_row(sql, [], |r| r.get::<_, i64>(0))
                .map(|n| n as u64)
                .map_err(|e| PalaceError::Storage(format!("kg_stats: {e}")))
        };
        let entities = count("SELECT COUNT(*) FROM entities")?;
        let triples = count("SELECT COUNT(*) FROM triples")?;
        let open_triples = count("SELECT COUNT(*) FROM triples WHERE valid_to IS NULL")?;
        let invalidated_triples = count("SELECT COUNT(*) FROM triples WHERE valid_to IS NOT NULL")?;
        Ok(KgStats { entities, triples, open_triples, invalidated_triples })
    }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    const TEST_KEY: &str = "correct horse battery staple";

    /// Hermetic store: in-memory sqlcipher DB + temp vault root.  No embedder is
    /// touched anywhere in these tests (crate rule: no environment-coupled
    /// tests) â€” the KG surface is pure SQL + CRDT, and `kg_rebuild_projection`
    /// reads only the `triples`/`entities` tables.
    fn store() -> (PalaceStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let s = PalaceStore::open_in_memory(tmp.path(), TEST_KEY).expect("open_in_memory");
        (s, tmp)
    }

    fn dt(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    fn facts(triples: &[KgTriple]) -> Vec<(String, String, String)> {
        triples
            .iter()
            .map(|t| (t.subject.clone(), t.predicate.clone(), t.object.clone()))
            .collect()
    }

    // â”€â”€ P3 gates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// P3 gate (brief Â§3): `as_of` is real time-travel.  A fact valid over
    /// `[t0, t1)` is returned for `as_of âˆˆ [t0, t1)` and NOT at or after `t1`.
    #[test]
    fn kg_query_entity_bitemporal_as_of() {
        let (s, _tmp) = store();
        let t0 = dt(2024, 1, 1);
        let t1 = dt(2025, 1, 1);
        let t2 = dt(2025, 6, 1);

        s.kg_add_triple("Mira", "lives_in", "Dili", Some(t0), None, None)
            .expect("assert fact");
        s.kg_invalidate("Mira", "lives_in", "Dili", t1)
            .expect("close the window at t1");

        // Inside the window [t0, t1): visible.
        for as_of in [t0, dt(2024, 6, 1)] {
            let got = s
                .kg_query_entity("Mira", Some(as_of), KgDirection::Outgoing)
                .expect("query inside window");
            assert_eq!(
                facts(&got),
                vec![("Mira".into(), "lives_in".into(), "Dili".into())],
                "fact must be visible at as_of={as_of} (inside [t0, t1))"
            );
        }

        // At and after t1: gone (upper bound is exclusive).
        for as_of in [t1, t2] {
            let got = s
                .kg_query_entity("Mira", Some(as_of), KgDirection::Outgoing)
                .expect("query at/after invalidation");
            assert!(
                got.is_empty(),
                "fact must NOT be visible at as_of={as_of} (>= t1); got {:?}",
                facts(&got)
            );
        }

        // Before t0: gone (lower bound).
        let before = s
            .kg_query_entity("Mira", Some(dt(2023, 1, 1)), KgDirection::Outgoing)
            .expect("query before valid_from");
        assert!(before.is_empty(), "fact must NOT be visible before valid_from");
    }

    /// P3 gate (brief Â§4 â€” THE load-bearing one): `kg_add_triple` is a CRDT
    /// write, not a bare INSERT.  Assert on A â†’ `produce_delta` â†’
    /// `apply_remote_delta` onto a FRESH store B â†’ the fact is queryable on B.
    /// This is the PC round-trip, for KG writes: it fails if the write ever
    /// stops routing through the `TripleRecord` + HLC + `write_triple` seam.
    #[test]
    fn kg_add_triple_is_crdt_consistent() {
        use simonides_core::sync::{Hlc, PeerClock, SyncableStore};

        let (a, _ta) = store();
        a.kg_add_triple("Mira", "works_on", "Cedar", Some(dt(2024, 1, 1)), None, Some(0.9))
            .expect("assert fact on A");

        // The row must carry a real HLC â€” an HLC-zero row is the tell that the
        // write bypassed the CRDT seam.
        let wall: i64 = {
            let g = a.conn.lock().unwrap();
            g.query_row("SELECT hlc_wall FROM triples", [], |r| r.get(0))
                .expect("read hlc_wall")
        };
        assert!(wall > 0, "kg_add_triple must stamp a real HLC, got hlc_wall = {wall}");

        let clock = PeerClock {
            peer_node_id: [0u8; 16],
            last_seen_hlc: Hlc { wall_time: 0, logical: 0, node_id: [0u8; 16] },
        };
        let delta = a.produce_delta(&clock).expect("produce_delta on A");

        let (b, _tb) = store();
        b.apply_remote_delta(delta).expect("apply_remote_delta on fresh B");

        let got = b
            .kg_query_entity("Mira", None, KgDirection::Outgoing)
            .expect("query the merged fact on B");
        assert_eq!(
            facts(&got),
            vec![("Mira".into(), "works_on".into(), "Cedar".into())],
            "the fact must survive produce_delta â†’ apply_remote_delta onto a fresh peer"
        );
        assert_eq!(got[0].valid_from, Some(dt(2024, 1, 1)), "validity window must survive the merge");
    }

    /// P3 gate (brief Â§3): invalidation CLOSES the window â€” it never deletes.
    /// The row is still on disk, `valid_to` is set, the now-query excludes it,
    /// and `kg_timeline` still tells its story.
    #[test]
    fn kg_invalidate_sets_valid_to_not_delete() {
        let (s, _tmp) = store();
        let ended = dt(2025, 3, 1);
        s.kg_add_triple("Sarah", "works_on", "ProjectX", Some(dt(2024, 1, 1)), None, None)
            .expect("assert fact");

        s.kg_invalidate("Sarah", "works_on", "ProjectX", ended).expect("invalidate");

        // The row is STILL THERE, with valid_to set.
        let (rows, valid_to): (i64, Option<String>) = {
            let g = s.conn.lock().unwrap();
            let n = g
                .query_row("SELECT COUNT(*) FROM triples", [], |r| r.get(0))
                .expect("count triples");
            let vt = g
                .query_row("SELECT valid_to FROM triples", [], |r| r.get(0))
                .expect("read valid_to");
            (n, vt)
        };
        assert_eq!(rows, 1, "invalidation must NOT delete the row");
        assert_eq!(
            valid_to.as_deref().and_then(parse_ts),
            Some(ended),
            "valid_to must be set to the ended timestamp"
        );

        // The currently-true view excludes it...
        let now_view = s
            .kg_query_entity("Sarah", None, KgDirection::Both)
            .expect("query now");
        assert!(now_view.is_empty(), "an invalidated fact must not appear in the now-view");

        // ...but the timeline still carries it, with the window it had.
        let timeline = s.kg_timeline("Sarah").expect("timeline");
        assert_eq!(
            facts(&timeline),
            vec![("Sarah".into(), "works_on".into(), "ProjectX".into())],
            "kg_timeline must include invalidated facts (full history)"
        );
        assert_eq!(timeline[0].valid_to, Some(ended), "timeline carries the closed window");

        // And the counts say so out loud: still 1 triple, now 0 open / 1 closed.
        let stats = s.kg_stats().expect("stats");
        assert_eq!(stats.triples, 1);
        assert_eq!(stats.open_triples, 0);
        assert_eq!(stats.invalidated_triples, 1);
    }

    /// P3 gate (brief Â§5, loud-not-silent): an entity the KG has never heard of
    /// is an explicit `NotFound`, NOT an empty Vec.  An empty Vec is reserved
    /// for "entity known, nothing matched the window/direction" â€” collapsing the
    /// two would let a typo read as "this person has no relationships."
    #[test]
    fn kg_query_unknown_entity_is_explicit() {
        let (s, _tmp) = store();
        s.kg_add_triple("Mira", "lives_in", "Dili", Some(dt(2024, 1, 1)), None, None)
            .expect("assert fact");

        // Unknown entity â†’ loud NotFound naming the entity.
        for probe in ["Ghost", "Miraa"] {
            let err = s
                .kg_query_entity(probe, None, KgDirection::Both)
                .expect_err("an unknown entity must be a loud error, not an empty Vec");
            match &err {
                PalaceError::NotFound { id } => assert!(
                    id.contains(probe),
                    "the error must name the unknown entity; got {id:?}"
                ),
                other => panic!("expected NotFound for unknown entity, got {other:?}"),
            }
            // kg_timeline holds the same line.
            assert!(
                matches!(s.kg_timeline(probe), Err(PalaceError::NotFound { .. })),
                "kg_timeline must also be loud on an unknown entity"
            );
        }

        // KNOWN entity, nothing in the window â†’ Ok(empty). The two cases are
        // distinguishable, which is the whole point.
        s.kg_invalidate("Mira", "lives_in", "Dili", dt(2025, 1, 1)).expect("invalidate");
        let known_but_empty = s
            .kg_query_entity("Mira", None, KgDirection::Both)
            .expect("a KNOWN entity with no matching triples is Ok(empty), not an error");
        assert!(known_but_empty.is_empty());
    }

    /// P3 gate: direction filtering is correct â€” outgoing matches on the subject
    /// side, incoming on the object side, both is the union.
    #[test]
    fn kg_direction_filter_outgoing_incoming_both() {
        let (s, _tmp) = store();
        // Mira --works_on--> Cedar   (outgoing from Mira)
        // Hanna --married_to--> Mira (incoming to Mira)
        s.kg_add_triple("Mira", "works_on", "Cedar", None, None, None).expect("outgoing fact");
        s.kg_add_triple("Hanna", "married_to", "Mira", None, None, None).expect("incoming fact");

        let outgoing = s
            .kg_query_entity("Mira", None, KgDirection::Outgoing)
            .expect("outgoing");
        assert_eq!(
            facts(&outgoing),
            vec![("Mira".into(), "works_on".into(), "Cedar".into())],
            "Outgoing must return only triples where Mira is the SUBJECT"
        );

        let incoming = s
            .kg_query_entity("Mira", None, KgDirection::Incoming)
            .expect("incoming");
        assert_eq!(
            facts(&incoming),
            vec![("Hanna".into(), "married_to".into(), "Mira".into())],
            "Incoming must return only triples where Mira is the OBJECT"
        );

        let both = s.kg_query_entity("Mira", None, KgDirection::Both).expect("both");
        let mut both_facts = facts(&both);
        both_facts.sort();
        assert_eq!(
            both_facts,
            vec![
                ("Hanna".to_string(), "married_to".to_string(), "Mira".to_string()),
                ("Mira".to_string(), "works_on".to_string(), "Cedar".to_string()),
            ],
            "Both must return the union of the two directions"
        );

        // Cedar is only ever an object; it has no outgoing edges â€” and that is
        // an empty Vec, not an error, because Cedar IS known to the KG.
        assert!(
            s.kg_query_entity("Cedar", None, KgDirection::Outgoing)
                .expect("Cedar is a known entity")
                .is_empty(),
            "a known entity with no outgoing edges yields Ok(empty)"
        );
    }

    // â”€â”€ Supporting behaviour â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Re-asserting the same fact key updates the one row in place (Scenario A
    /// coalesce), rather than spawning a duplicate.
    #[test]
    fn kg_add_triple_is_idempotent_on_fact_key() {
        let (s, _tmp) = store();
        let first = s
            .kg_add_triple("Mira", "lives_in", "Dili", Some(dt(2024, 1, 1)), None, Some(0.5))
            .expect("first assertion");
        let second = s
            .kg_add_triple("Mira", "lives_in", "Dili", Some(dt(2024, 1, 1)), None, Some(0.95))
            .expect("re-assertion");

        assert_eq!(first.id, second.id, "same fact key must reuse the same row id");
        let stats = s.kg_stats().expect("stats");
        assert_eq!(stats.triples, 1, "re-asserting a fact must not duplicate the row");
        assert_eq!(stats.entities, 2, "Mira + Dili, deduped");
        assert_eq!(
            second.confidence.map(|c| (c * 100.0).round() as i32),
            Some(95),
            "the re-assertion's confidence must land"
        );
    }

    // â”€â”€ Bitemporal gates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// THE regression gate. add [t0, open) â†’ invalidate at t1 â†’ re-assert with
    /// valid_from = t2 must PRESERVE the earlier [t0, t1) window as a distinct row
    /// rather than overwriting it. Before the fix, `kg_add_triple` reused the
    /// closed row's id and `write_triple`'s INSERT OR REPLACE ate the first window.
    #[test]
    fn kg_reassert_after_invalidate_preserves_earlier_window() {
        let (s, _tmp) = store();
        let t0 = dt(2024, 1, 1);
        let t1 = dt(2025, 1, 1);
        let t2 = dt(2025, 6, 1);

        s.kg_add_triple("Mira", "lives_in", "Dili", Some(t0), None, None)
            .expect("assert [t0, open)");
        s.kg_invalidate("Mira", "lives_in", "Dili", t1)
            .expect("invalidate â†’ [t0, t1)");
        s.kg_add_triple("Mira", "lives_in", "Dili", Some(t2), None, None)
            .expect("re-assert â†’ a NEW window [t2, open)");

        // Two rows now â€” the earlier window was NOT eaten.
        let stats = s.kg_stats().expect("stats");
        assert_eq!(
            stats.triples, 2,
            "re-assertion after invalidation must open a SECOND window, not overwrite the first"
        );
        assert_eq!(stats.open_triples, 1, "exactly the re-asserted window is open");
        assert_eq!(stats.invalidated_triples, 1, "the earlier window is still closed on file");

        // kg_timeline shows BOTH windows, oldest-first.
        let timeline = s.kg_timeline("Mira").expect("timeline");
        assert_eq!(timeline.len(), 2, "timeline must carry both windows");
        assert_eq!(timeline[0].valid_from, Some(t0));
        assert_eq!(timeline[0].valid_to, Some(t1), "first window is [t0, t1)");
        assert_eq!(timeline[1].valid_from, Some(t2));
        assert_eq!(timeline[1].valid_to, None, "second window is [t2, open)");

        // Time-travel resolves the right window at each instant.
        let inside = s
            .kg_query_entity("Mira", Some(dt(2024, 6, 1)), KgDirection::Outgoing)
            .expect("as_of inside [t0, t1)");
        assert_eq!(facts(&inside), vec![("Mira".into(), "lives_in".into(), "Dili".into())]);
        assert_eq!(inside[0].valid_to, Some(t1), "as_of in [t0, t1) resolves the earlier window");

        let now = s
            .kg_query_entity("Mira", None, KgDirection::Outgoing)
            .expect("as_of = now");
        assert_eq!(facts(&now), vec![("Mira".into(), "lives_in".into(), "Dili".into())]);
        assert_eq!(now[0].valid_from, Some(t2), "as_of = now resolves the re-asserted window");
        assert_eq!(now[0].valid_to, None);

        // Between the two windows the fact was false â€” a gap query sees nothing.
        let gap = s
            .kg_query_entity("Mira", Some(dt(2025, 3, 1)), KgDirection::Outgoing)
            .expect("as_of in the gap");
        assert!(gap.is_empty(), "between [t0,t1) and [t2,open) the fact is false");
    }

    /// A `dispatch_delta` PreserveBoth (Scenario B) applies to a peer as TWO rows
    /// on one fact key â€” both readable, neither overwriting the other. Exercises
    /// the real merge â†’ `apply_remote_delta` â†’ `write_triple` path end-to-end.
    #[test]
    fn preserve_both_lands_two_rows_on_merge() {
        use simonides_core::sync::{Hlc, SyncDelta, SyncableStore};
        use simonides_sync::delta::{encode_delta, PalaceDelta};

        let (s, _tmp) = store();
        let t0 = dt(2024, 1, 1);
        let t1 = dt(2025, 1, 1);
        let t2 = dt(2025, 6, 1);

        // Local history: (Mira, lives_in, Dili) asserted [t0, open) then closed â†’ [t0, t1).
        s.kg_add_triple("Mira", "lives_in", "Dili", Some(t0), None, None)
            .expect("assert");
        s.kg_invalidate("Mira", "lives_in", "Dili", t1).expect("invalidate");

        // Resolve local entity ids so the remote triple shares this store's fact key.
        let (subject_id, object_id): (String, String) = {
            let g = s.conn.lock().unwrap();
            let sid = g
                .query_row("SELECT id FROM entities WHERE canonical_name = 'Mira'", [], |r| r.get(0))
                .expect("mira id");
            let oid = g
                .query_row("SELECT id FROM entities WHERE canonical_name = 'Dili'", [], |r| r.get(0))
                .expect("dili id");
            (sid, oid)
        };

        // Remote delta re-asserts the fact as a FRESH OPEN window from t2 â€” the
        // dispatch_delta Scenario-B (PreserveBoth) case. Entities omitted: the triple
        // already carries this store's valid entity ids, so FKs resolve with no remap.
        let hlc = Hlc { wall_time: 9_999_999_999, logical: 0, node_id: [0u8; 16] };
        let remote = TripleRecord {
            id: Uuid::now_v7().to_string(),
            subject_id: subject_id.clone(),
            predicate: "lives_in".to_string(),
            object_id: object_id.clone(),
            valid_from: Some(t2),
            valid_to: None,
            confidence: 1.0,
            hlc,
            replaces: None,
            replaced_by: None,
        };
        let delta = PalaceDelta {
            drawers: vec![],
            palace_records: vec![],
            tunnels: vec![],
            triples: vec![remote],
            entities: vec![],
        };
        let payload = encode_delta(&delta).expect("encode_delta");
        s.apply_remote_delta(SyncDelta { payload, source_hlc: hlc })
            .expect("apply_remote_delta");

        // Both windows present on the one fact key â€” neither overwrote the other.
        let timeline = s.kg_timeline("Mira").expect("timeline");
        assert_eq!(timeline.len(), 2, "PreserveBoth must land TWO rows on one fact key");
        assert_eq!(timeline[0].valid_from, Some(t0));
        assert_eq!(timeline[0].valid_to, Some(t1), "earlier window [t0, t1) preserved");
        assert_eq!(timeline[1].valid_from, Some(t2));
        assert_eq!(timeline[1].valid_to, None, "later window [t2, open) landed");

        // Each is independently readable via time-travel.
        let earlier = s
            .kg_query_entity("Mira", Some(dt(2024, 6, 1)), KgDirection::Outgoing)
            .expect("earlier window");
        assert_eq!(earlier.len(), 1);
        assert_eq!(earlier[0].valid_to, Some(t1));
        let later = s
            .kg_query_entity("Mira", None, KgDirection::Outgoing)
            .expect("later window");
        assert_eq!(later.len(), 1);
        assert_eq!(later[0].valid_from, Some(t2));
    }

    /// With multiple windows on one fact key, `kg_invalidate` closes the OPEN one
    /// deterministically â€” never the already-closed window, never an arbitrary row.
    /// This is the read-path guarantee `query_current_triple` provides (brief Â§5).
    #[test]
    fn query_current_triple_returns_the_open_window_not_an_arbitrary_one() {
        let (s, _tmp) = store();
        let t0 = dt(2024, 1, 1);
        let t1 = dt(2025, 1, 1);
        let t2 = dt(2025, 6, 1);
        let t3 = dt(2025, 9, 1);

        // A CLOSED [t0, t1) and an OPEN [t2, ) window on one fact key.
        s.kg_add_triple("Mira", "lives_in", "Dili", Some(t0), None, None).expect("assert");
        s.kg_invalidate("Mira", "lives_in", "Dili", t1).expect("â†’ [t0, t1)");
        s.kg_add_triple("Mira", "lives_in", "Dili", Some(t2), None, None).expect("â†’ [t2, open)");

        // Invalidating must close the OPEN window at t3 â€” not the closed one.
        s.kg_invalidate("Mira", "lives_in", "Dili", t3).expect("close the open window");

        let timeline = s.kg_timeline("Mira").expect("timeline");
        assert_eq!(timeline.len(), 2, "still two windows");
        assert_eq!(timeline[0].valid_from, Some(t0));
        assert_eq!(
            timeline[0].valid_to,
            Some(t1),
            "the already-closed window must be left exactly as it was"
        );
        assert_eq!(timeline[1].valid_from, Some(t2));
        assert_eq!(
            timeline[1].valid_to,
            Some(t3),
            "kg_invalidate closed the OPEN window, deterministically"
        );

        // Nothing is open now; a further invalidate is loudly 'already invalidated'.
        let err = s
            .kg_invalidate("Mira", "lives_in", "Dili", dt(2026, 1, 1))
            .expect_err("no open window left to close");
        assert!(
            format!("{err}").contains("already invalidated"),
            "a fully-closed fact key invalidate must be loud; got {err}"
        );
    }

    /// Invalidating something the KG doesn't hold is loud in three distinct
    /// ways â€” never a silent no-op (brief Â§5).
    #[test]
    fn kg_invalidate_misses_are_loud() {
        let (s, _tmp) = store();
        s.kg_add_triple("Mira", "lives_in", "Dili", None, None, None).expect("assert fact");
        let ended = dt(2025, 1, 1);

        // Unknown entity.
        assert!(
            matches!(
                s.kg_invalidate("Ghost", "lives_in", "Dili", ended),
                Err(PalaceError::NotFound { .. })
            ),
            "unknown entity must be NotFound"
        );
        // Known entities, unknown fact.
        assert!(
            matches!(
                s.kg_invalidate("Mira", "works_on", "Dili", ended),
                Err(PalaceError::NotFound { .. })
            ),
            "unknown fact must be NotFound"
        );
        // Already invalidated.
        s.kg_invalidate("Mira", "lives_in", "Dili", ended).expect("first close");
        let err = s
            .kg_invalidate("Mira", "lives_in", "Dili", dt(2025, 6, 1))
            .expect_err("double-invalidate must be loud");
        assert!(
            format!("{err}").contains("already invalidated"),
            "double-invalidate must say so explicitly; got {err}"
        );
    }

    /// `kg_timeline` is the full history, oldest-first, including invalidated
    /// facts and facts from both directions.
    #[test]
    fn kg_timeline_is_full_history_oldest_first() {
        let (s, _tmp) = store();
        s.kg_add_triple("Mira", "lives_in", "Dili", Some(dt(2020, 1, 1)), None, None)
            .expect("fact 1");
        s.kg_add_triple("Mira", "works_on", "Cedar", Some(dt(2024, 1, 1)), None, None)
            .expect("fact 2");
        s.kg_add_triple("Hanna", "married_to", "Mira", Some(dt(2018, 1, 1)), None, None)
            .expect("fact 3");
        s.kg_invalidate("Mira", "lives_in", "Dili", dt(2025, 1, 1)).expect("close fact 1");

        let timeline = s.kg_timeline("Mira").expect("timeline");
        assert_eq!(
            facts(&timeline),
            vec![
                ("Mira".into(), "lives_in".into(), "Dili".into()),
                ("Mira".into(), "works_on".into(), "Cedar".into()),
                ("Hanna".into(), "married_to".into(), "Mira".into()),
            ],
            "timeline must be assertion-ordered (oldest first) and include BOTH \
             directions and the invalidated fact"
        );
        assert_eq!(
            timeline[0].valid_to,
            Some(dt(2025, 1, 1)),
            "the invalidated fact keeps its closed window in the timeline"
        );
    }

    /// `kg_list_entities` returns the full set (uncapped), filterable by type.
    #[test]
    fn kg_list_entities_filters_by_type_and_is_uncapped() {
        let (s, _tmp) = store();
        s.kg_add_triple("Mira", "lives_in", "Dili", None, None, None).expect("fact");
        s.kg_upsert_entity("Hanna", Some("person"), None).expect("typed entity");
        s.kg_upsert_entity("Cedar", Some("project"), None).expect("typed entity");

        let all = s.kg_list_entities(None).expect("list all");
        let names: Vec<&str> = all.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["Cedar", "Dili", "Hanna", "Mira"], "all entities, name-ordered");

        let people = s.kg_list_entities(Some("person")).expect("list people");
        assert_eq!(people.len(), 1);
        assert_eq!(people[0].name, "Hanna");

        assert_eq!(s.kg_stats().expect("stats").entities, 4);
    }

    #[test]
    fn extract_entities_finds_proper_nouns() {
        let entities = extract_entities("Mira works on Cedar and lives in Dili");
        assert!(entities.contains(&"Mira".to_string()), "must find Mira: {entities:?}");
        assert!(entities.contains(&"Cedar".to_string()), "must find Cedar: {entities:?}");
        assert!(entities.contains(&"Dili".to_string()), "must find Dili: {entities:?}");
    }

    #[test]
    fn extract_entities_filters_stop_words() {
        let entities = extract_entities("The team In On At For is working");
        // Common stop-words must not appear in entities.
        for e in &entities {
            assert!(
                !STOP_WORDS.contains(&e.as_str()),
                "stop word {e:?} must be filtered"
            );
        }
    }

    #[test]
    fn extract_entities_deduplicates() {
        let entities = extract_entities("Mira met Mira in Dili and Mira loved Dili");
        let mira_count = entities.iter().filter(|e| e.as_str() == "Mira").count();
        let dili_count = entities.iter().filter(|e| e.as_str() == "Dili").count();
        assert_eq!(mira_count, 1, "Mira must appear exactly once");
        assert_eq!(dili_count, 1, "Dili must appear exactly once");
    }

    #[test]
    fn extract_triples_finds_works_on() {
        let text = "Mira works on Cedar every day";
        let entities = extract_entities(text);
        let triples = extract_triples(text, &entities);
        assert!(
            triples.contains(&("Mira".to_string(), "works_on".to_string(), "Cedar".to_string())),
            "must find (Mira, works_on, Cedar); got: {triples:?}"
        );
    }

    #[test]
    fn extract_triples_finds_lives_in() {
        let text = "Hanna lives in Dili with her family";
        let entities = extract_entities(text);
        let triples = extract_triples(text, &entities);
        assert!(
            triples.contains(&("Hanna".to_string(), "lives_in".to_string(), "Dili".to_string())),
            "must find (Hanna, lives_in, Dili); got: {triples:?}"
        );
    }

    #[test]
    fn extract_triples_empty_when_no_pattern() {
        let entities = vec!["Alice".to_string(), "Bob".to_string()];
        let triples = extract_triples("Alice saw Bob at the store", &entities);
        // "saw" is not in the pattern list â€” no triples extracted.
        assert!(triples.is_empty(), "no triples expected without a known pattern");
    }

    #[test]
    fn extract_triples_gate_skips_dense_drawer_without_keyword() {
        // Regression (rebuild_kg stall): a drawer with MANY entities but NO
        // pattern keyword must return empty via the keyword-presence gate
        // WITHOUT running the O(k^2) pairing. Pre-fix this pegged a core for
        // minutes on real corpus drawers. 600 capitalized tokens, no predicate.
        let mut text = String::new();
        for i in 0..600 {
            text.push_str(&format!("Node{i} and "));
        }
        let entities = extract_entities(&text);
        assert!(entities.len() >= 500, "fixture must be entity-dense: {}", entities.len());
        let start = std::time::Instant::now();
        let triples = extract_triples(&text, &entities);
        let elapsed = start.elapsed();
        assert!(triples.is_empty(), "no keyword present -> no triples");
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "gate must short-circuit dense keyword-free drawers fast; took {elapsed:?}"
        );
    }

    #[test]
    fn extract_triples_bounded_on_dense_drawer_and_keeps_triple() {
        // Regression: even WITH a keyword present, a pathologically entity-dense
        // drawer must stay bounded by the linear occurrence scan and still surface the
        // genuine triple sitting at the front of the entity list.
        let mut text = String::from("Mira works on Cedar. ");
        for i in 0..1000 {
            text.push_str(&format!("Node{i} and "));
        }
        let entities = extract_entities(&text);
        assert!(entities.len() > 256, "fixture must exceed the cap: {}", entities.len());
        let start = std::time::Instant::now();
        let triples = extract_triples(&text, &entities);
        let elapsed = start.elapsed();
        assert!(
            triples.contains(&("Mira".to_string(), "works_on".to_string(), "Cedar".to_string())),
            "the real triple at the front must survive the cap; got {} triples",
            triples.len()
        );
        assert!(
            elapsed < std::time::Duration::from_secs(3),
            "capped pairing must stay bounded; took {elapsed:?}"
        );
    }

    #[test]
    fn extract_triples_no_substring_false_positive() {
        // "Cedar" must NOT match inside "Cedarfast" (the old substring test did);
        // the token-based scan matches whole tokens only.
        let text = "Mira works on Cedarfast";
        let entities =
            vec!["Mira".to_string(), "Cedar".to_string(), "Cedarfast".to_string()];
        let triples = extract_triples(text, &entities);
        assert!(
            !triples.contains(&("Mira".to_string(), "works_on".to_string(), "Cedar".to_string())),
            "must not match 'Cedar' inside 'Cedarfast'; got {triples:?}"
        );
        assert!(
            triples.contains(&(
                "Mira".to_string(),
                "works_on".to_string(),
                "Cedarfast".to_string()
            )),
            "must match the real object 'Cedarfast'; got {triples:?}"
        );
    }
}
