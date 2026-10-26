//! Hybrid retrieval for simonides — M3a (BM25) + M3b (vector + RRF).
//!
//! ## Algorithm
//!
//! 1. **Lexical (BM25)**: drawer content is chunked on write (synchronous, in
//!    the INSERT transaction).  FTS5 `chunks_fts` provides BM25 scores;
//!    parent-boost aggregates chunk scores to drawer level.
//!
//! 2. **Vector**: `rebuild_indexes()` (async/background) embeds each chunk with
//!    BGE-small-en-v1.5 and stores the 384-dim f32 BLOB inside the sqlcipher DB.
//!    At query time, brute-force cosine over decrypted-in-memory vectors (correct
//!    and fast at phone-scale; no external vector store).
//!
//! 3. **RRF fusion**: Reciprocal Rank Fusion (k=60) combines BM25 and vector
//!    ranked lists.  Parent-document grain preserved: one result per drawer.
//!    Graceful degradation: if no embeddings exist (rebuild not yet run), the
//!    vector list is empty and RRF degrades to BM25-only.

use crate::rerank::{RerankCandidate, Reranker};
use std::collections::HashMap;

use rusqlite::{params, Transaction};
use uuid::Uuid;

use crate::{DrawerState, PalaceError, PalaceStore};

type Result<T> = std::result::Result<T, PalaceError>;

/// Fixed character chunk size for lexical indexing.
pub const DEFAULT_CHUNK_SIZE: usize = 300;

/// Weight applied to sibling-chunk contributions when boosting a parent drawer.
pub const DEFAULT_PARENT_BOOST_WEIGHT: f32 = 0.35;

/// RRF smoothing constant.  Standard value; higher = flatter ranking.
const RRF_K: f32 = 60.0;

/// A ranked drawer plus the context assembled around it.
///
/// Derefs to [`crate::Drawer`], so a hit reads exactly like the drawer it wraps
/// (`hit.id`, `hit.content`, `hit.state`) — the context is strictly additive.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// The matched drawer — the system-of-record content.
    pub drawer: crate::Drawer,
    /// P4: the closets this drawer sits in, if the local projection has been
    /// built. Attached at **assembly time only** — closet membership changes no
    /// score and no ranking, it just tells the caller what company the hit
    /// keeps. Empty when no closets have been generated (the leg is additive
    /// and never required), so an absent projection degrades a hit's context,
    /// never the hit itself.
    pub closet_context: Vec<String>,
}

impl std::ops::Deref for SearchHit {
    type Target = crate::Drawer;
    fn deref(&self) -> &Self::Target {
        &self.drawer
    }
}

/// Result of a `PalaceStore::search()` call. Never a silent empty vec.
#[derive(Debug)]
pub enum SearchResult {
    /// One or more parent drawers ranked by hybrid BM25+vector RRF, best first.
    Hits(Vec<SearchHit>),
    /// No live drawers matched the query.
    Empty {
        /// The original query string, for diagnostic display.
        query: String,
    },
}

/// Insert chunks for `drawer_id` and `content` into the `chunks` table inside
/// `tx`. Called from `insert_impl` so every drawer is immediately keyword-
/// findable after the transaction commits. Idempotent: deletes any existing
/// chunks for the drawer before inserting (regeneration semantics).
pub(crate) fn chunk_in_tx(
    tx: &Transaction<'_>,
    drawer_id: &str,
    content: &str,
    chunk_size: usize,
) -> Result<()> {
    tx.execute("DELETE FROM chunks WHERE drawer_id = ?1", params![drawer_id])
        .map_err(|e| PalaceError::Storage(format!("chunk_in_tx delete: {e}")))?;

    if content.is_empty() {
        return Ok(());
    }

    let lookback = (chunk_size / 4).max(8);
    let total = content.len();
    let mut offset = 0usize;
    let mut chunk_index = 0i64;

    while offset < total {
        let end = (offset + chunk_size).min(total);
        let safe_end = find_chunk_break_point(content, end, lookback, offset);
        if safe_end <= offset {
            break;
        }
        let chunk_content = &content[offset..safe_end];
        let id = Uuid::now_v7().to_string();
        tx.execute(
            "INSERT INTO chunks \
             (id, drawer_id, chunk_index, content, char_offset, char_length) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                id,
                drawer_id,
                chunk_index,
                chunk_content,
                offset as i64,
                chunk_content.len() as i64,
            ],
        )
        .map_err(|e| PalaceError::Storage(format!("chunk_in_tx insert: {e}")))?;
        offset = safe_end;
        chunk_index += 1;
    }

    Ok(())
}

// ── Retrieval implementations ─────────────────────────────────────────────────

impl PalaceStore {
    /// BM25 leg of `search()`.  Returns up to `top_k` (drawer_id, score) pairs
    /// after parent-boost aggregation, ranked best-first.  Empty if no chunks
    /// match the query.
    pub(crate) fn search_bm25_ranked(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        // Each whitespace-separated word is its own quoted term, OR-ed: a drawer
        // matching more of the query ranks higher under BM25, but none has to
        // contain the whole query as one phrase. A word the tokenizer splits
        // further (`snake_case`, `e-mail`) stays a phrase, so identifiers still
        // match exactly. Quoting keeps FTS5 operators in user text inert.
        let terms: Vec<String> = query
            .split_whitespace()
            .filter(|w| w.chars().any(char::is_alphanumeric))
            .map(|w| format!("\"{}\"", w.replace('"', "\"\"")))
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let match_clause = terms.join(" OR ");

        // Over-fetch at chunk level so multi-chunk drawers aren't cut off before
        // parent-boost re-ranks them.
        let chunk_limit = (top_k * 20).max(100);

        let raw_hits: Vec<(String, f32)> = {
            let guard = self.conn.lock().unwrap();
            let sql = r#"
                SELECT chunks.drawer_id, bm25(chunks_fts) AS raw_score
                FROM chunks_fts
                JOIN chunks ON chunks.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?1
                ORDER BY raw_score ASC
                LIMIT ?2
            "#;
            let mut stmt = guard
                .prepare(sql)
                .map_err(|e| PalaceError::Storage(format!("bm25 prepare: {e}")))?;
            let mapped = stmt
                .query_map(params![match_clause, chunk_limit as i64], |row| {
                    let drawer_id: String = row.get(0)?;
                    let raw: f64 = row.get(1)?;
                    Ok((drawer_id, raw))
                })
                .map_err(|e| PalaceError::Storage(format!("bm25 query: {e}")))?;
            let mut out = Vec::new();
            for r in mapped {
                let (drawer_id, raw) = r.map_err(|e| PalaceError::Storage(e.to_string()))?;
                out.push((drawer_id, -(raw as f32)));
            }
            out
        };

        if raw_hits.is_empty() {
            return Ok(Vec::new());
        }

        // Aggregate per-drawer (sum of chunk scores + matching chunk count).
        let mut drawer_agg: HashMap<String, (f32, usize)> = HashMap::new();
        for (drawer_id, score) in &raw_hits {
            let entry = drawer_agg.entry(drawer_id.clone()).or_insert((0.0, 0));
            entry.0 += score;
            entry.1 += 1;
        }

        // Apply parent-boost: S * (1 + w * (N − 1)).
        let mut ranked: Vec<(String, f32)> = drawer_agg
            .into_iter()
            .map(|(drawer_id, (sum, n))| {
                let final_score =
                    sum * (1.0 + DEFAULT_PARENT_BOOST_WEIGHT * (n as f32 - 1.0));
                (drawer_id, final_score)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        Ok(ranked)
    }

    /// Vector leg of `search()`.  Brute-force cosine over all chunk embeddings
    /// in the DB; parent-collapse to best cosine per drawer.  Returns up to
    /// `top_k` (drawer_id, cosine) pairs, ranked best-first.  Empty if no
    /// embeddings have been stored yet.
    pub(crate) fn vector_search_ranked(
        &self,
        query_vec: &[f32],
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        // Load all chunk embeddings for live drawers.  sqlcipher decrypts on read;
        // cosine runs in Rust on the decrypted f32 bytes.
        let rows: Vec<(String, Vec<u8>)> = {
            let guard = self.conn.lock().unwrap();
            let mut stmt = guard
                .prepare(
                    "SELECT c.drawer_id, c.embedding \
                     FROM chunks c \
                     JOIN drawers d ON d.id = c.drawer_id \
                     WHERE c.embedding IS NOT NULL \
                       AND d.state = 'live'",
                )
                .map_err(|e| PalaceError::Storage(format!("vector_search prepare: {e}")))?;
            let mapped = stmt
                .query_map([], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
                })
                .map_err(|e| PalaceError::Storage(format!("vector_search query: {e}")))?;
            let mut out = Vec::new();
            for r in mapped {
                out.push(r.map_err(|e| PalaceError::Storage(e.to_string()))?);
            }
            out
        };

        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // Cosine per chunk, then parent-collapse to best score per drawer.
        let mut drawer_best: HashMap<String, f32> = HashMap::new();
        for (drawer_id, blob) in &rows {
            let chunk_vec = crate::embedder::blob_to_vec(blob);
            let score = crate::embedder::cosine_similarity(query_vec, &chunk_vec);
            let entry = drawer_best.entry(drawer_id.clone()).or_insert(f32::NEG_INFINITY);
            if score > *entry {
                *entry = score;
            }
        }

        let mut ranked: Vec<(String, f32)> = drawer_best.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        Ok(ranked)
    }

    /// Hybrid BM25 + vector search with Reciprocal Rank Fusion.
    ///
    /// Returns ranked parent drawers — one entry per drawer regardless of how
    /// many chunks matched.  Gracefully degrades to BM25-only when no
    /// embeddings exist or the embedder is unavailable.
    ///
    /// M4: a third KG-expansion leg is fused when the KG projection is
    /// available.  If the projection is absent or the query mentions no known
    /// entities, the leg is empty and RRF degrades to BM25+vector.
    ///
    /// Never returns a silent bare empty: `Empty { query }` makes no-match
    /// explicit and diagnosable.
    pub fn search(&self, query: &str, limit: usize) -> Result<SearchResult> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Ok(SearchResult::Empty { query: query.to_string() });
        }

        let top_k = (limit * 20).max(100);
        let fused = self.retrieve_fused(trimmed, top_k)?;

        if fused.is_empty() {
            return Ok(SearchResult::Empty { query: query.to_string() });
        }

        // Fetch parent drawers in fused rank order, live-only, up to limit.
        //
        // P4: each surviving hit is assembled with its closet context. This is a
        // pure read of the local projection — it happens after ranking, so it
        // cannot move a result up or down, and an absent/stale projection simply
        // yields empty context rather than failing the search.
        let mut hits = Vec::new();
        for (drawer_id, _) in fused.into_iter().take(limit) {
            if let Some(d) = self.get(&drawer_id)? {
                if d.state == DrawerState::Live {
                    let closet_context = self.closet_context_for_drawer(&d.id)?;
                    hits.push(SearchHit { drawer: d, closet_context });
                }
            }
        }

        if hits.is_empty() {
            return Ok(SearchResult::Empty { query: query.to_string() });
        }

        Ok(SearchResult::Hits(hits))
    }

    /// The recall-side retrieval pipeline: BM25 + vector + KG legs fused with RRF
    /// into a ranked `(drawer_id, score)` list, best-first. Extracted from
    /// `search()` so a reranker can operate on the fused head before truncation.
    /// Behaviour is exactly what `search()` used inline — this is a pure extract.
    pub(crate) fn retrieve_fused(&self, trimmed: &str, top_k: usize) -> Result<Vec<(String, f32)>> {
        // BM25 leg — always runs (lexical index is always current).
        let bm25_ranked = self.search_bm25_ranked(trimmed, top_k)?;

        // Vector leg — best-effort; empty on embedder failure or no embeddings.
        let vec_ranked: Vec<(String, f32)> = match self.embed_texts(&[trimmed]) {
            Ok(vecs) if !vecs.is_empty() => {
                self.vector_search_ranked(&vecs[0], top_k).unwrap_or_default()
            }
            Ok(_) => Vec::new(),
            Err(e) => {
                tracing::warn!("search: embedder unavailable (BM25-only fallback): {e}");
                Vec::new()
            }
        };

        // M4: KG expansion leg — depth-1, capped fan-out.
        // Check projection first; skip the entity-name DB query entirely when
        // projection is absent (no lock held across conn acquisitions).
        let kg_ranked: Vec<(String, f32)> = {
            let proj_present = self.kg_projection.lock().unwrap().is_some();
            if proj_present {
                let entity_names = self.kg_list_entity_names().unwrap_or_default();
                let expanded: Vec<(String, f32)> = {
                    let guard = self.kg_projection.lock().unwrap();
                    match guard.as_ref() {
                        Some(proj) => proj.expand(trimmed, &entity_names),
                        None => Vec::new(),
                    }
                }; // projection lock released here
                if expanded.is_empty() {
                    Vec::new()
                } else {
                    self.search_bm25_or_terms(&expanded, top_k).unwrap_or_default()
                }
            } else {
                Vec::new()
            }
        };

        // RRF fusion over all three legs.  Empty legs are no-ops in RRF.
        let fused = rrf_fusion(vec![bm25_ranked, vec_ranked, kg_ranked], RRF_K);
        Ok(fused)
    }

    /// §3.6 reranked search: retrieve (wide recall), rerank the top-N fused
    /// candidates with `reranker`, THEN truncate to `limit`. The rerank window is
    /// wider than `limit` so the reranker can promote a better content match from
    /// below the fold — the whole point of a reranking layer. Uses the SAME
    /// pipeline as `search()`; with an `IdentityReranker` this is identical to
    /// `search()` (proven path preserved). Retrieval + rerank are LOCAL — the
    /// reranker is on-device (membrane holds).
    pub fn search_with_reranker(
        &self,
        query: &str,
        limit: usize,
        reranker: &dyn Reranker,
    ) -> Result<SearchResult> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Ok(SearchResult::Empty { query: query.to_string() });
        }
        let top_k = (limit * 20).max(100);
        let fused = self.retrieve_fused(trimmed, top_k)?;
        if fused.is_empty() {
            return Ok(SearchResult::Empty { query: query.to_string() });
        }

        // Rerank window: fetch content for the top-N fused candidates (N > limit),
        // live-only, each drawer fetched ONCE and held for assembly.
        let window = (limit * 8).max(30);
        let mut held: std::collections::HashMap<String, crate::Drawer> =
            std::collections::HashMap::new();
        let mut candidates: Vec<RerankCandidate> = Vec::new();
        for (drawer_id, score) in fused.iter().take(window) {
            if let Some(d) = self.get(drawer_id)? {
                if d.state == DrawerState::Live {
                    candidates.push(RerankCandidate {
                        drawer_id: d.id.clone(),
                        content: d.content.clone(),
                        retrieval_score: *score,
                    });
                    held.insert(d.id.clone(), d);
                }
            }
        }
        if candidates.is_empty() {
            return Ok(SearchResult::Empty { query: query.to_string() });
        }

        let reranked = reranker.rerank(trimmed, candidates);

        let mut hits = Vec::new();
        for c in reranked.into_iter().take(limit) {
            if let Some(d) = held.remove(&c.drawer_id) {
                let closet_context = self.closet_context_for_drawer(&d.id)?;
                hits.push(SearchHit { drawer: d, closet_context });
            }
        }
        if hits.is_empty() {
            return Ok(SearchResult::Empty { query: query.to_string() });
        }
        Ok(SearchResult::Hits(hits))
    }

    /// BM25 OR-match across a set of expansion terms (entity names).
    ///
    /// Constructs an FTS5 `"term1" OR "term2" OR ...` clause so that drawers
    /// mentioning ANY of the expanded entities are returned.  Used by the M4
    /// KG expansion leg in `search()`.
    pub(crate) fn search_bm25_or_terms(
        &self,
        terms: &[(String, f32)],
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        // Build FTS5 OR match clause: "TermA" OR "TermB" …
        // Double-quotes inside a term name are escaped as "".
        let match_clause = terms
            .iter()
            .map(|(name, _)| format!("\"{}\"", name.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" OR ");

        let chunk_limit = (top_k * 20).max(100);

        let raw_hits: Vec<(String, f32)> = {
            let guard = self.conn.lock().unwrap();
            let sql = r#"
                SELECT chunks.drawer_id, bm25(chunks_fts) AS raw_score
                FROM chunks_fts
                JOIN chunks ON chunks.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?1
                ORDER BY raw_score ASC
                LIMIT ?2
            "#;
            let mut stmt = guard
                .prepare(sql)
                .map_err(|e| PalaceError::Storage(format!("kg or bm25 prepare: {e}")))?;
            let mapped = stmt
                .query_map(params![match_clause, chunk_limit as i64], |row| {
                    let drawer_id: String = row.get(0)?;
                    let raw: f64 = row.get(1)?;
                    Ok((drawer_id, raw))
                })
                .map_err(|e| PalaceError::Storage(format!("kg or bm25 query: {e}")))?;
            let mut out = Vec::new();
            for r in mapped {
                let (drawer_id, raw) = r.map_err(|e| PalaceError::Storage(e.to_string()))?;
                out.push((drawer_id, -(raw as f32)));
            }
            out
        };

        if raw_hits.is_empty() {
            return Ok(Vec::new());
        }

        // Parent-boost aggregation (same logic as search_bm25_ranked).
        let mut drawer_agg: HashMap<String, (f32, usize)> = HashMap::new();
        for (drawer_id, score) in &raw_hits {
            let entry = drawer_agg.entry(drawer_id.clone()).or_insert((0.0, 0));
            entry.0 += score;
            entry.1 += 1;
        }
        let mut ranked: Vec<(String, f32)> = drawer_agg
            .into_iter()
            .map(|(drawer_id, (sum, n))| {
                let s = sum * (1.0 + DEFAULT_PARENT_BOOST_WEIGHT * (n as f32 - 1.0));
                (drawer_id, s)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        Ok(ranked)
    }
}

/// Reciprocal Rank Fusion over multiple ranked lists.
///
/// `rankings`: each element is a ranked Vec<(drawer_id, score)> best-first.
/// `k`: smoothing constant (standard = 60).
///
/// Returns a new ranked Vec<(drawer_id, rrf_score)> best-first.
/// A drawer that appears in N lists receives N independent RRF contributions.
pub(crate) fn rrf_fusion(
    rankings: Vec<Vec<(String, f32)>>,
    k: f32,
) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for ranking in &rankings {
        for (rank, (drawer_id, _)) in ranking.iter().enumerate() {
            *scores.entry(drawer_id.clone()).or_insert(0.0) +=
                1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut result: Vec<(String, f32)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

// ── UTF-8-aware chunk boundary helpers ────────────────────────────────────────

fn find_char_boundary(s: &str, byte_idx: usize) -> usize {
    if byte_idx >= s.len() {
        return s.len();
    }
    let mut idx = byte_idx;
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx.max(1).min(s.len())
}

fn find_chunk_break_point(s: &str, target: usize, lookback: usize, min_offset: usize) -> usize {
    let total = s.len();
    if target >= total {
        return total;
    }
    let char_end = find_char_boundary(s, target);
    let lower_bound = char_end.saturating_sub(lookback).max(min_offset + 1);
    let bytes = s.as_bytes();
    if lower_bound < char_end {
        let mut idx = char_end;
        while idx > lower_bound {
            idx -= 1;
            if s.is_char_boundary(idx) {
                let b = bytes[idx];
                let is_ws = if b < 128 {
                    b == b' ' || b == b'\t' || b == b'\n' || b == b'\r'
                } else {
                    s[idx..].chars().next().map(|c| c.is_whitespace()).unwrap_or(false)
                };
                if is_ws {
                    let ws_len = if b < 128 {
                        1
                    } else {
                        s[idx..].chars().next().map(|c| c.len_utf8()).unwrap_or(1)
                    };
                    return idx + ws_len;
                }
            }
        }
    }
    char_end
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PalaceStore;
    use rusqlite::params;
    use tempfile::TempDir;

    const TEST_KEY: &str = "palace-retrieval-test-key-fixed";

    fn store() -> (PalaceStore, TempDir) {
        let tmp = TempDir::new().expect("tempdir");
        let s =
            PalaceStore::open_in_memory(tmp.path(), TEST_KEY).expect("open in-memory store");
        (s, tmp)
    }

    // ── Gate A: chunk schema + chunk-on-write ─────────────────────────────────

    #[test]
    fn chunks_tables_exist_in_schema() {
        let (s, _tmp) = store();
        let guard = s.conn.lock().unwrap();
        let tables: Vec<String> = guard
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert!(
            tables.contains(&"chunks".to_string()),
            "chunks table must exist; tables: {tables:?}"
        );
        assert!(
            tables.contains(&"chunks_fts".to_string()),
            "chunks_fts virtual table must exist; tables: {tables:?}"
        );
    }

    #[test]
    fn chunks_table_has_embedding_column() {
        let (s, _tmp) = store();
        let guard = s.conn.lock().unwrap();
        let cols: Vec<String> = guard
            .prepare("PRAGMA table_info(chunks)")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert!(
            cols.contains(&"embedding".to_string()),
            "chunks table must have `embedding` column (M3b schema); columns: {cols:?}"
        );
    }

    #[test]
    fn insert_drawer_creates_chunks() {
        let (s, _tmp) = store();
        let d = s
            .insert("test", "room", "Hello world this is some drawer content for retrieval indexing", "test")
            .unwrap();
        let guard = s.conn.lock().unwrap();
        let count: i64 = guard
            .query_row(
                "SELECT COUNT(*) FROM chunks WHERE drawer_id = ?1",
                params![d.id],
                |row| row.get(0),
            )
            .unwrap();
        assert!(count > 0, "insert must create at least one chunk for non-empty content");
    }

    #[test]
    fn tombstone_clears_chunks() {
        let (s, _tmp) = store();
        let d = s
            .insert("test", "room", "content that will be tombstoned and chunks removed", "test")
            .unwrap();
        let before: i64 = {
            let guard = s.conn.lock().unwrap();
            guard
                .query_row(
                    "SELECT COUNT(*) FROM chunks WHERE drawer_id = ?1",
                    params![d.id],
                    |row| row.get(0),
                )
                .unwrap()
        };
        assert!(before > 0, "chunks must exist before tombstone");

        s.tombstone(&d.id).unwrap();

        let after: i64 = {
            let guard = s.conn.lock().unwrap();
            guard
                .query_row(
                    "SELECT COUNT(*) FROM chunks WHERE drawer_id = ?1",
                    params![d.id],
                    |row| row.get(0),
                )
                .unwrap()
        };
        assert_eq!(after, 0, "tombstone must remove all chunks for the drawer");
    }

    #[test]
    fn fts5_under_sqlcipher_queryable() {
        let (s, _tmp) = store();
        s.insert("w", "r", "cedar privacy assistant mesh network tor routing", "test")
            .unwrap();
        let guard = s.conn.lock().unwrap();
        let count: i64 = guard
            .query_row(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'privacy'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(count > 0, "FTS5 must be queryable under sqlcipher; count={count}");
    }

    #[test]
    fn empty_content_drawer_has_no_chunks() {
        let (s, _tmp) = store();
        let d = s.insert("w", "r", "", "test").unwrap();
        let guard = s.conn.lock().unwrap();
        let count: i64 = guard
            .query_row(
                "SELECT COUNT(*) FROM chunks WHERE drawer_id = ?1",
                params![d.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0, "empty content must produce zero chunks");
    }

    // ── Gate B (M3a): BM25 + parent-boost ────────────────────────────────────

    #[test]
    fn search_returns_matching_drawer() {
        let (s, _tmp) = store();
        s.insert("w", "r", "cedar privacy assistant mesh network tor routing", "test")
            .unwrap();
        match s.search("privacy", 10).unwrap() {
            SearchResult::Hits(drawers) => {
                assert!(!drawers.is_empty());
                assert!(
                    drawers.iter().any(|d| d.content.contains("privacy")),
                    "search must return the drawer containing the query term"
                );
            }
            SearchResult::Empty { query } => {
                panic!("expected Hits, got Empty {{ query: {query:?} }}")
            }
        }
    }

    #[test]
    fn a_multi_word_query_matches_without_being_an_exact_phrase() {
        let (s, _tmp) = store();
        let d = s
            .insert("w", "r", "The method of loci places each memory in a room of an imagined building.", "test")
            .unwrap();
        s.insert("w", "r", "Unrelated note about lunch.", "test").unwrap();
        // Words out of order, one not present at all: still a lexical hit.
        let hits = s.search_bm25_ranked("imagined building memory palace", 10).unwrap();
        assert_eq!(hits.first().map(|(id, _)| id.as_str()), Some(d.id.as_str()), "{hits:?}");
        // FTS5 syntax in user text is inert, not an error.
        assert!(s.search_bm25_ranked("NEAR( \"loci\" OR * -", 10).is_ok());
        assert!(s.search_bm25_ranked("   ", 10).unwrap().is_empty());
    }
    #[test]
    fn parent_boost_ranks_multi_chunk_drawer_higher() {
        let (s, _tmp) = store();

        let content_a = "Fundraising plan for Q3. ".repeat(36);
        s.insert("w", "r", &content_a, "test").unwrap();
        s.insert("w", "r", "Fundraising note: see Q3 plan for details.", "test").unwrap();

        match s.search("fundraising", 10).unwrap() {
            SearchResult::Hits(drawers) => {
                assert!(drawers.len() >= 2, "both drawers must appear in results");
                assert!(
                    drawers[0].content.starts_with("Fundraising plan"),
                    "multi-chunk drawer A must rank above single-chunk drawer B; \
                     first result starts with: {:?}",
                    &drawers[0].content[..40.min(drawers[0].content.len())]
                );
            }
            SearchResult::Empty { query } => {
                panic!("expected Hits, got Empty {{ query: {query:?} }}")
            }
        }
    }

    #[test]
    fn search_empty_query_returns_loud_empty() {
        let (s, _tmp) = store();
        s.insert("w", "r", "some content", "test").unwrap();
        match s.search("", 10).unwrap() {
            SearchResult::Empty { .. } => {}
            SearchResult::Hits(_) => panic!("empty query must return Empty, not Hits"),
        }
    }

    #[test]
    fn search_no_match_returns_loud_empty() {
        let (s, _tmp) = store();
        s.insert("w", "r", "cedar mesh routing privacy", "test").unwrap();
        match s.search("xyzzy_no_such_term_8472", 10).unwrap() {
            SearchResult::Empty { .. } => {}
            SearchResult::Hits(d) => {
                panic!("no-match query must return Empty, got {} hits", d.len())
            }
        }
    }

    #[test]
    fn search_deduplicates_to_parent_grain() {
        let (s, _tmp) = store();
        let long_content = "privacy mesh routing cedar network. ".repeat(50);
        let d = s.insert("w", "r", &long_content, "test").unwrap();

        match s.search("privacy", 100).unwrap() {
            SearchResult::Hits(drawers) => {
                let id_set: std::collections::HashSet<&str> =
                    drawers.iter().map(|dr| dr.id.as_str()).collect();
                assert_eq!(
                    id_set.len(),
                    drawers.len(),
                    "search must return one result per parent drawer (no duplicates)"
                );
                assert!(
                    drawers.iter().any(|dr| dr.id == d.id),
                    "the inserted drawer must appear in results"
                );
            }
            SearchResult::Empty { query } => {
                panic!("expected Hits, got Empty {{ query: {query:?} }}")
            }
        }
    }

    #[test]
    fn search_excludes_tombstoned_drawers() {
        let (s, _tmp) = store();
        let d = s
            .insert("w", "r", "cedar privacy routing unique_term_xqz9283", "test")
            .unwrap();
        s.tombstone(&d.id).unwrap();

        match s.search("unique_term_xqz9283", 10).unwrap() {
            SearchResult::Empty { .. } => {}
            SearchResult::Hits(hits) => {
                assert!(
                    !hits.iter().any(|h| h.id == d.id),
                    "tombstoned drawer must not appear in search results"
                );
            }
        }
    }

    // ── Gate A (M3b): embedding stable 384-dim + blob round-trip ─────────────

    #[test]
    fn embedding_gate_stable_384dim() {
        let (s, _tmp) = store();
        s.insert("w", "r", "cedar privacy encryption mesh tor network", "test").unwrap();
        s.rebuild_indexes().expect("rebuild_indexes");

        let guard = s.conn.lock().unwrap();
        let blob: Vec<u8> = guard
            .query_row(
                "SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1",
                [],
                |r| r.get(0),
            )
            .expect("embedding row must exist after rebuild_indexes");
        drop(guard);

        let vec = crate::embedder::blob_to_vec(&blob);
        assert_eq!(
            vec.len(),
            crate::embedder::EMBEDDING_DIM,
            "BGE-small-en-v1.5 must produce exactly {} dimensions",
            crate::embedder::EMBEDDING_DIM
        );

        // Blob round-trip: encode → decode → cosine ≈ 1.0.
        let rt = crate::embedder::blob_to_vec(&crate::embedder::vec_to_blob(&vec));
        let sim = crate::embedder::cosine_similarity(&vec, &rt);
        assert!(
            (sim - 1.0_f32).abs() < 1e-5,
            "blob round-trip cosine must be 1.0; got {sim}"
        );
    }

    // ── Gate B (M3b): rebuild_indexes embeds + clears pending ────────────────

    #[test]
    fn rebuild_indexes_embeds_and_clears_pending() {
        let (s, _tmp) = store();
        let d = s
            .insert("w", "r", "cedar privacy assistant tor mesh network encryption", "test")
            .unwrap();
        assert!(d.reindex_pending, "new drawer must start reindex_pending=true");

        s.rebuild_indexes().expect("rebuild_indexes");

        let refetched = s.get(&d.id).unwrap().unwrap();
        assert!(!refetched.reindex_pending, "rebuild_indexes must clear reindex_pending");

        let guard = s.conn.lock().unwrap();
        let count: i64 = guard
            .query_row(
                "SELECT COUNT(*) FROM chunks WHERE drawer_id = ?1 AND embedding IS NOT NULL",
                params![d.id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            count > 0,
            "all chunks must have embeddings after rebuild_indexes"
        );
    }

    #[test]
    fn rebuild_indexes_sets_embedding_model_version() {
        let (s, _tmp) = store();
        s.insert("w", "r", "test content for embedding model version check", "test").unwrap();
        s.rebuild_indexes().expect("rebuild_indexes");

        let guard = s.conn.lock().unwrap();
        let ver: String = guard
            .query_row(
                "SELECT value FROM meta WHERE key = 'embedding_model_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            ver,
            crate::embedder::MODEL_ID,
            "meta.embedding_model_version must be set after rebuild_indexes"
        );
    }

    #[test]
    fn insert_makes_drawer_lexically_findable_before_rebuild() {
        // Lexical index is synchronous — drawer must be keyword-searchable
        // immediately after insert, before rebuild_indexes() runs.
        let (s, _tmp) = store();
        let d = s
            .insert("w", "r", "unique_term_zqp8847 cedar privacy", "test")
            .unwrap();
        assert!(d.reindex_pending, "must start pending (vector not yet embedded)");

        match s.search("unique_term_zqp8847", 5).unwrap() {
            SearchResult::Hits(hits) => {
                assert!(
                    hits.iter().any(|h| h.id == d.id),
                    "drawer must be lexically findable before rebuild_indexes"
                );
            }
            SearchResult::Empty { query } => {
                panic!("expected Hits for lexical query before rebuild; query={query:?}");
            }
        }
    }

    // ── Gate C (M3b): vector retrieval finds semantic match BM25 misses ───────

    #[test]
    fn vector_retrieval_finds_semantic_match_bm25_misses() {
        let (s, _tmp) = store();
        // "automobile"/"expressway" vs query "car crash road" — lexically disjoint.
        let d = s
            .insert(
                "w",
                "r",
                "The automobile collided with another vehicle on the expressway \
                 causing significant traffic congestion and property damage",
                "test",
            )
            .unwrap();
        s.insert(
            "w",
            "r",
            "recipe for making chocolate cake with flour butter eggs sugar vanilla",
            "test",
        )
        .unwrap();

        s.rebuild_indexes().expect("rebuild_indexes");

        // BM25 leg should miss (lexically disjoint vocabulary).
        let bm25 = s.search_bm25_ranked("car crash road", 10).unwrap();
        let bm25_found = bm25.iter().any(|(id, _)| id == &d.id);

        // Vector search should find the automobile drawer.
        let qvec = s.embed_texts(&["car crash road"]).expect("embed");
        let vec_hits = s
            .vector_search_ranked(&qvec[0], 10)
            .expect("vector_search_ranked");
        let vec_found = vec_hits.iter().any(|(id, _)| id == &d.id);

        assert!(
            vec_found,
            "vector search must find the automobile/expressway drawer via semantic similarity \
             (BM25 found: {bm25_found})"
        );
    }

    // ── Gate D (M3b): RRF fusion + parent grain ───────────────────────────────

    #[test]
    fn rrf_fusion_combines_signals() {
        // Unit test for rrf_fusion: an item present in both lists must rank first.
        let bm25: Vec<(String, f32)> = vec![
            ("drawer_a".into(), 2.0),
            ("drawer_b".into(), 1.0),
        ];
        let vec: Vec<(String, f32)> = vec![
            ("drawer_b".into(), 0.9),
            ("drawer_c".into(), 0.8),
        ];
        let fused = rrf_fusion(vec![bm25, vec], 60.0);

        assert!(!fused.is_empty());
        assert_eq!(
            fused[0].0, "drawer_b",
            "drawer_b (in both lists) must rank first via RRF; got {:?}",
            fused
        );

        // drawer_a at rank-0 in BM25 (score 1/61), drawer_c at rank-1 in vec (score 1/62).
        // drawer_a > drawer_c.
        let pos_a = fused.iter().position(|(id, _)| id == "drawer_a").unwrap();
        let pos_c = fused.iter().position(|(id, _)| id == "drawer_c").unwrap();
        assert!(pos_a < pos_c, "drawer_a (rank 0 in BM25) must beat drawer_c (rank 1 in vec)");
    }

    #[test]
    fn rrf_parent_grain_preserved_under_fusion() {
        // After hybrid search, each drawer ID appears at most once.
        let (s, _tmp) = store();
        let long = "privacy encryption mesh cedar network. ".repeat(50);
        s.insert("w", "r", &long, "test").unwrap();
        s.insert("w", "r", "cedar mesh routing privacy assistant", "test").unwrap();
        s.rebuild_indexes().expect("rebuild_indexes");

        match s.search("privacy", 20).unwrap() {
            SearchResult::Hits(drawers) => {
                assert!(
                    !drawers.is_empty(),
                    "expected Hits for fixture query 'privacy', got empty drawers vec"
                );
                let ids: Vec<&str> = drawers.iter().map(|d| d.id.as_str()).collect();
                let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
                assert_eq!(
                    unique.len(),
                    ids.len(),
                    "hybrid search must return one result per drawer (parent grain); got: {ids:?}"
                );
            }
            SearchResult::Empty { query } => {
                panic!("expected Hits for fixture query, got Empty {{ query: {query:?} }}")
            }
        }
    }

    // ── Gate G (M4): KG encryption + extraction ──────────────────────────────

    /// Encryption gate.
    ///
    /// Asserts:
    /// (i)  Positive whitelist: only {*.db, *-wal, *-shm, vault} on disk —
    ///      no graph/triple/entity directory outside the keyed DB.
    /// (ii) DB header ≠ SQLite magic (remains sqlcipher-encrypted).
    /// (iii) Triples present in the keyed DB after rebuild_indexes().
    #[test]
    fn kg_facts_in_encrypted_db_no_plaintext_graph_dir() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let db_path = dir.path().join("palace.db");
        let vault_root = dir.path().join("vault");
        const SQLITE_MAGIC: &[u8; 16] = b"SQLite format 3\0";

        let s = PalaceStore::open(&db_path, &vault_root, "m4-kg-gate-test-key-xz9")
            .expect("open");
        s.insert(
            "w", "r",
            "Mira works on Cedar and Hanna lives in Dili",
            "test",
        )
        .expect("insert");
        s.rebuild_indexes().expect("rebuild_indexes (extraction + embedding)");

        // (i) Positive whitelist.
        let children: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        for child in &children {
            assert!(
                child.ends_with(".db")
                    || child.ends_with("-wal")
                    || child.ends_with("-shm")
                    || child == "vault",
                "unexpected on-disk child {child:?} — KG facts must remain inside \
                 the encrypted DB; all children: {children:?}"
            );
        }

        // (ii) DB header is ciphertext.
        let raw = std::fs::read(&db_path).expect("read db");
        assert!(raw.len() >= 16, "db file too short");
        let header: &[u8; 16] = raw[..16].try_into().unwrap();
        assert_ne!(
            header, SQLITE_MAGIC,
            "DB header matches SQLite magic — KG facts are in a plaintext DB (M4 gate)"
        );

        // (iii) Triples present inside the keyed DB.
        let triple_count: i64 = {
            let guard = s.conn.lock().unwrap();
            guard
                .query_row("SELECT COUNT(*) FROM triples", [], |r| r.get(0))
                .expect("count triples")
        };
        assert!(
            triple_count > 0,
            "at least one triple must be stored in the encrypted DB after rebuild_indexes; \
             got {triple_count}"
        );
    }

    /// KG extraction produces entities and triples for known-pattern content.
    #[test]
    fn kg_extraction_stores_entities_and_triples() {
        let (s, _tmp) = store();
        s.insert("w", "r", "Mira works on Cedar every day", "test").unwrap();
        s.rebuild_indexes().expect("rebuild_indexes");

        let entity_names = s.kg_list_entity_names().expect("list entity names");
        assert!(
            entity_names.iter().any(|n| n == "Mira"),
            "Mira must be in entities after extraction; entities: {entity_names:?}"
        );
        assert!(
            entity_names.iter().any(|n| n == "Cedar"),
            "Cedar must be in entities after extraction; entities: {entity_names:?}"
        );

        let triple_count: i64 = {
            let guard = s.conn.lock().unwrap();
            guard
                .query_row("SELECT COUNT(*) FROM triples", [], |r| r.get(0))
                .unwrap()
        };
        assert!(
            triple_count > 0,
            "at least one triple must be extracted; got {triple_count}"
        );
    }

    /// KG projection is present after rebuild_indexes.
    #[test]
    fn kg_projection_built_after_rebuild_indexes() {
        let (s, _tmp) = store();
        s.insert("w", "r", "Hanna lives in Dili with her family", "test").unwrap();
        s.rebuild_indexes().expect("rebuild_indexes");

        let guard = s.kg_projection.lock().unwrap();
        assert!(
            guard.is_some(),
            "kg_projection must be Some after rebuild_indexes"
        );
    }

    /// KG expansion in search() surfaces a drawer about a related entity
    /// even when the query term doesn't appear in that drawer's text.
    #[test]
    fn kg_expansion_surfaces_related_entity_drawer() {
        let (s, _tmp) = store();

        // Drawer A: about Mira (who works on Cedar per KG).
        s.insert("w", "r", "Mira works on Cedar the privacy app", "test").unwrap();
        // Drawer B: about Cedar only — no mention of Mira.
        let d_cedar = s
            .insert("w", "r", "Cedar is a peer-to-peer encrypted communication platform", "test")
            .unwrap();
        // Drawer C: noise.
        s.insert("w", "r", "chocolate cake recipe with vanilla frosting", "test").unwrap();

        s.rebuild_indexes().expect("rebuild_indexes + KG extraction");

        // Query for "Mira" — KG should expand to "Cedar" and surface drawer B.
        match s.search("Mira", 10).unwrap() {
            SearchResult::Hits(drawers) => {
                assert!(
                    drawers.iter().any(|d| d.id == d_cedar.id),
                    "KG expansion must surface the Cedar drawer when querying Mira \
                     (Mira works_on Cedar); results: {:?}",
                    drawers.iter().map(|d| &d.content[..d.content.len().min(40)]).collect::<Vec<_>>()
                );
            }
            SearchResult::Empty { query } => {
                panic!("expected Hits for query='Mira', got Empty {{ query: {query:?} }}");
            }
        }
    }

    /// Parent-grain deduplication holds under KG expansion.
    #[test]
    fn kg_expansion_preserves_parent_grain() {
        let (s, _tmp) = store();
        s.insert("w", "r", "Mira works on Cedar, a privacy tool", "test").unwrap();
        s.insert("w", "r", "Cedar uses mesh networking for routing", "test").unwrap();
        s.rebuild_indexes().expect("rebuild_indexes");

        match s.search("Mira", 20).unwrap() {
            SearchResult::Hits(drawers) => {
                assert!(
                    !drawers.is_empty(),
                    "expected Hits for fixture query 'Mira', got empty drawers vec"
                );
                let ids: Vec<&str> = drawers.iter().map(|d| d.id.as_str()).collect();
                let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
                assert_eq!(
                    unique.len(),
                    ids.len(),
                    "KG expansion must not produce duplicate drawer entries; ids: {ids:?}"
                );
            }
            SearchResult::Empty { query } => {
                panic!("expected Hits for fixture query, got Empty {{ query: {query:?} }}")
            }
        }
    }

    // ── Gate F (M3b): eval recall@5 bar ──────────────────────────────────────

    /// Recall@5 >= 0.8 over 5 semantic query–drawer pairs where query
    /// vocabulary is lexically disjoint from drawer content.  Tests that the
    /// hybrid path provides genuine semantic retrieval.
    ///
    /// Bar = 0.8 (4/5).  BGE-small-en-v1.5 on clear semantic pairs should
    /// achieve 5/5; the bar gives one miss of slack for borderline pairs.
    #[test]
    fn eval_recall_at_5_bar() {
        const RECALL_BAR: f32 = 0.8;

        let (s, _tmp) = store();

        let d1 = s
            .insert(
                "w", "r",
                "automobile vehicle collision on expressway causing major congestion",
                "eval",
            )
            .unwrap();
        let d2 = s
            .insert(
                "w", "r",
                "neural network training gradient descent backpropagation optimization",
                "eval",
            )
            .unwrap();
        let d3 = s
            .insert(
                "w", "r",
                "preparing pasta dish with olive oil garlic herbs for dinner",
                "eval",
            )
            .unwrap();
        let d4 = s
            .insert(
                "w", "r",
                "stock market portfolio equity holdings investment strategy",
                "eval",
            )
            .unwrap();
        let d5 = s
            .insert(
                "w", "r",
                "hiking through mountain trail wilderness nature trekking",
                "eval",
            )
            .unwrap();

        s.rebuild_indexes().expect("rebuild_indexes for eval");

        let cases: &[(&str, &str)] = &[
            ("car crash road accident",                      &d1.id),
            ("deep learning AI model optimization",          &d2.id),
            ("cooking spaghetti Italian recipe ingredients", &d3.id),
            ("investing in bonds financial returns",         &d4.id),
            ("outdoor forest adventure trail",               &d5.id),
        ];

        let mut correct = 0usize;
        for (query, expected_id) in cases {
            match s.search(query, 5).unwrap() {
                SearchResult::Hits(drawers) => {
                    if drawers.iter().any(|d| &d.id == expected_id) {
                        correct += 1;
                    }
                }
                SearchResult::Empty { .. } => {}
            }
        }

        let recall = correct as f32 / cases.len() as f32;
        assert!(
            recall >= RECALL_BAR,
            "recall@5 = {correct}/{} = {:.2}, below bar {RECALL_BAR}. \
             Queries and expected drawers: {cases:?}",
            cases.len(),
            recall,
        );
    }

    // ── Independent positive gate: insert-then-search must return Hits ────────

    /// Dedicated non-empty guard: insert a drawer with a unique token, rebuild,
    /// search that token, assert Hits are returned and include the drawer.
    /// Independent of all other tests — a retrieval regression that silently
    /// returns Empty for matching content fails here regardless of test order.
    #[test]
    fn retrieval_returns_hits_for_matching_insert() {
        let (s, _tmp) = store();
        let d = s
            .insert(
                "w",
                "r",
                "unique_retrieval_positive_guard_term_zx7293 cedar privacy encryption",
                "test",
            )
            .unwrap();
        s.rebuild_indexes().expect("rebuild_indexes");

        match s.search("unique_retrieval_positive_guard_term_zx7293", 5).unwrap() {
            SearchResult::Hits(drawers) => {
                assert!(
                    !drawers.is_empty(),
                    "search must return at least one hit for matching content"
                );
                assert!(
                    drawers.iter().any(|h| h.id == d.id),
                    "the inserted drawer must appear in search results; \
                     got ids: {:?}",
                    drawers.iter().map(|h| &h.id).collect::<Vec<_>>()
                );
            }
            SearchResult::Empty { query } => {
                panic!(
                    "expected Hits for known-matching query, got Empty {{ query: {query:?} }}"
                )
            }
        }
    }

    #[test]
    fn search_with_reranker_identity_equals_proven_search() {
        // Gated-OFF path: IdentityReranker must produce exactly what search() does.
        let (s, _t) = store();
        s.insert("w", "r", "quantum physics and entanglement", "t").unwrap();
        s.insert("w", "r", "quantum computing basics", "t").unwrap();
        s.insert("w", "r", "a note about gardening", "t").unwrap();
        let base = match s.search("quantum", 5).unwrap() {
            SearchResult::Hits(h) => h.iter().map(|x| x.drawer.id.clone()).collect::<Vec<_>>(),
            SearchResult::Empty { .. } => panic!("expected hits"),
        };
        let rr = match s.search_with_reranker("quantum", 5, &crate::rerank::IdentityReranker).unwrap() {
            SearchResult::Hits(h) => h.iter().map(|x| x.drawer.id.clone()).collect::<Vec<_>>(),
            SearchResult::Empty { .. } => panic!("expected hits"),
        };
        assert_eq!(base, rr, "identity reranker must not change the proven result order");
    }

    #[test]
    fn search_with_reranker_promotes_highest_content_overlap() {
        // The reranking win: the full content match is promoted to the head even
        // if retrieval didn't rank it first.
        let (s, _t) = store();
        let full = s.insert("w", "r", "quantum physics entanglement lecture", "t").unwrap();
        let _partial = s.insert("w", "r", "quantum computing hardware", "t").unwrap();
        let hits = match s
            .search_with_reranker("quantum physics", 5, &crate::rerank::LexicalOverlapReranker)
            .unwrap()
        {
            SearchResult::Hits(h) => h,
            SearchResult::Empty { .. } => panic!("expected hits"),
        };
        assert_eq!(hits[0].drawer.id, full.id, "highest query-term overlap must rank first after rerank");
    }
}
