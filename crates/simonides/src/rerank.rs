//! §3.6 learned-reranking seam — rerank the top-K hybrid-retrieval candidates by
//! query-relevance before truncating to the caller's limit.
//!
//! WHERE THIS LIVES in the pipeline: `search()` runs BM25 + vector + KG legs,
//! fuses them with RRF into a ranked list, then truncates to `limit`. A reranker
//! sits BETWEEN fusion and truncation: rescore the top-N fused candidates, reorder,
//! then take `limit`. Retrieval casts a wide net (recall); the reranker sharpens
//! the head (precision) — the k=5 precision win the graph-vs-flat A/B pointed at.
//!
//! WHAT PLUGS IN HERE:
//!  * The production **learned cross-encoder** (§3.6): a small bundled model
//!    (~150MB, lazy-loaded only when search fires) scoring (query, drawer) pairs
//!    on the same ONNX runtime the embedder already uses. It reranks on CONTENT
//!    semantics, independent of the KG — which is why it's the first real win.
//!  * A **graph-signal reranker** (the A/B's literal finding) is a further follow,
//!    but it is bottlenecked by the extractor, NOT this architecture: on the real
//!    corpus the KG is degenerate (13 triples), so graph proximity reorders almost
//!    nothing until the v2 (LLM) extractor lands. The cross-encoder does not depend
//!    on the KG, so it pays off now; the graph reranker pays off after the extractor
//!    upgrade. Both implement the same [`Reranker`] trait and drop in unchanged.
//!
//! This module is the seam + a hermetic, deterministic reference reranker so the
//! plumbing is proven and testable WITHOUT a 150MB model dependency. The learned
//! model is wired behind the trait against measurement (the eval bar), never on
//! intuition — same discipline as the retrieval eval-to-a-bar.

/// A retrieval candidate handed to a reranker: the drawer id, its content (the
/// reranker's signal), and the fused retrieval score (recall-side rank, used as a
/// stable tie-break so reranking is deterministic).
#[derive(Debug, Clone, PartialEq)]
pub struct RerankCandidate {
    pub drawer_id: String,
    pub content: String,
    pub retrieval_score: f32,
}

/// Rerank the top-K retrieval candidates for a query. Implementations MUST be
/// deterministic (same inputs -> same order) and MUST NOT drop candidates — a
/// reranker reorders, it never silently loses a hit (truncation to `limit` is the
/// caller's job, after reranking).
pub trait Reranker: Send + Sync {
    /// Return the candidates reordered best-first. Same length as the input.
    fn rerank(&self, query: &str, candidates: Vec<RerankCandidate>) -> Vec<RerankCandidate>;
}

/// No-op reranker: preserves the retrieval order exactly. This is the identity
/// used when reranking is gated OFF — search() with `IdentityReranker` is
/// byte-identical to search() without a reranker (the proven path never regresses).
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityReranker;

impl Reranker for IdentityReranker {
    fn rerank(&self, _query: &str, candidates: Vec<RerankCandidate>) -> Vec<RerankCandidate> {
        candidates
    }
}

/// Deterministic reference reranker: scores each candidate by the fraction of
/// query terms present in its content (a lexical proxy for the learned
/// cross-encoder's semantic relevance). It demonstrably reorders the head — a
/// candidate that is a better *content* match than its retrieval rank suggests is
/// promoted — with the retrieval score as a stable tie-break. Hermetic (no model),
/// so the rerank plumbing is gated by a real assertion, not a model download.
///
/// NOT the shipped reranker — the production cross-encoder replaces the scoring
/// function behind the same trait. This exists to prove the seam and set the eval
/// baseline the learned model must beat.
#[derive(Debug, Clone, Copy, Default)]
pub struct LexicalOverlapReranker;

fn terms(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

impl LexicalOverlapReranker {
    /// Fraction of distinct query terms that appear in `content` (0.0..=1.0).
    fn overlap(query: &str, content: &str) -> f32 {
        let q: std::collections::BTreeSet<String> = terms(query).into_iter().collect();
        if q.is_empty() {
            return 0.0;
        }
        let c: std::collections::BTreeSet<String> = terms(content).into_iter().collect();
        let hit = q.iter().filter(|t| c.contains(*t)).count();
        hit as f32 / q.len() as f32
    }
}

impl Reranker for LexicalOverlapReranker {
    fn rerank(&self, query: &str, mut candidates: Vec<RerankCandidate>) -> Vec<RerankCandidate> {
        candidates.sort_by(|a, b| {
            let sa = Self::overlap(query, &a.content);
            let sb = Self::overlap(query, &b.content);
            // Primary: overlap (rerank signal). Tie-break: retrieval score, then
            // drawer_id — a total, deterministic order (no arbitrary ties).
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.retrieval_score
                        .partial_cmp(&a.retrieval_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.drawer_id.cmp(&b.drawer_id))
        });
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: &str, content: &str, score: f32) -> RerankCandidate {
        RerankCandidate { drawer_id: id.into(), content: content.into(), retrieval_score: score }
    }

    #[test]
    fn identity_preserves_retrieval_order_exactly() {
        let input = vec![cand("a", "x", 0.9), cand("b", "y", 0.8), cand("c", "z", 0.7)];
        let out = IdentityReranker.rerank("anything", input.clone());
        assert_eq!(out, input, "identity reranker must be a no-op (gated-off == proven path)");
    }

    #[test]
    fn overlap_promotes_better_content_match_above_retrieval_rank() {
        // "a" ranks higher on retrieval but is off-topic; "b" is the real content
        // match. The reranker must promote "b" — the whole point of reranking.
        let input = vec![
            cand("a", "the cat sat on the mat", 0.95),
            cand("b", "a lecture on quantum physics and entanglement", 0.80),
        ];
        let out = LexicalOverlapReranker.rerank("quantum physics", input);
        assert_eq!(out[0].drawer_id, "b", "content match must be promoted over retrieval rank");
        assert_eq!(out[1].drawer_id, "a");
    }

    #[test]
    fn overlap_ties_break_on_retrieval_score_then_id_deterministically() {
        // Neither matches the query -> equal overlap (0) -> retrieval score decides,
        // then id. Fully deterministic, no arbitrary ordering.
        let input = vec![
            cand("z", "unrelated one", 0.5),
            cand("a", "unrelated two", 0.9),
            cand("m", "unrelated three", 0.9),
        ];
        let out = LexicalOverlapReranker.rerank("nonmatching query", input);
        // 0.9 scores first (a before m by id), then 0.5.
        assert_eq!(out.iter().map(|c| c.drawer_id.as_str()).collect::<Vec<_>>(), vec!["a", "m", "z"]);
    }

    #[test]
    fn reranker_never_drops_candidates() {
        let input = vec![cand("a", "one", 0.3), cand("b", "two", 0.2), cand("c", "three", 0.1)];
        let n = input.len();
        assert_eq!(LexicalOverlapReranker.rerank("q", input.clone()).len(), n);
        assert_eq!(IdentityReranker.rerank("q", input).len(), n, "reorder never loses a hit");
    }

    #[test]
    fn empty_query_is_stable_not_a_panic() {
        let input = vec![cand("a", "one", 0.3), cand("b", "two", 0.2)];
        let out = LexicalOverlapReranker.rerank("", input.clone());
        // empty query -> overlap 0 for all -> retrieval-score order preserved.
        assert_eq!(out.iter().map(|c| c.drawer_id.clone()).collect::<Vec<_>>(),
                   vec!["a".to_string(), "b".to_string()]);
    }
}
