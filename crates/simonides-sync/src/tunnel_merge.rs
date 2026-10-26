//! Tunnel merge — OR-set with `(from, to, relation)` triple-key deduplication.
//!
//! Design:
//!
//! Tunnel uniqueness key: `(from_drawer_id, to_drawer_id, relation)`. Two tunnels
//! with the same triple ARE the same tunnel even with different IDs — they collapse
//! on merge.
//!
//! Merge rules:
//! - Concurrent adds of the same triple → dedupe to one tunnel. Pick the tunnel
//!   with the earlier HLC for ID; confidence is **averaged**
//!   (averaging is honest merge for soft signals; max overstates,
//!   LWW loses info).
//! - Concurrent adds of different triples → both persist.
//! - Add then delete → tombstoned.
//! - Delete then concurrent add (HLC-concurrent): add wins (tunnels are weak
//!   relationships; resurrection is cheap and rarely wrong).
//!
//! `divergence_from` tunnels created by the drawer-merge layer also flow through
//! this OR-set logic; concurrent divergence-detection on two devices emits two
//! divergence_from tunnels with the same `(sibling, original, "divergence_from")`
//! triple-key, which collapse here.

use crate::SyncResult;
use serde::{Deserialize, Serialize};
use simonides_core::sync::{ApplyOutcome, Hlc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelRecord {
    pub id: String,
    pub from_drawer_id: String,
    pub to_drawer_id: String,
    pub relation: String,
    pub confidence: f32,
    pub hlc: Hlc,
    pub tombstoned: bool,
}

impl TunnelRecord {
    /// The triple-key under which two tunnels are considered the same fact.
    pub fn triple_key(&self) -> (String, String, String) {
        (self.from_drawer_id.clone(), self.to_drawer_id.clone(), self.relation.clone())
    }
}

#[derive(Debug, Clone)]
pub enum TunnelMergeOutcome {
    /// The merged result is a new record (deduped + confidence averaged).
    Merged(TunnelRecord),
    /// Records differ on triple-key — both should persist separately.
    Distinct,
    /// Identical inputs — no work.
    NoChange,
}

/// Merge two views of (potentially) the same tunnel. If the triple-keys match,
/// returns a `Merged` record with the earlier-HLC id and averaged confidence.
/// If they differ, returns `Distinct` (caller persists both).
pub fn merge_tunnels(local: &TunnelRecord, remote: &TunnelRecord) -> TunnelMergeOutcome {
    if local.triple_key() != remote.triple_key() {
        return TunnelMergeOutcome::Distinct;
    }

    // Same triple-key. Identical state → no-op.
    if local.id == remote.id
        && (local.confidence - remote.confidence).abs() < f32::EPSILON
        && local.hlc == remote.hlc
        && local.tombstoned == remote.tombstoned
    {
        return TunnelMergeOutcome::NoChange;
    }

    // Tombstone resolution: delete-then-concurrent-add → add wins (locked decision).
    if local.tombstoned != remote.tombstoned {
        // The add side wins regardless of HLC ordering, per the "tunnels are weak;
        // resurrection is cheap" rule.
        let live = if local.tombstoned { remote } else { local };
        return TunnelMergeOutcome::Merged(live.clone());
    }

    // Both live, same triple-key, differ on confidence / hlc / id. Dedupe:
    // - id = earlier-HLC's id (deterministic; both devices arrive at same id)
    // - confidence = average
    // - hlc = max (total-order) — represents "most recent merge"
    let (earlier_id, later_hlc) = match local.hlc.total_order_cmp(&remote.hlc) {
        std::cmp::Ordering::Less => (local.id.clone(), remote.hlc),
        std::cmp::Ordering::Greater => (remote.id.clone(), local.hlc),
        std::cmp::Ordering::Equal => (local.id.clone(), local.hlc),
    };
    let averaged_confidence = (local.confidence + remote.confidence) / 2.0;

    TunnelMergeOutcome::Merged(TunnelRecord {
        id: earlier_id,
        from_drawer_id: local.from_drawer_id.clone(),
        to_drawer_id: local.to_drawer_id.clone(),
        relation: local.relation.clone(),
        confidence: averaged_confidence,
        hlc: later_hlc,
        tombstoned: false,
    })
}

/// Tunnel merges never surface ConflictEvents — same-triple dedupe is silent
/// per §4, and resurrection-of-tombstoned is silent per §4. Always returns Clean.
pub fn aggregate_tunnel_outcomes(_outcomes: Vec<TunnelMergeOutcome>) -> SyncResult<ApplyOutcome> {
    Ok(ApplyOutcome::Clean)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hlc(wall: u64, logical: u32, node_seed: u8) -> Hlc {
        let mut node_id = [0u8; 16];
        node_id[0] = node_seed;
        Hlc { wall_time: wall, logical, node_id }
    }

    fn tunnel(id: &str, from: &str, to: &str, rel: &str, conf: f32, h: Hlc) -> TunnelRecord {
        TunnelRecord {
            id: id.to_string(),
            from_drawer_id: from.to_string(),
            to_drawer_id: to.to_string(),
            relation: rel.to_string(),
            confidence: conf,
            hlc: h,
            tombstoned: false,
        }
    }

    #[test]
    fn distinct_triple_keys_persist_both() {
        let a = tunnel("t1", "dX", "dY", "supersedes", 0.8, hlc(100, 0, 0));
        let b = tunnel("t2", "dX", "dY", "refines", 0.7, hlc(100, 0, 0));
        let outcome = merge_tunnels(&a, &b);
        assert!(matches!(outcome, TunnelMergeOutcome::Distinct));
    }

    #[test]
    fn same_triple_key_dedupes_and_averages_confidence() {
        // Worked example: device A creates supersedes(X→Y, conf=0.8),
        // device B creates supersedes(X→Y, conf=0.9). Same triple-key. Dedupe with
        // earlier-HLC id; confidence averages to 0.85.
        let a = tunnel("t1", "dX", "dY", "supersedes", 0.8, hlc(100, 0, 0xAA));
        let b = tunnel("t2", "dX", "dY", "supersedes", 0.9, hlc(100, 0, 0xBB));
        let outcome = merge_tunnels(&a, &b);
        match outcome {
            TunnelMergeOutcome::Merged(merged) => {
                assert_eq!(merged.id, "t1"); // earlier total-order (node_id 0xAA < 0xBB)
                assert!((merged.confidence - 0.85).abs() < 0.001);
            }
            _ => panic!("expected Merged"),
        }
    }

    #[test]
    fn delete_then_concurrent_add_resurrects() {
        // device A: tombstoned; device B: live. Add wins.
        let mut a = tunnel("t1", "dX", "dY", "supersedes", 0.8, hlc(100, 0, 0xAA));
        a.tombstoned = true;
        let b = tunnel("t2", "dX", "dY", "supersedes", 0.9, hlc(100, 0, 0xBB));
        let outcome = merge_tunnels(&a, &b);
        match outcome {
            TunnelMergeOutcome::Merged(merged) => {
                assert!(!merged.tombstoned);
                assert_eq!(merged.id, "t2"); // live side wins
            }
            _ => panic!("expected Merged"),
        }
    }

    #[test]
    fn identical_inputs_no_change() {
        let a = tunnel("t1", "dX", "dY", "supersedes", 0.8, hlc(100, 0, 0xAA));
        let outcome = merge_tunnels(&a, &a);
        assert!(matches!(outcome, TunnelMergeOutcome::NoChange));
    }

    #[test]
    fn divergence_from_tunnels_collapse() {
        // Worked example: concurrent divergence-detection
        // on two devices produces two divergence_from tunnels for the same
        // (sibling, original) pair. They collapse here.
        let a = tunnel("auto-1", "sibling-abc", "original-xyz", "divergence_from", 1.0, hlc(200, 0, 0xAA));
        let b = tunnel("auto-2", "sibling-abc", "original-xyz", "divergence_from", 1.0, hlc(200, 0, 0xBB));
        let outcome = merge_tunnels(&a, &b);
        assert!(matches!(outcome, TunnelMergeOutcome::Merged(_)));
    }
}
