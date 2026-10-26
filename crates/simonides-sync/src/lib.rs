//! simonides-sync — CRDT-based cross-device sync layer.
//!
//! Implements the per-table merge semantics. `simonides-sync` consumes the
//! `SyncableStore` trait from `simonides-core::sync`; the `simonides` store
//! implements it.
//!
//! Per-table semantics:
//! - Drawers + entries (verbatim content): Automerge-document-per-record with
//!   sibling-on-divergence detection. See `drawer_merge`.
//! - Palace structure (wings, rooms, closets): LWW-element-set with tombstones.
//!   See `palace_merge`.
//! - Tunnels: OR-set with `(from, to, relation)` triple-key deduplication.
//!   See `tunnel_merge`.
//! - KG entities + triples: bitemporal-aware custom merge preserving divergent
//!   timelines. See `kg_merge`.
//! - Embeddings: NOT synced. Each device regenerates from synced drawer content.
//!
//! Maturity: every table's merge rules are implemented and unit-tested, the store
//! implements `SyncableStore`, and two stores converge in an in-process test. Not
//! yet done: `produce_delta` returns a full snapshot rather than an incremental
//! delta, and this repository contains no network transport. Syncing real devices
//! has not been proven end-to-end, so treat sync as experimental.

pub mod delta;
pub mod drawer_merge;
pub mod palace_merge;
pub mod tunnel_merge;
pub mod kg_merge;
pub mod error;

pub use error::{SyncError, SyncResult};

use simonides_core::sync::{ApplyOutcome, ConflictEvent, SyncTable};

// Re-export delta types at crate root for callers that use simonides_sync::delta::*.
pub use delta::{
    decode_delta, encode_delta, dispatch_delta, DrawerDeltaItem, DrawerLocalState, DrawerWrite,
    LocalState, PalaceDelta, SiblingDrawer,
};

/// Combines multiple per-table outcomes into a single `ApplyOutcome` for the
/// top-level `apply_remote_delta` return path. Conflicts dominate divergence;
/// divergence dominates clean.
pub fn aggregate_outcomes(outcomes: Vec<ApplyOutcome>) -> ApplyOutcome {
    let mut conflicts: Vec<ConflictEvent> = Vec::new();
    let mut divergence: Option<(String, String, String)> = None;

    for outcome in outcomes {
        match outcome {
            ApplyOutcome::Clean => {}
            ApplyOutcome::DivergencePreserved {
                original_drawer_id,
                sibling_drawer_id,
                tunnel_id,
            } => {
                if divergence.is_none() {
                    divergence = Some((original_drawer_id, sibling_drawer_id, tunnel_id));
                }
            }
            ApplyOutcome::ConflictSurfaced { conflicts: mut c } => {
                conflicts.append(&mut c);
            }
        }
    }

    if !conflicts.is_empty() {
        ApplyOutcome::ConflictSurfaced { conflicts }
    } else if let Some((o, s, t)) = divergence {
        ApplyOutcome::DivergencePreserved {
            original_drawer_id: o,
            sibling_drawer_id: s,
            tunnel_id: t,
        }
    } else {
        ApplyOutcome::Clean
    }
}

/// Helper enum used by per-table modules to tag intermediate merge results.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum TableTag {
    Drawer,
    Entry,
    Wing,
    Room,
    Closet,
    Tunnel,
    KgEntity,
    KgTriple,
}

impl From<TableTag> for SyncTable {
    fn from(t: TableTag) -> Self {
        match t {
            TableTag::Drawer => SyncTable::Drawer,
            TableTag::Entry => SyncTable::Entry,
            TableTag::Wing => SyncTable::Wing,
            TableTag::Room => SyncTable::Room,
            TableTag::Closet => SyncTable::Closet,
            TableTag::Tunnel => SyncTable::Tunnel,
            TableTag::KgEntity => SyncTable::KgEntity,
            TableTag::KgTriple => SyncTable::KgTriple,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simonides_core::sync::Hlc;

    #[test]
    fn aggregate_empty_is_clean() {
        let result = aggregate_outcomes(vec![]);
        assert!(matches!(result, ApplyOutcome::Clean));
    }

    #[test]
    fn aggregate_only_clean_is_clean() {
        let result = aggregate_outcomes(vec![ApplyOutcome::Clean, ApplyOutcome::Clean]);
        assert!(matches!(result, ApplyOutcome::Clean));
    }

    #[test]
    fn aggregate_conflict_dominates_clean() {
        let event = ConflictEvent::DrawerDeleteVsEdit {
            drawer_id: "d1".to_string(),
            edit_hlc: Hlc { wall_time: 100, logical: 0, node_id: [0u8; 16] },
            delete_hlc: Hlc { wall_time: 110, logical: 0, node_id: [1u8; 16] },
            ghost_preserved: true,
        };
        let result = aggregate_outcomes(vec![
            ApplyOutcome::Clean,
            ApplyOutcome::ConflictSurfaced { conflicts: vec![event] },
        ]);
        match result {
            ApplyOutcome::ConflictSurfaced { conflicts } => assert_eq!(conflicts.len(), 1),
            _ => panic!("conflict should dominate clean"),
        }
    }

    #[test]
    fn aggregate_divergence_dominates_clean() {
        let result = aggregate_outcomes(vec![
            ApplyOutcome::Clean,
            ApplyOutcome::DivergencePreserved {
                original_drawer_id: "o".into(),
                sibling_drawer_id: "s".into(),
                tunnel_id: "t".into(),
            },
        ]);
        assert!(matches!(result, ApplyOutcome::DivergencePreserved { .. }));
    }
}
