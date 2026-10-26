//! Drawer + entry merge — Automerge-document-per-record with sibling-on-divergence detection.
//!
//! Design:
//!
//! - Each drawer is wrapped in one Automerge document.
//! - On apply_remote_delta, we run Automerge's native merge. Automerge always succeeds
//!   syntactically, but its text-merge can produce content NEITHER side authored when
//!   both edited the same region concurrently. That violates the store's verbatim guarantee.
//! - Divergence detection: after merge, compare merged content to (a) the local
//!   pre-merge content and (b) the remote pre-merge content. If neither matches,
//!   we have a synthesized merge that no human authored.
//! - On divergence: materialize the synthesized version as a SIBLING drawer with
//!   `source = Manual`, `added_by = "simonides-sync (merge)"`, same wing/room/closet
//!   metadata as the original. Create a `divergence_from` tunnel from sibling →
//!   original. The original keeps the local pre-merge content. Both persist.
//! - Drawer delete vs concurrent edit: edit wins, ghost preserved, ConflictEvent
//!   surfaced — losing content is worse than carrying a ghost until the user
//!   confirms the delete.
//!
//! This module currently contains the merge-shape logic and the divergence-detection
//! algorithm against simulated string state. Wiring through to real Automerge documents
//! lands during per-table implementation (post-lancedb integration per the §11 work plan).

use crate::SyncResult;
use simonides_core::sync::{ApplyOutcome, ConflictEvent, Hlc};

/// Pre-merge + post-merge content snapshot used by the divergence detector.
#[derive(Debug, Clone)]
pub struct DrawerMergeInputs<'a> {
    pub drawer_id: &'a str,
    pub local_pre: &'a str,
    pub remote_pre: &'a str,
    pub merged: &'a str,
}

/// Action the divergence detector recommends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DivergenceAction {
    /// Merged content matches one of the sides — no divergence, just take it.
    Converged,
    /// Merged content matches neither side — preserve as sibling drawer.
    PreserveAsSibling {
        sibling_id: String,
        tunnel_id: String,
    },
}

/// Run the divergence detection.
///
/// Returns `Converged` if Automerge's merge produced content that exactly equals
/// one of the two pre-merge versions (the common case — one side strictly extends
/// or precedes the other in causal order). Returns `PreserveAsSibling` if the merge
/// synthesized content NEITHER side authored — the store's verbatim guarantee forbids
/// auto-merging in this case.
///
/// Sibling and tunnel IDs are generated deterministically from the merge HLC so
/// concurrent divergence-detection on two devices collapses to the same sibling
/// after sync (see the worked example in the tests).
pub fn detect_divergence(inputs: &DrawerMergeInputs<'_>, merge_hlc: &Hlc) -> DivergenceAction {
    if inputs.merged == inputs.local_pre || inputs.merged == inputs.remote_pre {
        DivergenceAction::Converged
    } else {
        DivergenceAction::PreserveAsSibling {
            sibling_id: derive_sibling_id(inputs.drawer_id, merge_hlc),
            tunnel_id: derive_divergence_tunnel_id(inputs.drawer_id, merge_hlc),
        }
    }
}

/// Run drawer delete-vs-edit conflict resolution per the design above
/// in the deeper-memo ack ("losing content is worse than carrying a ghost").
///
/// Edit always wins; ghost preserved; `ConflictEvent::DrawerDeleteVsEdit` surfaced
/// for assistant UX.
pub fn resolve_delete_vs_edit(
    drawer_id: &str,
    edit_hlc: Hlc,
    delete_hlc: Hlc,
) -> ApplyOutcome {
    ApplyOutcome::ConflictSurfaced {
        conflicts: vec![ConflictEvent::DrawerDeleteVsEdit {
            drawer_id: drawer_id.to_string(),
            edit_hlc,
            delete_hlc,
            ghost_preserved: true,
        }],
    }
}

/// Apply a drawer-level merge. Returns the appropriate `ApplyOutcome`.
///
/// SCAFFOLD: production version reads/writes actual Automerge documents. This shape
/// shows the post-merge decision tree.
pub fn apply_drawer_merge(
    inputs: &DrawerMergeInputs<'_>,
    merge_hlc: Hlc,
) -> SyncResult<ApplyOutcome> {
    match detect_divergence(inputs, &merge_hlc) {
        DivergenceAction::Converged => Ok(ApplyOutcome::Clean),
        DivergenceAction::PreserveAsSibling { sibling_id, tunnel_id } => {
            Ok(ApplyOutcome::DivergencePreserved {
                original_drawer_id: inputs.drawer_id.to_string(),
                sibling_drawer_id: sibling_id,
                tunnel_id,
            })
        }
    }
}

// ─── ID derivation helpers ──────────────────────────────────────────────────

/// Deterministic sibling-id derivation: same drawer + same merge HLC produces the
/// same sibling-id on every device that runs the same merge. This is what lets
/// the worked example in the tests converge — both devices independently
/// detect divergence, both generate the same sibling ID, the two sibling-drawer
/// creations collapse to one drawer in Automerge.
fn derive_sibling_id(drawer_id: &str, merge_hlc: &Hlc) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    drawer_id.hash(&mut hasher);
    merge_hlc.wall_time.hash(&mut hasher);
    merge_hlc.logical.hash(&mut hasher);
    // NOTE: deliberately NOT hashing node_id — same merge result on different
    // devices should collapse, so node_id must not contribute to the sibling id.
    "sibling".hash(&mut hasher);
    format!("drawer-sibling-{:x}", hasher.finish())
}

fn derive_divergence_tunnel_id(drawer_id: &str, merge_hlc: &Hlc) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    drawer_id.hash(&mut hasher);
    merge_hlc.wall_time.hash(&mut hasher);
    merge_hlc.logical.hash(&mut hasher);
    "divergence-tunnel".hash(&mut hasher);
    format!("tunnel-divergence-{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hlc(wall: u64, logical: u32) -> Hlc {
        Hlc { wall_time: wall, logical, node_id: [0u8; 16] }
    }

    #[test]
    fn converged_when_merged_equals_local() {
        let inputs = DrawerMergeInputs {
            drawer_id: "d1",
            local_pre: "lunch at noon",
            remote_pre: "lunch at noon",
            merged: "lunch at noon",
        };
        assert_eq!(detect_divergence(&inputs, &hlc(100, 0)), DivergenceAction::Converged);
    }

    #[test]
    fn converged_when_merged_equals_remote_extension() {
        // Remote extended; local was the prefix. Automerge's merge takes the longer
        // version since the local content is a strict prefix. That's a "converged"
        // case — the user is reading what remote authored.
        let inputs = DrawerMergeInputs {
            drawer_id: "d1",
            local_pre: "lunch at noon",
            remote_pre: "lunch at noon downtown",
            merged: "lunch at noon downtown",
        };
        assert_eq!(detect_divergence(&inputs, &hlc(100, 0)), DivergenceAction::Converged);
    }

    #[test]
    fn divergence_detected_when_merge_synthesizes() {
        // Worked example: phone says "lunch at 12:30", desktop says
        // "lunch at 1pm", Automerge's text merge synthesizes something like
        // "lunch at 12:130pmpm" that neither device authored. Divergence detected.
        let inputs = DrawerMergeInputs {
            drawer_id: "d1",
            local_pre: "lunch at 12:30",
            remote_pre: "lunch at 1pm",
            merged: "lunch at 12:130pmpm",
        };
        let result = detect_divergence(&inputs, &hlc(200, 0));
        assert!(matches!(result, DivergenceAction::PreserveAsSibling { .. }));
    }

    #[test]
    fn sibling_id_is_deterministic_across_runs() {
        let h = hlc(200, 0);
        let id1 = derive_sibling_id("d1", &h);
        let id2 = derive_sibling_id("d1", &h);
        assert_eq!(id1, id2);
    }

    #[test]
    fn sibling_id_collapses_across_devices() {
        // Same merge HLC but different node_ids — both devices should derive the
        // same sibling-id so the two divergence-detection events collapse to one
        // sibling drawer in Automerge per the worked example above.
        let mut h1 = hlc(200, 0);
        h1.node_id[0] = 0xAA;
        let mut h2 = hlc(200, 0);
        h2.node_id[0] = 0xBB;
        assert_eq!(derive_sibling_id("d1", &h1), derive_sibling_id("d1", &h2));
    }

    #[test]
    fn sibling_id_differs_for_different_drawers() {
        let h = hlc(200, 0);
        assert_ne!(derive_sibling_id("d1", &h), derive_sibling_id("d2", &h));
    }

    #[test]
    fn delete_vs_edit_preserves_ghost() {
        let result = resolve_delete_vs_edit("d1", hlc(100, 0), hlc(110, 0));
        match result {
            ApplyOutcome::ConflictSurfaced { conflicts } => {
                assert_eq!(conflicts.len(), 1);
                match &conflicts[0] {
                    ConflictEvent::DrawerDeleteVsEdit { ghost_preserved, drawer_id, .. } => {
                        assert!(ghost_preserved);
                        assert_eq!(drawer_id, "d1");
                    }
                    _ => panic!("wrong conflict variant"),
                }
            }
            _ => panic!("expected ConflictSurfaced"),
        }
    }
}
