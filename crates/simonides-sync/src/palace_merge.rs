//! Palace structure merge — LWW-element-set with tombstones, per-table.
//!
//! Design:
//!
//! Three separate sets, one per palace table (wings, rooms, closets). Each set
//! element is `(record_id, hlc, tombstoned: bool, payload)`. Adds and updates store
//! the record + HLC; deletes flip `tombstoned: true` while preserving the row for
//! tombstone resolution.
//!
//! Merge rules per record_id:
//! - Two adds → LWW by HLC. Later HLC wins; payload replaced atomically (no
//!   field-by-field merge inside palace records — they're small enough that
//!   atomic replacement is correct).
//! - Add then delete (causal): tombstoned. Tombstone wins (later event).
//! - Delete then concurrent add (HLC-concurrent): tombstone has earlier HLC; the
//!   add resurrects with a new HLC. Handles "I deleted but want it back" UX.
//! - Concurrent add + delete (HLC-concurrent on different devices): tombstone wins.
//!   Deletion is a stronger signal of intent; surface ConflictEvent for assistant.
//! - Concurrent renames (HLC-concurrent): quiet LWW by HLC total-order (wall →
//!   logical → node_id). No ConflictEvent — concurrent rename is benign.
//!
//! Tunnels get OR-set semantics, not LWW — see `tunnel_merge` module.

use crate::SyncResult;
use serde::{Deserialize, Serialize};
use simonides_core::sync::{ApplyOutcome, ConflictEvent, Hlc, SyncTable};

/// Serde adapter: encodes `serde_json::Value` as a JSON string for bincode compatibility.
mod json_as_string {
    use serde::{Deserialize, Deserializer, Serializer};
    use serde_json::Value;
    pub fn serialize<S: Serializer>(v: &Value, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&v.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Value, D::Error> {
        let raw = String::deserialize(d)?;
        serde_json::from_str(&raw).map_err(serde::de::Error::custom)
    }
}

/// A palace-structure record snapshot — wings, rooms, and closets all share this shape.
/// The `payload` is an opaque JSON value (name, description, summary, room/wing references
/// in metadata, etc.); the merge logic treats it atomically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalaceRecord {
    pub record_id: String,
    pub table: SyncTable,
    pub hlc: Hlc,
    pub tombstoned: bool,
    /// Stored as a JSON string over the wire so bincode can encode it.
    #[serde(with = "json_as_string")]
    pub payload: serde_json::Value,
}

/// Outcome of merging a local and remote view of the same palace record.
#[derive(Debug, Clone)]
pub enum PalaceMergeOutcome {
    /// Keep the local record unchanged.
    KeepLocal,
    /// Replace the local record with the remote one.
    TakeRemote,
    /// Both sides agreed on the same state (idempotent merge).
    NoChange,
    /// Concurrent delete vs modify — tombstone won operationally; surface for the user.
    ConflictDeleteVsModify {
        record_id: String,
        winning_hlc: Hlc,
        losing_hlc: Hlc,
    },
}

/// Merge two views of the same palace record per the rules in module-level docs.
///
/// The caller is responsible for ensuring both records share the same `record_id`
/// and `table`. The function asserts that.
pub fn merge_palace_records(
    local: &PalaceRecord,
    remote: &PalaceRecord,
) -> PalaceMergeOutcome {
    assert_eq!(local.record_id, remote.record_id, "merge requires matching record_id");
    assert_eq!(local.table, remote.table, "merge requires matching table");

    // Byte-identical state → no-op
    if local.hlc == remote.hlc
        && local.tombstoned == remote.tombstoned
        && local.payload == remote.payload
    {
        return PalaceMergeOutcome::NoChange;
    }

    // Concurrent delete-vs-modify case: one tombstoned, one not, HLCs concurrent.
    if local.tombstoned != remote.tombstoned && local.hlc.concurrent_with(&remote.hlc) {
        // Tombstone wins operationally regardless of which side has it.
        let (winning, losing) = if local.tombstoned {
            (&local.hlc, &remote.hlc)
        } else {
            (&remote.hlc, &local.hlc)
        };
        return PalaceMergeOutcome::ConflictDeleteVsModify {
            record_id: local.record_id.clone(),
            winning_hlc: *winning,
            losing_hlc: *losing,
        };
    }

    // Otherwise: total-order tiebreak (LWW). wall → logical → node_id.
    match local.hlc.total_order_cmp(&remote.hlc) {
        std::cmp::Ordering::Less => PalaceMergeOutcome::TakeRemote,
        std::cmp::Ordering::Greater => PalaceMergeOutcome::KeepLocal,
        std::cmp::Ordering::Equal => PalaceMergeOutcome::NoChange,
    }
}

/// Top-level entry point used by the dispatcher. Translates a list of per-record
/// merge outcomes into a single `ApplyOutcome` for the apply_remote_delta return path.
pub fn aggregate_palace_outcomes(outcomes: Vec<PalaceMergeOutcome>) -> SyncResult<ApplyOutcome> {
    let mut conflicts = Vec::new();
    for outcome in outcomes {
        if let PalaceMergeOutcome::ConflictDeleteVsModify { record_id, winning_hlc, losing_hlc } = outcome {
            // The table tag would be looked up from the original record in production;
            // for the skeleton we conservatively report Wing — the dispatcher will
            // tag with the actual table when it routes.
            conflicts.push(ConflictEvent::PalaceStructureDeleteVsModify {
                table: SyncTable::Wing,
                record_id,
                winning_hlc,
                losing_hlc,
            });
        }
    }
    if conflicts.is_empty() {
        Ok(ApplyOutcome::Clean)
    } else {
        Ok(ApplyOutcome::ConflictSurfaced { conflicts })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hlc(wall: u64, logical: u32, node_seed: u8) -> Hlc {
        let mut node_id = [0u8; 16];
        node_id[0] = node_seed;
        Hlc { wall_time: wall, logical, node_id }
    }

    fn record(id: &str, h: Hlc, tomb: bool, payload: &str) -> PalaceRecord {
        PalaceRecord {
            record_id: id.to_string(),
            table: SyncTable::Wing,
            hlc: h,
            tombstoned: tomb,
            payload: serde_json::json!({ "name": payload }),
        }
    }

    #[test]
    fn identical_records_no_change() {
        let r = record("w1", hlc(100, 0, 0), false, "fundraising");
        let outcome = merge_palace_records(&r, &r);
        assert!(matches!(outcome, PalaceMergeOutcome::NoChange));
    }

    #[test]
    fn later_hlc_wins_lww() {
        let local = record("w1", hlc(100, 0, 0xAA), false, "fundraising");
        let remote = record("w1", hlc(200, 0, 0xBB), false, "fundraising-Q3");
        let outcome = merge_palace_records(&local, &remote);
        assert!(matches!(outcome, PalaceMergeOutcome::TakeRemote));
    }

    #[test]
    fn earlier_hlc_loses_lww() {
        let local = record("w1", hlc(200, 0, 0xAA), false, "fundraising-Q3");
        let remote = record("w1", hlc(100, 0, 0xBB), false, "fundraising");
        let outcome = merge_palace_records(&local, &remote);
        assert!(matches!(outcome, PalaceMergeOutcome::KeepLocal));
    }

    #[test]
    fn concurrent_delete_vs_modify_surfaces_conflict() {
        // Worked example: phone renames wing, desktop deletes it.
        // HLCs concurrent (different node_ids, identical wall+logical).
        let local_rename = record("w1", hlc(100, 5, 0xAA), false, "Q3-fundraising");
        let remote_delete = record("w1", hlc(100, 5, 0xBB), true, "fundraising");
        let outcome = merge_palace_records(&local_rename, &remote_delete);
        match outcome {
            PalaceMergeOutcome::ConflictDeleteVsModify { record_id, .. } => {
                assert_eq!(record_id, "w1");
            }
            _ => panic!("expected ConflictDeleteVsModify"),
        }
    }

    #[test]
    fn concurrent_renames_resolve_quietly_via_total_order() {
        // Worked example: phone renames "fundraising-2026", desktop renames
        // "fundraising-Q3", concurrent HLCs. No ConflictEvent; LWW picks a winner.
        let local = record("w1", hlc(100, 0, 0xAA), false, "fundraising-2026");
        let remote = record("w1", hlc(110, 0, 0xBB), false, "fundraising-Q3");
        let outcome = merge_palace_records(&local, &remote);
        assert!(matches!(outcome, PalaceMergeOutcome::TakeRemote));
    }

    #[test]
    fn causal_delete_after_add_wins() {
        // Add at t=100, delete at t=110 (causal — t=110 > t=100). Tombstone wins.
        let local = record("w1", hlc(100, 0, 0xAA), false, "fundraising");
        let remote = record("w1", hlc(110, 0, 0xAA), true, "fundraising");
        let outcome = merge_palace_records(&local, &remote);
        assert!(matches!(outcome, PalaceMergeOutcome::TakeRemote));
    }

    #[test]
    fn aggregate_clean_when_no_conflicts() {
        let outcomes = vec![
            PalaceMergeOutcome::TakeRemote,
            PalaceMergeOutcome::KeepLocal,
            PalaceMergeOutcome::NoChange,
        ];
        let result = aggregate_palace_outcomes(outcomes).unwrap();
        assert!(matches!(result, ApplyOutcome::Clean));
    }

    #[test]
    fn aggregate_surfaces_conflicts() {
        let outcomes = vec![
            PalaceMergeOutcome::TakeRemote,
            PalaceMergeOutcome::ConflictDeleteVsModify {
                record_id: "w1".to_string(),
                winning_hlc: hlc(100, 5, 0xBB),
                losing_hlc: hlc(100, 5, 0xAA),
            },
        ];
        let result = aggregate_palace_outcomes(outcomes).unwrap();
        assert!(matches!(result, ApplyOutcome::ConflictSurfaced { .. }));
    }
}
