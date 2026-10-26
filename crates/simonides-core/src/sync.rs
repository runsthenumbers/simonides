//! Sync — trait contract between the store (implementer) and `simonides-sync` (consumer).
//!
//! Lives in `simonides-core` because both the store and `simonides-sync` depend on it,
//! so the trait must live in a crate both can use without a circular dependency.
//!
//! The `SyncableStore` trait is the seal between the row-shaped storage layer (SQLCipher,
//! owned by the store) and the CRDT layer (Automerge documents + per-table merge
//! semantics, owned by `simonides-sync`). Implementations of this trait expose enough surface
//! for `simonides-sync` to: subscribe to local writes, produce deltas for peers, apply deltas
//! from peers, snapshot for initial pairing, and read the current HLC for ordering.
//!
//! Per-table merge semantics are owned by `simonides-sync` and not visible at this interface.
//! `SyncDelta`, `SyncSnapshot`, and `PeerClock` are opaque from the store's perspective —
//! they are bytes produced and consumed by the merge logic on the sync side.

use crate::SimonidesResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

// ─── Hybrid Logical Clock ────────────────────────────────────────────────────

/// Hybrid Logical Clock — the per-event timestamp used across every syncable table.
///
/// Composed of (wall_time, logical, node_id) where wall_time is milliseconds-since-epoch
/// in UTC, logical is a monotonic counter that distinguishes events at the same wall_time
/// on the same device, and node_id is a per-device random identifier generated at first
/// launch.
///
/// Why HLC over alternatives: pure wall-clock drops events from skewed-clock devices;
/// pure Lamport clocks lose the human-time approximation the UX needs; vector clocks grow
/// O(devices) in metadata per event which becomes painful at scale. HLC gives wall-clock
/// approximation + monotonic-ordering correctness + O(1) metadata.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct Hlc {
    /// Milliseconds since Unix epoch, UTC.
    pub wall_time: u64,
    /// Monotonic counter for events sharing the same wall_time on the same node.
    pub logical: u32,
    /// 16-byte per-device random identifier.
    pub node_id: [u8; 16],
}

impl Hlc {
    /// Strict happens-before test:
    ///
    /// `A.happens_before(&B)` iff `A.wall_time < B.wall_time`
    /// OR (`A.wall_time == B.wall_time` AND `A.logical < B.logical`).
    ///
    /// Note: this test deliberately ignores `node_id` for the happens-before relation.
    /// Two events on different nodes with identical (wall_time, logical) are CONCURRENT,
    /// not ordered — even though node_id provides a deterministic tiebreak for LWW
    /// resolution downstream.
    pub fn happens_before(&self, other: &Self) -> bool {
        if self.wall_time < other.wall_time {
            return true;
        }
        if self.wall_time == other.wall_time && self.logical < other.logical {
            return true;
        }
        false
    }

    /// Two HLCs are concurrent iff neither happens-before the other.
    pub fn concurrent_with(&self, other: &Self) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }

    /// Total-order tiebreak used by LWW resolution: wall_time → logical → node_id (lexicographic).
    /// Unlike `happens_before`, this never returns Equal except for byte-identical HLCs.
    pub fn total_order_cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match self.wall_time.cmp(&other.wall_time) {
            Ordering::Equal => match self.logical.cmp(&other.logical) {
                Ordering::Equal => self.node_id.cmp(&other.node_id),
                ord => ord,
            },
            ord => ord,
        }
    }
}

impl fmt::Display for Hlc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "hlc({},{},{:02x}{:02x}{:02x}{:02x})",
            self.wall_time, self.logical, self.node_id[0], self.node_id[1], self.node_id[2], self.node_id[3]
        )
    }
}

// ─── Per-event records ───────────────────────────────────────────────────────

/// Identifies which logical table an event came from / applies to.
///
/// Stable serialization names so deltas remain interpretable across crate-version drift.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SyncTable {
    Drawer,
    Entry,
    Wing,
    Room,
    Closet,
    Tunnel,
    KgEntity,
    KgTriple,
}

/// Local write event — what the store emits to `simonides-sync` over the `WriteSink`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteEvent {
    pub table: SyncTable,
    pub record_id: String,
    pub hlc: Hlc,
}

/// Subscription token returned by `subscribe_writes`; used to unsubscribe.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SubscriptionId(pub u64);

/// Callback contract for receiving local write events.
///
/// `simonides-sync` implements this; the store calls `on_write` when a record changes.
/// Implementations must be cheap — the sink runs synchronously in the write path.
/// Heavy work (delta computation, peer broadcast) belongs in a background task that
/// `simonides-sync` spawns after enqueueing the event.
pub trait WriteSink: Send + Sync {
    fn on_write(&self, event: WriteEvent);
}

// ─── Opaque CRDT payloads ────────────────────────────────────────────────────

/// A delta of changes to apply to a `SyncableStore`. Opaque from the store's
/// perspective — produced and consumed by `simonides-sync`'s per-table merge logic
/// (Automerge documents under the hood for drawers + entries + KG triples;
/// LWW-element-set / OR-set encodings for palace structure / tunnels).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncDelta {
    /// Encoded change payload — format is internal to `simonides-sync`.
    pub payload: Vec<u8>,
    /// Source peer's HLC at delta production time.
    pub source_hlc: Hlc,
}

/// Full corpus snapshot for initial peer pairing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncSnapshot {
    /// Encoded snapshot payload — format is internal to `simonides-sync`.
    pub payload: Vec<u8>,
    /// HLC at snapshot time.
    pub snapshot_hlc: Hlc,
}

/// Per-peer clock used to request incremental deltas ("send me everything since this HLC").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerClock {
    pub peer_node_id: [u8; 16],
    pub last_seen_hlc: Hlc,
}

// ─── Apply outcomes + conflict surfacing ─────────────────────────────────────

/// What happened when a remote delta was applied to the local store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApplyOutcome {
    /// The merge applied without any divergence or conflict surfacing.
    Clean,

    /// A drawer-merge produced text that neither side authored; we materialized
    /// the merged content as a sibling drawer and recorded a `divergence_from`
    /// tunnel from sibling → original. Per locked decision #9 (verbatim guarantee
    /// under concurrent drawer edits).
    DivergencePreserved {
        original_drawer_id: String,
        sibling_drawer_id: String,
        tunnel_id: String,
    },

    /// One or more conflicts surfaced that the assistant should resolve with the user.
    /// Operational state is consistent (the merge picked a safe default); the conflicts
    /// are emitted so the UI can ask the user to confirm.
    ConflictSurfaced { conflicts: Vec<ConflictEvent> },
}

/// A conflict that the assistant-side UX should surface to the user.
///
/// The merge always picks an operational default (per per-table semantics), so the
/// store stays consistent without user input. These events let the UI surface "two
/// devices disagreed about X — please confirm Y" prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictEvent {
    /// One device modified (rename, payload update) a palace structure record while
    /// another device tombstoned it. Tombstone wins per LWW-with-deletion-precedence;
    /// surface so the user can restore if needed.
    PalaceStructureDeleteVsModify {
        table: SyncTable,
        record_id: String,
        winning_hlc: Hlc,
        losing_hlc: Hlc,
    },

    /// Two devices set different values for the same property key on the same KG entity.
    /// LWW picks one operationally; surface the other so the user can override.
    KgEntityPropertyDisagreement {
        entity_id: String,
        property_key: String,
        local_value: serde_json::Value,
        remote_value: serde_json::Value,
        winning_hlc: Hlc,
    },

    /// Two devices invalidated the same KG triple at different wall-clock end-dates.
    /// `min(valid_to)` is taken as the operational answer (conservative — "no later
    /// than" both claims); the assistant should ask the user which end-date is correct.
    KgTripleConcurrentInvalidation {
        triple_subject: String,
        triple_predicate: String,
        triple_object: String,
        local_valid_to: DateTime<Utc>,
        remote_valid_to: DateTime<Utc>,
        operational_choice: DateTime<Utc>,
    },

    /// One device deleted a drawer while another edited it. Edit wins (ghost preserved):
    /// losing content is worse than carrying a ghost. Surface so the user can confirm.
    DrawerDeleteVsEdit {
        drawer_id: String,
        edit_hlc: Hlc,
        delete_hlc: Hlc,
        ghost_preserved: bool,
    },
}

// ─── The trait itself ────────────────────────────────────────────────────────

/// Trait the storage layer (the store's `PalaceStore`) implements to participate
/// in cross-device sync. `simonides-sync` consumes this trait and never reaches into
/// `MemoryStore`'s internals.
///
/// `Send + Sync` because `simonides-sync`'s background tasks may invoke methods concurrently.
pub trait SyncableStore: Send + Sync {
    /// Register a sink to receive write events. Returns a `SubscriptionId` for unsubscribe.
    ///
    /// Implementations should call `sink.on_write(event)` synchronously on every successful
    /// mutation to a syncable table. Multiple subscribers are supported.
    fn subscribe_writes(&self, sink: Arc<dyn WriteSink>) -> SimonidesResult<SubscriptionId>;

    /// Drop a previously-registered sink. After this call, the sink will no longer receive events.
    fn unsubscribe(&self, id: SubscriptionId) -> SimonidesResult<()>;

    /// Apply a peer's delta to the local store. The returned `ApplyOutcome` distinguishes
    /// clean merges from divergence-preserved-as-sibling cases from cases that produced
    /// `ConflictEvent`s for the assistant to surface.
    ///
    /// Implementations are responsible for atomicity — either the entire delta lands or
    /// no part of it does. Partial application is not allowed.
    fn apply_remote_delta(&self, delta: SyncDelta) -> SimonidesResult<ApplyOutcome>;

    /// Produce a delta covering everything that has changed locally since `since.last_seen_hlc`
    /// from the perspective of `since.peer_node_id`.
    ///
    /// May return a delta that is empty (no changes since last pairing) — sync layer
    /// distinguishes empty-deltas from no-op cases for keepalive purposes.
    fn produce_delta(&self, since: &PeerClock) -> SimonidesResult<SyncDelta>;

    /// Full corpus snapshot, used during initial pairing. Typically much larger than a
    /// delta; transferred once per peer-pair and then kept in sync via deltas afterward.
    fn snapshot(&self) -> SimonidesResult<SyncSnapshot>;

    /// The store's current HLC — used by `simonides-sync` for delta-production correctness
    /// and to stamp outgoing events. Implementations should return a fresh HLC reading,
    /// not a cached value.
    fn current_hlc(&self) -> Hlc;
}

// ─── Unit tests for the leaf logic (HLC) ─────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn hlc(wall: u64, logical: u32, node_seed: u8) -> Hlc {
        let mut node_id = [0u8; 16];
        node_id[0] = node_seed;
        Hlc { wall_time: wall, logical, node_id }
    }

    #[test]
    fn happens_before_strict_wall() {
        let a = hlc(100, 0, 0xAA);
        let b = hlc(200, 0, 0xBB);
        assert!(a.happens_before(&b));
        assert!(!b.happens_before(&a));
    }

    #[test]
    fn happens_before_same_wall_logical_breaks_tie() {
        let a = hlc(100, 5, 0xAA);
        let b = hlc(100, 6, 0xBB);
        assert!(a.happens_before(&b));
        assert!(!b.happens_before(&a));
    }

    #[test]
    fn happens_before_same_wall_same_logical_is_concurrent() {
        // Different node_id, identical (wall, logical) → concurrent per the strict test.
        let a = hlc(100, 5, 0xAA);
        let b = hlc(100, 5, 0xBB);
        assert!(!a.happens_before(&b));
        assert!(!b.happens_before(&a));
        assert!(a.concurrent_with(&b));
        assert!(b.concurrent_with(&a));
    }

    #[test]
    fn happens_before_byte_identical_neither_before() {
        let a = hlc(100, 5, 0xAA);
        let b = hlc(100, 5, 0xAA);
        // happens_before is strict — byte-identical HLCs are neither before nor after.
        assert!(!a.happens_before(&b));
        assert!(!b.happens_before(&a));
        // But they are concurrent (vacuously — they're the same event from the rule's perspective).
        assert!(a.concurrent_with(&b));
    }

    #[test]
    fn total_order_cmp_is_total() {
        let a = hlc(100, 5, 0xAA);
        let b = hlc(100, 5, 0xBB);
        // Different node_id breaks the tie for total ordering even though they're
        // concurrent for happens-before purposes.
        assert_eq!(a.total_order_cmp(&b), std::cmp::Ordering::Less);
        assert_eq!(b.total_order_cmp(&a), std::cmp::Ordering::Greater);
    }

    #[test]
    fn concurrency_is_symmetric() {
        let a = hlc(100, 5, 0xAA);
        let b = hlc(100, 5, 0xBB);
        assert_eq!(a.concurrent_with(&b), b.concurrent_with(&a));
    }

    #[test]
    fn sync_table_round_trips_serde() {
        let original = SyncTable::KgTriple;
        let json = serde_json::to_string(&original).unwrap();
        let decoded: SyncTable = serde_json::from_str(&json).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn apply_outcome_round_trips_serde() {
        let original = ApplyOutcome::DivergencePreserved {
            original_drawer_id: "orig-123".to_string(),
            sibling_drawer_id: "sib-456".to_string(),
            tunnel_id: "tun-789".to_string(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let decoded: ApplyOutcome = serde_json::from_str(&json).unwrap();
        match decoded {
            ApplyOutcome::DivergencePreserved { original_drawer_id, sibling_drawer_id, tunnel_id } => {
                assert_eq!(original_drawer_id, "orig-123");
                assert_eq!(sibling_drawer_id, "sib-456");
                assert_eq!(tunnel_id, "tun-789");
            }
            _ => panic!("wrong variant after round-trip"),
        }
    }
}
