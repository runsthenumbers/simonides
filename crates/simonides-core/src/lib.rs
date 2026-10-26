//! simonides-core - the small set of types the other crates share: errors, the
//! hybrid logical clock and merge contract used by sync, and the memory-model
//! types.

pub mod error;
pub mod memory;
pub mod sync;

pub use error::{SimonidesError, SimonidesResult};
pub use sync::{
    ApplyOutcome, ConflictEvent, Hlc, PeerClock, SubscriptionId, SyncDelta, SyncSnapshot,
    SyncTable, SyncableStore, WriteEvent, WriteSink,
};