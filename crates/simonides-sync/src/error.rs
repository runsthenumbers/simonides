//! Sync-specific error types.

use thiserror::Error;

pub type SyncResult<T> = Result<T, SyncError>;

#[derive(Debug, Error)]
pub enum SyncError {
    #[error("automerge error: {0}")]
    Automerge(String),

    #[error("invalid delta payload: {0}")]
    InvalidDelta(String),

    #[error("invalid snapshot payload: {0}")]
    InvalidSnapshot(String),

    #[error("merge conflict could not be reconciled: {0}")]
    UnreconciledConflict(String),

    #[error("HLC integrity error: {0}")]
    HlcIntegrity(String),

    #[error("storage layer error: {0}")]
    Storage(String),

    #[error("internal sync error: {0}")]
    Internal(String),
}

impl From<simonides_core::SimonidesError> for SyncError {
    fn from(e: simonides_core::SimonidesError) -> Self {
        SyncError::Storage(e.to_string())
    }
}
