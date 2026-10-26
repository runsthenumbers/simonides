//! Canonical error type for the engine.
//!
//! Errors that leave the engine are mapped to a small set of variants — the
//! caller maps each variant to a user-facing message. Platform
//! errors arrive as `Platform(code, details)`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SimonidesError {
    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("permission denied: {0}")]
    PermissionDenied(String),

    #[error("cancelled")]
    Cancelled,

    #[error("unavailable: {0}")]
    Unavailable(String),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("transport error: {0}")]
    Transport(String),

    #[error("crypto error: {0}")]
    Crypto(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("platform shim error [{code}]: {details}")]
    Platform { code: String, details: String },

    #[error("internal: {0}")]
    Internal(String),
}

pub type SimonidesResult<T> = Result<T, SimonidesError>;
