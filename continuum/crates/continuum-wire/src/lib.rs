//! Continuum's socket wire format, shared by the broker and the client SDKs.
//!
//! Every message is a `ControlEnvelope` FlatBuffer (schemas/continuum.fbs) sent as one length-prefixed frame
//! ([`frame`]). The broker passes arena file descriptors alongside `Welcome` with SCM_RIGHTS ([`fds`]). Nothing here
//! knows about sessions or requests; that's the broker's and the clients' business.

pub mod fds;
pub mod frame;

use continuum_core::schema::continuum::ControlEnvelope;
use flatbuffers::InvalidFlatbuffer;

pub use fds::{MAX_FDS, read_frame_with_fds, write_frame_with_fds};
pub use frame::{FrameError, MAX_FRAME, read_frame, write_frame};

/// Reads and verifies a `ControlEnvelope`. The schema file's root type is `PayloadHeader`, so envelopes are read
/// with [`flatbuffers::root`] rather than a generated `root_as_*` function.
///
/// # Errors
///
/// An [`InvalidFlatbuffer`] if `bytes` isn't a valid `ControlEnvelope`.
pub fn read_envelope(bytes: &[u8]) -> Result<ControlEnvelope<'_>, InvalidFlatbuffer> {
    flatbuffers::root::<ControlEnvelope>(bytes)
}
