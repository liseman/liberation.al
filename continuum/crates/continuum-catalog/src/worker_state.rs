//! Worker checkpoints as one keyed row per worker.
//!
//! A durable worker's newest snapshot lives in `WORKER_STATE`, keyed by its owner and contract, and a save overwrites
//! that row in place. Before, each save wrote a manifest and a few dozen parts as payloads and a trim purged the old
//! ones minutes later: about 76 MB/h written into `payloads` and `locations` and deleted again, which left the two
//! tables' pages full of holes.
//!
//! The guarantees stay those of the payload checkpoints:
//!
//! - A save names the invalidation revision of each input stream it read. It is refused (Conflict) when any of them
//!   has moved since, checked in the write that stores the row, so state derived from data a purge took is never
//!   kept.
//! - A load returns the row only while every input's revision still equals the stored one.
//! - A purge that advances an input stream's revision deletes the rows of every worker that names that stream, in
//!   the same transaction (`WORKER_STATE_INPUTS` is the stream-to-worker index that finds them). The worker starts
//!   again from its durable cursor.
//! - A row holds at most [`MAX_WORKER_STATE_BYTES`] of state.

use continuum_core::StreamId;

use crate::codec::{In, Out};
use crate::{CatalogError, Result};

/// The most state one worker may keep (its encoded body), well inside one control frame.
pub const MAX_WORKER_STATE_BYTES: usize = 8 << 20;
/// The most input streams one worker's state may name.
pub const MAX_WORKER_STATE_INPUTS: usize = 64;
const MAX_OWNER: usize = 256;
const MAX_CONTRACT: usize = 1024;
const MAX_STREAM_NAME: usize = 256;

/// One durable worker's checkpoint: where its input cursor stood, when it was made, the invalidation revision of each
/// input stream it was built from, and the worker's own encoded state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerState {
    /// The worker (a service name, say).
    pub owner: String,
    /// Which of its checkpoint formats and settings this is: a worker with another contract never reads it.
    pub contract: String,
    /// The input position the state includes.
    pub position: u64,
    /// When the state was made (UTC ns).
    pub created_ns: u64,
    /// Each input stream's name and the invalidation revision the state was read under, sorted and unique.
    pub inputs: Vec<(String, u64)>,
    /// The worker's encoded state.
    pub body: Vec<u8>,
}

/// Refuses an owner, contract, or input list out of bounds (Invalid).
pub(crate) fn check_key(owner: &str, contract: &str) -> Result<()> {
    if owner.is_empty() || owner.len() > MAX_OWNER || contract.is_empty() || contract.len() > MAX_CONTRACT {
        return Err(CatalogError::Invalid("a worker state's owner and contract are bounded nonempty strings".into()));
    }
    Ok(())
}

/// Refuses a state out of bounds (Invalid): its key, at least one and at most [`MAX_WORKER_STATE_INPUTS`] inputs,
/// sorted and unique, with bounded names, and a body of at most [`MAX_WORKER_STATE_BYTES`].
pub(crate) fn check(state: &WorkerState) -> Result<()> {
    check_key(&state.owner, &state.contract)?;
    if state.inputs.is_empty() || state.inputs.len() > MAX_WORKER_STATE_INPUTS {
        return Err(CatalogError::Invalid(format!(
            "a worker state names 1 to {MAX_WORKER_STATE_INPUTS} input streams"
        )));
    }
    if state.inputs.iter().any(|(name, _)| name.is_empty() || name.len() > MAX_STREAM_NAME)
        || state.inputs.windows(2).any(|w| w[0].0 >= w[1].0)
    {
        return Err(CatalogError::Invalid("a worker state's inputs are sorted, unique, bounded names".into()));
    }
    if state.body.len() > MAX_WORKER_STATE_BYTES {
        return Err(CatalogError::Invalid(format!(
            "a worker state is {} bytes; at most {MAX_WORKER_STATE_BYTES} are kept",
            state.body.len()
        )));
    }
    Ok(())
}

/// A revision at its ceiling is reported as unknown (the catalog's accessor returns None), so nothing may be saved or
/// trusted under it.
pub(crate) fn usable(revision: u64) -> bool {
    revision != u64::MAX
}

/// The `WORKER_STATE` key: owner, then contract, each length-prefixed.
pub(crate) fn key(owner: &str, contract: &str) -> Vec<u8> {
    Out::new().str(owner).str(contract).done()
}

/// The `WORKER_STATE_INPUTS` key: the input stream, then the worker's key.
pub(crate) fn input_key(stream: StreamId, worker: &[u8]) -> Vec<u8> {
    Out::new().u32(stream.0).raw(worker).done()
}

/// The prefix of every `WORKER_STATE_INPUTS` key of one stream.
pub(crate) fn input_prefix(stream: StreamId) -> Vec<u8> {
    Out::new().u32(stream.0).done()
}

/// A stored row: the state and the stream id each input resolved to when it was saved.
pub(crate) struct Row {
    pub(crate) state: WorkerState,
    pub(crate) ids: Vec<StreamId>,
}

pub(crate) fn encode(state: &WorkerState, ids: &[StreamId]) -> Vec<u8> {
    let count = u32::try_from(state.inputs.len()).unwrap_or(u32::MAX);
    let mut o = Out::new().u64(state.position).u64(state.created_ns).u32(count);
    for ((name, revision), id) in state.inputs.iter().zip(ids) {
        o = o.str(name).u32(id.0).u64(*revision);
    }
    o.bytes(&state.body).done()
}

pub(crate) fn decode(owner: &str, contract: &str, value: &[u8]) -> Result<Row> {
    let mut i = In::new(value);
    let (position, created_ns) = (i.u64()?, i.u64()?);
    let count = i.u32()? as usize;
    if count > MAX_WORKER_STATE_INPUTS {
        return Err(CatalogError::Storage("a worker state row names too many inputs".into()));
    }
    let (mut inputs, mut ids) = (Vec::with_capacity(count), Vec::with_capacity(count));
    for _ in 0..count {
        let name = i.string()?;
        ids.push(StreamId(i.u32()?));
        inputs.push((name, i.u64()?));
    }
    let body = i.bytes()?.to_vec();
    if !i.is_empty() {
        return Err(CatalogError::Storage("a worker state row has bytes after its body".into()));
    }
    let state =
        WorkerState { owner: owner.to_owned(), contract: contract.to_owned(), position, created_ns, inputs, body };
    Ok(Row { state, ids })
}

/// The owner and contract a `WORKER_STATE` key names.
pub(crate) fn split_key(key: &[u8]) -> Result<(String, String)> {
    let mut i = In::new(key);
    let (owner, contract) = (i.string()?, i.string()?);
    if !i.is_empty() {
        return Err(CatalogError::Storage("a worker state key has bytes after its contract".into()));
    }
    Ok((owner, contract))
}
