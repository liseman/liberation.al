//! `continuum_py.RingReader`: following a ring of fixed-size records (PCM blocks, IMU samples), synchronously or
//! live from asyncio.
//!
//! The live path is the audio path (voice activity, speech recognition, barge-in), whose target is under 25 ms from
//! capture to reader. An awaited read waits on the ring's sequence word (`SeqWait`: a futex on Linux,
//! `os_sync_wait_on_address` on macOS) on one of tokio's blocking threads, so the event loop keeps running, the GIL
//! is free, and nothing polls: the writer's publish wakes it. A reader that falls behind the ring never slows the
//! writer; it skips to the oldest record still there and reports how many it lost.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use continuum_shm::RingReader;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{IntoPyObjectExt, exceptions::PyValueError};

/// The longest one blocking wait lasts. A publish wakes it at once whatever this is; it only bounds how long a
/// cancelled wait keeps its blocking thread.
const WAIT_SLICE: Duration = Duration::from_millis(50);

/// A record: its sequence number, its bytes, and how many records were skipped just before it.
type Got = (u64, Vec<u8>, u64);

/// A reader of a ring, with its own cursor. Read it with `read()`, wait with `wait()`, or follow it live with
/// `async for seq, record, lost in reader` (or `await reader.next_record(timeout_ms)`).
#[pyclass(name = "RingReader", module = "continuum_py", frozen)]
pub(crate) struct PyRingReader {
    reader: Mutex<RingReader>,
    lost: AtomicU64,
    stride: usize,
}

/// Reads the next record, if there is one, counting what was skipped.
fn try_read(reader: &Mutex<RingReader>, lost: &AtomicU64) -> PyResult<Option<Got>> {
    let mut reader = reader.lock().unwrap_or_else(PoisonError::into_inner);
    let mut buf = vec![0u8; reader.stride()];
    let Some(record) = reader.read(&mut buf).map_err(crate::to_py)? else { return Ok(None) };
    lost.fetch_add(record.lost, Ordering::SeqCst);
    Ok(Some((record.seq, buf, record.lost)))
}

fn record_tuple(py: Python<'_>, (seq, bytes, lost): Got) -> PyResult<Py<PyAny>> {
    (seq, PyBytes::new(py, &bytes), lost).into_py_any(py)
}

impl PyRingReader {
    pub(crate) fn new(reader: RingReader) -> Self {
        let stride = reader.stride();
        Self { reader: Mutex::new(reader), lost: AtomicU64::new(0), stride }
    }

    /// An awaitable that resolves to True once a record this reader hasn't read is published, or False when
    /// `timeout` passes. It only waits (on a blocking thread, no GIL, no polling); it never reads, so cancelling it
    /// can't lose a record. continuum_py._ring builds next_record and the async iterator on it.
    fn ready_awaitable<'py>(&self, py: Python<'py>, timeout: Duration) -> PyResult<Bound<'py, PyAny>> {
        let waiter = self.reader.lock().unwrap_or_else(PoisonError::into_inner).waiter();
        crate::awaitable(py, async move {
            let deadline = continuum_shm::deadline_after(timeout);
            let waiter = Arc::new(waiter);
            loop {
                let left = deadline.saturating_duration_since(Instant::now());
                let slice = left.min(WAIT_SLICE);
                let w = Arc::clone(&waiter);
                let ready = tokio::task::spawn_blocking(move || w.wait(slice))
                    .await
                    .map_err(|e| PyValueError::new_err(format!("the ring wait failed: {e}")))?;
                if ready || left <= slice {
                    return Ok(ready);
                }
            }
        })
    }
}

#[pymethods]
impl PyRingReader {
    /// Bytes per record.
    #[getter]
    fn stride(&self) -> usize {
        self.stride
    }

    /// Records skipped so far because the writer overwrote them before this reader got to them.
    #[getter]
    fn lost(&self) -> u64 {
        self.lost.load(Ordering::SeqCst)
    }

    /// Blocks until a record this reader hasn't read is published, or `timeout_ms` passes, without polling and with
    /// the GIL released. Returns whether one is ready.
    fn wait(&self, py: Python<'_>, timeout_ms: u64) -> bool {
        // Snapshot the position and let go of the reader before releasing the GIL: holding it across the wait would
        // deadlock against a thread that holds the GIL and wants the reader (read() or stride).
        let waiter = self.reader.lock().unwrap_or_else(PoisonError::into_inner).waiter();
        py.detach(|| waiter.wait(Duration::from_millis(timeout_ms)))
    }

    /// The next record as (sequence number, bytes), or None if the reader is caught up. Records are small, so this
    /// copies; big payloads go in slots, whose handles travel through a ring.
    fn read<'py>(&self, py: Python<'py>) -> PyResult<Option<(u64, Bound<'py, PyBytes>)>> {
        Ok(try_read(&self.reader, &self.lost)?.map(|(seq, bytes, _)| (seq, PyBytes::new(py, &bytes))))
    }

    /// The next record as (seq, bytes, lost), or None if the reader is caught up. lost counts the records skipped
    /// just before this one because the reader fell behind.
    fn read_record(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        try_read(&self.reader, &self.lost)?.map(|got| record_tuple(py, got)).transpose()
    }

    /// An awaitable: True once a record this reader hasn't read is published, False after `timeout_ms`. It never
    /// reads, so cancelling it loses nothing. (next_record and `async for` are built on it.)
    fn _ready<'py>(&self, py: Python<'py>, timeout_ms: u64) -> PyResult<Bound<'py, PyAny>> {
        self.ready_awaitable(py, Duration::from_millis(timeout_ms))
    }

    /// Awaits the next record as (seq, bytes, lost), or None if `timeout_ms` passes first (None waits as long as it
    /// takes). Doesn't block the event loop, holds no GIL while it waits, doesn't poll, and loses no record if
    /// cancelled.
    #[pyo3(signature = (timeout_ms = None))]
    fn next_record<'py>(slf: &Bound<'py, Self>, timeout_ms: Option<u64>) -> PyResult<Bound<'py, PyAny>> {
        slf.py().import("continuum_py._ring")?.call_method1("next_record", (slf, timeout_ms))
    }

    /// `async for seq, record, lost in reader`: follows the ring live, forever (a ring has no end).
    fn __aiter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py().import("continuum_py._ring")?.call_method1("follow", (slf,))
    }
}

/// The single writer of a ring (`Client.ring_writer`). `push(data)` appends one record, `data` zero-padded to the
/// stride; a record longer than the stride is a ValueError. Its claim is released when the writer is dropped or its
/// session ends, so the stream's next writer can take the ring.
#[pyclass(name = "RingWriter", module = "continuum_py", frozen)]
pub(crate) struct PyRingWriter {
    writer: Mutex<continuum_shm::RingWriter>,
    stride: usize,
}

impl PyRingWriter {
    pub(crate) fn new(writer: continuum_shm::RingWriter) -> Self {
        let stride = writer.stride();
        Self { writer: Mutex::new(writer), stride }
    }
}

#[pymethods]
impl PyRingWriter {
    /// Bytes per record.
    #[getter]
    fn stride(&self) -> usize {
        self.stride
    }

    /// Appends one record, zero-padded to the stride, overwriting the oldest when the ring is full, and wakes its
    /// readers. Returns its sequence number.
    fn push(&self, data: &[u8]) -> PyResult<u64> {
        if data.len() > self.stride {
            return Err(PyValueError::new_err(format!("{} bytes don't fit a {}-byte record", data.len(), self.stride)));
        }
        let mut record = vec![0u8; self.stride];
        record[..data.len()].copy_from_slice(data);
        self.writer.lock().unwrap_or_else(PoisonError::into_inner).push(&record).map_err(crate::to_py)
    }
}
