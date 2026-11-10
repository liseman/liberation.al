//! Python bindings for the Tier 0 arena (`continuum_py`): open an arena, lease a slot, and read it in place as a
//! memoryview, a NumPy array, or a DLPack capsule, with no copies.
//!
//! A [`PyLease`] holds the pin. Every view of it keeps it alive: a memoryview or NumPy array through the buffer
//! protocol (so `release()` refuses while one is open), and a DLPack capsule through its own reference to the lease,
//! dropped by the capsule's deleter. The pin in shared memory is released when the last of those goes.

mod client;
mod h264;
mod promise;
mod recent;
mod ring;
mod search;
mod segments;
mod subscribe;
mod wire;
mod worker;

/// Set by `_shutdown`, which `continuum_py` registers with `atexit`: the interpreter is about to finalize.
static SHUTTING_DOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// An asyncio awaitable for a Rust future, run on the tokio runtime: the future the running event loop makes, resolved
/// with the Rust future's result (or exception).
///
/// Not `pyo3_async_runtimes::tokio::future_into_py`: that attaches a tokio blocking thread to Python to look at the
/// Python future when the Rust side finishes or is cancelled, and when the interpreter finalizes with an awaitable
/// still pending, its cancel path does that after Python is gone (a segfault at exit, in
/// `pyo3_async_runtimes::generic::cancelled`, seen with a client that had requests in flight as its process
/// exited). Here:
///
/// - The result is handed over with `call_soon_threadsafe` from a `Python::try_attach`, never once `_shutdown` has
///   run (atexit) or the interpreter is gone; a pending awaitable at exit just never completes.
/// - Cancelling the Python future aborts the Rust task from a done callback, which Python runs: cancelling never
///   needs a tokio thread to touch Python.
/// - A panic in the Rust future becomes a RuntimeError, as `future_into_py`'s RustPanic was, instead of an awaitable
///   that never completes.
/// - The handoff waits for the GIL on a blocking-pool thread, so a busy Python side never stalls a runtime worker
///   (and the client's own I/O tasks on it).
///
/// One window is left: a handoff that read the flag just before `_shutdown` set it waits for the GIL through the rest
/// of exit. On Python 3.13 and later a thread that wakes up into finalization is parked for good, which is harmless
/// at exit; the package needs 3.14.
pub(crate) fn awaitable<'py, F, T>(py: Python<'py>, fut: F) -> PyResult<Bound<'py, PyAny>>
where
    F: std::future::Future<Output = PyResult<T>> + Send + 'static,
    T: for<'a> IntoPyObject<'a> + Send + 'static,
{
    if SHUTTING_DOWN.load(Ordering::SeqCst) {
        // Made during exit (an atexit handler running after ours): it couldn't complete, so say so now.
        return Err(pyo3::exceptions::PyRuntimeError::new_err("continuum_py is shutting down with the interpreter"));
    }
    let event_loop = py.import("asyncio")?.call_method0("get_running_loop")?;
    let py_fut = event_loop.call_method0("create_future")?;
    let (loop_ref, fut_ref) = (event_loop.unbind(), py_fut.clone().unbind());
    let task = pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
        let out = CatchPanic(Box::pin(fut)).await;
        if SHUTTING_DOWN.load(Ordering::SeqCst) {
            return;
        }
        let _ = tokio::task::spawn_blocking(move || {
            Python::try_attach(|py| -> PyResult<()> {
                if SHUTTING_DOWN.load(Ordering::SeqCst) {
                    return Ok(());
                }
                let (value, failed) = match out.and_then(|v| v.into_py_any(py)) {
                    Ok(v) => (v, false),
                    Err(e) => (e.into_value(py).into_any(), true),
                };
                let resolve = wrap_pyfunction!(_resolve, py)?;
                loop_ref.bind(py).call_method1("call_soon_threadsafe", (resolve, fut_ref.bind(py), value, failed))?;
                Ok(())
            })
        })
        .await;
    });
    py_fut.call_method1("add_done_callback", (AbortOnCancel { task: task.abort_handle() },))?;
    Ok(py_fut)
}

/// A future whose panic becomes a `RuntimeError` result.
struct CatchPanic<F>(std::pin::Pin<Box<F>>);

impl<F, T> std::future::Future for CatchPanic<F>
where
    F: std::future::Future<Output = PyResult<T>>,
{
    type Output = PyResult<T>;

    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.0.as_mut().poll(cx))) {
            Ok(poll) => poll,
            Err(panic) => {
                let what = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "a panic".to_owned());
                std::task::Poll::Ready(Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "continuum_py panicked: {what}"
                ))))
            }
        }
    }
}

/// Settles an awaitable on its event loop (scheduled by `awaitable`), unless it was cancelled meanwhile.
#[pyfunction]
fn _resolve(fut: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>, failed: bool) -> PyResult<()> {
    if fut.call_method0("done")?.is_truthy()? {
        return Ok(());
    }
    fut.call_method1(if failed { "set_exception" } else { "set_result" }, (value,))?;
    Ok(())
}

/// The done callback that stops an awaitable's Rust task when its Python future is cancelled.
#[pyclass(frozen)]
struct AbortOnCancel {
    task: tokio::task::AbortHandle,
}

#[pymethods]
impl AbortOnCancel {
    fn __call__(&self, fut: &Bound<'_, PyAny>) -> PyResult<()> {
        if fut.call_method0("cancelled")?.is_truthy()? {
            self.task.abort();
        }
        Ok(())
    }
}

/// `Python::attach` for a tokio thread after an await: once the interpreter is shutting down (or finalizing), the
/// future stops here for good instead of attaching (see `awaitable`).
pub(crate) async fn attach<R>(f: impl for<'py> FnOnce(Python<'py>) -> R) -> R {
    if !SHUTTING_DOWN.load(Ordering::SeqCst)
        && let Some(r) = Python::try_attach(f)
    {
        return r;
    }
    std::future::pending().await
}

/// Called at interpreter exit (`continuum_py` registers it with `atexit`): from now on no awaitable completes.
#[pyfunction]
fn _shutdown() {
    SHUTTING_DOWN.store(true, Ordering::SeqCst);
}

use std::ffi::c_int;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use continuum_shm::{Arena, ArenaConfig, DEFAULT_LEASE, Error, Lease, ReadFrom, RingSpec, SlabClass, SlotHandle};
use dlpark::ffi::{DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned};
use dlpark::python::producer::{DlpackExporter, ExportRequest, export_dlpack};
use dlpark::{DlpackFlags, Managed, ManagedTensorBase};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyBufferError, PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::{create_exception, ffi};

create_exception!(continuum_py, ContinuumError, PyException, "Base class for Continuum arena errors.");
create_exception!(
    continuum_py,
    StaleError,
    ContinuumError,
    "The slot was reclaimed or reused since the handle was made, or the lease lapsed."
);
create_exception!(continuum_py, NotReadyError, ContinuumError, "The arena exists but isn't initialized yet.");
/// A write larger than this copies with the GIL released.
const DETACH_ABOVE: usize = 64 * 1024;

create_exception!(continuum_py, FullError, ContinuumError, "Every slot in the class is in use.");

pub(crate) fn to_py(err: Error) -> PyErr {
    match err {
        Error::Io(e) => PyErr::from(e),
        Error::BadName(_) | Error::BadConfig(_) | Error::BadLease(_) | Error::NoSuchSlot | Error::NoSuchRing => {
            PyValueError::new_err(err.to_string())
        }
        Error::NotReady => NotReadyError::new_err(err.to_string()),
        Error::Stale | Error::Expired => StaleError::new_err(err.to_string()),
        Error::Full => FullError::new_err(err.to_string()),
        other => ContinuumError::new_err(other.to_string()),
    }
}

fn ttl(ttl_ms: u64) -> Duration {
    Duration::from_millis(ttl_ms)
}

const DEFAULT_TTL_MS: u64 = DEFAULT_LEASE.as_millis() as u64;

/// A Tier 0 shared memory arena.
#[pyclass(name = "Arena", module = "continuum_py", frozen)]
pub(crate) struct PyArena {
    pub(crate) arena: Arena,
}

#[pymethods]
impl PyArena {
    /// Opens an existing arena by name. Raises NotReadyError while its creator is still setting it up.
    #[staticmethod]
    fn open(name: &str) -> PyResult<Self> {
        Ok(Self { arena: Arena::open(name).map_err(to_py)? })
    }

    /// Creates an arena: `classes` is a list of (slot_size, slot_count), `rings` a list of (stride, capacity).
    #[staticmethod]
    #[pyo3(signature = (name, classes, rings = Vec::new(), generation = 0, reclaim_requires_flush = false))]
    fn create(
        name: &str,
        classes: Vec<(usize, u32)>,
        rings: Vec<(u32, u32)>,
        generation: u32,
        reclaim_requires_flush: bool,
    ) -> PyResult<Self> {
        let config = ArenaConfig {
            classes: classes.into_iter().map(|(slot_size, slot_count)| SlabClass { slot_size, slot_count }).collect(),
            rings: rings.into_iter().map(|(stride, capacity)| RingSpec { stride, capacity }).collect(),
            generation,
            reclaim_requires_flush,
            // Rings carved at registration are the broker's; an arena made from Python has only its fixed rings.
            ring_region_bytes: 0,
        };
        Ok(Self { arena: Arena::create(name, &config).map_err(to_py)? })
    }

    /// Removes an arena's name; processes that have it open keep their mappings.
    #[staticmethod]
    fn unlink(name: &str) -> PyResult<()> {
        Arena::unlink(name).map_err(to_py)
    }

    /// The shared memory name.
    #[getter]
    fn name(&self) -> &str {
        self.arena.name()
    }

    /// The arena's size in bytes.
    #[getter]
    fn size(&self) -> usize {
        self.arena.len()
    }

    /// The arena's generation.
    #[getter]
    fn generation(&self) -> u32 {
        self.arena.generation()
    }

    /// Where the arena is mapped in this process (16 KB aligned).
    #[getter]
    fn base_address(&self) -> usize {
        self.arena.base_ptr() as usize
    }

    /// A slab class's (slot_size, slot_count).
    fn slab_class(&self, slab_class: u16) -> PyResult<(usize, u32)> {
        let c = self.arena.class(slab_class).map_err(to_py)?;
        Ok((c.slot_size, c.slot_count))
    }

    /// How many of a slab class's slots are in use now (being written, ready, or being reclaimed), out of its
    /// slot_count. A snapshot.
    fn used(&self, slab_class: u16) -> PyResult<u32> {
        self.arena.used(slab_class).map_err(to_py)
    }

    /// How many of a slab class's slots can't be taken back without losing something now: being written or
    /// reclaimed, or ready with a live pin or no Tier 1 copy yet. A capture class runs full by design; this is what
    /// says whether it's healthy. A snapshot.
    fn held(&self, slab_class: u16) -> PyResult<u32> {
        self.arena.held(slab_class).map_err(to_py)
    }

    /// How many slab classes the arena has.
    #[getter]
    fn class_count(&self) -> usize {
        self.arena.class_count()
    }

    /// Pins a published slot for reading for `ttl_ms` milliseconds. Raises StaleError if it was reclaimed or
    /// reused.
    #[pyo3(signature = (slab_class, index, generation, ttl_ms = DEFAULT_TTL_MS))]
    fn lease(&self, slab_class: u16, index: u32, generation: u32, ttl_ms: u64) -> PyResult<PyLease> {
        let handle = SlotHandle { class: slab_class, index, generation };
        let lease = self.arena.lease(handle, ttl(ttl_ms)).map_err(to_py)?;
        Ok(PyLease::new(lease))
    }

    /// Claims a free slot in `slab_class`, copies `data` into it, and publishes it, the way a producer does: for a
    /// producer written in Python (a test camera, a synthetic source). `data` is any C-contiguous buffer (bytes, a
    /// memoryview, a NumPy array), read in place without a copy into bytes first; a copy over 64 KiB runs with the GIL
    /// released. `owner` tags the claim (a stream id, say; 0 for none, as on the Rust side). Returns the slot's
    /// (index, generation, offset); with the arena's name and generation, the class, and the length, that's an
    /// `IngestNotice`'s Tier 0 locator. Raises FullError when the class has no free slot, and ValueError when `data`
    /// doesn't fit a slot or isn't contiguous.
    #[pyo3(signature = (slab_class, data, owner = 0))]
    fn write(
        &self,
        py: Python<'_>,
        slab_class: u16,
        data: &Bound<'_, PyAny>,
        owner: u64,
    ) -> PyResult<(u32, u32, usize)> {
        let buf = pyo3::buffer::PyBuffer::<u8>::get(data)?;
        if !buf.is_c_contiguous() {
            return Err(PyValueError::new_err("data must be a C-contiguous buffer"));
        }
        let len = buf.len_bytes();
        let mut slot = self.arena.claim(slab_class, owner).map_err(to_py)?;
        if len > slot.capacity() {
            let msg = format!("{len} bytes don't fit a {}-byte slot", slot.capacity());
            return Err(PyValueError::new_err(msg));
        }
        let src = buf.buf_ptr() as usize;
        let dst = &mut slot.as_mut_slice()[..len];
        let copy = move |dst: &mut [u8]| {
            // SAFETY: `buf` keeps the exporter's memory alive and in place until it's dropped below, it's
            // C-contiguous, and `len` is its length in bytes. The slot is claimed (Writing), so nothing else writes
            // it. A Python thread that mutates the source while the GIL is released races only on its own bytes.
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst.as_mut_ptr(), dst.len()) };
        };
        if len > DETACH_ABOVE {
            py.detach(|| copy(dst));
        } else {
            copy(dst);
        }
        drop(buf);
        // The claim and the publish are each a compare-and-swap of the slot's word in continuum-shm: no reader can
        // lease it until it's published, Ready, with its length.
        let offset = slot.offset();
        let handle = slot.publish(len).map_err(to_py)?;
        Ok((handle.index, handle.generation, offset))
    }

    /// How many pins the slot holds (0 if the handle is stale).
    fn pin_count(&self, slab_class: u16, index: u32, generation: u32) -> PyResult<u32> {
        self.arena.pin_count(SlotHandle { class: slab_class, index, generation }).map_err(to_py)
    }

    /// A reader for ring `ring`, starting at the oldest record still there, or at the next one written.
    #[pyo3(signature = (ring, from_oldest = true))]
    fn ring_reader(&self, ring: usize, from_oldest: bool) -> PyResult<ring::PyRingReader> {
        let from = if from_oldest { ReadFrom::Oldest } else { ReadFrom::Latest };
        let reader = self.arena.ring_reader(ring, from).map_err(to_py)?;
        Ok(ring::PyRingReader::new(reader))
    }
}

/// A pin on a published slot. Read it with `memoryview(lease)`, `lease.numpy()`, or `numpy.from_dlpack(lease)`.
#[pyclass(name = "Lease", module = "continuum_py", frozen)]
struct PyLease {
    lease: Mutex<Option<Arc<Lease>>>,
    exports: AtomicUsize,
    handle: SlotHandle,
    offset: usize,
    address: usize,
    nbytes: usize,
}

impl PyLease {
    fn new(lease: Lease) -> Self {
        Self {
            handle: lease.handle(),
            offset: lease.offset(),
            address: lease.as_ptr() as usize,
            nbytes: lease.len(),
            lease: Mutex::new(Some(Arc::new(lease))),
            exports: AtomicUsize::new(0),
        }
    }

    fn guard(&self) -> MutexGuard<'_, Option<Arc<Lease>>> {
        self.lease.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn live(&self) -> PyResult<Arc<Lease>> {
        self.guard().clone().ok_or_else(|| PyValueError::new_err("the lease was released"))
    }
}

#[pymethods]
impl PyLease {
    /// The slot's size class.
    #[getter]
    fn slab_class(&self) -> u16 {
        self.handle.class
    }

    /// The slot's index in its class.
    #[getter]
    fn index(&self) -> u32 {
        self.handle.index
    }

    /// The slot's generation when it was leased.
    #[getter]
    fn generation(&self) -> u32 {
        self.handle.generation
    }

    /// The slot's offset from the start of the arena (a multiple of 16,384).
    #[getter]
    fn offset(&self) -> usize {
        self.offset
    }

    /// The slot's address in this process (a multiple of 16,384).
    #[getter]
    fn address(&self) -> usize {
        self.address
    }

    /// The published length in bytes.
    #[getter]
    fn nbytes(&self) -> usize {
        self.nbytes
    }

    fn __len__(&self) -> usize {
        self.nbytes
    }

    /// Whether the lease still holds the slot. Check after reading if the reader may have stalled past its ttl.
    fn is_valid(&self) -> bool {
        self.guard().as_ref().is_some_and(|l| l.is_valid())
    }

    /// Pushes the deadline out to `ttl_ms` from now. Raises StaleError if the lease already lapsed.
    fn renew(&self, ttl_ms: u64) -> PyResult<()> {
        self.live()?.renew(ttl(ttl_ms)).map_err(to_py)
    }

    /// Drops this object's hold on the pin and returns whether the lease was still valid. The pin itself goes once
    /// any DLPack capsules made from the lease are gone too. Raises BufferError while a memoryview or NumPy array
    /// of it is open.
    fn release(&self) -> PyResult<bool> {
        // Checked under the lease's lock, which __getbuffer__ also holds while it exports, so a buffer can't be
        // handed out between the check and the release (free-threaded Python runs both at once).
        let mut lease = self.guard();
        if self.exports.load(Ordering::SeqCst) > 0 {
            return Err(PyBufferError::new_err("a memoryview or NumPy array still uses this lease"));
        }
        Ok(lease.take().is_some_and(|l| l.is_valid()))
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Releases the lease. If a NumPy array or memoryview still uses it, raises BufferError, unless the block is
    /// already raising, whose exception then goes on unmasked (the lease is released when the last view goes).
    fn __exit__(
        &self,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        match self.release() {
            Err(e) if exc_type.is_none() => Err(e),
            _ => Ok(()),
        }
    }

    /// The bytes as a read-only NumPy array of `dtype`, reshaped to `shape` if given. No copy: the array reads the
    /// slot in place and keeps the lease from being released while it lives.
    #[pyo3(signature = (dtype = "uint8", shape = None))]
    fn numpy<'py>(slf: &Bound<'py, Self>, dtype: &str, shape: Option<Vec<isize>>) -> PyResult<Bound<'py, PyAny>> {
        let np = slf.py().import("numpy")?;
        let array = np.call_method1("frombuffer", (slf, dtype))?;
        match shape {
            Some(shape) => array.call_method1("reshape", (shape,)),
            None => Ok(array),
        }
    }

    /// The DLPack protocol: a capsule over the slot's bytes (1-D uint8, CPU, read-only), with no copy.
    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<&Bound<'py, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(u32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let request = ExportRequest::parse(stream, max_version, dl_device, copy)?;
        let exporter = SlotExport { lease: self.live()?, address: self.address, nbytes: self.nbytes };
        export_dlpack(&exporter, py, request)
    }

    /// The DLPack protocol: (device type, device id); always (1, 0), the CPU.
    fn __dlpack_device__(&self) -> (u32, i32) {
        (1, 0)
    }

    unsafe fn __getbuffer__(slf: Bound<'_, Self>, view: *mut ffi::Py_buffer, flags: c_int) -> PyResult<()> {
        if flags & ffi::PyBUF_WRITABLE == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("leased slots are read-only"));
        }
        let this = slf.get();
        // Hold the lease's lock until the export is counted, so release() can't drop the pin in between.
        let lease = this.guard();
        if lease.is_none() {
            return Err(PyValueError::new_err("the lease was released"));
        }
        // SAFETY: view is the buffer Python asked us to fill. The address and length are the leased slot's, which
        // stays mapped while this object lives (the view holds a reference to it) and pinned while exports > 0.
        let rc = unsafe {
            ffi::PyBuffer_FillInfo(view, slf.as_ptr(), this.address as *mut _, this.nbytes as ffi::Py_ssize_t, 1, flags)
        };
        if rc != 0 {
            return Err(PyErr::fetch(slf.py()));
        }
        this.exports.fetch_add(1, Ordering::SeqCst);
        drop(lease);
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        self.exports.fetch_sub(1, Ordering::SeqCst);
    }
}

/// One DLPack export of a leased slot. Each capsule holds its own reference to the lease.
struct SlotExport {
    lease: Arc<Lease>,
    address: usize,
    nbytes: usize,
}

impl SlotExport {
    fn managed<M: ManagedTensorBase>(&self) -> PyResult<Managed<M>> {
        let len = i64::try_from(self.nbytes).map_err(|_| PyValueError::new_err("slot too large for DLPack"))?;
        let prepared = dlpark::metadata::Fixed::new([len], [1])
            .prepare_as::<M>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut tensor = prepared.initialize(Arc::clone(&self.lease));
        tensor.set_data(self.address as *mut std::ffi::c_void);
        tensor.set_dtype(DLDataType::U8);
        tensor.set_device(DLDevice::CPU);
        // SAFETY: READ_ONLY is accurate (readers never write a leased slot), and the data pointer, shape, and
        // strides describe the slot, which the lease in the context keeps pinned until the deleter runs.
        unsafe {
            tensor.set_flags_unchecked(DlpackFlags::READ_ONLY);
            Ok(tensor.finish())
        }
    }
}

impl DlpackExporter for SlotExport {
    fn device(&self) -> DLDevice {
        DLDevice::CPU
    }

    fn flags(&self) -> DlpackFlags {
        DlpackFlags::READ_ONLY
    }

    fn prepare_export(&self, _py: Python<'_>, _request: &ExportRequest<'_>) -> PyResult<()> {
        Ok(())
    }

    fn export_legacy(&self, _py: Python<'_>) -> PyResult<Managed<DLManagedTensor>> {
        self.managed()
    }

    fn export_versioned(&self, _py: Python<'_>) -> PyResult<Managed<DLManagedTensorVersioned>> {
        self.managed()
    }
}

/// Continuum's Tier 0 shared memory arena, from Python.
#[pymodule]
#[pyo3(name = "_native")]
fn continuum_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<h264::PyH264Decoder>()?;
    m.add_function(wrap_pyfunction!(h264::h264_decoder_available, m)?)?;
    m.add_class::<PyArena>()?;
    m.add_class::<PyLease>()?;
    m.add_class::<ring::PyRingReader>()?;
    m.add_class::<ring::PyRingWriter>()?;
    m.add("ContinuumError", py.get_type::<ContinuumError>())?;
    m.add("StaleError", py.get_type::<StaleError>())?;
    m.add("NotReadyError", py.get_type::<NotReadyError>())?;
    m.add("FullError", py.get_type::<FullError>())?;
    m.add_class::<client::PyClient>()?;
    m.add_class::<recent::PyH264Window>()?;
    m.add_function(wrap_pyfunction!(recent::open_h264_window, m)?)?;
    m.add_class::<promise::PyPromise>()?;
    m.add_function(wrap_pyfunction!(_shutdown, m)?)?;
    m.add_function(wrap_pyfunction!(wire::encode_header, m)?)?;
    m.add_function(wrap_pyfunction!(wire::decode_header, m)?)?;
    m.add_function(wrap_pyfunction!(wire::decode_ingest_rejected, m)?)?;
    m.add_class::<worker::PyWorkLease>()?;
    m.add_class::<subscribe::PySubscription>()?;
    m.add_class::<segments::PySegmentWriter>()?;
    m.add_class::<segments::PySegmentFollower>()?;
    m.add_class::<segments::PySegmentFollowers>()?;
    m.add_function(wrap_pyfunction!(segments::read_segments, m)?)?;
    m.add_function(wrap_pyfunction!(segments::forget_segments, m)?)?;
    m.add_function(wrap_pyfunction!(segments::forget_segments_where, m)?)?;
    m.add_function(wrap_pyfunction!(segments::segment_deletions, m)?)?;
    m.add_function(wrap_pyfunction!(segments::segment_revision, m)?)?;
    m.add_function(wrap_pyfunction!(segments::compact_segments, m)?)?;
    m.add_function(wrap_pyfunction!(segments::segment_tail, m)?)?;
    m.add("LeaseDenied", py.get_type::<worker::LeaseDenied>())?;
    m.add_function(wrap_pyfunction!(worker::decode_event, m)?)?;
    m.add_function(wrap_pyfunction!(worker::stage_id, m)?)?;
    m.add_function(wrap_pyfunction!(search::encode_vector_query, m)?)?;
    m.add_function(wrap_pyfunction!(search::encode_text_query, m)?)?;
    m.add_function(wrap_pyfunction!(search::decode_hits, m)?)?;
    m.add_function(wrap_pyfunction!(search::encode_hits, m)?)?;
    m.add_function(wrap_pyfunction!(client::decode_flac, m)?)?;
    m.add_function(wrap_pyfunction!(client::read_chunk_records, m)?)?;
    m.add_function(wrap_pyfunction!(client::v1_gop_units, m)?)?;
    m.add("BrokerError", py.get_type::<client::BrokerError>())?;
    m.add("ConnectionClosedError", py.get_type::<client::ConnectionClosedError>())?;
    m.add("PAGE_ALIGN", continuum_core::PAGE_ALIGN)?;
    Ok(())
}
