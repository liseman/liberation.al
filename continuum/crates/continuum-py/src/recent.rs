//! The recent window of a camera's native H.264 in Python (#1741): `await open_h264_window(client, live, stream,
//! from_ns)` returns an `H264Window`, whose `await next()` gives one dict per access unit or gap, history first and
//! then live, and `None` once the live subscription ends. See `continuum_tier::recent` for the contract.
//!
//! An access unit's dict holds only what capture attested. A value capture didn't give is `None`, never a guess:
//! `device_seq` and `capture_epoch` (zero epoch means unknown), and `width` and `height`. `header` is the frame's whole
//! original header (bytes, for `decode_header`) when it was read from Tier 0, and `None` from a Tier 1 chunk, which
//! keeps the frame's id, times, sequence, provenance, and flip but not the rest of its header (arrival time, the
//! monotonic capture time, the device timestamp, the stride). `tier` says which.

use std::sync::Arc;
use std::time::Duration;

use continuum_tier::recent::{Gap, GapWhy, H264Window, Item};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::client::PyClient;

fn why(why: GapWhy) -> &'static str {
    match why {
        GapWhy::Missing => "missing",
        GapWhy::EpochChanged => "epoch_changed",
        GapWhy::ContinuityUnknown => "continuity_unknown",
        GapWhy::Unreadable => "unreadable",
        GapWhy::BeforeRandomAccess => "before_random_access",
    }
}

fn item_dict(py: Python<'_>, item: Item) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    match item {
        Item::Au(au) => {
            d.set_item("type", "au")?;
            d.set_item("id", au.id.to_string())?;
            d.set_item("start_ns", au.start_ns)?;
            d.set_item("end_ns", au.end_ns)?;
            d.set_item("seq", au.seq)?;
            d.set_item("device_seq", au.device_seq)?;
            d.set_item("capture_epoch", (au.epoch != 0).then_some(au.epoch))?;
            d.set_item("width", (au.width != 0).then_some(au.width))?;
            d.set_item("height", (au.height != 0).then_some(au.height))?;
            d.set_item("flip", au.flip)?;
            d.set_item("random_access", au.random_access)?;
            d.set_item("tier", if au.header.is_some() { "tier0" } else { "tier1" })?;
            d.set_item("header", au.header.as_ref().map(|h| PyBytes::new(py, h.as_bytes())))?;
            d.set_item("bytes", PyBytes::new(py, &au.bytes))?;
        }
        Item::Gap(Gap { from_ns, to_ns, why: w }) => {
            d.set_item("type", "gap")?;
            d.set_item("from_ns", from_ns)?;
            d.set_item("to_ns", to_ns)?;
            d.set_item("why", why(w))?;
        }
    }
    d.into_py_any(py)
}

/// A camera's recent window: `async for item in window`, or `await window.next()` (None once it ends).
#[pyclass(name = "H264Window", module = "continuum_py", frozen)]
pub(crate) struct PyH264Window {
    inner: Arc<tokio::sync::Mutex<H264Window>>,
}

impl PyH264Window {
    fn next_future<'py>(&self, py: Python<'py>, stop: bool) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        crate::awaitable(py, async move {
            let item = inner.lock().await.next().await;
            crate::attach(|py| match item {
                Some(item) => item_dict(py, item),
                None if stop => Err(PyStopAsyncIteration::new_err(())),
                None => Ok(py.None()),
            })
            .await
        })
    }
}

#[pymethods]
impl PyH264Window {
    /// The next access unit or gap, or None once the live subscription ended. An awaitable.
    fn next<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.next_future(py, false)
    }

    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.next_future(py, true)
    }
}

/// Opens `stream`'s recent window at `from_ns` (UTC ns), which must be within `window_s` seconds of `now_ns` (now,
/// by default). `client` answers queries and maps the arena; `live` is a connection of its own for the live
/// subscription, made with `subscription_overflow="backpressure"`. `data_root` is the state directory that holds
/// `continuum/`, which relative chunk locators name files under (the working directory when None). An awaitable.
#[pyfunction]
#[pyo3(signature = (client, live, stream, from_ns, window_s = 300.0, now_ns = None, data_root = None))]
#[allow(clippy::too_many_arguments)] // one per Python parameter, most of them keywords with defaults
pub(crate) fn open_h264_window<'py>(
    py: Python<'py>,
    client: &PyClient,
    live: &PyClient,
    stream: String,
    from_ns: u64,
    window_s: f64,
    now_ns: Option<u64>,
    data_root: Option<std::path::PathBuf>,
) -> PyResult<Bound<'py, PyAny>> {
    let root = data_root.unwrap_or_default();
    let window = Duration::try_from_secs_f64(window_s).map_err(|_| {
        PyValueError::new_err(format!("window_s must be a nonnegative number of seconds, not {window_s}"))
    })?;
    let now_ns = match now_ns {
        Some(now) => now,
        None => u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .as_nanos(),
        )
        .unwrap_or(u64::MAX),
    };
    let (client, live) = (client.shared(), live.shared());
    crate::awaitable(py, async move {
        let window = H264Window::open(client, live, &root, &stream, from_ns, window, now_ns)
            .await
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyH264Window { inner: Arc::new(tokio::sync::Mutex::new(window)) })
    })
}
