//! Subscriptions in Python: `Client.subscribe` returns a `Subscription`, an async iterator of event dicts with
//! `ack` and `unsubscribe` (and `async with`, which unsubscribes on the way out).

use std::sync::Arc;

use continuum_client::{Start, SubEvent, SubscribeSpec, Subscription, SubscriptionHandle};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::client::to_py;
use crate::worker::{locator_dict, uuid_v7};

/// A subscription's event as a dict: {"type": "Event", "position", "item": {"header", "locator"}},
/// {"type": "CaughtUp", "position"}, or {"type": "Undecodable", "position", "reason"}.
fn event_dict(py: Python<'_>, event: SubEvent) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    match event {
        SubEvent::Event { position, item } => {
            d.set_item("type", "Event")?;
            d.set_item("position", position)?;
            let i = PyDict::new(py);
            i.set_item("header", PyBytes::new(py, item.header.as_bytes()))?;
            i.set_item("locator", item.locator.as_ref().map(|l| locator_dict(py, l)).transpose()?)?;
            d.set_item("item", i)?;
        }
        SubEvent::CaughtUp { position } => {
            d.set_item("type", "CaughtUp")?;
            d.set_item("position", position)?;
        }
        SubEvent::Undecodable { position, reason } => {
            d.set_item("type", "Undecodable")?;
            d.set_item("position", position)?;
            d.set_item("reason", reason)?;
        }
    }
    d.into_py_any(py)
}

/// A live subscription: `async for event in sub`, `await sub.next()` (None once it ends), `sub.ack(position)` for a
/// durable one, and `await sub.unsubscribe()` (from any task; a waiting `next` then ends). `async with sub:`
/// unsubscribes on the way out; prefer it, since a `next()` still pending keeps the subscription alive after the
/// last reference to it goes. Cancelling a `next()` (a timeout) just as an event arrives can drop that event: a
/// durable subscriber then sees it again after a restart, since only acked positions are saved.
#[pyclass(name = "Subscription", module = "continuum_py", frozen)]
pub(crate) struct PySubscription {
    /// Locked across `next`'s wait; only reading needs it, so ack and unsubscribe never wait on a reader.
    inner: Arc<tokio::sync::Mutex<Subscription>>,
    handle: SubscriptionHandle,
    id: u64,
    start_position: u64,
    live_at_start: bool,
    rings: Vec<(continuum_core::StreamId, continuum_client::StreamLease)>,
}

impl PySubscription {
    fn next_future<'py>(&self, py: Python<'py>, stop: bool) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        crate::awaitable(py, async move {
            let event = inner.lock().await.next().await;
            crate::attach(|py| match event {
                Some(e) => event_dict(py, e),
                None if stop => Err(PyStopAsyncIteration::new_err(())),
                None => Ok(py.None()),
            })
            .await
        })
    }
}

#[pymethods]
impl PySubscription {
    /// The client's number for it.
    #[getter]
    fn id(&self) -> u64 {
        self.id
    }

    /// The position it started after (0 for everything).
    #[getter]
    fn start_position(&self) -> u64 {
        self.start_position
    }

    /// The followed ring streams' rings: [{"stream_id", "lease"}], each lease for `Client.ring_reader`, to read a ring
    /// live instead of waiting for its batches.
    #[getter]
    fn rings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let out = pyo3::types::PyList::empty(py);
        for (stream, lease) in &self.rings {
            let d = PyDict::new(py);
            d.set_item("stream_id", stream.0)?;
            d.set_item("lease", crate::client::lease_dict(py, lease)?)?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// Whether it was live from the start (nothing to catch up on, so no CaughtUp comes).
    #[getter]
    fn live_at_start(&self) -> bool {
        self.live_at_start
    }

    /// The next event dict, or None once the subscription or the connection ends. An awaitable.
    fn next<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.next_future(py, false)
    }

    /// Blocks this thread, with the GIL released, until an event arrives, then returns it with every other event
    /// already waiting, up to `max` in all: a list of event dicts, `[]` once `timeout_s` passes with none, or None
    /// once the subscription or the connection ends. Not for the event loop's thread, which it would stall: for a
    /// thread of its own, such as a camera's decoder, which then wakes once per batch, with no handoff to
    /// asyncio for each event. An `unsubscribe` from another task ends a waiting `take` with None.
    #[pyo3(signature = (timeout_s, max = 64))]
    fn take(&self, py: Python<'_>, timeout_s: f64, max: usize) -> PyResult<Option<Vec<Py<PyAny>>>> {
        let timeout = std::time::Duration::try_from_secs_f64(timeout_s)
            .map_err(|_| PyValueError::new_err("timeout_s must be a finite nonnegative number of seconds"))?;
        if max == 0 {
            return Err(PyValueError::new_err("max must be at least 1"));
        }
        let inner = Arc::clone(&self.inner);
        let events = py.detach(|| {
            pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
                let mut subscription = inner.lock().await;
                match tokio::time::timeout(timeout, subscription.next()).await {
                    Err(_) => Some(Vec::new()),
                    Ok(None) => None,
                    Ok(Some(first)) => {
                        let mut events = vec![first];
                        while events.len() < max
                            && let Some(event) = subscription.try_next()
                        {
                            events.push(event);
                        }
                        Some(events)
                    }
                }
            })
        });
        events.map(|events| events.into_iter().map(|event| event_dict(py, event)).collect()).transpose()
    }

    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.next_future(py, true)
    }

    /// Says everything up to `position` is handled: a durable subscription resumes after it. One way; not
    /// awaitable.
    fn ack(&self, position: u64) -> PyResult<()> {
        self.handle.ack(position).map_err(to_py)
    }

    /// Ends it and waits for the broker's answer; a durable one keeps its saved position. Ending one that already
    /// ended does nothing. An awaitable.
    fn unsubscribe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let handle = self.handle.clone();
        crate::awaitable(py, async move { handle.unsubscribe().await.map_err(to_py) })
    }

    fn __aenter__<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.unbind();
        crate::awaitable(py, async move { Ok(this) })
    }

    fn __aexit__<'py>(
        &self,
        py: Python<'py>,
        _exc_type: Option<Bound<'py, PyAny>>,
        _exc: Option<Bound<'py, PyAny>>,
        _tb: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let handle = self.handle.clone();
        crate::awaitable(py, async move {
            // A closed connection already ended it; the block's own exception, if any, is what matters.
            let _ = handle.unsubscribe().await;
            Ok(())
        })
    }
}

/// `Client.subscribe`'s body (see its docs in client.rs).
#[allow(clippy::too_many_arguments)]
pub(crate) fn subscribe<'py>(
    py: Python<'py>,
    client: Arc<continuum_client::Client>,
    streams: Vec<String>,
    start: &str,
    start_ns: u64,
    durable_name: Option<String>,
    pinned: bool,
    subjects: Vec<String>,
    shape_name: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let start = match start {
        "Live" => Start::Live,
        "Time" => Start::Time(start_ns),
        "Saved" if durable_name.is_some() => Start::Saved,
        "Saved" => return Err(PyValueError::new_err("a Saved start needs a durable_name")),
        other => return Err(PyValueError::new_err(format!("start must be Live, Time, or Saved, not {other:?}"))),
    };
    let mut spec = SubscribeSpec::new(streams, start);
    spec.durable_name = durable_name;
    spec.pinned = pinned;
    spec.subjects = subjects.iter().map(|s| uuid_v7(s)).collect::<PyResult<_>>()?;
    spec.shape_name = shape_name;
    crate::awaitable(py, async move {
        let sub = client.subscribe(&spec).await.map_err(to_py)?;
        Ok(PySubscription {
            handle: sub.handle(),
            id: sub.id(),
            start_position: sub.start_position(),
            live_at_start: sub.live_at_start(),
            rings: sub.rings().to_vec(),
            inner: Arc::new(tokio::sync::Mutex::new(sub)),
        })
    })
}

/// An ordinary item, shared by full query replies and projected-query fallback rows.
fn query_item_dict<'py>(
    py: Python<'py>,
    item: &continuum_client::Item,
    commit_seq: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("header", PyBytes::new(py, item.header.as_bytes()))?;
    out.set_item("commit_seq", commit_seq)?;
    out.set_item("locator", item.locator.as_ref().map(|l| locator_dict(py, l)).transpose()?)?;
    Ok(out)
}

/// A query result as a dict: {"items": [{"header", "locator"}], "complete", "truncated", "missing": [(start, end)], "tail":
/// {"arena", "ring_offset", "first_seq", "last_seq"} or None}.
pub(crate) fn query_result_dict(py: Python<'_>, r: continuum_client::QueryResult) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    let items = pyo3::types::PyList::empty(py);
    for (n, it) in r.items.iter().enumerate() {
        items.append(query_item_dict(py, it, r.seqs.get(n).copied().unwrap_or(0))?)?;
    }
    d.set_item("items", items)?;
    d.set_item("complete", r.complete)?;
    d.set_item("truncated", r.truncated)?;
    d.set_item("missing", r.missing)?;
    let tail = r
        .tail
        .map(|t| -> PyResult<Bound<'_, PyDict>> {
            let td = PyDict::new(py);
            td.set_item("arena", t.arena)?;
            td.set_item("ring_offset", t.ring_offset)?;
            td.set_item("first_seq", t.first_seq)?;
            td.set_item("last_seq", t.last_seq)?;
            Ok(td)
        })
        .transpose()?;
    d.set_item("tail", tail)?;
    d.set_item("next_after", r.next_after)?;
    d.into_py_any(py)
}

/// `Client.query`'s body (see its docs in client.rs).
#[allow(clippy::too_many_arguments)]
pub(crate) fn query<'py>(
    py: Python<'py>,
    client: Arc<continuum_client::Client>,
    stream: String,
    start_ns: u64,
    end_ns: u64,
    all: bool,
    max_items: u32,
    band: &str,
    subjects: Vec<String>,
    ids: Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let spec = query_spec(stream, start_ns, end_ns, all, max_items, band, &subjects, &ids)?;
    crate::awaitable(py, async move {
        let r = if all { client.query_all(&spec).await } else { client.query(&spec).await }.map_err(to_py)?;
        crate::attach(|py| query_result_dict(py, r)).await
    })
}

/// Shared argument validation and query options for full and compact query replies.
#[allow(clippy::too_many_arguments)]
fn query_spec(
    stream: String,
    start_ns: u64,
    end_ns: u64,
    all: bool,
    max_items: u32,
    band: &str,
    subjects: &[String],
    ids: &[String],
) -> PyResult<continuum_client::QuerySpec> {
    if end_ns <= start_ns {
        // An empty range matches nothing; sent as is, the paged query would drop it and read the whole stream.
        return Err(PyValueError::new_err("end_ns must be after start_ns"));
    }
    if !all && (!subjects.is_empty() || !ids.is_empty()) {
        // The broker applies these filters only to the paged (sequence) query; a single time query would ignore them
        // and return everything in the range.
        return Err(PyValueError::new_err("subjects and ids filter only with all=True"));
    }
    let mut spec = continuum_client::QuerySpec::hit_only(stream, (start_ns, end_ns));
    spec.max_items = max_items;
    spec.band = crate::worker::band(band)?;
    spec.subjects = subjects.iter().map(|s| uuid_v7(s)).collect::<PyResult<_>>()?;
    spec.ids = ids.iter().map(|s| uuid_v7(s)).collect::<PyResult<_>>()?;
    Ok(spec)
}

/// A fully drained query in query_all order: inline identity/span/bytes tuples, other items unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn query_inline_records<'py>(
    py: Python<'py>,
    client: Arc<continuum_client::Client>,
    stream: String,
    start_ns: u64,
    end_ns: u64,
    max_items: u32,
    band: &str,
    subjects: Vec<String>,
    ids: Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let spec = query_spec(stream, start_ns, end_ns, true, max_items, band, &subjects, &ids)?;
    crate::awaitable(py, async move {
        let result = client.query_all(&spec).await.map_err(to_py)?;
        crate::attach(|py| {
            let out = pyo3::types::PyList::empty(py);
            for (n, item) in result.items.iter().enumerate() {
                if let Some(continuum_core::StorageLocator::Inline { bytes }) = &item.locator {
                    let header = item.header.header();
                    out.append((
                        item.header.id().to_string(),
                        header.start_ns(),
                        header.end_ns(),
                        PyBytes::new(py, bytes),
                    ))?;
                } else {
                    out.append(query_item_dict(py, item, result.seqs.get(n).copied().unwrap_or(0))?)?;
                }
            }
            out.into_py_any(py)
        })
        .await
    })
}
