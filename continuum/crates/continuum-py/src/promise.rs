//! The consumer's promises for Python: `Client.query_quoted`, `Client.commit_promise`, and `Client.cancel_promise`
//! (continuum_client's `promise` module), as dicts.

use std::sync::Arc;

use continuum_client::{Item, Promise, PromiseAccepted, PromiseQuote};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::client::to_py;

/// A quote as a dict: {"promise_id", "status", "options": [{"steps": [{"read_tier", "worker_id", "produces",
/// "items", "est_ms"}], "est_latency_ms", "cost_units", "exceeds_budget", "expected_confidence"}],
/// "shares_in_flight", "expires_at_ns"}.
pub(crate) fn quote_dict<'py>(py: Python<'py>, q: &PromiseQuote) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("promise_id", q.promise_id.to_string())?;
    d.set_item("status", q.status.variant_name().unwrap_or("unknown"))?;
    let options = PyList::empty(py);
    for o in &q.options {
        let od = PyDict::new(py);
        let steps = PyList::empty(py);
        for s in &o.steps {
            let sd = PyDict::new(py);
            sd.set_item("read_tier", s.read_tier.variant_name().unwrap_or("unknown"))?;
            sd.set_item("worker_id", &s.worker_id)?;
            sd.set_item("produces", &s.produces)?;
            sd.set_item("items", s.items)?;
            sd.set_item("est_ms", s.est_ms)?;
            steps.append(sd)?;
        }
        od.set_item("steps", steps)?;
        od.set_item("est_latency_ms", o.est_latency_ms)?;
        od.set_item("cost_units", o.cost_units)?;
        od.set_item("exceeds_budget", o.exceeds_budget)?;
        od.set_item("expected_confidence", o.expected_confidence)?;
        options.append(od)?;
    }
    d.set_item("options", options)?;
    d.set_item("shares_in_flight", q.shares_in_flight)?;
    d.set_item("expires_at_ns", q.expires_at_ns)?;
    Ok(d)
}

/// Where a promise stands, as {"promise_id", "life"} ("Armed", "Dispatching", "Cancelled", or "Failed").
pub(crate) fn accepted_dict(py: Python<'_>, a: PromiseAccepted) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("promise_id", a.promise_id.to_string())?;
    d.set_item("life", a.life.variant_name().unwrap_or("unknown"))?;
    d.into_py_any(py)
}

fn items_list<'py>(py: Python<'py>, items: &[Item]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for it in items {
        let i = PyDict::new(py);
        i.set_item("header", PyBytes::new(py, it.header.as_bytes()))?;
        i.set_item("locator", it.locator.as_ref().map(|l| crate::worker::locator_dict(py, l)).transpose()?)?;
        list.append(i)?;
    }
    Ok(list)
}

/// A committed promise (from `Client.commit_promise`): its outputs as they come, routed here by its id, so they
/// never arrive from next_event. `id` and `life` say which promise and where it stood when committed.
#[pyclass(name = "Promise", module = "continuum_py", frozen)]
pub(crate) struct PyPromise {
    promise: Arc<tokio::sync::Mutex<Promise>>,
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    life: String,
}

impl PyPromise {
    pub(crate) fn new(accepted: PromiseAccepted, promise: Promise) -> Self {
        Self {
            id: accepted.promise_id.to_string(),
            life: accepted.life.variant_name().unwrap_or("unknown").to_owned(),
            promise: Arc::new(tokio::sync::Mutex::new(promise)),
        }
    }
}

#[pymethods]
impl PyPromise {
    /// The next part of the outputs, {"items": [{"header", "locator"}], "final_part"}, or None after the final
    /// part. ContinuumError when the promise failed or the connection ended. An awaitable.
    fn next_part<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let promise = Arc::clone(&self.promise);
        crate::awaitable(py, async move {
            let part = promise.lock().await.next_part().await.map_err(to_py)?;
            crate::attach(move |py| -> PyResult<Py<PyAny>> {
                let Some(part) = part else { return Ok(py.None()) };
                let d = PyDict::new(py);
                d.set_item("items", items_list(py, &part.items)?)?;
                d.set_item("final_part", part.final_part)?;
                d.into_py_any(py)
            })
            .await
        })
    }

    /// Every output, [{"header", "locator"}], gathered through the final part. ContinuumError when the promise
    /// failed or the connection ended. An awaitable.
    fn fulfilled<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let promise = Arc::clone(&self.promise);
        crate::awaitable(py, async move {
            let mut items = Vec::new();
            let mut p = promise.lock().await;
            while let Some(part) = p.next_part().await.map_err(to_py)? {
                items.extend(part.items);
            }
            drop(p);
            crate::attach(move |py| items_list(py, &items).and_then(|l| l.into_py_any(py))).await
        })
    }
}
