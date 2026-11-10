//! The worker side in Python: `Client.register_worker`, `Client.quote` returning a `WorkLease`, the lease's
//! renew, keep-alive, commit, abort, and release, `decode_event` for what the broker pushes, and `stage_id`.
//! Dicts use the schema's field names; enums are their schema names (`"GPU"`, `"P3Background"`, `"Produce"`).

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use continuum_client::{
    BrokerEvent, CapabilitySpec, Cost, CoverageDone, JobLease, OutputSlot, Renewer, Requirement, WorkLease,
};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, StorageLocator, Uuid};
use pyo3::create_exception;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::ContinuumError;
use crate::client::to_py;

create_exception!(
    continuum_py,
    LeaseDenied,
    ContinuumError,
    "The broker didn't grant the lease now. args[0] is how many milliseconds to wait before asking again."
);

fn enum_named<T: Copy>(values: &[T], name: impl Fn(T) -> Option<&'static str>, what: &str, s: &str) -> PyResult<T> {
    values.iter().copied().find(|v| name(*v) == Some(s)).ok_or_else(|| {
        let names: Vec<_> = values.iter().filter_map(|v| name(*v)).collect();
        PyValueError::new_err(format!("{what} must be one of {names:?}, not {s:?}"))
    })
}

pub(crate) fn domain(s: &str) -> PyResult<fb::ComputeDomain> {
    enum_named(fb::ComputeDomain::ENUM_VALUES, |v| v.variant_name(), "compute_domain", s)
}

pub(crate) fn band(s: &str) -> PyResult<fb::PriorityBand> {
    enum_named(fb::PriorityBand::ENUM_VALUES, |v| v.variant_name(), "band", s)
}

pub(crate) fn voice(s: &str) -> PyResult<fb::VoiceTurnState> {
    enum_named(fb::VoiceTurnState::ENUM_VALUES, |v| v.variant_name(), "voice_turn", s)
}

fn role(s: &str) -> PyResult<fb::CapabilityRole> {
    enum_named(fb::CapabilityRole::ENUM_VALUES, |v| v.variant_name(), "role", s)
}

fn get<'py, T: for<'a> FromPyObject<'a, 'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    let v = d.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key.to_owned()))?;
    v.extract::<T>().map_err(|e| PyValueError::new_err(format!("{key}: {}", Into::<PyErr>::into(e))))
}

fn opt<'py, T: for<'a> FromPyObject<'a, 'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>> {
    match d.get_item(key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => {
            v.extract::<T>().map(Some).map_err(|e| PyValueError::new_err(format!("{key}: {}", Into::<PyErr>::into(e))))
        }
    }
}

pub(crate) fn capability(d: &Bound<'_, PyDict>) -> PyResult<CapabilitySpec> {
    let mut c = CapabilitySpec::new(get::<String>(d, "produces")?, domain(&get::<String>(d, "compute_domain")?)?);
    if let Some(reqs) = opt::<Vec<Bound<'_, PyDict>>>(d, "requires")? {
        c.requires = reqs
            .iter()
            .map(|r| {
                Ok(Requirement {
                    stream: get(r, "stream")?,
                    before_ns: opt(r, "before_ns")?.unwrap_or(0),
                    after_ns: opt(r, "after_ns")?.unwrap_or(0),
                })
            })
            .collect::<PyResult<_>>()?;
    }
    c.ram_mb = opt(d, "ram_mb")?.unwrap_or(0);
    c.vram_mb = opt(d, "vram_mb")?.unwrap_or(0);
    c.est_ms_per_item = opt(d, "est_ms_per_item")?.unwrap_or(0.0);
    c.cold_start_ms = opt(d, "cold_start_ms")?.unwrap_or(0);
    c.max_batch = opt(d, "max_batch")?.unwrap_or(1);
    c.preemptible = opt(d, "preemptible")?.unwrap_or(true);
    if let Some(r) = opt::<String>(d, "role")? {
        c.role = role(&r)?;
    }
    c.output_bytes = opt(d, "output_bytes")?.unwrap_or(0);
    c.outputs_per_job = opt(d, "outputs_per_job")?.unwrap_or(1);
    c.expected_confidence = opt(d, "expected_confidence")?;
    Ok(c)
}

pub(crate) fn uuid_v7(s: &str) -> PyResult<Uuid> {
    let b = crate::wire::uuid_from_str(s)?;
    Ok(Uuid(b))
}

/// A granted lease. End it with `await commit(...)`, `await abort(reason)`, or `release()`; leaving it unended
/// gives it back when it's collected. As `async with lease:`, an exception aborts it and otherwise it's released
/// unless committed. Watch `next_event()` for a LeasePreempted naming `lease.id` and stop when one comes.
#[pyclass(name = "WorkLease", module = "continuum_py", frozen)]
pub(crate) struct PyWorkLease {
    /// The lease until it ends. The lock is only held briefly, never across a round trip.
    lease: Arc<Mutex<Option<WorkLease>>>,
    renewer: Renewer,
    id: u64,
}

impl PyWorkLease {
    pub(crate) fn new(lease: WorkLease) -> Self {
        let (id, renewer) = (lease.id(), lease.renewer());
        Self { lease: Arc::new(Mutex::new(Some(lease))), renewer, id }
    }

    fn ended() -> PyErr {
        PyValueError::new_err("the lease already ended")
    }

    fn take(lease: &Mutex<Option<WorkLease>>) -> PyResult<WorkLease> {
        lease.lock().unwrap_or_else(PoisonError::into_inner).take().ok_or_else(Self::ended)
    }

    fn live(&self) -> bool {
        self.lease.lock().unwrap_or_else(PoisonError::into_inner).is_some()
    }
}

fn outputs(
    items: Vec<(Bound<'_, PyAny>, Option<Bound<'_, PyDict>>)>,
) -> PyResult<Vec<(HeaderBuf, Option<StorageLocator>)>> {
    items
        .into_iter()
        .map(|(header, at)| {
            let bytes = pyo3::buffer::PyBuffer::<u8>::get(&header)?.to_vec(header.py())?;
            let header = HeaderBuf::new(bytes).map_err(|e| PyValueError::new_err(format!("header: {e}")))?;
            let at = at.map(|d| crate::wire::shm_location(&d)).transpose()?;
            Ok((header, at))
        })
        .collect()
}

#[pymethods]
impl PyWorkLease {
    /// The lease's id (what a LeasePreempted names).
    #[getter]
    fn id(&self) -> u64 {
        self.id
    }

    /// When it expires, UTC ns (None once it ended).
    #[getter]
    fn expires_at_ns(&self) -> Option<u64> {
        self.live().then(|| self.renewer.expires_at_ns())
    }

    /// Renews the lease by `extend_ms` from now (the broker caps it); returns its new expiry, UTC ns. An awaitable.
    #[pyo3(signature = (extend_ms = 30_000))]
    fn renew<'py>(&self, py: Python<'py>, extend_ms: u64) -> PyResult<Bound<'py, PyAny>> {
        if !self.live() {
            return Err(Self::ended());
        }
        let renewer = self.renewer.clone();
        crate::awaitable(py, async move { renewer.renew(Duration::from_millis(extend_ms)).await.map_err(to_py) })
    }

    /// Renews in the background every `every_s` seconds (at least 0.01) until the lease ends. It doesn't stop on a
    /// preemption (the broker still renews a lease it's taking back): watch next_event for a LeasePreempted naming
    /// this lease and end it then.
    #[pyo3(signature = (every_s, extend_ms = 30_000))]
    fn keep_alive(&self, every_s: f64, extend_ms: u64) -> PyResult<()> {
        let every = Duration::try_from_secs_f64(every_s)
            .ok()
            .filter(|d| *d >= Duration::from_millis(10))
            .ok_or_else(|| PyValueError::new_err("every_s must be at least 0.01"))?;
        let mut guard = self.lease.lock().unwrap_or_else(PoisonError::into_inner);
        let l = guard.as_mut().ok_or_else(Self::ended)?;
        l.keep_alive(every, Duration::from_millis(extend_ms));
        Ok(())
    }

    /// Commits the job: `outputs` is a list of (header bytes from encode_header, a ShmLocation dict or None),
    /// `coverage` a {"source_stream_id", "stage", "range"} dict, and `cost` a ComputeCost dict. Returns once the
    /// outputs are durable. If the broker refuses, raises BrokerError and the lease is over (released), except when
    /// the broker was only busy ("Unavailable"): then the lease stays live and the commit can be retried. An
    /// awaitable.
    #[pyo3(signature = (outputs = Vec::new(), *, coverage = None, cost = None))]
    fn commit<'py>(
        &self,
        py: Python<'py>,
        outputs: Vec<(Bound<'py, PyAny>, Option<Bound<'py, PyDict>>)>,
        coverage: Option<Bound<'py, PyDict>>,
        cost: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let outputs = self::outputs(outputs)?;
        let coverage = coverage
            .map(|c| -> PyResult<CoverageDone> {
                Ok(CoverageDone {
                    source_stream_id: get(&c, "source_stream_id")?,
                    stage: get(&c, "stage")?,
                    range: get(&c, "range")?,
                })
            })
            .transpose()?;
        let cost = cost
            .map(|c| -> PyResult<Cost> {
                Ok(Cost {
                    wall_time_us: opt(&c, "wall_time_us")?.unwrap_or(0),
                    queue_wait_us: opt(&c, "queue_wait_us")?.unwrap_or(0),
                    ram_bytes: opt(&c, "ram_bytes")?.unwrap_or(0),
                    vram_bytes: opt(&c, "vram_bytes")?.unwrap_or(0),
                    device: opt(&c, "device")?.unwrap_or_default(),
                    items: opt(&c, "items")?.unwrap_or(0),
                })
            })
            .transpose()?;
        let lease = Arc::clone(&self.lease);
        crate::awaitable(py, async move {
            let l = Self::take(&lease)?;
            match l.commit(&outputs, coverage.as_ref(), cost.as_ref()).await {
                Ok(()) => Ok(()),
                Err(e) => {
                    if let Some(back) = e.lease {
                        *lease.lock().unwrap_or_else(PoisonError::into_inner) = Some(back);
                    }
                    Err(to_py(e.error))
                }
            }
        })
    }

    /// Gives up the job, with a reason. An awaitable.
    fn abort<'py>(&self, py: Python<'py>, reason: String) -> PyResult<Bound<'py, PyAny>> {
        let lease = Arc::clone(&self.lease);
        crate::awaitable(py, async move { Self::take(&lease)?.abort(&reason).await.map_err(to_py) })
    }

    /// Gives the lease back without committing (after a preemption, or when done early). Not awaitable. A lease
    /// that already ended is left alone.
    fn release(&self) -> PyResult<()> {
        let taken = self.lease.lock().unwrap_or_else(PoisonError::into_inner).take();
        if let Some(l) = taken {
            l.release();
        }
        Ok(())
    }

    fn __aenter__<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.unbind();
        crate::awaitable(py, async move { Ok(this) })
    }

    /// Leaving `async with`: an exception aborts the lease (with the exception's text); otherwise an uncommitted
    /// lease is released. The exception goes on.
    fn __aexit__<'py>(
        &self,
        py: Python<'py>,
        exc_type: Option<Bound<'py, PyAny>>,
        exc: Option<Bound<'py, PyAny>>,
        _tb: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let reason = match (&exc_type, &exc) {
            // The text, or the type's name when the text is empty (CancelledError, a bare raise of a class).
            (Some(t), Some(e)) => Some(
                e.str()
                    .ok()
                    .map(|s| s.to_string())
                    .filter(|s| !s.is_empty())
                    .or_else(|| t.getattr("__name__").ok().map(|n| n.to_string()))
                    .unwrap_or_else(|| "an exception".to_owned()),
            ),
            (Some(_), None) => Some("an exception".to_owned()),
            _ => None,
        };
        let lease = Arc::clone(&self.lease);
        crate::awaitable(py, async move {
            let taken = lease.lock().unwrap_or_else(PoisonError::into_inner).take();
            let Some(l) = taken else { return Ok(()) };
            match reason {
                Some(r) => {
                    let _ = l.abort(&r).await; // the exception is what the caller needs to see
                }
                None => l.release(),
            }
            Ok(())
        })
    }
}

/// What the broker pushed (an envelope from next_event) as a dict with "type": LeasePreempted {lease_id, reason,
/// by_band, revoked}, SystemStateChanged {voice_turn, changed_at_ns}, IngestRejected {stream_id, payload_id, reason},
/// DispatchJob {job_id, promise_ids, capability, capability_id, inputs, output_range, band, deadline_ns, lease},
/// PurgeNotice {purge_id, payload_ids}, Fulfilled {promise_id, items, final_part} (a part of a committed promise's
/// outputs), PromiseFailed {promise_id, reason, delivered}, Error {code, message} (a refused one-way send), Undecodable {of, reason,
/// job_id, lease_id} (one of these it couldn't read; a job's ids when they could be read, so it can still be
/// rejected), or another type name with no other keys.
#[pyfunction]
pub(crate) fn decode_event<'py>(py: Python<'py>, envelope: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let bytes = pyo3::buffer::PyBuffer::<u8>::get(envelope)?.to_vec(py)?;
    let message = continuum_client::Message::from_bytes(bytes)
        .map_err(|e| PyValueError::new_err(format!("not a ControlEnvelope: {e}")))?;
    let event = message.event();
    let d = PyDict::new(py);
    match &event {
        BrokerEvent::LeasePreempted { lease_id, reason, by_band } => {
            d.set_item("type", "LeasePreempted")?;
            d.set_item("lease_id", lease_id)?;
            d.set_item("reason", reason)?;
            d.set_item("by_band", by_band.variant_name().unwrap_or("unknown"))?;
            d.set_item("revoked", event.is_revocation())?;
        }
        BrokerEvent::SystemStateChanged { voice_turn, changed_at_ns } => {
            d.set_item("type", "SystemStateChanged")?;
            d.set_item("voice_turn", voice_turn.variant_name().unwrap_or("unknown"))?;
            d.set_item("changed_at_ns", changed_at_ns)?;
        }
        BrokerEvent::IngestRejected { stream_id, payload_id, reason } => {
            d.set_item("type", "IngestRejected")?;
            d.set_item("stream_id", stream_id)?;
            d.set_item("payload_id", payload_id.map(|u| u.to_string()))?;
            d.set_item("reason", reason)?;
        }
        BrokerEvent::DispatchJob(job) => {
            d.set_item("type", "DispatchJob")?;
            d.set_item("job_id", job.job_id.to_string())?;
            d.set_item("promise_ids", job.promise_ids.iter().map(ToString::to_string).collect::<Vec<_>>())?;
            d.set_item("capability", &job.capability)?;
            d.set_item("capability_id", job.capability_id)?;
            let inputs = PyList::empty(py);
            for it in &job.inputs {
                let i = PyDict::new(py);
                i.set_item("header", PyBytes::new(py, it.header.as_bytes()))?;
                i.set_item("locator", it.locator.as_ref().map(|l| locator_dict(py, l)).transpose()?)?;
                inputs.append(i)?;
            }
            d.set_item("inputs", inputs)?;
            d.set_item("output_range", job.output_range)?;
            d.set_item("band", job.band.variant_name().unwrap_or("unknown"))?;
            d.set_item("deadline_ns", job.deadline_ns)?;
            let lease = match &job.lease {
                None => None,
                Some(l) => {
                    let ld = PyDict::new(py);
                    ld.set_item("lease_id", l.lease_id)?;
                    ld.set_item("expires_at_ns", l.expires_at_ns)?;
                    let slots = PyList::empty(py);
                    for s in &l.slots {
                        slots.append(slot_dict(py, s)?)?;
                    }
                    ld.set_item("slots", slots)?;
                    Some(ld)
                }
            };
            d.set_item("lease", lease)?;
        }
        BrokerEvent::PurgeNotice { purge_id, payload_ids, spans } => {
            d.set_item("type", "PurgeNotice")?;
            d.set_item("purge_id", purge_id.to_string())?;
            d.set_item("payload_ids", payload_ids.iter().map(ToString::to_string).collect::<Vec<_>>())?;
            let spans: Vec<_> = spans
                .iter()
                .map(|s| -> PyResult<Bound<'_, PyDict>> {
                    let sd = PyDict::new(py);
                    sd.set_item("stream_id", s.stream_id)?;
                    sd.set_item("range", s.range)?;
                    Ok(sd)
                })
                .collect::<PyResult<_>>()?;
            d.set_item("spans", spans)?;
        }
        BrokerEvent::Fulfilled { promise_id, items, final_part } => {
            d.set_item("type", "Fulfilled")?;
            d.set_item("promise_id", promise_id.to_string())?;
            let list = PyList::empty(py);
            for it in items {
                let i = PyDict::new(py);
                i.set_item("header", PyBytes::new(py, it.header.as_bytes()))?;
                i.set_item("locator", it.locator.as_ref().map(|l| locator_dict(py, l)).transpose()?)?;
                list.append(i)?;
            }
            d.set_item("items", list)?;
            d.set_item("final_part", final_part)?;
        }
        BrokerEvent::PromiseFailed { promise_id, reason, delivered } => {
            d.set_item("type", "PromiseFailed")?;
            d.set_item("promise_id", promise_id.to_string())?;
            d.set_item("reason", reason)?;
            d.set_item("delivered", delivered)?;
        }
        BrokerEvent::Undecodable { kind, reason, job_id, lease_id } => {
            d.set_item("type", "Undecodable")?;
            d.set_item("of", kind.variant_name().unwrap_or("unknown"))?;
            d.set_item("reason", reason)?;
            d.set_item("job_id", job_id.map(|u| u.to_string()))?;
            d.set_item("lease_id", lease_id)?;
        }
        BrokerEvent::Error { code, message } => {
            d.set_item("type", "Error")?;
            d.set_item("code", code.variant_name().unwrap_or("unknown"))?;
            d.set_item("message", message)?;
        }
        BrokerEvent::Other(kind) => d.set_item("type", kind.variant_name().unwrap_or("unknown"))?,
    }
    Ok(d)
}

fn slot_dict<'py>(py: Python<'py>, s: &OutputSlot) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("arena", &s.arena)?;
    d.set_item("arena_generation", s.arena_generation)?;
    d.set_item("slab_class", s.slab_class)?;
    d.set_item("slot", s.slot)?;
    d.set_item("slot_generation", s.slot_generation)?;
    d.set_item("offset", s.offset)?;
    d.set_item("length", s.length)?;
    Ok(d)
}

/// A locator as a dict with "kind" ("Shm", "Ring", "Inline", or "Chunk") and that kind's fields.
pub(crate) fn locator_dict<'py>(py: Python<'py>, l: &StorageLocator) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match l {
        StorageLocator::Shm { arena, arena_generation, slab_class, slot, slot_generation, offset, length } => {
            d.set_item("kind", "Shm")?;
            d.set_item("arena", arena)?;
            d.set_item("arena_generation", arena_generation)?;
            d.set_item("slab_class", slab_class)?;
            d.set_item("slot", slot)?;
            d.set_item("slot_generation", slot_generation)?;
            d.set_item("offset", offset)?;
            d.set_item("length", length)?;
        }
        StorageLocator::Ring { arena, arena_generation, ring, first_seq, count } => {
            d.set_item("kind", "Ring")?;
            d.set_item("arena", arena)?;
            d.set_item("arena_generation", arena_generation)?;
            d.set_item("ring", ring)?;
            d.set_item("first_seq", first_seq)?;
            d.set_item("count", count)?;
        }
        StorageLocator::Inline { bytes } => {
            d.set_item("kind", "Inline")?;
            d.set_item("bytes", PyBytes::new(py, bytes))?;
        }
        StorageLocator::Chunk { tier, uri, offset, length } => {
            d.set_item("kind", "Chunk")?;
            d.set_item("tier", tier.variant_name().unwrap_or("unknown"))?;
            d.set_item("uri", uri)?;
            d.set_item("offset", offset)?;
            d.set_item("length", length)?;
        }
    }
    Ok(d)
}

/// A job's output lease from the "lease" dict decode_event gives a DispatchJob.
pub(crate) fn job_lease_from(d: &Bound<'_, PyDict>) -> PyResult<JobLease> {
    let slots = opt::<Vec<Bound<'_, PyDict>>>(d, "slots")?
        .unwrap_or_default()
        .iter()
        .map(|s| {
            Ok(OutputSlot {
                arena: get(s, "arena")?,
                arena_generation: get(s, "arena_generation")?,
                slab_class: get(s, "slab_class")?,
                slot: get(s, "slot")?,
                slot_generation: get(s, "slot_generation")?,
                offset: get(s, "offset")?,
                length: get(s, "length")?,
            })
        })
        .collect::<PyResult<_>>()?;
    Ok(JobLease { lease_id: get(d, "lease_id")?, slots, expires_at_ns: get(d, "expires_at_ns")? })
}

/// A stage's id: the 32-bit FNV-1a hash of its name (the broker computes the same).
#[pyfunction]
pub(crate) fn stage_id(name: &str) -> u32 {
    continuum_client::stage_id(name).0
}
