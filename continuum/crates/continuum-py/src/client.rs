//! `continuum_py.Client`: the asyncio client, wrapping continuum-client.
//!
//! Every network call returns an awaitable, run on a tokio runtime that pyo3-async-runtimes keeps for the process,
//! so the event loop never blocks on the socket. Ids cross the boundary as UUID strings and times as UTC
//! nanoseconds (docs/CONTINUUM.md, decisions 8 and 9).

use std::sync::Arc;
use std::time::Duration;

use continuum_client::{
    Client, ClientOptions, Error as ClientError, Quote, QuoteSpec, StreamLease, StreamRegistered, StreamSpec,
    StreamStorage, Telemetry, WorkerSpec,
};
use continuum_core::schema::continuum as fb;
use pyo3::exceptions::{PyConnectionError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3::{IntoPyObjectExt, create_exception};

use crate::{ContinuumError, PyArena};

create_exception!(
    continuum_py,
    BrokerError,
    ContinuumError,
    "The broker refused a request. args are (code, message); code is the ErrorCode's name, like 'Unavailable'."
);
create_exception!(continuum_py, ConnectionClosedError, PyConnectionError, "The connection to the broker is closed.");

pub(crate) fn to_py(err: ClientError) -> PyErr {
    match err {
        ClientError::Broker { code, message } => {
            BrokerError::new_err((code.variant_name().unwrap_or("Unknown").to_owned(), message))
        }
        ClientError::Closed => ConnectionClosedError::new_err("the connection to the broker is closed"),
        ClientError::Timeout(t) => PyTimeoutError::new_err(format!("no reply from the broker within {t:?}")),
        ClientError::Io(e) => PyErr::from(e),
        other => ContinuumError::new_err(other.to_string()),
    }
}

fn uuid_string(id: continuum_core::Uuid) -> String {
    id.to_string()
}

fn parse_kind(kind: &str) -> PyResult<fb::StreamKind> {
    match kind {
        "sensor" => Ok(fb::StreamKind::Sensor),
        "derived" => Ok(fb::StreamKind::Derived),
        other => Err(PyValueError::new_err(format!("kind must be 'sensor' or 'derived', not {other:?}"))),
    }
}

pub(crate) fn lease_dict<'py>(py: Python<'py>, l: &StreamLease) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("lease_id", l.lease_id)?;
    d.set_item("arena", &l.arena)?;
    d.set_item("arena_generation", l.arena_generation)?;
    d.set_item("first_slot", l.first_slot)?;
    d.set_item("slot_count", l.slot_count)?;
    d.set_item("slot_bytes", l.slot_bytes)?;
    d.set_item("ring_offset", l.ring_offset)?;
    d.set_item("ring_records", l.ring_records)?;
    d.set_item("ring_index", l.ring_index)?;
    d.set_item("ring_data_offset", l.ring_data_offset)?;
    d.set_item("ring_stride", l.ring_stride)?;
    d.set_item("ring_generation", l.ring_generation)?;
    Ok(d)
}

/// A carved ring from a lease dict (from register_stream or a Subscription's rings), and the arena it's in.
fn carved_of(lease: &Bound<'_, PyDict>) -> PyResult<(String, u32, continuum_shm::CarvedRing)> {
    let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
        lease.get_item(k)?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(k.to_owned()))
    };
    let ring = continuum_shm::CarvedRing {
        ctrl_offset: get("ring_offset")?.extract()?,
        data_offset: get("ring_data_offset")?.extract()?,
        stride: get("ring_stride")?.extract()?,
        capacity: get("ring_records")?.extract()?,
        generation: get("ring_generation")?.extract()?,
    };
    if ring.stride == 0 || ring.capacity == 0 {
        return Err(PyValueError::new_err("the lease is for slots, not a ring"));
    }
    Ok((get("arena")?.extract()?, get("arena_generation")?.extract()?, ring))
}

fn registered_dict(py: Python<'_>, r: &StreamRegistered) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("stream_id", r.stream_id)?;
    match &r.lease {
        Some(l) => d.set_item("lease", lease_dict(py, l)?)?,
        None => d.set_item("lease", py.None())?,
    }
    d.into_py_any(py)
}

/// The shortest heartbeat interval or request timeout accepted, so a typo can't make the client flood the broker.
const MIN_INTERVAL: Duration = Duration::from_millis(10);

/// A connection to the broker. Make one with `await Client.connect(socket)`.
///
/// Every method that talks to the broker returns an awaitable bound to the running event loop. Cancelling one
/// abandons only the reply: a request already sent (a registration, say) still takes effect. Leaving one un-awaited
/// when the loop closes logs a harmless "Event loop is closed" from the runtime.
#[pyclass(name = "Client", module = "continuum_py", frozen)]
pub(crate) struct PyClient {
    client: Arc<Client>,
}

impl PyClient {
    /// The shared connection, for a binding that keeps it past the call (the recent H.264 window).
    pub(crate) fn shared(&self) -> Arc<Client> {
        Arc::clone(&self.client)
    }
}

#[pymethods]
impl PyClient {
    /// Connects to the broker's socket, says Hello, and maps the arenas it hands over. An awaitable.
    #[staticmethod]
    #[pyo3(signature = (socket, client_name = "python", heartbeat_interval_s = 5.0, request_timeout_s = 10.0, subscription_overflow = "close"))]
    fn connect<'py>(
        py: Python<'py>,
        socket: std::path::PathBuf,
        client_name: &str,
        heartbeat_interval_s: f64,
        request_timeout_s: f64,
        subscription_overflow: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let seconds = |s: f64, what: &str| {
            Duration::try_from_secs_f64(s)
                .ok()
                .filter(|d| *d >= MIN_INTERVAL)
                .ok_or_else(|| PyValueError::new_err(format!("{what} must be at least {MIN_INTERVAL:?}, not {s} s")))
        };
        let options = ClientOptions {
            client_name: client_name.to_owned(),
            heartbeat_interval: seconds(heartbeat_interval_s, "heartbeat_interval_s")?,
            request_timeout: seconds(request_timeout_s, "request_timeout_s")?,
            subscription_overflow: match subscription_overflow {
                "close" => continuum_client::SubscriptionOverflow::Close,
                "backpressure" => continuum_client::SubscriptionOverflow::Backpressure,
                _ => return Err(PyValueError::new_err("subscription_overflow must be close or backpressure")),
            },
        };
        crate::awaitable(py, async move {
            let client = Client::connect(socket, options).await.map_err(to_py)?;
            Ok(PyClient { client: Arc::new(client) })
        })
    }

    /// The session id the broker assigned, as a UUID string.
    #[getter]
    fn session_id(&self) -> String {
        uuid_string(self.client.session_id())
    }

    /// The broker's schema version, (major, minor).
    #[getter]
    fn broker_schema(&self) -> (u16, u16) {
        self.client.broker_schema()
    }

    /// Latest voice turn, tracked even without reading events; None after disconnection.
    #[getter]
    fn voice_turn(&self) -> Option<&'static str> {
        self.client.voice_turn().and_then(|state| state.variant_name())
    }

    /// A live reader over a ring stream's ring: `lease` is the ring's lease dict (from register_stream, or a
    /// Subscription's `rings`). It reads from the next record written, or with `from_oldest` from the oldest still in
    /// the ring. The ring must be in an arena this client mapped (Welcome's), at that arena's generation; ValueError
    /// otherwise.
    #[pyo3(signature = (lease, *, from_oldest = false))]
    fn ring_reader(&self, lease: &Bound<'_, PyDict>, from_oldest: bool) -> PyResult<crate::ring::PyRingReader> {
        let (arena, generation, ring) = carved_of(lease)?;
        let mapped = self
            .client
            .arenas()
            .iter()
            .find(|a| a.info.name == arena)
            .ok_or_else(|| PyValueError::new_err(format!("this client hasn't mapped arena {arena:?}")))?;
        if mapped.info.generation != generation {
            return Err(PyValueError::new_err(format!(
                "arena {arena:?} is at generation {}, the lease names {generation}",
                mapped.info.generation
            )));
        }
        let from = if from_oldest { continuum_shm::ReadFrom::Oldest } else { continuum_shm::ReadFrom::Latest };
        let reader = mapped.arena.carved_ring_reader(&ring, from).map_err(crate::to_py)?;
        Ok(crate::ring::PyRingReader::new(reader))
    }

    /// Takes a ring stream's single writer: `lease` is the ring's lease dict from `register_stream`. ValueError when
    /// the ring isn't in an arena this client mapped at that generation; an error when another writer holds it.
    fn ring_writer(&self, lease: &Bound<'_, PyDict>) -> PyResult<crate::ring::PyRingWriter> {
        let (arena, generation, ring) = carved_of(lease)?;
        let mapped =
            self.client.arenas().iter().find(|a| a.info.name == arena && a.info.generation == generation).ok_or_else(
                || PyValueError::new_err(format!("this client hasn't mapped arena {arena:?} at {generation}")),
            )?;
        let writer = mapped.arena.carved_ring_writer(&ring).map_err(crate::to_py)?;
        Ok(crate::ring::PyRingWriter::new(writer))
    }

    /// The names of the arenas the broker handed over, in Welcome order.
    #[getter]
    fn arena_names(&self) -> Vec<String> {
        self.client.arenas().iter().map(|a| a.info.name.clone()).collect()
    }

    /// The mapped arena with this name (already mapped; this doesn't touch the socket). KeyError if there's none.
    fn arena(&self, name: &str) -> PyResult<PyArena> {
        self.client
            .arena(name)
            .map(|a| PyArena { arena: a.clone() })
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.to_owned()))
    }

    /// Sends a heartbeat, with a worker's telemetry if given, and returns the broker's clock in UTC nanoseconds.
    /// An awaitable. The client already sends heartbeats on its own.
    #[pyo3(signature = (*, queue_depth = None, models_resident = None, ram_bytes = 0, vram_bytes = 0))]
    fn heartbeat<'py>(
        &self,
        py: Python<'py>,
        queue_depth: Option<u32>,
        models_resident: Option<Vec<String>>,
        ram_bytes: u64,
        vram_bytes: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let telemetry =
            (queue_depth.is_some() || models_resident.is_some() || ram_bytes > 0 || vram_bytes > 0).then(|| {
                Telemetry {
                    queue_depth: queue_depth.unwrap_or(0),
                    models_resident: models_resident.unwrap_or_default(),
                    ram_bytes,
                    vram_bytes,
                }
            });
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move { client.heartbeat(telemetry.as_ref()).await.map_err(to_py) })
    }

    /// Registers a stream by name. Give `slab_class` for a stream of slots, or `ring_stride_bytes` (and optionally
    /// `ring_seconds`) for a ring. `anchor` is a sensor's clock anchor: (boot id UUID string, monotonic ns, UTC ns).
    /// `format` and `layer` are the schema's names (`"RawBytes"`, `"Layer1Raw"`). Re-registering a name with the same
    /// kind, layer, format, and shape returns the same id; a different one raises BrokerError("Conflict", ...).
    /// Returns {"stream_id": int, "lease": dict | None}. An awaitable.
    #[pyo3(signature = (
        name, *, kind, format, shape_name, shape_version = 1, nominal_rate_hz = 0.0, slab_class = None,
        ring_stride_bytes = None, ring_seconds = 0.0, anchor = None, layer = "Layer1Raw",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn register_stream<'py>(
        &self,
        py: Python<'py>,
        name: String,
        kind: &str,
        format: &str,
        shape_name: String,
        shape_version: u32,
        nominal_rate_hz: f32,
        slab_class: Option<u32>,
        ring_stride_bytes: Option<u32>,
        ring_seconds: f32,
        anchor: Option<(String, u64, u64)>,
        layer: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let storage = match (slab_class, ring_stride_bytes) {
            (Some(class), None) => StreamStorage::Slab {
                class: u16::try_from(class)
                    .map_err(|_| PyValueError::new_err(format!("slab_class must be at most 65535, not {class}")))?,
            },
            (None, Some(stride_bytes)) => StreamStorage::Ring { stride_bytes, seconds: ring_seconds },
            _ => return Err(PyValueError::new_err("give exactly one of slab_class or ring_stride_bytes")),
        };
        let anchor = anchor
            .map(|(boot_id, monotonic_ns, utc_ns)| {
                Ok::<_, PyErr>(fb::ClockAnchor::new(
                    &fb::Uuid::new(&crate::wire::uuid_from_str(&boot_id)?),
                    monotonic_ns,
                    utc_ns,
                ))
            })
            .transpose()?;
        let spec = StreamSpec {
            name,
            kind: parse_kind(kind)?,
            format: crate::wire::format_from_name(format)?,
            shape_name,
            shape_version,
            nominal_rate_hz,
            storage,
            anchor,
            layer: crate::wire::layer_from_name(layer)?,
        };
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let registered = client.register_stream(&spec).await.map_err(to_py)?;
            crate::attach(|py| registered_dict(py, &registered)).await
        })
    }

    /// Registers this session as a worker. `capabilities` are dicts with the schema's Capability fields (`produces`
    /// and `compute_domain` required). Register the output streams first. Returns {"accepted": [{"index", "id",
    /// "runnable"}], "refused_reasons": [...]}. `keeps_copies` says the worker keeps its own copies of payloads, so
    /// purges send it a PurgeNotice and wait for its PurgeAck. An awaitable.
    #[pyo3(signature = (worker_id, capabilities, *, version_hash = String::new(), device_index = None, keeps_copies = false))]
    fn register_worker<'py>(
        &self,
        py: Python<'py>,
        worker_id: String,
        capabilities: Vec<Bound<'py, PyDict>>,
        version_hash: String,
        device_index: Option<u8>,
        keeps_copies: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let capabilities = capabilities.iter().map(crate::worker::capability).collect::<PyResult<Vec<_>>>()?;
        let spec = WorkerSpec { worker_id, version_hash, capabilities, device_index, keeps_copies };
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let r = client.register_worker(&spec).await.map_err(to_py)?;
            crate::attach(|py| {
                let d = PyDict::new(py);
                let accepted = PyList::empty(py);
                for a in &r.accepted {
                    let ad = PyDict::new(py);
                    ad.set_item("index", a.index)?;
                    ad.set_item("id", a.id)?;
                    ad.set_item("runnable", a.runnable)?;
                    accepted.append(ad)?;
                }
                d.set_item("accepted", accepted)?;
                d.set_item("refused_reasons", &r.refused_reasons)?;
                d.into_py_any(py)
            })
            .await
        })
    }

    /// Asks for a lease for work this worker starts itself, covering `range` (UTC ns, half open) for capability
    /// `capability_id` (from register_worker). Returns a WorkLease, or raises LeaseDenied(retry_after_ms). An
    /// awaitable.
    #[pyo3(signature = (
        capability_id, *, stage, range, band = "P3Background", compute_domain = "CPU", output_bytes = 0,
        est_duration_ms = 0, pin_parents = Vec::new(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn quote<'py>(
        &self,
        py: Python<'py>,
        capability_id: u32,
        stage: String,
        range: (u64, u64),
        band: &str,
        compute_domain: &str,
        output_bytes: u64,
        est_duration_ms: u32,
        pin_parents: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let spec = QuoteSpec {
            capability_id,
            stage,
            band: crate::worker::band(band)?,
            compute_domain: crate::worker::domain(compute_domain)?,
            output_bytes,
            est_duration_ms,
            pin_parents: pin_parents.iter().map(|s| crate::worker::uuid_v7(s)).collect::<PyResult<_>>()?,
            range,
        };
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            match client.quote(&spec).await.map_err(to_py)? {
                Quote::Granted(lease) => Ok(crate::worker::PyWorkLease::new(lease)),
                Quote::Denied { retry_after_ms } => Err(crate::worker::LeaseDenied::new_err(retry_after_ms)),
            }
        })
    }

    /// What `stream` holds over [start_ns, end_ns) (UTC ns): {"items": [{"header", "locator"}], "complete",
    /// "truncated", "missing", "tail", "next_after"}. With `all` (the default), every item, paged by the broker's cursor and
    /// sorted by start time (missing and tail then empty); otherwise one time query of at most `max_items` (0 for
    /// the broker's limit). `subjects` and `ids` (UUID strings) filter. Only what exists (HitOnly). An awaitable.
    /// `truncated` means an item or inline-byte limit cut the page; `complete` can also be false for time gaps.
    /// A drained `all=True` result has `truncated=False`; single time queries have no continuation cursor.
    #[pyo3(signature = (
        stream, start_ns, end_ns, *, all = true, max_items = 0, band = "P2Nearline", subjects = Vec::new(),
        ids = Vec::new(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn query<'py>(
        &self,
        py: Python<'py>,
        stream: String,
        start_ns: u64,
        end_ns: u64,
        all: bool,
        max_items: u32,
        band: &str,
        subjects: Vec<String>,
        ids: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::subscribe::query(
            py,
            Arc::clone(&self.client),
            stream,
            start_ns,
            end_ns,
            all,
            max_items,
            band,
            subjects,
            ids,
        )
    }

    /// Current bounded chunk snapshot: its frames and locations. An awaitable.
    fn get_chunk<'py>(&self, py: Python<'py>, chunk_id: &str) -> PyResult<Bound<'py, PyAny>> {
        let id = crate::worker::uuid_v7(chunk_id)?;
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let info = client.get_chunk(id).await.map_err(to_py)?;
            if info.frames.len() > 4096 || info.locations.len() > 16 {
                return Err(PyValueError::new_err("chunk snapshot exceeds bounds"));
            }
            crate::attach(move |py| {
                let d = PyDict::new(py);
                d.set_item("chunk_id", uuid_string(info.chunk_id))?;
                d.set_item("seq", info.seq)?;
                d.set_item("frames", info.frames.iter().map(|f| (f.start_ns, f.cut)).collect::<Vec<_>>())?;
                let locations = PyList::empty(py);
                for location in &info.locations {
                    locations.append(crate::worker::locator_dict(py, location)?)?;
                }
                d.set_item("locations", locations)?;
                d.into_py_any(py)
            })
            .await
        })
    }

    /// Every item over [start_ns, end_ns), in the same order as query(all=True). Inline items become
    /// (payload_id, start_ns, end_ns, bytes); all other locators, including None, remain ordinary item dicts.
    /// Drains every page or raises; max_items limits each page, not the total. Does not decode JSON or read slots/files.
    #[pyo3(signature = (
        stream, start_ns, end_ns, *, max_items = 0, band = "P2Nearline", subjects = Vec::new(), ids = Vec::new(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn query_inline_records<'py>(
        &self,
        py: Python<'py>,
        stream: String,
        start_ns: u64,
        end_ns: u64,
        max_items: u32,
        band: &str,
        subjects: Vec<String>,
        ids: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::subscribe::query_inline_records(
            py,
            Arc::clone(&self.client),
            stream,
            start_ns,
            end_ns,
            max_items,
            band,
            subjects,
            ids,
        )
    }

    /// Subscribes to `streams` (names). `start` is "Live" (from the next commit), "Time" (from `start_ns`, UTC ns,
    /// catching up first), or "Saved" (after the position saved under `durable_name`, which it needs). `pinned`
    /// asks for each payload's locator as well as its header; `subjects` (UUID strings) and `shape_name` filter.
    /// Returns a Subscription. An awaitable.
    #[pyo3(signature = (
        streams, *, start = "Live", start_ns = 0, durable_name = None, pinned = false, subjects = Vec::new(),
        shape_name = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn subscribe<'py>(
        &self,
        py: Python<'py>,
        streams: Vec<String>,
        start: &str,
        start_ns: u64,
        durable_name: Option<String>,
        pinned: bool,
        subjects: Vec<String>,
        shape_name: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::subscribe::subscribe(
            py,
            Arc::clone(&self.client),
            streams,
            start,
            start_ns,
            durable_name,
            pinned,
            subjects,
            shape_name,
        )
    }

    /// What `stream` holds over [start_ns, end_ns) (UTC ns), as `query` with all=False, and the quote for what's
    /// missing (QuoteIfMissing): the result dict plus "quote", a dict ({"promise_id", "status", "options",
    /// "shares_in_flight", "expires_at_ns"}; option 0 is the broker's pick) or None when nothing is missing or
    /// nothing missing can be made. Commit it with commit_promise before it expires; its outputs then come from
    /// next_event as Fulfilled events (the last with final_part), or a PromiseFailed. An awaitable.
    #[pyo3(signature = (stream, start_ns, end_ns, *, band = "P2Nearline", latency_budget_ms = 0, max_items = 0))]
    #[allow(clippy::too_many_arguments)]
    fn query_quoted<'py>(
        &self,
        py: Python<'py>,
        stream: String,
        start_ns: u64,
        end_ns: u64,
        band: &str,
        latency_budget_ms: u32,
        max_items: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        if end_ns <= start_ns {
            return Err(pyo3::exceptions::PyValueError::new_err("end_ns must be after start_ns"));
        }
        let mut spec = continuum_client::QuerySpec::hit_only(stream, (start_ns, end_ns));
        spec.mode = continuum_core::schema::continuum::QueryMode::QuoteIfMissing;
        spec.band = crate::worker::band(band)?;
        spec.latency_budget_ms = latency_budget_ms;
        spec.max_items = max_items;
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let (result, quote) = client.query_quoted(&spec).await.map_err(to_py)?;
            crate::attach(move |py| {
                let d = crate::subscribe::query_result_dict(py, result)?;
                let q = quote.as_ref().map(|q| crate::promise::quote_dict(py, q)).transpose()?;
                d.bind(py).set_item("quote", q)?;
                Ok(d)
            })
            .await
        })
    }

    /// Commits a quoted promise: runs option `option` of its quote (0, the broker's pick) at `band`. Returns a
    /// Promise (`id`, `life`; `await promise.fulfilled()` for its outputs). NotFound for a promise this session
    /// wasn't quoted or whose quote lapsed. An awaitable.
    #[pyo3(signature = (promise_id, option = 0, band = "P2Nearline"))]
    fn commit_promise<'py>(
        &self,
        py: Python<'py>,
        promise_id: &str,
        option: u32,
        band: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let id = crate::worker::uuid_v7(promise_id)?;
        let band = crate::worker::band(band)?;
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let (accepted, promise) = client.commit_promise(id, option, band).await.map_err(to_py)?;
            Ok(crate::promise::PyPromise::new(accepted, promise))
        })
    }

    /// Calls a promise off: its jobs are revoked and nothing more comes. Returns {"promise_id", "life"}. NotFound
    /// for one that's gone. An awaitable.
    fn cancel_promise<'py>(&self, py: Python<'py>, promise_id: &str) -> PyResult<Bound<'py, PyAny>> {
        let id = crate::worker::uuid_v7(promise_id)?;
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let a = client.cancel_promise(id).await.map_err(to_py)?;
            crate::attach(move |py| crate::promise::accepted_dict(py, a)).await
        })
    }

    /// Takes on a dispatched job: `job` is the dict decode_event gave for its DispatchJob. Returns a WorkLease to
    /// commit or abort like a quoted one. It sets job["lease"] to None, so calling it again raises instead of making
    /// a second handle (whose collection would release the lease under the first). ValueError if the job has no
    /// output lease (none came, or it was taken).
    fn job_lease(&self, job: &Bound<'_, PyDict>) -> PyResult<crate::worker::PyWorkLease> {
        let lease = job.get_item("lease")?.filter(|l| !l.is_none()).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("the job has no output lease (none came, or it was taken)")
        })?;
        let lease = crate::worker::job_lease_from(lease.cast::<PyDict>()?)?;
        let work = self.client.lease_for(lease).map_err(to_py)?;
        job.set_item("lease", job.py().None())?;
        Ok(crate::worker::PyWorkLease::new(work))
    }

    /// Turns down a dispatched job, with why (JobRejected); the broker replans its promises. One way.
    fn reject_job(&self, job_id: &str, reason: &str) -> PyResult<()> {
        let id = crate::worker::uuid_v7(job_id)?;
        self.client.reject_job(id, reason).map_err(to_py)
    }

    /// Says this worker dropped its copies of a PurgeNotice's payloads (PurgeAck). One way.
    fn ack_purge(&self, purge_id: &str) -> PyResult<()> {
        let id = crate::worker::uuid_v7(purge_id)?;
        self.client.ack_purge(id).map_err(to_py)
    }

    /// Sets the voice turn ("Idle", "Listening", "Thinking", "Speaking"): the broker preempts what it outranks and
    /// tells every session. For the voice service. One way; not awaitable.
    fn set_system_state(&self, voice_turn: &str) -> PyResult<()> {
        self.client.set_system_state(crate::worker::voice(voice_turn)?).map_err(to_py)
    }

    /// Tells the broker a payload is written: `header` is its encoded header (`encode_header`), and its bytes are
    /// `inline` (up to 64 KiB, stored in the catalog), or in a shared memory slot (`shm`), or nowhere (a header-only
    /// payload). One way: nothing comes back for a notice the broker accepts; one it refuses arrives from
    /// `next_event()` as ("IngestRejected", envelope), which `decode_ingest_rejected` reads. Returns once queued (not
    /// awaitable): a notice queued just before a disconnect is lost without a rejection.
    #[pyo3(signature = (header, *, inline = None, shm = None))]
    fn ingest(
        &self,
        py: Python<'_>,
        header: &Bound<'_, PyAny>,
        inline: Option<Vec<u8>>,
        shm: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        let bytes = pyo3::buffer::PyBuffer::<u8>::get(header)?.to_vec(py)?;
        let header =
            continuum_core::HeaderBuf::new(bytes).map_err(|e| PyValueError::new_err(format!("header: {e}")))?;
        let at = match (inline, shm) {
            (Some(_), Some(_)) => return Err(PyValueError::new_err("give at most one of inline and shm")),
            (Some(bytes), None) => {
                if bytes.len() > continuum_core::MAX_INLINE_BYTES {
                    return Err(PyValueError::new_err(format!(
                        "inline payloads are at most {} bytes, not {}",
                        continuum_core::MAX_INLINE_BYTES,
                        bytes.len()
                    )));
                }
                Some(continuum_core::StorageLocator::Inline { bytes })
            }
            (None, Some(d)) => Some(crate::wire::shm_location(d)?),
            (None, None) => None,
        };
        self.client.ingest(&header, at.as_ref()).map_err(to_py)
    }

    /// As ingest, but waits for the broker's answer and returns the payload's commit sequence (its position in
    /// commit order). Sending the same header again returns the same sequence; a different header under the same
    /// id raises BrokerError (Conflict). `dependencies` (minor 31, #1535): (stream_id, start_ns, end_ns, revision)
    /// stretches the payload was made from, each with the stream's invalidation_revision read before checking what was
    /// used; stored with it in one commit, refused (Conflict) if a revision moved, and a later purge removing anything
    /// on such a stream inside the stretch takes the payload too. An awaitable.
    #[pyo3(signature = (header, *, inline = None, dependencies = Vec::new()))]
    fn ingest_acked<'py>(
        &self,
        py: Python<'py>,
        header: &Bound<'py, PyAny>,
        inline: Option<Vec<u8>>,
        dependencies: Vec<(u32, u64, u64, u64)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let dependencies: Vec<continuum_client::Dependency> = dependencies
            .into_iter()
            .map(|(stream_id, start_ns, end_ns, revision)| {
                if end_ns <= start_ns {
                    return Err(PyValueError::new_err(format!("dependency range [{start_ns}, {end_ns}) is empty")));
                }
                Ok(continuum_client::Dependency { stream_id, start_ns, end_ns, revision })
            })
            .collect::<PyResult<_>>()?;
        let bytes = pyo3::buffer::PyBuffer::<u8>::get(header)?.to_vec(py)?;
        let header =
            continuum_core::HeaderBuf::new(bytes).map_err(|e| PyValueError::new_err(format!("header: {e}")))?;
        let at = match inline {
            Some(bytes) => Some(continuum_core::StorageLocator::inline(bytes).ok_or_else(|| {
                PyValueError::new_err(format!("inline payloads are at most {} bytes", continuum_core::MAX_INLINE_BYTES))
            })?),
            None => None,
        };
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            client.ingest_dependent(&header, at.as_ref(), &dependencies).await.map_err(to_py)
        })
    }

    /// One page of `stream`'s payloads in commit order after `after_seq` (0 for the beginning), that the filters
    /// keep: `subjects` and `ids` (UUID strings), and `start_before` (UTC ns, 0 for none). At most `max_items` (0 for
    /// the broker's limit) after scanning at most `max_scan` rows (0 for its default). Returns query's dict; each
    /// item has its "commit_seq", and "next_after" is where the next page starts (0 when nothing remains). An
    /// awaitable.
    #[pyo3(signature = (
        stream, *, after_seq = 0, max_items = 0, subjects = Vec::new(), ids = Vec::new(), start_before = 0,
        max_scan = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn query_by_seq<'py>(
        &self,
        py: Python<'py>,
        stream: String,
        after_seq: u64,
        max_items: u32,
        subjects: Vec<String>,
        ids: Vec<String>,
        start_before: u64,
        max_scan: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let parse = |v: &[String]| -> PyResult<Vec<continuum_core::Uuid>> {
            v.iter().map(|s| crate::wire::uuid_from_str(s).map(continuum_core::Uuid)).collect()
        };
        let mut spec = continuum_client::QuerySpec::hit_only(stream, (0, 0));
        spec.by_seq = true;
        spec.after_seq = after_seq;
        spec.max_items = max_items;
        spec.subjects = parse(&subjects)?;
        spec.ids = parse(&ids)?;
        spec.start_before = start_before;
        spec.max_scan = max_scan;
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let r = client.query(&spec).await.map_err(to_py)?;
            crate::attach(|py| crate::subscribe::query_result_dict(py, r)).await
        })
    }

    /// Every registered stream, in id order: [{"stream_id", "name", "kind", "layer", "format", "shape_name",
    /// "shape_version", "live", "invalidation_revision" (int or None), "forget_revision" (int or None, minor 33)}]. An
    /// awaitable.
    fn list_streams<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let streams = client.list_streams().await.map_err(to_py)?;
            crate::attach(|py| {
                let out = PyList::empty(py);
                for s in streams {
                    let d = PyDict::new(py);
                    d.set_item("stream_id", s.id.0)?;
                    d.set_item("name", s.name)?;
                    d.set_item("kind", s.kind.variant_name().unwrap_or("Unknown"))?;
                    d.set_item("layer", s.layer.variant_name().unwrap_or("Unknown"))?;
                    d.set_item("format", s.format.variant_name().unwrap_or("Unknown"))?;
                    d.set_item("shape_name", s.shape_name)?;
                    d.set_item("shape_version", s.shape_version)?;
                    d.set_item("live", s.live)?;
                    d.set_item("invalidation_revision", s.invalidation_revision)?;
                    d.set_item("forget_revision", s.forget_revision)?;
                    out.append(d)?;
                }
                out.into_py_any(py)
            })
            .await
        })
    }

    /// Purges exactly one of: a range ([start_ns, end_ns) on `streams`, every stream when empty), an `entity` (UUID
    /// string), or `ids` (UUID strings: those payloads and everything derived from them). `reason` and the JSON
    /// `detail` are kept in the purge record for good, so the detail must not name a person. Returns {"purge_id",
    /// "removed": [(stream_id, tier, count)]} once the purge's steps have run. An awaitable.
    #[pyo3(signature = (
        *, start_ns = None, end_ns = None, streams = Vec::new(), entity = None, ids = Vec::new(),
        reason = "purge".to_owned(), detail = String::new(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn purge<'py>(
        &self,
        py: Python<'py>,
        start_ns: Option<u64>,
        end_ns: Option<u64>,
        streams: Vec<String>,
        entity: Option<String>,
        ids: Vec<String>,
        reason: String,
        detail: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Both bounds or neither: a lone bound beside an entity or ids would otherwise be dropped, and the purge would
        // take everything about them instead of being refused.
        if start_ns.is_some() != end_ns.is_some() {
            return Err(PyValueError::new_err(
                "purge exactly one of: start_ns and end_ns (with streams), entity, or ids; give both bounds or neither",
            ));
        }
        let target = match (start_ns.zip(end_ns), entity, ids.is_empty()) {
            (Some(range), None, true) => continuum_client::PurgeTarget::Range { range, streams },
            (None, Some(e), true) if streams.is_empty() => {
                continuum_client::PurgeTarget::Entity(continuum_core::Uuid(crate::wire::uuid_from_str(&e)?))
            }
            (None, None, false) if streams.is_empty() => continuum_client::PurgeTarget::Ids(
                ids.iter()
                    .map(|s| crate::wire::uuid_from_str(s).map(continuum_core::Uuid))
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            _ => {
                return Err(PyValueError::new_err(
                    "purge exactly one of: start_ns and end_ns (with streams), entity, or ids",
                ));
            }
        };
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let done = client.purge(&target, &reason, &detail).await.map_err(to_py)?;
            crate::attach(|py| {
                let d = PyDict::new(py);
                d.set_item("purge_id", uuid_string(done.purge_id))?;
                let removed: Vec<(u32, String, u64)> = done
                    .removed
                    .iter()
                    .map(|(s, t, n)| (s.0, t.variant_name().unwrap_or("Unknown").to_owned(), *n))
                    .collect();
                d.set_item("removed", removed)?;
                d.into_py_any(py)
            })
            .await
        })
    }

    /// Ages out `ids` (UUID strings): only those payloads go, none of what was derived from them, and only those
    /// committed at or before `committed_before_ns` by the commit clock (an import committed later stays, whatever
    /// its own time). Their spans are recorded as aged out. `reason` and the JSON `detail` are kept for good. Returns
    /// {"purge_id", "removed": [(stream_id, tier, count)]}. Needs broker schema minor 23. `aged_gap_ns` joins aged
    /// spans at most that far apart on a stream into one run (minor 29; 0 keeps each payload's span). An awaitable.
    #[pyo3(signature = (ids, committed_before_ns, *, reason = "aged out".to_owned(), detail = String::new(), aged_gap_ns = 0))]
    fn age_out<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<String>,
        committed_before_ns: u64,
        reason: String,
        detail: String,
        aged_gap_ns: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        if ids.is_empty() {
            return Err(PyValueError::new_err("age_out needs at least one id"));
        }
        let ids = ids
            .iter()
            .map(|s| crate::wire::uuid_from_str(s).map(continuum_core::Uuid))
            .collect::<PyResult<Vec<_>>>()?;
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let done = client.age_out(&ids, committed_before_ns, aged_gap_ns, &reason, &detail).await.map_err(to_py)?;
            crate::attach(|py| {
                let d = PyDict::new(py);
                d.set_item("purge_id", uuid_string(done.purge_id))?;
                let removed: Vec<(u32, String, u64)> = done
                    .removed
                    .iter()
                    .map(|(s, t, n)| (s.0, t.variant_name().unwrap_or("Unknown").to_owned(), *n))
                    .collect();
                d.set_item("removed", removed)?;
                d.into_py_any(py)
            })
            .await
        })
    }

    /// The spans of `stream` recorded as aged out within [start_ns, end_ns), at most `limit` (0 for the broker's
    /// default): [(start_ns, end_ns)]. An awaitable.
    #[pyo3(signature = (stream, start_ns, end_ns, *, limit = 0))]
    fn aged_out<'py>(
        &self,
        py: Python<'py>,
        stream: String,
        start_ns: u64,
        end_ns: u64,
        limit: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let spans = client.aged_out(&stream, (start_ns, end_ns), limit).await.map_err(to_py)?;
            crate::attach(|py| spans.into_py_any(py)).await
        })
    }

    /// Where the catalog file's bytes go, from one read snapshot on the broker's readers (slow on a large catalog):
    /// {"page_size", "file_bytes", "disk_bytes", "tables": [{"name", "rows", "height", "leaf_pages", "branch_pages",
    /// "stored_bytes", "metadata_bytes", "fragmented_bytes"}]}, largest table first. Read-only. Needs broker schema
    /// minor 30. An awaitable.
    fn catalog_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let stats = client.catalog_stats().await.map_err(to_py)?;
            crate::attach(|py| {
                let d = PyDict::new(py);
                d.set_item("page_size", stats.page_size)?;
                d.set_item("file_bytes", stats.file_bytes)?;
                d.set_item("disk_bytes", stats.disk_bytes)?;
                let tables = PyList::empty(py);
                for t in &stats.tables {
                    let row = PyDict::new(py);
                    row.set_item("name", &t.name)?;
                    row.set_item("rows", t.rows)?;
                    row.set_item("height", t.height)?;
                    row.set_item("leaf_pages", t.leaf_pages)?;
                    row.set_item("branch_pages", t.branch_pages)?;
                    row.set_item("stored_bytes", t.stored_bytes)?;
                    row.set_item("metadata_bytes", t.metadata_bytes)?;
                    row.set_item("fragmented_bytes", t.fragmented_bytes)?;
                    tables.append(row)?;
                }
                d.set_item("tables", tables)?;
                d.into_py_any(py)
            })
            .await
        })
    }

    /// The highest commit sequence known committed at or before `at_ns` (0 when none is that old). An awaitable.
    fn commit_seq_at<'py>(&self, py: Python<'py>, at_ns: u64) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let seq = client.commit_seq_at(at_ns).await.map_err(to_py)?;
            crate::attach(|py| seq.into_py_any(py)).await
        })
    }

    /// Which chunk holds each of `ids` (canonical UUID strings, at most 256), in the asked order: [{"id", "state"
    /// ("unknown", "payload", or "absorbed"), "stream_id", "seq", "chunk_id", "frame_index", "start_ns"}], with
    /// chunk_id None unless absorbed. What an analysis receipt names as its sources (#1394). An awaitable.
    fn resolve_frames<'py>(&self, py: Python<'py>, ids: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let ids = ids
            .iter()
            .map(|s| crate::wire::uuid_from_str(s).map(continuum_core::Uuid))
            .collect::<PyResult<Vec<_>>>()?;
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let frames = client.resolve_frames(&ids).await.map_err(to_py)?;
            crate::attach(|py| {
                let out = PyList::empty(py);
                for f in &frames {
                    out.append(resolved_frame_dict(py, f)?)?;
                }
                out.into_py_any(py)
            })
            .await
        })
    }

    /// Purge summaries, oldest first, after the purge `after` (from the start when None), at most `limit` (0 for the
    /// broker's default): [{"purge_id", "at_ns", "kind", "range", "stream_ids", "id_count", "reason", "detail",
    /// "removed", "finished", "completed_ns"}]. A summary never names a person. An awaitable.
    #[pyo3(signature = (*, after = None, limit = 0))]
    fn list_purges<'py>(&self, py: Python<'py>, after: Option<String>, limit: u32) -> PyResult<Bound<'py, PyAny>> {
        let after = after.map(|s| crate::wire::uuid_from_str(&s).map(continuum_core::Uuid)).transpose()?;
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let purges = client.list_purges(after, limit).await.map_err(to_py)?;
            crate::attach(|py| {
                let out = PyList::empty(py);
                for p in &purges {
                    out.append(purge_summary_dict(py, p)?)?;
                }
                out.into_py_any(py)
            })
            .await
        })
    }

    /// Records that a person was forgotten: `hash` is the name, hashed by the caller (32 bytes; the
    /// broker never sees the name), kept in the catalog's FORGOTTEN, which no purge removes. An awaitable.
    fn remember_forgotten<'py>(&self, py: Python<'py>, hash: Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        let hash = forgotten_hash(&hash)?;
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            client.remember_forgotten(hash).await.map_err(to_py)?;
            crate::attach(|py| Ok(py.None())).await
        })
    }

    /// Stores a durable worker's checkpoint in place of its last (#1547): `inputs` is each input stream's name and the
    /// invalidation revision the state was read under (sorted, unique, 1 to 64), `body` the encoded state (at most
    /// 8 MiB). Refused with a Conflict BrokerError when an input's revision has moved or it isn't registered. A purge
    /// that later advances one of them deletes it. Needs broker schema minor 32. An awaitable.
    #[pyo3(signature = (owner, contract, position, created_ns, inputs, body))]
    #[allow(clippy::too_many_arguments)]
    fn save_worker_state<'py>(
        &self,
        py: Python<'py>,
        owner: String,
        contract: String,
        position: u64,
        created_ns: u64,
        inputs: Vec<(String, u64)>,
        body: Vec<u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        let state = continuum_client::WorkerState { owner, contract, position, created_ns, inputs, body };
        crate::awaitable(py, async move {
            client.save_worker_state(&state).await.map_err(to_py)?;
            crate::attach(|py| Ok(py.None())).await
        })
    }

    /// A worker's checkpoint, if it has one whose inputs' revisions all still equal the stored ones: {"position",
    /// "created_ns", "inputs": [(stream, revision)], "body": bytes}, or None. Needs broker schema minor 32. An
    /// awaitable.
    fn load_worker_state<'py>(&self, py: Python<'py>, owner: String, contract: String) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let state = client.load_worker_state(&owner, &contract).await.map_err(to_py)?;
            crate::attach(|py| match state {
                None => Ok(py.None()),
                Some(s) => {
                    let d = PyDict::new(py);
                    d.set_item("position", s.position)?;
                    d.set_item("created_ns", s.created_ns)?;
                    d.set_item("inputs", s.inputs)?;
                    d.set_item("body", PyBytes::new(py, &s.body))?;
                    d.into_py_any(py)
                }
            })
            .await
        })
    }

    /// Deletes a worker's checkpoint; resolves to whether it had one. Needs broker schema minor 32. An awaitable.
    fn forget_worker_state<'py>(
        &self,
        py: Python<'py>,
        owner: String,
        contract: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let removed = client.forget_worker_state(&owner, &contract).await.map_err(to_py)?;
            crate::attach(|py| removed.into_py_any(py)).await
        })
    }

    /// Read-only lexicographic forgotten-hash page, with an exclusive 32-byte cursor.
    #[pyo3(signature = (after=None, limit=256))]
    fn list_forgotten<'py>(&self, py: Python<'py>, after: Option<Vec<u8>>, limit: u32) -> PyResult<Bound<'py, PyAny>> {
        let after = after.as_deref().map(forgotten_hash).transpose()?;
        if !(1..=1024).contains(&limit) {
            return Err(PyValueError::new_err("forgotten page limit must be 1..=1024"));
        }
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let page = client.list_forgotten(after, limit).await.map_err(to_py)?;
            crate::attach(|py| {
                let result = PyDict::new(py);
                let hashes = PyList::empty(py);
                for hash in page.hashes {
                    hashes.append(PyBytes::new(py, &hash))?;
                }
                result.set_item("hashes", hashes)?;
                result.set_item("next", page.next.map(|hash| PyBytes::new(py, &hash)))?;
                result.into_py_any(py)
            })
            .await
        })
    }

    /// Whether a forgotten name's hash (32 bytes) is in FORGOTTEN. An awaitable resolving to a bool.
    fn is_forgotten<'py>(&self, py: Python<'py>, hash: Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        let hash = forgotten_hash(&hash)?;
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let forgotten = client.is_forgotten(hash).await.map_err(to_py)?;
            crate::attach(|py| forgotten.into_py_any(py)).await
        })
    }

    /// Drops a keeper's name (a worker that registered with keeps_copies and was retired or renamed), so purges stop
    /// waiting for it. Resolves to how many purges were waiting on it. An awaitable.
    fn forget_keeper<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let released = client.forget_keeper(&name).await.map_err(to_py)?;
            crate::attach(|py| released.into_py_any(py)).await
        })
    }

    /// One purge's record: its summary's keys, plus "removed_by": [(stream_id, tier, count)], "removed_frames", and
    /// "partial_chunks". An awaitable.
    fn get_purge<'py>(&self, py: Python<'py>, purge_id: String) -> PyResult<Bound<'py, PyAny>> {
        let id = continuum_core::Uuid(crate::wire::uuid_from_str(&purge_id)?);
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let r = client.get_purge(id).await.map_err(to_py)?;
            crate::attach(|py| {
                let d = purge_summary_dict(py, &r.summary)?;
                let removed: Vec<(u32, String, u64)> = r
                    .removed
                    .iter()
                    .map(|(s, t, n)| (s.0, t.variant_name().unwrap_or("Unknown").to_owned(), *n))
                    .collect();
                d.set_item("removed_by", removed)?;
                d.set_item("removed_frames", r.removed_frames)?;
                d.set_item("partial_chunks", r.partial_chunks)?;
                d.into_py_any(py)
            })
            .await
        })
    }

    /// Purges `[start_ns, end_ns)` on the named streams, or on every stream when `streams` is empty, and waits for
    /// the broker's answer: {"purge_id", "removed": [(stream_id, tier, count)]}, `tier` being the StorageTier's name.
    /// Workers that keep copies are told after; the purge is complete once they acknowledge. An awaitable.
    #[pyo3(signature = (start_ns, end_ns, *, streams = Vec::new(), reason = String::from("purge")))]
    fn purge_range<'py>(
        &self,
        py: Python<'py>,
        start_ns: u64,
        end_ns: u64,
        streams: Vec<String>,
        reason: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let done = client.purge_range((start_ns, end_ns), &streams, &reason).await.map_err(to_py)?;
            crate::attach(|py| {
                let d = PyDict::new(py);
                d.set_item("purge_id", uuid_string(done.purge_id))?;
                let removed: Vec<(u32, String, u64)> = done
                    .removed
                    .iter()
                    .map(|(s, t, n)| (s.0, t.variant_name().unwrap_or("Unknown").to_owned(), *n))
                    .collect();
                d.set_item("removed", removed)?;
                d.into_py_any(py)
            })
            .await
        })
    }

    /// Commits a chunk file as a Tier 1 (or `tier = 2`) chunk absorbing `absorbed` (payload UUID strings, in frame
    /// order): `header` is the chunk's own header (encode_header), `uri` and `length` the file, and `frames` its
    /// index as (start_ns, offset, length, cut). Only a registered worker commits chunks. Returns {"chunk_id",
    /// "commit_seq", "duplicate", "hidden"}; a repeat answers duplicate. An awaitable.
    #[pyo3(signature = (header, uri, length, frames, absorbed, *, tier = 1))]
    #[expect(clippy::too_many_arguments, reason = "the chunk's parts, as Python passes them")]
    fn commit_chunk<'py>(
        &self,
        py: Python<'py>,
        header: &Bound<'py, PyAny>,
        uri: String,
        length: u64,
        frames: Vec<(u64, u64, u64, bool)>,
        absorbed: Vec<String>,
        tier: u8,
    ) -> PyResult<Bound<'py, PyAny>> {
        let bytes = pyo3::buffer::PyBuffer::<u8>::get(header)?.to_vec(py)?;
        let header =
            continuum_core::HeaderBuf::new(bytes).map_err(|e| PyValueError::new_err(format!("header: {e}")))?;
        let tier = match tier {
            1 => continuum_core::StorageTier::Tier1Warm,
            2 => continuum_core::StorageTier::Tier2Cold,
            other => return Err(PyValueError::new_err(format!("tier must be 1 or 2, not {other}"))),
        };
        let chunk = continuum_client::NewChunk {
            header,
            location: continuum_core::StorageLocator::Chunk { tier, uri, offset: 0, length },
            frames: frames
                .into_iter()
                .map(|(start_ns, offset, length, cut)| continuum_client::ChunkFrame { start_ns, offset, length, cut })
                .collect(),
        };
        let absorbed = absorbed
            .iter()
            .map(|s| crate::wire::uuid_from_str(s).map(continuum_core::Uuid))
            .collect::<PyResult<Vec<_>>>()?;
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let c = client.commit_chunk(&chunk, &absorbed).await.map_err(to_py)?;
            crate::attach(|py| {
                let d = PyDict::new(py);
                d.set_item("chunk_id", uuid_string(c.chunk_id))?;
                d.set_item("commit_seq", c.commit_seq)?;
                d.set_item("duplicate", c.duplicate)?;
                d.set_item("hidden", c.hidden)?;
                d.into_py_any(py)
            })
            .await
        })
    }

    /// Registers one v1 FLAC file (starting at `start_ns`) in place as a payload and a Tier 1 chunk on `stream_id`,
    /// a stream this session registered for the v1 archive (continuum_tier::v1audio). The session must have
    /// registered a worker. Idempotent. Returns a dict as `register_v1_index`'s entries. An awaitable.
    fn register_v1_flac<'py>(
        &self,
        py: Python<'py>,
        stream_id: u32,
        path: std::path::PathBuf,
        start_ns: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let r = continuum_tier::v1audio::register_v1_flac(
                &client,
                continuum_core::StreamId(stream_id),
                &path,
                start_ns,
            )
            .await
            .map_err(v1_to_py)?;
            crate::attach(|py| v1_dict(py, &path, &r)?.into_py_any(py)).await
        })
    }

    /// Registers every file a v1 device directory's index.jsonl lists, on stream `stream_name` (registered here),
    /// in place. The session must have registered a worker. Idempotent. Returns one dict per index line: {"path",
    /// "registered": True, "payload_id", "chunk_id", "start_ns", "end_ns", "duplicate"} or {"path", "registered":
    /// False, "skipped": reason}. An awaitable.
    fn register_v1_index<'py>(
        &self,
        py: Python<'py>,
        stream_name: String,
        device_dir: std::path::PathBuf,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let all = continuum_tier::v1audio::register_v1_index(&client, &stream_name, &device_dir)
                .await
                .map_err(v1_to_py)?;
            crate::attach(|py| {
                let list = PyList::empty(py);
                for (path, r) in &all {
                    list.append(v1_dict(py, path, r)?)?;
                }
                list.into_py_any(py)
            })
            .await
        })
    }

    /// Registers every recording of camera `cam` that v1's `video/index.jsonl` (in `video_dir`) lists, in place, on
    /// `<cam>.video_v1` (continuum_tier::v1video): one payload per GOP and one chunk per file. The session must have
    /// registered a worker. Idempotent. Returns one dict per file, as register_v1_index's. An awaitable.
    fn register_v1_video_index<'py>(
        &self,
        py: Python<'py>,
        cam: String,
        video_dir: std::path::PathBuf,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let all =
                continuum_tier::v1video::register_v1_video_index(&client, &cam, &video_dir).await.map_err(v1_to_py)?;
            crate::attach(|py| {
                let list = PyList::empty(py);
                for (path, r) in &all {
                    list.append(v1_dict(py, path, r)?)?;
                }
                list.into_py_any(py)
            })
            .await
        })
    }

    /// The next message the broker sent on its own, as (payload type name, envelope bytes), or None once the
    /// connection is closed. An awaitable. File descriptors that come with an event are closed here for now: the
    /// broker sends none after Welcome yet, and an announced arena will be mapped into the client when it does.
    fn next_event<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            let event = client.next_event().await;
            crate::attach(|py| -> PyResult<Py<PyAny>> {
                match event {
                    Some(m) => {
                        let kind = m.kind().variant_name().unwrap_or("unknown").to_owned();
                        (kind, PyBytes::new(py, m.as_bytes())).into_py_any(py)
                    }
                    None => Ok(py.None()),
                }
            })
            .await
        })
    }

    /// Says Goodbye and closes the session; the broker drops its pins and leases. An awaitable.
    #[pyo3(signature = (reason = String::from("closed")))]
    fn close<'py>(&self, py: Python<'py>, reason: String) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move { client.close(&reason).await.map_err(to_py) })
    }

    fn __aenter__<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.unbind();
        crate::awaitable(py, async move { Ok(this) })
    }

    #[pyo3(signature = (*_exc))]
    fn __aexit__<'py>(&self, py: Python<'py>, _exc: &Bound<'py, pyo3::types::PyTuple>) -> PyResult<Bound<'py, PyAny>> {
        let client = Arc::clone(&self.client);
        crate::awaitable(py, async move {
            // Returns None, never True: an exception raised in the block always propagates.
            match client.close("closed").await {
                Ok(()) | Err(ClientError::Closed) => Ok(()),
                Err(e) => Err(to_py(e)),
            }
        })
    }
}

fn v1_to_py(err: continuum_tier::v1audio::V1Error) -> PyErr {
    match err {
        continuum_tier::v1audio::V1Error::Client(e) => to_py(e),
        other => ContinuumError::new_err(other.to_string()),
    }
}

fn v1_dict<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    r: &continuum_tier::v1audio::V1Registered,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("path", path.to_string_lossy())?;
    match r {
        continuum_tier::v1audio::V1Registered::Registered { payload_id, chunk_id, span, duplicate } => {
            d.set_item("registered", true)?;
            d.set_item("payload_id", uuid_string(*payload_id))?;
            d.set_item("chunk_id", uuid_string(*chunk_id))?;
            d.set_item("start_ns", span.0)?;
            d.set_item("end_ns", span.1)?;
            d.set_item("duplicate", *duplicate)?;
        }
        continuum_tier::v1audio::V1Registered::Skipped(why) => {
            d.set_item("registered", false)?;
            d.set_item("skipped", why)?;
        }
    }
    Ok(d)
}

/// Decodes FLAC bytes (a v1 archive file, as hydrated) to (rate, channels, samples), the samples interleaved f32
/// little-endian bytes scaled as capture's are (numpy.frombuffer(samples, "<f4")).
/// A chunk's live records whose time overlaps `[start_ns, end_ns)`, read from the local file `uri` (a query item's
/// Chunk locator, which names the whole chunk): [(payload_id, start_ns, end_ns, bytes)], each in the stream's native
/// form, records a purge cut left out. None when the file is gone (a purge took the chunk). A v1 FLAC file
/// registered in place is one record, the whole file, with zero times and the nil id. Raises ValueError for a uri
/// that isn't a local file or a record that won't decode, and OSError for a corrupt chunk or a failed read. The file
/// is open only during the call; the GIL is released while it reads.
/// One record: (payload_id, start_ns, end_ns, bytes, optional (rate_hz, channels, ring_stride)).
type ChunkRecord<'py> = (String, u64, u64, Bound<'py, PyBytes>, Option<(u32, u16, u32)>);

#[pyfunction]
pub(crate) fn read_chunk_records<'py>(
    py: Python<'py>,
    uri: &str,
    start_ns: u64,
    end_ns: u64,
) -> PyResult<Option<Vec<ChunkRecord<'py>>>> {
    let path = uri.strip_prefix("file://").unwrap_or(uri);
    if path.contains("://") || !std::path::Path::new(path).is_absolute() {
        return Err(PyValueError::new_err(format!("{uri}: not a local file (only local chunk files can be read)")));
    }
    let path = std::path::PathBuf::from(path);
    let read = py.detach(|| continuum_tier::read_chunk(&path, start_ns, end_ns)).map_err(|e| match e {
        continuum_tier::TierError::Codec(_) => PyValueError::new_err(format!("{}: {e}", path.display())),
        _ => pyo3::exceptions::PyOSError::new_err(format!("{}: {e}", path.display())),
    })?;
    Ok(read.map(|records| {
        records
            .into_iter()
            .map(|h| {
                (
                    uuid_string(h.record.id),
                    h.record.start_ns,
                    h.record.end_ns,
                    PyBytes::new(py, &h.bytes),
                    h.pcm.map(|p| (p.sample_rate_hz, p.layout.channels, p.layout.stride)),
                )
            })
            .collect()
    }))
}

#[pyfunction]
pub(crate) fn decode_flac<'py>(py: Python<'py>, flac: &Bound<'py, PyAny>) -> PyResult<(u32, u16, Bound<'py, PyBytes>)> {
    let bytes = pyo3::buffer::PyBuffer::<u8>::get(flac)?.to_vec(py)?;
    // Decoding minutes of audio takes a while: other Python threads run meanwhile.
    let (rate, channels, samples) =
        py.detach(|| continuum_tier::v1audio::decode_flac(&bytes)).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out = PyBytes::new_with(py, samples.len() * 4, |buf| {
        for (dst, x) in buf.chunks_exact_mut(4).zip(&samples) {
            dst.copy_from_slice(&x.to_le_bytes());
        }
        Ok(())
    })?;
    Ok((rate, channels, out))
}

fn resolved_frame_dict<'py>(py: Python<'py>, f: &continuum_client::ResolvedFrame) -> PyResult<Bound<'py, PyDict>> {
    use continuum_client::FrameSource;
    let d = PyDict::new(py);
    d.set_item("id", uuid_string(f.id))?;
    let (state, stream, seq, chunk, frame_index, start_ns) = match f.source {
        FrameSource::Unknown => ("unknown", 0, 0, None, 0, 0),
        FrameSource::Payload { stream, seq } => ("payload", stream.0, seq, None, 0, 0),
        FrameSource::Absorbed { chunk, stream, seq, frame_index, start_ns } => {
            ("absorbed", stream.0, seq, Some(uuid_string(chunk)), frame_index, start_ns)
        }
    };
    d.set_item("state", state)?;
    d.set_item("stream_id", stream)?;
    d.set_item("seq", seq)?;
    d.set_item("chunk_id", chunk)?;
    d.set_item("frame_index", frame_index)?;
    d.set_item("start_ns", start_ns)?;
    Ok(d)
}

fn purge_summary_dict<'py>(py: Python<'py>, p: &continuum_client::PurgeSummary) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("purge_id", uuid_string(p.purge_id))?;
    d.set_item("at_ns", p.at_ns)?;
    let kind = match p.kind {
        continuum_client::PurgeKind::Range => "Range",
        continuum_client::PurgeKind::Entity => "Entity",
        continuum_client::PurgeKind::Ids => "Ids",
        continuum_client::PurgeKind::AgeOut => "AgeOut",
    };
    d.set_item("kind", kind)?;
    d.set_item("range", p.range)?;
    d.set_item("stream_ids", p.stream_ids.clone())?;
    d.set_item("id_count", p.id_count)?;
    d.set_item("reason", &p.reason)?;
    d.set_item("detail", &p.detail)?;
    d.set_item("removed", p.removed)?;
    d.set_item("finished", p.finished)?;
    d.set_item("completed_ns", p.completed_ns)?;
    Ok(d)
}

/// The access units of GOP `gop` (its place in the chunk's frame index) of the v1 recording at `path`, which starts
/// at `start_ns`, in decode order: a list of dicts {"bytes" (Annex B, the SPS and PPS on the keyframe), "pts_ns",
/// "end_ns", "dts_ns", "key"}. It reads only the file's moov and that GOP's bytes.
#[pyfunction]
pub(crate) fn v1_gop_units<'py>(
    py: Python<'py>,
    path: std::path::PathBuf,
    start_ns: u64,
    gop: usize,
) -> PyResult<Bound<'py, PyList>> {
    let units = py
        .detach(|| continuum_tier::v1video::gop_units(&path, start_ns, gop))
        .map_err(|e| ContinuumError::new_err(e.to_string()))?;
    let list = PyList::empty(py);
    for u in units {
        let d = PyDict::new(py);
        d.set_item("bytes", PyBytes::new(py, &u.bytes))?;
        d.set_item("pts_ns", u.pts_ns)?;
        d.set_item("end_ns", u.end_ns)?;
        d.set_item("dts_ns", u.dts_ns)?;
        d.set_item("key", u.key)?;
        list.append(d)?;
    }
    Ok(list)
}

/// A forgotten name's hash as the wire wants it: exactly 32 bytes.
fn forgotten_hash(bytes: &[u8]) -> PyResult<[u8; 32]> {
    <[u8; 32]>::try_from(bytes)
        .map_err(|_| PyValueError::new_err(format!("a forgotten name's hash is 32 bytes, got {}", bytes.len())))
}
