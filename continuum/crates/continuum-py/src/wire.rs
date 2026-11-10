//! The schema's payload types in Python: `PayloadHeader`, `Producer`, `EvidenceRef`, `ComputeCost`, and
//! `CaptureInfo` as plain dicts with the .fbs field names (typed as TypedDicts in continuum_py.pyi), and
//! `encode_header` / `decode_header` between those dicts and a header's FlatBuffer bytes (root `PayloadHeader`, the
//! bytes the catalog keeps and `IngestNotice.header_bytes` carries).
//!
//! At this boundary ids are canonical UUID strings, times are UTC nanoseconds, and enums are their schema names
//! (`"Layer1Raw"`, `"Tier0Hot"`, `"RawBytes"`), so a dict reads the same as the schema.

use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, Uuid};
use flatbuffers::FlatBufferBuilder;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

/// A canonical UUID string as its 16 bytes.
pub(crate) fn uuid_from_str(s: &str) -> PyResult<[u8; 16]> {
    let canonical = s.len() == 36
        && s.char_indices().all(|(i, c)| {
            if matches!(i, 8 | 13 | 18 | 23) { c == '-' } else { c.is_ascii_hexdigit() && !c.is_ascii_uppercase() }
        });
    if !canonical {
        return Err(PyValueError::new_err(format!("not a canonical UUID string (lowercase, hyphenated): {s:?}")));
    }
    let hex: String = s.chars().filter(|c| *c != '-').collect();
    let mut out = [0u8; 16];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("checked above");
    }
    Ok(out)
}

fn enum_from_name<T: Copy>(
    values: &[T],
    name_of: impl Fn(T) -> Option<&'static str>,
    what: &str,
    s: &str,
) -> PyResult<T> {
    values.iter().copied().find(|v| name_of(*v) == Some(s)).ok_or_else(|| {
        let names: Vec<_> = values.iter().filter_map(|v| name_of(*v)).collect();
        PyValueError::new_err(format!("{what} must be one of {names:?}, not {s:?}"))
    })
}

pub(crate) fn layer_from_name(s: &str) -> PyResult<fb::DomainLayer> {
    enum_from_name(fb::DomainLayer::ENUM_VALUES, |v| v.variant_name(), "layer", s)
}

pub(crate) fn format_from_name(s: &str) -> PyResult<fb::PayloadFormat> {
    enum_from_name(fb::PayloadFormat::ENUM_VALUES, |v| v.variant_name(), "format", s)
}

fn tier_from_name(s: &str) -> PyResult<fb::StorageTier> {
    enum_from_name(fb::StorageTier::ENUM_VALUES, |v| v.variant_name(), "source_tier", s)
}

/// A conversion error, naming the key it was for.
fn keyed(key: &str, e: PyErr) -> PyErr {
    // An int that doesn't fit the field (negative, or past u32/u64) is a bad value, not a bad type.
    let overflow = Python::attach(|py| e.is_instance_of::<pyo3::exceptions::PyOverflowError>(py));
    if overflow {
        PyValueError::new_err(format!("{key}: out of range: {e}"))
    } else {
        PyTypeError::new_err(format!("{key}: {e}"))
    }
}

/// A required key.
fn get<'py, T: for<'a> FromPyObject<'a, 'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    let v = d.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key.to_owned()))?;
    v.extract::<T>().map_err(|e| keyed(key, e.into()))
}

/// An optional key: missing or None gives None.
fn opt<'py, T: for<'a> FromPyObject<'a, 'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>> {
    match d.get_item(key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => v.extract::<T>().map(Some).map_err(|e| keyed(key, e.into())),
    }
}

/// A payload or entity id: a canonical UUIDv7 string (decision 8).
fn id_v7(key: &str, s: &str) -> PyResult<fb::Uuid> {
    let b = uuid_from_str(s)?;
    if b[6] >> 4 != 7 {
        return Err(PyValueError::new_err(format!("{key}: not a UUIDv7 (version {}): {s:?}", b[6] >> 4)));
    }
    Ok(fb::Uuid::new(&b))
}

fn ids(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<fb::Uuid>> {
    let strings: Vec<String> = opt(d, key)?.unwrap_or_default();
    strings.iter().map(|s| id_v7(key, s)).collect()
}

/// An entity id (a subject): any RFC 9562 UUID. A subject is an identifier, not a time-ordered key, so it needn't
/// be a v7: a person's id is one, and a voiceprint's or faceprint's is a v5 named from its ref.
fn entity_id(key: &str, s: &str) -> PyResult<fb::Uuid> {
    let b = uuid_from_str(s)?;
    if b[8] >> 6 != 0b10 || !(1..=8).contains(&(b[6] >> 4)) {
        return Err(PyValueError::new_err(format!("{key}: not an RFC 9562 UUID: {s:?}")));
    }
    Ok(fb::Uuid::new(&b))
}

/// The optional confidence, which the schema keeps as an f32: refuses NaN, infinities, and values past f32's range
/// rather than storing them as infinity.
fn confidence(header: &Bound<'_, PyDict>) -> PyResult<Option<f32>> {
    let Some(c) = opt::<f64>(header, "confidence")? else { return Ok(None) };
    if !c.is_finite() || c.abs() > f64::from(f32::MAX) {
        return Err(PyValueError::new_err(format!("confidence must be a finite f32, not {c}")));
    }
    Ok(Some(c as f32))
}

/// Builds a header's FlatBuffer bytes from a `PayloadHeader` dict.
///
/// Required: `id`, `stream_id`, `start_ns`, `end_ns`, `layer`, `format`, `shape_name`. Everything else is optional
/// and defaults like the schema (no producer, no parents, `Tier0Hot`, and so on).
#[pyfunction]
pub(crate) fn encode_header<'py>(py: Python<'py>, header: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyBytes>> {
    let id = id_v7("id", &get::<String>(header, "id")?)?;
    let start_ns: u64 = get(header, "start_ns")?;
    let end_ns: u64 = get(header, "end_ns")?;
    if end_ns < start_ns {
        return Err(PyValueError::new_err(format!("end_ns {end_ns} is before start_ns {start_ns}")));
    }
    let layer = layer_from_name(&get::<String>(header, "layer")?)?;
    let format = format_from_name(&get::<String>(header, "format")?)?;
    let source_tier = tier_from_name(&opt::<String>(header, "source_tier")?.unwrap_or_else(|| "Tier0Hot".into()))?;
    let parents = ids(header, "parents")?;
    let subjects: Vec<String> = opt(header, "subjects")?.unwrap_or_default();
    let subjects = subjects.iter().map(|s| entity_id("subjects", s)).collect::<PyResult<Vec<_>>>()?;

    let mut b = FlatBufferBuilder::new();
    let shape = b.create_string(&get::<String>(header, "shape_name")?);
    let producer = match opt::<Bound<'py, PyDict>>(header, "producer")? {
        None => None,
        Some(p) => {
            let mut s = |key: &str| -> PyResult<Option<flatbuffers::WIPOffset<&str>>> {
                Ok(opt::<String>(&p, key)?.map(|v| b.create_string(&v)))
            };
            let args = fb::ProducerArgs {
                producer_id: s("producer_id")?,
                model: s("model")?,
                version: s("version")?,
                version_hash: s("version_hash")?,
                prompt_id: s("prompt_id")?,
                prompt_hash: s("prompt_hash")?,
            };
            Some(fb::Producer::create(&mut b, &args))
        }
    };
    let evidence = match opt::<Vec<Bound<'py, PyDict>>>(header, "evidence")? {
        None => None,
        Some(refs) => {
            let refs = refs
                .iter()
                .map(|r| {
                    let payload_id = id_v7("evidence payload_id", &get::<String>(r, "payload_id")?)?;
                    let args = fb::EvidenceRefArgs {
                        payload_id: Some(&payload_id),
                        start_ns: opt(r, "start_ns")?,
                        end_ns: opt(r, "end_ns")?,
                    };
                    Ok(fb::EvidenceRef::create(&mut b, &args))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Some(b.create_vector(&refs))
        }
    };
    let compute = match opt::<Bound<'py, PyDict>>(header, "compute")? {
        None => None,
        Some(c) => {
            let device = opt::<String>(&c, "device")?.map(|d| b.create_string(&d));
            let args = fb::ComputeCostArgs {
                wall_time_us: opt(&c, "wall_time_us")?.unwrap_or(0),
                queue_wait_us: opt(&c, "queue_wait_us")?.unwrap_or(0),
                ram_bytes: opt(&c, "ram_bytes")?.unwrap_or(0),
                vram_bytes: opt(&c, "vram_bytes")?.unwrap_or(0),
                device,
                items: opt(&c, "items")?.unwrap_or(0),
            };
            Some(fb::ComputeCost::create(&mut b, &args))
        }
    };
    let capture = match opt::<Bound<'py, PyDict>>(header, "capture")? {
        None => None,
        Some(c) => {
            let args = fb::CaptureInfoArgs {
                capture_mono_ns: opt(&c, "capture_mono_ns")?.unwrap_or(0),
                device_ts: opt(&c, "device_ts")?.unwrap_or(0),
                arrival_ns: opt(&c, "arrival_ns")?.unwrap_or(0),
                stride: opt(&c, "stride")?.unwrap_or(0),
                width: opt(&c, "width")?.unwrap_or(0),
                height: opt(&c, "height")?.unwrap_or(0),
                flip: opt(&c, "flip")?.unwrap_or(false),
                device_seq: opt(&c, "device_seq")?.unwrap_or(0),
                sample_rate_hz: opt(&c, "sample_rate_hz")?.unwrap_or(0),
                capture_epoch: opt(&c, "capture_epoch")?.unwrap_or(0),
            };
            Some(fb::CaptureInfo::create(&mut b, &args))
        }
    };
    let parents = (!parents.is_empty()).then(|| b.create_vector(&parents));
    let subjects = (!subjects.is_empty()).then(|| b.create_vector(&subjects));
    let h = fb::PayloadHeader::create(
        &mut b,
        &fb::PayloadHeaderArgs {
            id: Some(&id),
            stream_id: get(header, "stream_id")?,
            start_ns,
            end_ns,
            seq: opt(header, "seq")?.unwrap_or(0),
            layer,
            format,
            shape_name: Some(shape),
            shape_version: opt(header, "shape_version")?.unwrap_or(0),
            producer,
            parents,
            evidence,
            source_tier,
            compute,
            confidence: confidence(header)?,
            subjects,
            capture,
        },
    );
    fb::finish_payload_header_buffer(&mut b, h);
    // The same checks the broker and catalog make (an id, end not before start), so a header that encodes here is
    // one they accept.
    let checked = HeaderBuf::new(b.finished_data().to_vec()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, checked.as_bytes()))
}

fn id_list<'py>(py: Python<'py>, v: Option<flatbuffers::Vector<'_, fb::Uuid>>) -> PyResult<Bound<'py, PyList>> {
    let strings: Vec<String> = v.map(|v| v.iter().map(|u| Uuid::from(u).to_string()).collect()).unwrap_or_default();
    PyList::new(py, strings)
}

/// A header's FlatBuffer bytes as a `PayloadHeader` dict, with every key present (None or empty for what's unset).
/// Takes any bytes-like object (bytes, bytearray, memoryview).
#[pyfunction]
pub(crate) fn decode_header<'py>(py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let bytes = pyo3::buffer::PyBuffer::<u8>::get(data)?.to_vec(py)?;
    let buf = HeaderBuf::new(bytes).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let h = buf.header();
    let d = PyDict::new(py);
    d.set_item("id", buf.id().to_string())?;
    d.set_item("stream_id", h.stream_id())?;
    d.set_item("start_ns", h.start_ns())?;
    d.set_item("end_ns", h.end_ns())?;
    d.set_item("seq", h.seq())?;
    d.set_item("layer", h.layer().variant_name().unwrap_or("unknown"))?;
    d.set_item("format", h.format().variant_name().unwrap_or("unknown"))?;
    d.set_item("shape_name", h.shape_name().unwrap_or(""))?;
    d.set_item("shape_version", h.shape_version())?;
    match h.producer() {
        None => d.set_item("producer", py.None())?,
        Some(p) => {
            let pd = PyDict::new(py);
            pd.set_item("producer_id", p.producer_id().unwrap_or(""))?;
            pd.set_item("model", p.model().unwrap_or(""))?;
            pd.set_item("version", p.version().unwrap_or(""))?;
            pd.set_item("version_hash", p.version_hash().unwrap_or(""))?;
            pd.set_item("prompt_id", p.prompt_id())?;
            pd.set_item("prompt_hash", p.prompt_hash())?;
            d.set_item("producer", pd)?;
        }
    }
    d.set_item("parents", id_list(py, h.parents())?)?;
    let evidence = PyList::empty(py);
    for r in h.evidence().iter().flatten() {
        let rd = PyDict::new(py);
        let id = r.payload_id().ok_or_else(|| PyValueError::new_err("an evidence ref without a payload id"))?;
        rd.set_item("payload_id", Uuid::from(id).to_string())?;
        rd.set_item("start_ns", r.start_ns())?;
        rd.set_item("end_ns", r.end_ns())?;
        evidence.append(rd)?;
    }
    d.set_item("evidence", evidence)?;
    d.set_item("source_tier", h.source_tier().variant_name().unwrap_or("unknown"))?;
    match h.compute() {
        None => d.set_item("compute", py.None())?,
        Some(c) => {
            let cd = PyDict::new(py);
            cd.set_item("wall_time_us", c.wall_time_us())?;
            cd.set_item("queue_wait_us", c.queue_wait_us())?;
            cd.set_item("ram_bytes", c.ram_bytes())?;
            cd.set_item("vram_bytes", c.vram_bytes())?;
            cd.set_item("device", c.device().unwrap_or(""))?;
            cd.set_item("items", c.items())?;
            d.set_item("compute", cd)?;
        }
    }
    d.set_item("confidence", h.confidence())?;
    d.set_item("subjects", id_list(py, h.subjects())?)?;
    match h.capture() {
        None => d.set_item("capture", py.None())?,
        Some(c) => {
            let cd = PyDict::new(py);
            cd.set_item("capture_mono_ns", c.capture_mono_ns())?;
            cd.set_item("device_ts", c.device_ts())?;
            cd.set_item("arrival_ns", c.arrival_ns())?;
            cd.set_item("stride", c.stride())?;
            cd.set_item("width", c.width())?;
            cd.set_item("height", c.height())?;
            cd.set_item("flip", c.flip())?;
            cd.set_item("device_seq", c.device_seq())?;
            cd.set_item("sample_rate_hz", c.sample_rate_hz())?;
            cd.set_item("capture_epoch", c.capture_epoch())?;
            d.set_item("capture", cd)?;
        }
    }
    Ok(d)
}

/// An shm locator from a `ShmLocation` dict (every key required).
pub(crate) fn shm_location(d: &Bound<'_, PyDict>) -> PyResult<continuum_core::StorageLocator> {
    Ok(continuum_core::StorageLocator::Shm {
        arena: get(d, "arena")?,
        arena_generation: get(d, "arena_generation")?,
        slab_class: get(d, "slab_class")?,
        slot: get(d, "slot")?,
        slot_generation: get(d, "slot_generation")?,
        offset: get(d, "offset")?,
        length: get(d, "length")?,
    })
}

/// An `IngestRejected` event's envelope (from `Client.next_event`) as {"stream_id", "payload_id", "reason"}, the
/// payload id a UUID string (None if the broker couldn't read one from the notice). ValueError for anything else.
#[pyfunction]
pub(crate) fn decode_ingest_rejected<'py>(
    py: Python<'py>,
    envelope: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let bytes = pyo3::buffer::PyBuffer::<u8>::get(envelope)?.to_vec(py)?;
    let env = flatbuffers::root::<fb::ControlEnvelope>(&bytes)
        .map_err(|e| PyValueError::new_err(format!("not a ControlEnvelope: {e}")))?;
    let r = env
        .payload_as_ingest_rejected()
        .ok_or_else(|| PyValueError::new_err(format!("not an IngestRejected but {:?}", env.payload_type())))?;
    let d = PyDict::new(py);
    d.set_item("stream_id", r.stream_id())?;
    d.set_item("payload_id", r.payload_id().map(|u| Uuid::from(u).to_string()))?;
    d.set_item("reason", r.reason().unwrap_or(""))?;
    Ok(d)
}
