//! The search payloads (continuum-search) for Python: encode a vector or text query, and read (or, for a stand-in
//! worker in tests, write) a result. The same codecs the vector worker uses, so the two can't drift.

use continuum_search::{Filter, Hit, Key, TextQuery, VectorQuery, decode_hits as decode, encode_hits as encode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use uuid::Uuid;

fn invalid(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn uuid(s: &str) -> PyResult<Uuid> {
    Uuid::parse_str(s).map_err(|e| PyValueError::new_err(format!("bad id {s:?}: {e}")))
}

fn filter(during: Option<(u64, u64)>, subjects: Vec<String>, labelled: Option<bool>) -> PyResult<Filter> {
    let subjects = subjects.iter().map(|s| uuid(s)).collect::<PyResult<Vec<_>>>()?;
    Ok(Filter { during, subjects, labelled })
}

/// A `vector.query` payload's bytes.
#[pyfunction]
#[pyo3(signature = (model, version, vector, k, *, during = None, subjects = Vec::new(), labelled = None))]
#[allow(clippy::too_many_arguments)] // the Python keywords
pub(crate) fn encode_vector_query<'py>(
    py: Python<'py>,
    model: String,
    version: String,
    vector: Vec<f32>,
    k: u32,
    during: Option<(u64, u64)>,
    subjects: Vec<String>,
    labelled: Option<bool>,
) -> PyResult<Bound<'py, PyBytes>> {
    continuum_search::check_k(k).map_err(invalid)?;
    let q = VectorQuery { model, version, k, vector, filter: filter(during, subjects, labelled)? };
    Ok(PyBytes::new(py, &q.encode().map_err(invalid)?))
}

/// A `text.query` payload's bytes.
#[pyfunction]
#[pyo3(signature = (text, k, *, during = None, subjects = Vec::new(), labelled = None))]
pub(crate) fn encode_text_query<'py>(
    py: Python<'py>,
    text: String,
    k: u32,
    during: Option<(u64, u64)>,
    subjects: Vec<String>,
    labelled: Option<bool>,
) -> PyResult<Bound<'py, PyBytes>> {
    continuum_search::check_k(k).map_err(invalid)?;
    let q = TextQuery { text, k, filter: filter(during, subjects, labelled)? };
    Ok(PyBytes::new(py, &q.encode().map_err(invalid)?))
}

/// A `vector.search` or `text.search` payload's hits, nearest or most relevant first, as dicts.
#[pyfunction]
pub(crate) fn decode_hits<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for h in decode(data).map_err(invalid)? {
        let d = PyDict::new(py);
        let id = |u: Option<Uuid>| u.map(|u| u.hyphenated().to_string());
        d.set_item("payload_id", h.key.payload_id.hyphenated().to_string())?;
        d.set_item("entity_id", id(h.key.entity_id))?;
        d.set_item("parent_id", id(h.key.parent_id))?;
        d.set_item("start_ns", h.key.start_ns)?;
        d.set_item("end_ns", h.key.end_ns)?;
        d.set_item("score", h.score)?;
        out.append(d)?;
    }
    Ok(out)
}

/// A result payload's bytes from hits (dicts with decode_hits' keys): for a stand-in worker in tests.
#[pyfunction]
pub(crate) fn encode_hits<'py>(py: Python<'py>, hits: Vec<Bound<'py, PyDict>>) -> PyResult<Bound<'py, PyBytes>> {
    let get_id = |d: &Bound<'py, PyDict>, key: &str| -> PyResult<Option<Uuid>> {
        match d.get_item(key)? {
            Some(v) if !v.is_none() => Ok(Some(uuid(&v.extract::<String>()?)?)),
            _ => Ok(None),
        }
    };
    let num = |d: &Bound<'py, PyDict>, key: &str| -> PyResult<Bound<'py, PyAny>> {
        d.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("a hit needs {key}")))
    };
    let hits = hits
        .iter()
        .map(|d| {
            Ok(Hit {
                key: Key {
                    payload_id: get_id(d, "payload_id")?
                        .ok_or_else(|| PyValueError::new_err("a hit needs payload_id"))?,
                    entity_id: get_id(d, "entity_id")?,
                    parent_id: get_id(d, "parent_id")?,
                    start_ns: num(d, "start_ns")?.extract()?,
                    end_ns: num(d, "end_ns")?.extract()?,
                },
                score: num(d, "score")?.extract()?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyBytes::new(py, &encode(&hits).map_err(invalid)?))
}
