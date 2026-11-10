//! Optional software or NVDEC H.264 decoder for live vision. It retains codec state, not camera connections or decoded history.

#[cfg(not(feature = "ffmpeg"))]
use pyo3::exceptions::PyNotImplementedError;
#[cfg(feature = "ffmpeg")]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(feature = "ffmpeg")]
use pyo3::types::PyBytes;

#[pyclass(name = "H264Decoder", unsendable)]
pub(crate) struct PyH264Decoder {
    #[cfg(feature = "ffmpeg")]
    decoder: continuum_tier::video::H264StreamDecoder,
}

#[cfg(not(feature = "ffmpeg"))]
const UNSUPPORTED: &str = "RTSP vision requires continuum_py built with the optional ffmpeg feature";

type Picture<'py> = (u64, u32, u32, Bound<'py, pyo3::types::PyBytes>);
type RichPicture<'py> = (
    u64,
    u32,
    u32,
    Bound<'py, pyo3::types::PyBytes>,
    &'static str,
    &'static str,
    Option<Bound<'py, pyo3::types::PyBytes>>,
);

#[pymethods]
impl PyH264Decoder {
    #[new]
    #[pyo3(signature = (backend = "software", gpu = 0, codec = "h264"))]
    fn new(backend: &str, gpu: u32, codec: &str) -> PyResult<Self> {
        #[cfg(feature = "ffmpeg")]
        {
            use continuum_tier::video::{H264DecodeBackend, H264StreamDecoder};
            // An archived chunk's AV1 units (#1479) decode in software, one temporal unit per push.
            match (codec, backend, gpu) {
                ("av1", "software", 0) => return Ok(Self { decoder: H264StreamDecoder::av1().map_err(invalid)? }),
                ("av1", ..) => {
                    return Err(PyValueError::new_err("AV1 decodes in software only (backend software, gpu 0)"));
                }
                ("h264", ..) => {}
                _ => return Err(PyValueError::new_err("codec must be h264 or av1")),
            }
            let selected = match backend {
                "software" if gpu == 0 => H264DecodeBackend::Software,
                "cuda" => H264DecodeBackend::Cuda { gpu },
                _ => return Err(PyValueError::new_err("backend must be software (gpu=0) or cuda with a GPU ordinal")),
            };
            Ok(Self { decoder: H264StreamDecoder::with_backend(selected).map_err(invalid)? })
        }
        #[cfg(not(feature = "ffmpeg"))]
        {
            let _ = (backend, gpu, codec);
            Err(PyNotImplementedError::new_err(UNSUPPORTED))
        }
    }

    #[staticmethod]
    fn supported() -> bool {
        cfg!(feature = "ffmpeg")
    }

    #[getter]
    fn waiting_for_keyframe(&self) -> bool {
        #[cfg(feature = "ffmpeg")]
        {
            self.decoder.waiting_for_keyframe()
        }
        #[cfg(not(feature = "ffmpeg"))]
        {
            true
        }
    }

    #[getter]
    fn dropped_tags(&self) -> Vec<u64> {
        #[cfg(feature = "ffmpeg")]
        {
            self.decoder.dropped_tags().to_vec()
        }
        #[cfg(not(feature = "ffmpeg"))]
        {
            Vec::new()
        }
    }

    fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        #[cfg(feature = "ffmpeg")]
        {
            py.detach(|| self.decoder.reset()).map_err(invalid)
        }
        #[cfg(not(feature = "ffmpeg"))]
        {
            let _ = py;
            Err(PyNotImplementedError::new_err(UNSUPPORTED))
        }
    }

    #[pyo3(signature = (data, tag, presentation_ns = None))]
    fn push<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        tag: u64,
        presentation_ns: Option<i64>,
    ) -> PyResult<Vec<Picture<'py>>> {
        #[cfg(feature = "ffmpeg")]
        {
            let pictures = self.decode_input(py, data, tag, presentation_ns, false, true)?;
            Ok(pictures
                .into_iter()
                .map(|picture| (picture.tag, picture.width, picture.height, PyBytes::new(py, &picture.nv12)))
                .collect())
        }
        #[cfg(not(feature = "ffmpeg"))]
        {
            let _ = (py, data, tag, presentation_ns);
            Err(PyNotImplementedError::new_err(UNSUPPORTED))
        }
    }

    /// `keep` false decodes the unit for its references only: its picture returns with empty NV12 and no BGR.
    #[pyo3(signature = (data, tag, presentation_ns = None, want_bgr = false, keep = true))]
    fn push_outputs<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        tag: u64,
        presentation_ns: Option<i64>,
        want_bgr: bool,
        keep: bool,
    ) -> PyResult<Vec<RichPicture<'py>>> {
        #[cfg(feature = "ffmpeg")]
        {
            let pictures = self.decode_input(py, data, tag, presentation_ns, want_bgr, keep)?;
            Ok(rich_pictures(py, pictures))
        }
        #[cfg(not(feature = "ffmpeg"))]
        {
            let _ = (py, data, tag, presentation_ns, want_bgr, keep);
            Err(PyNotImplementedError::new_err(UNSUPPORTED))
        }
    }

    #[pyo3(signature = (want_bgr = false))]
    fn finish_outputs<'py>(&mut self, py: Python<'py>, want_bgr: bool) -> PyResult<Vec<RichPicture<'py>>> {
        #[cfg(feature = "ffmpeg")]
        {
            let pictures = py.detach(|| self.decoder.finish_outputs(want_bgr)).map_err(invalid)?;
            Ok(rich_pictures(py, pictures))
        }
        #[cfg(not(feature = "ffmpeg"))]
        {
            let _ = (py, want_bgr);
            Err(PyNotImplementedError::new_err(UNSUPPORTED))
        }
    }

    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Picture<'py>>> {
        #[cfg(feature = "ffmpeg")]
        {
            let pictures = py.detach(|| self.decoder.finish()).map_err(invalid)?;
            Ok(pictures
                .into_iter()
                .map(|picture| (picture.tag, picture.width, picture.height, PyBytes::new(py, &picture.nv12)))
                .collect())
        }
        #[cfg(not(feature = "ffmpeg"))]
        {
            let _ = py;
            Err(PyNotImplementedError::new_err(UNSUPPORTED))
        }
    }
}

#[cfg(feature = "ffmpeg")]
impl PyH264Decoder {
    fn decode_input(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        tag: u64,
        presentation_ns: Option<i64>,
        want_bgr: bool,
        keep: bool,
    ) -> PyResult<Vec<continuum_tier::video::TaggedFrame>> {
        let buffer = pyo3::buffer::PyBuffer::<u8>::get(data)?;
        if buffer.len_bytes() == 0 || buffer.len_bytes() > 16 * 1024 * 1024 {
            return Err(PyValueError::new_err("H.264 access unit must contain at most 16 MiB"));
        }
        let bytes = buffer.to_vec(py)?;
        py.detach(|| self.decoder.push_outputs(&bytes, tag, presentation_ns, want_bgr, keep)).map_err(invalid)
    }
}

#[cfg(feature = "ffmpeg")]
fn rich_pictures<'py>(py: Python<'py>, pictures: Vec<continuum_tier::video::TaggedFrame>) -> Vec<RichPicture<'py>> {
    pictures
        .into_iter()
        .map(|picture| {
            (
                picture.tag,
                picture.width,
                picture.height,
                PyBytes::new(py, &picture.nv12),
                picture.color_range,
                picture.color_matrix,
                picture.bgr.as_ref().map(|bytes| PyBytes::new(py, bytes)),
            )
        })
        .collect()
}

#[cfg(feature = "ffmpeg")]
fn invalid(error: continuum_tier::TierError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction]
pub(crate) fn h264_decoder_available() -> bool {
    cfg!(feature = "ffmpeg")
}
