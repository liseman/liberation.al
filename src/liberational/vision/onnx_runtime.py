"""Shared ONNX detector environment setup without importing optional runtimes."""

import logging
from typing import Any

_LOG = logging.getLogger(__name__)
_preload_failures: set[tuple[type[Exception], str]] = set()


class DetectorUnavailable(RuntimeError):
    """A detector's model or runtime environment cannot currently run inference."""


def preload_gpu_libraries(ort: Any) -> None:
    """Try optional CUDA library preloading on every attempt; warn once per failure.

    Failed preloading can still permit CPU inference. Never cache a failed
    attempt as completed: installed or repaired libraries must be retried.

    Every CUDA session builder calls this first, so it also sets ONNX Runtime's
    own logger to errors only: its warnings bypass this package's format, and
    the only one the logs ever carried was "No registered plugin EP device
    found" once per CUDA session, about 30 raw lines per see-actions start
    Where a model runs is logged by its loader.
    """
    if hasattr(ort, "set_default_logger_severity"):
        ort.set_default_logger_severity(3)  # 0 verbose, 1 info, 2 warning, 3 error, 4 fatal
    if not hasattr(ort, "preload_dlls"):
        return
    try:
        ort.preload_dlls()
    except Exception as exc:
        failure = (type(exc), str(exc))
        if failure not in _preload_failures:
            _preload_failures.add(failure)
            _LOG.warning("Could not preload CUDA libraries (%s); detection may fall back to CPU", exc)
