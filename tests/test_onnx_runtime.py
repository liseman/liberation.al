"""Optional runtime preloading stays retryable without hardware or models."""

from types import SimpleNamespace

from liberational.vision import onnx_runtime


def test_runtime_without_preload_support_is_allowed():
    onnx_runtime.preload_gpu_libraries(SimpleNamespace())


def test_repeated_preload_failure_warns_once_but_retries_after_repair(monkeypatch, caplog):
    monkeypatch.setattr(onnx_runtime, "_preload_failures", set())
    runtime = SimpleNamespace(broken=True, attempts=0, loaded=False)

    def preload_dlls():
        runtime.attempts += 1
        if runtime.broken:
            raise OSError("fixture CUDA library missing")
        runtime.loaded = True

    runtime.preload_dlls = preload_dlls
    onnx_runtime.preload_gpu_libraries(runtime)
    onnx_runtime.preload_gpu_libraries(runtime)
    assert runtime.attempts == 2
    assert len(caplog.records) == 1
    runtime.broken = False
    onnx_runtime.preload_gpu_libraries(runtime)
    assert runtime.attempts == 3
    assert runtime.loaded
    assert len(caplog.records) == 1


def test_runtime_logger_is_set_to_errors_only_before_sessions_are_built():
    severities = []
    runtime = SimpleNamespace(set_default_logger_severity=severities.append)
    onnx_runtime.preload_gpu_libraries(runtime)
    assert severities == [3]  # ONNX Runtime's ERROR: its per-session warnings stay out of the logs (#3147)
