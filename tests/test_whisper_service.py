"""Native readiness and cleanup must describe the actual server process, never its supervising client."""

import contextlib
import json
import os
from types import SimpleNamespace
from unittest.mock import create_autospec

import psutil

from liberational import config as cfg
from liberational.speech import whisper_service as service


def test_readiness_requires_the_child_to_own_the_listening_port(monkeypatch):
    process = create_autospec(psutil.Process, instance=True)
    process.net_connections.return_value = []
    monkeypatch.setattr(service.psutil, "Process", lambda pid: process)
    request = create_autospec(service.urllib.request.urlopen)
    monkeypatch.setattr(service.urllib.request, "urlopen", request)
    assert not service.ready(123, "localhost", 8999)
    request.assert_not_called()


def test_a_loading_child_is_not_a_successful_model_load(monkeypatch):
    process = create_autospec(psutil.Process, instance=True)
    process.net_connections.return_value = [
        SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(port=8999))
    ]
    monkeypatch.setattr(service.psutil, "Process", lambda pid: process)

    @contextlib.contextmanager
    def response(*args, **kwargs):
        yield SimpleNamespace(status=200, read=lambda limit: json.dumps({"status": "loading model"}).encode())

    monkeypatch.setattr(service.urllib.request, "urlopen", response)
    assert not service.ready(123, "localhost", 8999)


def test_verified_native_load_survives_probe_failure_until_child_exit(tmp_path, monkeypatch):
    from liberational import model_runtime

    seen = []

    class Child:
        pid = os.getpid()
        returncode = 0

        def poll(self):
            return 0 if len(seen) == 3 else None

        def wait(self, timeout=None):
            return 0

    class Event:
        def is_set(self):
            return False

        def wait(self, timeout):
            seen.append(model_runtime.snapshot({os.getpid(): "whisper"})["processes"])

    health = iter([False, True, False])
    monkeypatch.setattr(service.subprocess, "Popen", create_autospec(service.subprocess.Popen, return_value=Child()))
    monkeypatch.setattr(service, "threading", SimpleNamespace(Event=Event))
    monkeypatch.setattr(service, "ready", lambda *args: next(health))
    monkeypatch.setattr(model_runtime, "_query_gpu_memory", lambda: None)
    monkeypatch.setattr(model_runtime, "_gpu_cache", (0.0, None))
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        assert service.supervise(["native-server", "-m", "model.bin", "--host", "localhost", "--port", "8999"]) == 0
    assert seen[0] == []
    assert seen[1][0]["models"][0]["key"] == "models.files.asr_whisper"
    assert len(seen[2]) == 1
    assert seen[2][0]["models"] == seen[1][0]["models"]
    assert list((tmp_path / "model_loads").glob("*.json")) == []


def test_termination_is_forwarded_and_child_is_reaped(tmp_path, monkeypatch):
    import signal

    from liberational import model_runtime

    handlers = {}
    events = []

    class Child:
        pid = os.getpid()
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, signum):
            events.append(("signal", signum))
            self.returncode = -signum

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            return self.returncode

    def install(signum, handler):
        handlers[signum] = handler
        return signal.SIG_DFL

    def interrupted(*args):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return False

    monkeypatch.setattr(service.subprocess, "Popen", create_autospec(service.subprocess.Popen, return_value=Child()))
    monkeypatch.setattr(
        service, "signal", SimpleNamespace(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, signal=install)
    )
    monkeypatch.setattr(service, "ready", interrupted)
    monkeypatch.setattr(model_runtime, "_query_gpu_memory", lambda: None)
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        assert (
            service.supervise(["native-server", "-m", "model.bin", "--host", "localhost", "--port", "8999"])
            == -signal.SIGTERM
        )
    assert events == [("signal", signal.SIGTERM), ("wait", None)]
    assert handlers == {signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_DFL}
