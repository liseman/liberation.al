"""Restart requests are mocked at the executable boundary, without services or hardware."""

import json
import subprocess
from unittest.mock import patch

import pytest

from liberational.dashboard import capture_control


@pytest.fixture
def invocation(monkeypatch, tmp_path):
    monkeypatch.setattr(capture_control.config, "path", lambda _: tmp_path)
    with (
        patch.object(capture_control, "capture_binary", autospec=True, return_value="/bin/capture"),
        patch.object(
            capture_control.subprocess,
            "run",
            autospec=True,
            return_value=subprocess.CompletedProcess(
                [],
                0,
                json.dumps(
                    {
                        "status": "restarting",
                        "device": "cam0",
                        "message": "restart scheduled",
                    }
                ),
                "",
            ),
        ) as run,
    ):
        yield run


def test_accepted_request_uses_only_named_camera_and_configured_state(invocation, tmp_path):
    result = capture_control.restart_camera("cam0")
    assert result == capture_control.RestartResult("restarting", "restart scheduled")
    invocation.assert_called_once_with(
        ["/bin/capture", "--restart-camera", "cam0", "--state-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )


@pytest.mark.parametrize("code,status", [(3, "unavailable"), (1, "failed")])
def test_supervisor_rejection_and_unavailability_remain_distinct(invocation, code, status):
    invocation.return_value.returncode = code
    invocation.return_value.stdout = json.dumps({"status": status, "device": "cam0", "message": "reason"})
    assert capture_control.restart_camera("cam0") == capture_control.RestartResult(status, "reason")


def test_baseline_old_cli_usage_is_unavailable(invocation):
    invocation.return_value = subprocess.CompletedProcess(
        [],
        2,
        "",
        "usage: continuum-capture --manifest P (--state-dir D | --socket S)\n",
    )
    assert capture_control.restart_camera("cam0").status == "unavailable"


@pytest.mark.parametrize(
    "stdout,stderr",
    [
        ("", "unrelated failure"),
        ('{"status":"unavailable"}', "usage: continuum-capture --manifest P"),
        ("", "usage: other-program"),
    ],
)
def test_arbitrary_usage_failure_cannot_authorize_global_fallback(invocation, stdout, stderr):
    invocation.return_value = subprocess.CompletedProcess([], 2, stdout, stderr)
    assert capture_control.restart_camera("cam0").status == "failed"


@pytest.mark.parametrize(
    "code,reply",
    [
        (0, "not json"),
        (0, "[]"),
        (0, '{"status":"restarting","device":"cam1","message":"accepted"}'),
        (0, '{"status":"restarting","device":"cam0","message":null}'),
        (3, '{"status":"restarting","device":"cam0","message":"accepted"}'),
        (0, '{"status":"unavailable","device":"cam0","message":"no"}'),
        (0, '{"status":"restarted","device":"cam0","message":"yes"}'),
        (1, '{"status":"failed","device":"cam0","message":"no"}\n{}'),
    ],
)
def test_malformed_or_mismatched_reply_is_failed_not_unavailable(invocation, code, reply):
    invocation.return_value = subprocess.CompletedProcess([], code, reply, "")
    assert capture_control.restart_camera("cam0").status == "failed"


@pytest.mark.parametrize(
    "device", ["", "  ", "cam\n0", "cam\x000", "cam\x7f0", "cam\x850", None, False, "x" * 257, "é" * 129, "bad\ud800"]
)
def test_invalid_device_never_launches_capture(invocation, device):
    assert capture_control.restart_camera(device).status == "failed"
    invocation.assert_not_called()


def test_device_is_an_exact_argument_without_shell_interpretation(invocation):
    device = "camera ; $(anything)"
    invocation.return_value.stdout = json.dumps({"status": "restarting", "device": device, "message": "scheduled"})
    assert capture_control.restart_camera(device).status == "restarting"
    assert invocation.call_args.args[0][2] == device


@pytest.mark.parametrize("device", ["--camera", "x" * 256, "é" * 128])
@pytest.mark.parametrize("code,status", [(0, "restarting"), (1, "failed")])
def test_literal_and_maximum_length_names_reach_supervisor_without_fallback(invocation, device, code, status):
    invocation.return_value = subprocess.CompletedProcess(
        [],
        code,
        json.dumps({"status": status, "device": device, "message": "supervisor response"}),
        "",
    )
    assert capture_control.restart_camera(device).status == status
    assert invocation.call_args.args[0][2] == device


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("capture", 5), PermissionError("denied")])
def test_transport_failure_may_have_been_accepted_and_never_falls_back(invocation, error):
    invocation.side_effect = error
    assert capture_control.restart_camera("cam0").status == "failed"
    assert invocation.call_count == 1


def test_disappearing_executable_is_unavailable(invocation):
    invocation.side_effect = FileNotFoundError("capture disappeared")
    assert capture_control.restart_camera("cam0").status == "unavailable"


def test_missing_executable_does_not_launch_or_restart_anything(invocation, monkeypatch):
    with monkeypatch.context() as local:
        local.setattr(capture_control, "capture_binary", lambda: None)
        assert capture_control.restart_camera("cam0").status == "unavailable"
    invocation.assert_not_called()
