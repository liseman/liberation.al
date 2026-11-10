"""The capture stop-and-start check (tools/capture_restart_check.py, #1388): it reads capture's teardown lines (#1452)
and tells a teardown that hung from a device the OS kept wedged."""

import importlib.util
import pathlib
import sys

PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "capture_restart_check.py"
spec = importlib.util.spec_from_file_location("capture_restart_check", PATH)
assert spec is not None and spec.loader is not None
check = importlib.util.module_from_spec(spec)
sys.modules["capture_restart_check"] = check
spec.loader.exec_module(check)

EARLIER = """\
2026-11-03T15:58:36-08:00 capture supervisor: starting worker 6 for cam0 on 0x1100000
2026-11-03T15:58:38-08:00 capture supervisor: starting worker 7 for respeaker on BuiltInMicrophoneDevice
"""
CLEAN_STOP = """\
capture worker cam0: teardown begins
capture worker cam0: teardown session stopRunning took 140 ms
capture worker cam0: teardown detach frame delegate took 0 ms
capture worker cam0: teardown done in 152 ms
capture worker respeaker: teardown begins
capture worker respeaker: teardown stop audio unit took 12 ms
capture worker respeaker: teardown release audio unit took 30 ms
capture worker respeaker: teardown done in 43 ms
"""
REOPENED = """\
capture supervisor: starting worker 1 for cam0 on 0x1100000
capture worker cam0: 1920x1080 MJPEG at 33333333 ns a frame, frame duration pinned and verified
capture worker respeaker: 16000 Hz, 6 channels, PCM_F32LEx6
"""


def test_a_clean_stop_and_start_is_ok_with_each_steps_time() -> None:
    report = check.read(CLEAN_STOP, REOPENED, EARLIER)
    assert report.ok
    assert report.devices["cam0"].steps == [("session stopRunning", 140), ("detach frame delegate", 0)]
    assert report.devices["cam0"].done_ms == 152
    assert report.devices["respeaker"].opened == "16000 Hz, 6 channels, PCM_F32LEx6"


def test_a_teardown_killed_mid_step_names_that_step_and_the_kill() -> None:
    stop = """\
capture worker respeaker: teardown begins
capture worker respeaker: teardown still in stop audio unit after 1000 ms
capture worker respeaker: teardown still in stop audio unit after 9000 ms
capture supervisor: worker 7 ignored the stop: killing it
"""
    report = check.read(stop, "", EARLIER)
    mic = report.devices["respeaker"]
    assert (mic.stuck, mic.killed, mic.done_ms, mic.opened) == (("stop audio unit", 9000), True, None, None)
    assert not report.ok
    text = check.describe(report)
    assert "HUNG in stop audio unit (9000 ms" in text
    assert "the supervisor killed it" in text


def test_a_device_that_tore_down_cleanly_but_never_reopened_is_blamed_on_the_os() -> None:
    report = check.read(CLEAN_STOP, "capture worker respeaker: 16000 Hz, 6 channels, PCM_F32LEx6\n", EARLIER)
    assert not report.ok
    assert report.devices["respeaker"].ok
    assert "the OS kept the device wedged" in check.describe(report).split("respeaker:")[0]


def test_a_kill_for_a_worker_started_before_the_log_window_keeps_its_number() -> None:
    report = check.read("capture supervisor: worker 9 ignored the stop: killing it\n", "")
    assert report.devices["worker 9"].killed


def test_a_supervisor_that_gave_up_on_a_worker_fails_the_check() -> None:
    stop = CLEAN_STOP + "capture supervisor: ERROR workers did not exit after SIGKILL: [WorkerId(6)]; check OS\n"
    report = check.read(stop, REOPENED, EARLIER)
    assert report.unkillable
    assert not report.ok


def test_no_teardown_lines_says_so() -> None:
    assert "No teardown lines" in check.describe(check.read("", REOPENED))
