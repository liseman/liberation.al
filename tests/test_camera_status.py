"""Camera status (dashboard/cameras.py `states`): read from the broker at most every STATUS_EVERY_S, and a stale one
answered at once while it is read again in the background (#2568)."""

import threading
import types

import pytest

from liberational.dashboard import cameras


class Link:
    """A broker link whose query answers `frames` headers for each camera, counted; `gate` holds a read."""

    def __init__(self) -> None:
        self.frames = 3
        self.calls = 0
        self.gate: threading.Event | None = None
        self.fail = False

    def call(self, fn):
        self.calls += 1
        if self.gate is not None:
            assert self.gate.wait(5)
        if self.fail:
            raise ConnectionError("broker gone")
        return {"items": [{"header": {"capture": {"width": 640, "height": 480}}}] * self.frames}


@pytest.fixture
def status(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        cameras,
        "time",
        types.SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[0], time_ns=lambda: int(now[0] * 1e9)),
    )
    monkeypatch.setattr(cameras, "_cache", {"at": -1e9, "value": None})
    monkeypatch.setattr(cameras, "_status_read", None)
    monkeypatch.setattr(cameras, "manifest_cameras", lambda: [{"name": "cam0"}])
    monkeypatch.setattr(cameras, "manifest_devices", lambda kind: [])
    import continuum_py

    monkeypatch.setattr(continuum_py, "decode_header", lambda header: header)
    return now, Link()


def settled() -> None:
    read = cameras._status_read
    if read is not None:
        read.join(5)


def test_the_first_read_waits_for_the_broker_and_the_next_comes_from_it(status):
    now, link = status
    assert cameras.states(link)["cameras"] == {"cam0": {"state": "on", "size": [640, 480], "fps": 1}}
    now[0] += cameras.STATUS_EVERY_S - 0.1
    cameras.states(link)
    assert link.calls == 1


def test_a_stale_status_answers_at_once_and_is_read_again_behind_it(status):
    now, link = status
    first = cameras.states(link)
    link.frames, link.gate = 9, threading.Event()
    now[0] += cameras.STATUS_EVERY_S
    assert cameras.states(link) == first, "at once, while the broker is asked again"
    assert cameras.states(link) == first and link.calls == 2, "one read behind it, however many ask"
    link.gate.set()
    settled()
    assert cameras.states(link)["cameras"]["cam0"]["fps"] == 3


def test_a_background_read_that_fails_keeps_the_last_status_and_the_next_one_tries_again(status, monkeypatch):
    now, link = status
    first = cameras.states(link)

    def unreadable():
        raise OSError("devices.toml unreadable")

    monkeypatch.setattr(cameras, "manifest_cameras", unreadable)
    now[0] += cameras.STATUS_EVERY_S
    cameras.states(link)
    settled()
    assert cameras.states(link)["cameras"] == first["cameras"]
    settled()
    assert cameras._status_read is None, "a failure doesn't leave a read standing"


def test_a_broker_that_answers_nothing_reads_as_no_camera_on(status):
    _, link = status
    link.fail = True  # each camera's query fails: nothing came from it, as before
    assert cameras.states(link)["cameras"] == {}
