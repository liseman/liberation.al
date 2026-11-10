"""The shell's status on the live feed (#52): dashboard/shell_status.py, and which kinds a viewer's feed carries."""

from email.message import Message

import pytest

from liberational.dashboard import server, shell_status


@pytest.fixture(autouse=True)
def no_kept_reads(monkeypatch):
    monkeypatch.setattr(shell_status, "_kept", {})


class Source:
    """A view that answers `value` and counts how often it was read."""

    def __init__(self, value):
        self.value = value
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.value


def test_a_kind_is_sent_first_and_then_only_when_its_answer_changes():
    privacy = Source({"ok": True, "devices": []})
    status = shell_status.Status({"privacy": privacy})
    assert status.changed(now=100.0) == [("privacy", {"ok": True, "devices": []})]
    assert status.changed(now=101.0) == []
    privacy.value = {"ok": True, "devices": [{"name": "cam0", "state": "pausing"}]}
    assert status.changed(now=102.0) == [("privacy", privacy.value)]
    assert status.changed(now=103.0) == []


def test_a_kind_is_read_once_per_period_however_often_the_feed_asks():
    services = Source({"ok": True, "services": []})
    status = shell_status.Status({"services": services})
    status.changed(now=100.0)
    status.changed(now=100.5)
    status.changed(now=114.9)
    assert services.reads == 1
    status.changed(now=115.0)
    assert services.reads == 2


def test_a_view_that_is_not_ready_is_asked_again_soon():
    services = Source({"ok": False, "error": "Checking service processes…", "services": []})
    status = shell_status.Status({"services": services})
    assert [kind for kind, _ in status.changed(now=100.0)] == ["services"]
    assert status.changed(now=101.9) == []
    assert services.reads == 1
    services.value = {"ok": True, "services": [{"name": "listen"}]}
    assert status.changed(now=102.0) == [("services", services.value)]


def test_a_view_that_fails_is_skipped_and_the_others_still_go_out():
    def broken():
        raise OSError("research store is locked")

    privacy = Source({"ok": True, "devices": []})
    status = shell_status.Status({"research": broken, "privacy": privacy})
    assert status.changed(now=100.0) == [("privacy", {"ok": True, "devices": []})]
    status.sources["research"] = Source({"jobs": []})
    assert status.changed(now=105.0) == [("research", {"jobs": []})]


def test_feeds_open_together_share_one_read_until_it_is_older_than_the_period(monkeypatch):
    clock = [50.0]
    monkeypatch.setattr(shell_status.time, "monotonic", lambda: clock[0])
    read = Source({"jobs": []})
    assert shell_status.shared("research", 5.0, read) == {"jobs": []}
    clock[0] = 54.9
    shell_status.shared("research", 5.0, read)
    assert read.reads == 1
    clock[0] = 55.0
    shell_status.shared("research", 5.0, read)
    assert read.reads == 2
    shell_status.shared(("services", True), 5.0, read)  # another key is another read
    assert read.reads == 3


def test_a_shared_answer_that_was_not_ready_is_read_again_for_the_retry(monkeypatch):
    clock = [50.0]
    monkeypatch.setattr(shell_status.time, "monotonic", lambda: clock[0])
    read = Source({"ok": False, "error": "Checking service processes…", "services": []})
    shell_status.shared("services", 15.0, read)
    read.value = {"ok": True, "services": []}
    clock[0] = 52.0
    assert shell_status.shared("services", 15.0, read) == {"ok": True, "services": []}


class Viewer(server.H):
    """A feed's request, without a socket: where it came from, and whether it may control this machine."""

    def __init__(self, address: str, forwarded: bool = False) -> None:
        self.client_address = (address, 50000)
        self.headers = Message()
        if forwarded:
            self.headers["X-Forwarded-For"] = "198.51.100.7"

    def _can_control(self) -> bool:
        return False


@pytest.mark.parametrize(
    ("viewer", "password", "kinds"),
    [
        pytest.param(Viewer("127.0.0.1"), "", ["privacy", "services", "research"], id="this-machine"),
        pytest.param(Viewer("192.0.2.9"), "", ["privacy", "services"], id="remote-without-a-password"),
        pytest.param(Viewer("127.0.0.1", forwarded=True), "", ["privacy", "services"], id="proxied-without-a-password"),
        pytest.param(Viewer("192.0.2.9"), "secret", ["privacy", "services", "research"], id="remote-past-the-password"),
    ],
)
def test_research_rides_the_feed_only_for_a_viewer_its_route_would_answer(monkeypatch, viewer, password, kinds):
    monkeypatch.setattr(server, "DASH_PASSWORD", password)
    assert list(viewer._shell_status().sources) == kinds
