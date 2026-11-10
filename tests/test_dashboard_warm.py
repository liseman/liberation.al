"""The dashboard does its slow imports at start, not inside the first request that needs them."""

import http.client
import http.server
import json
import logging
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

import liberational.vision
from liberational import config, knowledge, voice_turn
from liberational.dashboard import server


def test_the_faces_helpers_are_imported_before_the_first_people_request(monkeypatch):
    # The first GET /people imported vision/faces.py and cv2 itself and waited 17 s after a start
    monkeypatch.delitem(sys.modules, "liberational.vision.faces", raising=False)
    monkeypatch.delattr(liberational.vision, "faces", raising=False)
    server._warm_faces().join(60)
    assert "liberational.vision.faces" in sys.modules


def test_ready_server_starts_one_warm_up_without_blocking_requests(monkeypatch):
    started, release, done = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def warm():
        calls.append("warm")
        started.set()
        try:
            assert release.wait(5)
        finally:
            done.set()

    monkeypatch.setattr(server, "_warm_people", warm)
    with server.DashboardServer(("127.0.0.1", 0), server.H) as srv:
        serving = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True)
        serving.start()
        try:
            assert started.wait(5)
            connection = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=2)
            try:
                connection.request("GET", "/health")
                response = connection.getresponse()
                assert response.status == 200
                response.read()
            finally:
                connection.close()
            srv.service_actions()
            assert calls == ["warm"]
        finally:
            release.set()
            srv.shutdown()
            serving.join(5)
            assert done.wait(5)
        assert not serving.is_alive()


@pytest.fixture
def warm_sources(monkeypatch):
    calls, originals = [], {}
    original_wait = voice_turn.wait_quiet
    monkeypatch.setattr(server, "_low_priority", lambda priority: calls.append(("priority", priority)))
    monkeypatch.setattr(
        voice_turn, "wait_quiet", create_autospec(voice_turn.wait_quiet, side_effect=lambda: calls.append("gate"))
    )
    for attr, name, reply in (
        ("mic_status", "microphones", {"devices": {}}),
        ("scene_entries", "scene", []),
        ("camera_status", "cameras", {"cameras": {}}),
        ("people_view", "roster", {"people": []}),
        ("clones_list", "clones", {"clones": [], "error": None}),
    ):
        fn = getattr(server, attr)
        originals[attr] = fn
        monkeypatch.setattr(
            server,
            attr,
            create_autospec(fn, side_effect=lambda *_, name=name, reply=reply, **__: calls.append(name) or reply),
        )
    voices = server._Fresh(
        "voice test", lambda: calls.append("voice groups") or (200, {"groups": []}), lambda: 0, lambda: 0
    )
    faces = server._Fresh(
        "face test", lambda: calls.append("face groups") or (200, {"groups": []}), lambda: 0, lambda: 0
    )
    monkeypatch.setattr(server, "_VOICE_GROUPS", voices)
    monkeypatch.setattr(server, "_FACE_GROUPS", faces)
    logged = create_autospec(server.log)
    monkeypatch.setattr(server, "log", logged)
    return SimpleNamespace(
        calls=calls, voices=voices, faces=faces, logged=logged, originals=originals, wait=original_wait
    )


def test_warm_up_admits_each_stage_at_low_priority_and_leaves_faces_last(warm_sources):
    server._warm_people()
    assert warm_sources.calls == [
        ("priority", 19),
        "gate",
        "microphones",
        "gate",
        "scene",
        "gate",
        "cameras",
        "gate",
        "roster",
        "gate",
        "clones",
        "gate",
        "voice groups",
        "gate",
        "face groups",
    ]
    warm_sources.logged.assert_called_once()
    assert warm_sources.logged.call_args.kwargs["level"] == server.logs.INFO


def test_warm_up_logs_one_failure_and_still_warms_the_other_caches(monkeypatch, warm_sources):
    monkeypatch.setattr(
        server,
        "people_view",
        create_autospec(warm_sources.originals["people_view"], side_effect=RuntimeError("broker not ready")),
    )
    warm_sources.voices.compute = lambda: warm_sources.calls.append("voice groups") or (500, {"error": "unavailable"})
    server._warm_people()
    assert warm_sources.calls == [
        ("priority", 19),
        "gate",
        "microphones",
        "gate",
        "scene",
        "gate",
        "cameras",
        "gate",
        "gate",
        "clones",
        "gate",
        "voice groups",
        "gate",
        "face groups",
    ]
    warm_sources.logged.assert_called_once()
    assert warm_sources.logged.call_args.kwargs["level"] == server.logs.WARN
    status = server._people_warm_status()
    assert status["state"] == "incomplete"
    assert len(status["errors"]) == 2
    assert "roster" in status["errors"][0]
    assert "voice groups" in status["errors"][1]
    status["errors"].append("caller must not mutate internal errors")
    assert len(server._people_warm_status()["errors"]) == 2


def test_unpublished_success_is_retried_once_then_reports_incomplete(monkeypatch, warm_sources):
    owner = {"revision": 0}

    def unpublished():
        warm_sources.calls.append("face groups")
        owner["revision"] += 1
        return 200, {"groups": []}

    faces = server._Fresh("changing face owner", unpublished, lambda: owner["revision"], lambda: 0)
    monkeypatch.setattr(server, "_FACE_GROUPS", faces)
    server._warm_people()
    assert warm_sources.calls.count("face groups") == 2
    assert faces.value is None
    assert server._people_warm_status()["state"] == "incomplete"
    assert any("face groups" in error for error in server._people_warm_status()["errors"])
    warm_sources.logged.assert_called_once()
    assert warm_sources.logged.call_args.kwargs["level"] == server.logs.WARN


def test_group_retry_can_publish_before_the_warm_up_reports_done(monkeypatch, warm_sources):
    owner, attempts = {"revision": 0}, []

    def eventually_published():
        attempts.append("face")
        if len(attempts) == 1:
            owner["revision"] += 1
        return 200, {"groups": []}

    faces = server._Fresh("changing face owner", eventually_published, lambda: owner["revision"], lambda: 0)
    monkeypatch.setattr(server, "_FACE_GROUPS", faces)
    server._warm_people()
    assert attempts == ["face", "face"]
    assert faces.value == (200, {"groups": []})
    status = server._people_warm_status()
    assert status["state"] == "done"
    assert status["stage"] is None
    assert status["errors"] == []
    assert status["elapsed_s"] >= 0
    warm_sources.logged.assert_called_once()
    assert warm_sources.logged.call_args.kwargs["level"] == server.logs.INFO


def test_request_joins_the_in_flight_people_warm_up(monkeypatch):
    computing, release, joining = threading.Event(), threading.Event(), threading.Event()
    reads = server._PeopleReads()
    calls = []

    def read():
        calls.append("read")
        computing.set()
        assert release.wait(5)
        return {"people": [{"name": "Alice"}]}

    monkeypatch.setattr(server, "_PEOPLE_READS", reads)
    monkeypatch.setattr(server, "_people_view", read)
    monkeypatch.setattr(server, "_voices_key", lambda: 0)
    monkeypatch.setattr(server, "_faces_owner_key", lambda **_: 0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        warm = pool.submit(server.people_view)
        try:
            assert computing.wait(5)
            with reads.lock:
                [pending] = reads.pending.values()
            result = pending.result

            def waiting(timeout=None):
                joining.set()
                return result(timeout)

            monkeypatch.setattr(pending, "result", create_autospec(result, side_effect=waiting))
            request = pool.submit(server.people_view)
            assert joining.wait(5)
            assert not request.done()
        finally:
            release.set()
        assert warm.result(timeout=5) == {"people": [{"name": "Alice"}]}
        assert request.result(timeout=5) == {"people": [{"name": "Alice"}]}
    assert calls == ["read"]
    assert server.people_view() == {"people": [{"name": "Alice"}]}
    settle(reads)
    assert calls == ["read", "read"]  # no permanent roster snapshot: each answer from the last read refreshes it


def settle(reads):
    """Wait for the people reads running in the background."""
    while True:
        with reads.lock:
            pending = list(reads.pending.values())
        if not pending:
            return
        for future in pending:
            future.exception(timeout=5)


class Speech:
    def __init__(self):
        self.lines = 0

    def mark(self):
        return self.lines, None


@pytest.fixture
def roster(monkeypatch):
    """people_view over a fake read that a test can hold, with speech and identity inputs it moves."""
    state = {"voices": 0, "version": 0, "calls": 0}
    held = threading.Event()
    held.set()
    speech = Speech()

    def read():
        state["calls"] += 1
        version = state["version"]
        assert held.wait(5)
        return {"version": version}

    reads = server._PeopleReads()
    monkeypatch.setattr(server, "_PEOPLE_READS", reads)
    monkeypatch.setattr(server, "_people_view", read)
    monkeypatch.setattr(server, "_voices_key", lambda: state["voices"])
    monkeypatch.setattr(server, "_faces_owner_key", lambda **_: 0)
    monkeypatch.setattr(server.speech_log, "shared", lambda: speech)
    return state, held, speech, reads


def test_new_speech_answers_from_the_last_read_and_refreshes_it_once_in_the_background(roster):
    state, held, speech, reads = roster
    assert server.people_view() == {"version": 0}
    state["version"], speech.lines = 1, 1
    held.clear()  # the refresh can take as long as it likes: no request waits for it
    assert server.people_view(wait_s=None) == {"version": 0}
    speech.lines = 2
    assert server.people_view(wait_s=None) == {"version": 0}
    assert state["calls"] == 2, "one refresh at a time, however fast speech moves"
    held.set()
    settle(reads)
    assert server.people_view() == {"version": 1}


def test_new_speech_refreshes_the_last_read_at_most_once_per_refresh_interval(roster, monkeypatch):
    state, _held, speech, reads = roster
    monkeypatch.setattr(server, "PEOPLE_REFRESH_S", 60.0)
    assert server.people_view() == {"version": 0}
    state["version"], speech.lines = 1, 1
    assert server.people_view(wait_s=None) == {"version": 0}
    settle(reads)
    assert state["calls"] == 1, "speech alone, inside the interval: the last read answers and none starts"
    reads.last_at -= 61.0  # the interval has passed since that read
    assert server.people_view(wait_s=None) == {"version": 0}
    settle(reads)
    assert state["calls"] == 2
    assert server.people_view(wait_s=None) == {"version": 1}


def test_a_changed_name_reads_at_once_inside_the_refresh_interval(roster, monkeypatch):
    state, _held, _speech, reads = roster
    monkeypatch.setattr(server, "PEOPLE_REFRESH_S", 60.0)
    assert server.people_view() == {"version": 0}
    state["voices"], state["version"] = 1, 1  # a voice was named
    assert server.people_view(wait_s=None) == {"version": 1}
    settle(reads)


def test_a_changed_name_waits_for_its_own_read_instead_of_answering_from_the_last(roster):
    state, _held, _speech, reads = roster
    assert server.people_view() == {"version": 0}
    state["voices"], state["version"] = 1, 1  # a voice was named
    assert server.people_view(wait_s=None) == {"version": 1}
    settle(reads)


def test_a_failed_people_warm_up_leaves_the_next_read_working():
    reads = server._PeopleReads()
    read = create_autospec(server._people_view, side_effect=[RuntimeError("broker not ready"), {"people": []}])
    with pytest.raises(RuntimeError, match="broker not ready"):
        reads.get((0,), read)
    assert reads.get((0,), read) == {"people": []}


def test_identity_corrections_do_not_join_an_older_people_warm_up(monkeypatch):
    computing, release = threading.Event(), threading.Event()
    owner = {"revision": 0}

    def read():
        revision = owner["revision"]
        if revision == 0:
            computing.set()
            assert release.wait(5)
        return {"revision": revision}

    monkeypatch.setattr(server, "_PEOPLE_READS", server._PeopleReads())
    monkeypatch.setattr(server, "_people_view", read)
    monkeypatch.setattr(server, "_voices_key", lambda: owner["revision"])
    monkeypatch.setattr(server, "_faces_owner_key", lambda **_: 0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        warm = pool.submit(server.people_view)
        try:
            assert computing.wait(5)
            owner["revision"] = 1
            assert server.people_view() == {"revision": 1}
        finally:
            release.set()
        assert warm.result(timeout=5) == {"revision": 0}


def test_health_reports_waiting_warm_up_without_waiting_for_the_voice_gate(monkeypatch, warm_sources):
    waiting, release, done = threading.Event(), threading.Event(), threading.Event()
    real_warm = server._warm_people

    def wait_for_voice():
        waiting.set()
        assert release.wait(5), "test did not release the warm-up voice gate"

    def warm():
        try:
            real_warm()
        finally:
            done.set()

    monkeypatch.setattr(voice_turn, "wait_quiet", create_autospec(warm_sources.wait, side_effect=wait_for_voice))
    monkeypatch.setattr(server, "_warm_people", create_autospec(real_warm, side_effect=warm))
    with server.DashboardServer(("127.0.0.1", 0), server.H) as srv:
        serving = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True)
        serving.start()
        try:
            assert waiting.wait(5)
            connection = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=2)
            try:
                connection.request("GET", "/health")
                response = connection.getresponse()
                assert response.status == 200
                status = json.loads(response.read())["people_warmup"]
            finally:
                connection.close()
            assert status["state"] == "waiting"
            assert status["stage"] == "microphones"
            assert status["errors"] == []
            assert set(status) == {"state", "stage", "elapsed_s", "errors"}
            assert warm_sources.calls == [("priority", 19), ("priority", 19)]  # crop worker and People warm-up
        finally:
            release.set()
            srv.shutdown()
            serving.join(5)
            assert done.wait(5)
    assert server._people_warm_status()["state"] == "done"


def test_faces_import_is_low_priority_and_waits_for_voice_off_the_serving_thread(monkeypatch):
    calls = []
    waiting, release = threading.Event(), threading.Event()

    def wait_for_voice():
        calls.append("gate")
        waiting.set()
        assert release.wait(5)

    monkeypatch.setattr(server, "_low_priority", lambda priority: calls.append(("priority", priority)))
    monkeypatch.setattr(voice_turn, "wait_quiet", create_autospec(voice_turn.wait_quiet, side_effect=wait_for_voice))
    monkeypatch.setattr(
        server, "_faces_mod", create_autospec(server._faces_mod, side_effect=lambda: calls.append("faces imported"))
    )
    worker = server._warm_faces()
    try:
        assert waiting.wait(5)
        assert calls == [("priority", 19), "gate"]
        assert worker.is_alive()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert calls == [("priority", 19), "gate", "faces imported"]


@pytest.mark.parametrize(
    "path, cache", [("/faces/unknown/groups", "_FACE_GROUPS"), ("/voices/unknown/groups", "_VOICE_GROUPS")]
)
@pytest.mark.parametrize("blocked", ["owner", "compute"])
def test_cold_group_http_releases_the_read_slot_before_sources_finish(monkeypatch, path, cache, blocked):
    entered, release = threading.Event(), threading.Event()

    def source():
        entered.set()
        assert release.wait(5)
        return 0

    def compute():
        if blocked == "compute":
            source()
        return 200, {"groups": [], "count": 0, "skipped": []}

    groups = server._Fresh("cold endpoint", compute, source if blocked == "owner" else lambda: 0, lambda: 0)
    monkeypatch.setattr(server, cache, groups)
    with server.DashboardServer(("127.0.0.1", 0), server.H) as srv:
        srv._people_warm_started = True
        serving = threading.Thread(target=srv.serve_forever, daemon=True)
        serving.start()

        def get():
            connection = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=2)
            try:
                connection.request("GET", path)
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=1) as pool:
            request = pool.submit(get)
            try:
                assert entered.wait(2)
                code, body = request.result(timeout=1)
                assert code == 202
                assert body == {"pending": True, "groups": [], "count": None, "skipped": []}
            finally:
                release.set()
                request.result(timeout=5)
                srv.shutdown()
                serving.join(5)
        assert groups.get() == (200, {"groups": [], "count": 0, "skipped": []})


def get(srv, path):
    connection = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), json.loads(response.read() or b"null")
    finally:
        connection.close()


@pytest.fixture
def serving(monkeypatch):
    monkeypatch.setattr(server, "_warm_people", lambda: None)
    with server.DashboardServer(("127.0.0.1", 0), server.H) as srv:
        thread = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        try:
            yield srv
        finally:
            srv.shutdown()
            thread.join(5)


def test_a_route_the_broker_is_away_for_answers_503_starting_with_one_warning(monkeypatch, serving):
    from liberational import continuum_link

    outage_log = create_autospec(logging.Logger, instance=True)
    monkeypatch.setattr(continuum_link, "OUTAGE", continuum_link.Outage(outage_log))
    exceptions = []
    monkeypatch.setattr(server.LOG, "exception", lambda *a, **k: exceptions.append(a))

    def away(wait_s=None, cold=False):
        raise FileNotFoundError("No such file or directory (os error 2)")

    monkeypatch.setattr(server, "people_view", away)
    for _ in range(3):
        status, headers, body = get(serving, "/people")
        assert status == 503 and headers["Retry-After"] == "1"
        assert body["starting"] is True and "broker" in body["error"]
    assert outage_log.warning.call_count == 1 and exceptions == []


def test_a_route_that_fails_otherwise_still_logs_its_traceback(monkeypatch, serving):
    exceptions = []
    monkeypatch.setattr(server.LOG, "exception", lambda *a, **k: exceptions.append(a))
    monkeypatch.setattr(server, "people_view", create_autospec(server.people_view, side_effect=ValueError("bug")))
    connection = http.client.HTTPConnection("127.0.0.1", serving.server_port, timeout=5)
    try:
        connection.request("GET", "/people")
        with pytest.raises(http.client.RemoteDisconnected):
            connection.getresponse()
    finally:
        connection.close()
    assert exceptions == [("GET /people failed",)]


def test_a_slow_people_read_answers_from_the_last_roster_marked_stale():
    reads, release = server._PeopleReads(), threading.Event()
    assert reads.get((0,), lambda: {"people": ["Alice"]}, wait_s=5) == {"people": ["Alice"]}

    def slow():
        assert release.wait(5)
        return {"people": ["Alice", "Dana"]}

    try:
        assert reads.get((1,), slow, wait_s=0.05) == {"people": ["Alice"], "stale": True}
    finally:
        release.set()
    assert reads.get((1,), slow, wait_s=5) == {"people": ["Alice", "Dana"]}


def test_a_cold_people_read_that_cant_wait_says_starting_and_goes_on():
    reads, release, finished = server._PeopleReads(), threading.Event(), threading.Event()

    def slow():
        assert release.wait(5)
        finished.set()
        return {"people": ["Alice"]}

    with pytest.raises(server.Starting, match="still loading"):
        reads.get((0,), slow, wait_s=0.05)
    release.set()
    assert finished.wait(5)
    assert reads.get((0,), slow, wait_s=5) == {"people": ["Alice"]}


def test_a_people_read_that_times_out_itself_is_not_hidden_as_stale():
    reads = server._PeopleReads()
    reads.get((0,), lambda: {"people": []}, wait_s=5)

    def broker_timeout():
        raise TimeoutError("no reply from the broker within 10s")

    with pytest.raises(TimeoutError, match="no reply from the broker"):
        reads.get((1,), broker_timeout, wait_s=5)


def test_a_cold_people_request_says_starting_without_waiting_on_a_busy_registry(monkeypatch):
    # After a start the warm-ups keep the registry busy, and working out /people's key reads it.
    busy = threading.Event()
    monkeypatch.setattr(server, "_PEOPLE_READS", server._PeopleReads())
    monkeypatch.setattr(server, "PEOPLE_WAIT_S", 0.05)
    monkeypatch.setattr(server, "_people_view", lambda: {"people": ["Alice"]})
    monkeypatch.setattr(server, "_voices_key", lambda: busy.wait(5))
    monkeypatch.setattr(server, "_faces_owner_key", lambda **_: 0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(server.people_view, cold=True)
        try:
            with pytest.raises(server.Starting, match="still loading"):
                request.result(timeout=1)
        finally:
            busy.set()
    for _ in range(100):  # the shared cold read goes on and publishes
        if server._PEOPLE_READS.last is not None:
            break
        time.sleep(0.02)
    assert server.people_view() == {"people": ["Alice"]}


@pytest.mark.parametrize("warmed", [True, False], ids=["after-the-people-warm-up", "or-at-most-the-cap"])
def test_later_warm_ups_start_after_the_people_warm_up(monkeypatch, warmed):
    started = threading.Event()
    monkeypatch.setattr(server, "_PEOPLE_WARMED", threading.Event())
    monkeypatch.setattr(server, "WARM_FIRST_MAX_S", 5.0 if warmed else 0.05)
    thread = server._after_people_warm("test", started.set)
    if warmed:
        assert not started.wait(0.2), "nothing starts while the people warm-up runs"
        server._PEOPLE_WARMED.set()
    thread.join(5)
    assert started.is_set()


def test_a_change_made_during_a_cold_start_answers_with_its_own_read(roster, monkeypatch):
    # A name given while the first roster read is still out: its answer must not be that older read.
    state, held, _, _ = roster
    monkeypatch.setattr(server, "PEOPLE_WAIT_S", 2.0)  # as served (the suite waits reads out)
    held.clear()
    with pytest.raises(server.Starting):
        server.people_view(wait_s=0.05, cold=True)  # the page's GET starts the shared cold read of version 0
    for _ in range(250):  # it is reading version 0
        if state["calls"]:
            break
        time.sleep(0.02)
    assert state["calls"] == 1
    state["voices"], state["version"] = 1, 1  # then the owner names someone
    with ThreadPoolExecutor(max_workers=1) as pool:
        change = pool.submit(server.people_view)  # the naming handler's answer, as it asks
        time.sleep(0.3)  # long enough to join the cold read, if it would (nothing observable marks that it has)
        held.set()
        assert change.result(timeout=5) == {"version": 1}


def test_the_faces_check_runs_in_a_child_process_and_saves_its_result(tmp_path, monkeypatch):
    # In the dashboard's own process it held every request up for seconds every 10 minutes.
    monkeypatch.setattr(server, "STATE", tmp_path)
    imported_before = "liberational.vision.faces" in sys.modules
    check = server._run_check()
    assert (check["groups"], check["status"]) == (0, "unchecked")  # an install with no tags yet
    saved = json.loads((config.path("paths.faces") / "check.json").read_text())
    assert saved["ts"] == check["ts"], "the child saved what it reported, for live naming and autoname"
    assert ("liberational.vision.faces" in sys.modules) == imported_before, "nothing ran in this process"


def test_a_faces_check_child_that_fails_says_why(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(
        server.subprocess,
        "run",
        create_autospec(
            server.subprocess.run,
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="Traceback ...\nStorageError: broken"),
        ),
    )
    with pytest.raises(RuntimeError, match=r"exit 1: (?s:.*)StorageError: broken"):
        server._run_check()


# An owner change a patch can apply is served at once, patched, while the full answer is worked out behind it.
def _slow_kept(patch):
    """A _Fresh whose first answer is published at once and whose later computes wait for `release`."""
    owner = {"key": "a"}
    computing, release = threading.Event(), threading.Event()
    computed: list[str] = []

    def compute():
        computed.append(owner["key"])
        if len(computed) > 1:
            computing.set()
            assert release.wait(5)
        return 200, {"for": computed[-1]}

    kept = server._Fresh("patched", compute, lambda: owner["key"], lambda: 0, patch=patch)
    assert kept.get() == (200, {"for": "a"})
    return kept, owner, computing, release, computed


def test_an_owner_change_the_patch_applies_is_served_at_once_then_worked_out_behind_it():
    seen = []

    def patch(answer, before, now):
        seen.append((before, now))
        return 200, {**answer[1], "patched_to": now}

    kept, owner, computing, release, computed = _slow_kept(patch)
    owner["key"] = "b"
    assert kept.get() == (200, {"for": "a", "patched_to": "b"})  # without waiting for the slow compute
    assert seen == [("a", "b")]
    assert computing.wait(5)  # the full answer is under way in the background
    assert kept.get() == (200, {"for": "a", "patched_to": "b"})  # one background compute, not one per read
    release.set()
    deadline = time.monotonic() + 5
    while kept.get() != (200, {"for": "b"}) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert kept.get() == (200, {"for": "b"})
    assert computed == ["a", "b"]


def test_an_owner_change_the_patch_refuses_waits_for_the_full_answer():
    kept, owner, computing, release, _ = _slow_kept(lambda answer, before, now: None)
    owner["key"] = "b"
    with ThreadPoolExecutor(1) as pool:
        read = pool.submit(kept.get)
        assert computing.wait(5)
        assert not read.done()  # the read waits on the compute
        release.set()
        assert read.result(5) == (200, {"for": "b"})


def test_a_background_answer_is_published_even_when_the_owner_moved_on_while_it_ran():
    kept, owner, computing, release, _ = _slow_kept(lambda answer, before, now: (200, {**answer[1], "now": now}))
    owner["key"] = "b"
    kept.get()
    assert computing.wait(5)
    owner["key"] = "c"  # another name while "b" is worked out
    release.set()
    deadline = time.monotonic() + 5
    while kept.keys is None or kept.keys[0] != "b":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert kept.value == (200, {"for": "b"})  # kept under the key it was worked out for
    assert kept.get() == (200, {"for": "b", "now": "c"})  # and patched to the newest


FACE_ANSWER = (
    200,
    {
        "groups": [{"id": 5, "shots": []}, {"id": 6, "shots": []}],
        "count": 3,
        "skipped": [9],
        "_counted": [5, 6, 7],
    },
)


def _face_key(ids=(5, 6, 7), mark=1, rejects=1, purge=1, state="s", nots=1):
    return state, mark, ("skips",), ids, rejects, purge, nots


@pytest.fixture
def face_skips(monkeypatch):
    skips = {"face": [9.0], "voice": [], "line": []}
    monkeypatch.setattr(server, "inbox_skips", lambda: skips)
    return skips


@pytest.mark.parametrize(
    "now, skipped, groups, count",
    [
        pytest.param(_face_key(ids=(6, 7), mark=2), [9.0], [6], 2, id="named"),
        pytest.param(_face_key(ids=(5, 6), mark=2), [9.0], [5, 6], 2, id="unlisted-group-merged"),
        pytest.param(_face_key(), [9.0, 6.0], [5], 2, id="put-aside"),
        pytest.param(_face_key(ids=(5, 6, 7, 8)), [9.0], [5, 6], 3, id="new-group-waits-for-the-full-answer"),
    ],
)
def test_face_patch_takes_named_and_put_aside_groups_out_of_the_rows_and_the_count(
    face_skips, now, skipped, groups, count
):
    face_skips["face"] = skipped
    patched = server._patch_face_groups(FACE_ANSWER, _face_key(), now)
    assert patched is not None
    code, body = patched
    assert code == 200
    assert [g["id"] for g in body["groups"]] == groups
    assert body["count"] == count
    assert body["skipped"] == sorted(int(s) for s in skipped)
    assert body["refreshing"] is True


@pytest.mark.parametrize(
    "now",
    [
        pytest.param(_face_key(purge=2), id="purge"),
        pytest.param(_face_key(rejects=2), id="pictures-taken-out"),
        pytest.param(_face_key(state="other"), id="other-state"),
        pytest.param(_face_key(ids=None), id="groups-unreadable"),
    ],
)
def test_face_patch_refuses_what_it_cannot_apply(face_skips, now):
    assert server._patch_face_groups(FACE_ANSWER, _face_key(), now) is None


def test_face_patch_applies_a_not_mark_to_the_rows_at_once(face_skips, monkeypatch):
    """ "Not Kara" on Face 5 takes Kara off its guess in the patched answer, without the 33 s recompute."""
    from liberational.people import face_nots

    monkeypatch.setattr(face_nots, "names", lambda state: {5: {"Kara"}})
    answer = (
        200,
        {
            **FACE_ANSWER[1],
            "groups": [
                {"id": 5, "shots": [], "guess": {"name": "Kara"}, "score": 80, "band": "probable"},
                {"id": 6, "shots": [], "guess": {"name": "Sam"}, "score": 60, "band": "maybe"},
            ],
        },
    )
    patched = server._patch_face_groups(answer, _face_key(), _face_key(nots=2))
    assert patched is not None
    five, six = patched[1]["groups"]
    assert (five["guess"], five["band"], five["not"]) == (None, "none", ["Kara"])
    assert (six["guess"], six["not"]) == ({"name": "Sam"}, [])


def test_face_patch_refuses_an_answer_that_failed(face_skips):
    assert server._patch_face_groups((500, {"error": "x"}), _face_key(), _face_key(mark=2)) is None


def test_two_patches_in_a_row_count_from_the_kept_answer(face_skips):
    once = server._patch_face_groups(FACE_ANSWER, _face_key(), _face_key(ids=(6, 7), mark=2))
    assert once is not None
    twice = server._patch_face_groups(FACE_ANSWER, _face_key(), _face_key(ids=(7,), mark=3))
    assert twice is not None
    assert (twice[1]["groups"], twice[1]["count"]) == ([], 1)


def test_the_reply_leaves_the_patch_bookkeeping_out(monkeypatch):
    monkeypatch.setattr(server, "_shot_revision", lambda: 0)
    monkeypatch.setattr(server.face_shots, "saved", lambda *a, **k: set())
    code, body = server._ready_face_groups(*FACE_ANSWER)
    assert code == 200
    assert "_counted" not in body
    assert body["count"] == 3


def _clip(line, ts):
    return {"ts": ts, "line": line, "text": "hi", "dur": 2.0, "url": f"/clip?line={line}"}


VOICE_ANSWER = (
    200,
    {
        "groups": [
            {"id": 5, "label": "Speaker 5", "clips": [_clip("mic@100.000", 100.0), _clip("mic@101.000", 101.0)]},
            {"id": 6, "label": "Speaker 6", "clips": [_clip("mic@200.000", 200.0)]},
        ],
        "count": 2,
        "skipped": [9],
    },
)


def _voice_key(voiceprints=1, skips=1, clones=1, mark=1, purge=1, review=1, state="s"):
    return state, voiceprints, skips, clones, mark, purge, review


@pytest.fixture
def voice_state(monkeypatch, tmp_path):
    """A real event log under a throwaway state directory, and the put-aside voices."""
    from liberational.people import registry

    skips = {"face": [], "voice": [9.0], "line": []}
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "inbox_skips", lambda: skips)
    yield registry.at(tmp_path), skips
    registry.close_all()


@pytest.mark.parametrize(
    "named, skipped, now, groups",
    [
        pytest.param([6], [9.0], _voice_key(mark=2), [5], id="named"),
        pytest.param([], [9.0, 5.0], _voice_key(skips=2), [6], id="put-aside"),
        pytest.param([], [9.0], _voice_key(voiceprints=2), [5, 6], id="new-voiceprint-handed-out"),
        pytest.param([], [9.0], _voice_key(clones=2), [5, 6], id="clone-made"),
    ],
)
def test_voice_patch_takes_named_and_put_aside_voices_out_of_the_rows_and_the_count(
    voice_state, named, skipped, now, groups
):
    from liberational.people import registry

    reg, skips = voice_state
    for n in named:
        reg.claim(registry.voiceprint(n), "Alice", registry.Basis.OWNER_TAG, "test")
    skips["voice"] = skipped
    patched = server._patch_voice_groups(VOICE_ANSWER, _voice_key(), now)
    assert patched is not None
    code, body = patched
    assert code == 200
    assert [g["id"] for g in body["groups"]] == groups
    assert body["count"] == len(groups)
    assert body["skipped"] == sorted(int(s) for s in skipped)
    assert body["refreshing"] is True


def test_voice_patch_refuses_another_state(voice_state):
    assert server._patch_voice_groups(VOICE_ANSWER, _voice_key(), _voice_key(state="other")) is None


@pytest.fixture
def voice_lines():
    """The speech lines VOICE_ANSWER's clips play, in the speech log every test has (tests/conftest.py)."""
    from speech_fake import fake

    for ts in (100.0, 101.0, 200.0):
        fake().say({"ts_start": ts, "device": "mic", "text": "hi"})
    return fake()


def test_after_a_purge_the_patch_drops_the_purged_lines_clips_at_once(voice_state, voice_lines):
    from liberational import speech_log

    voice_lines.purge(speech_log.UTTERANCES, lambda e: e["ts_start"] == 100.0)  # a forgotten person's line
    patched = server._patch_voice_groups(VOICE_ANSWER, _voice_key(), _voice_key(purge=2))
    assert patched is not None
    code, body = patched
    assert code == 200
    assert {g["id"]: [c["line"] for c in g["clips"]] for g in body["groups"]} == {
        5: ["mic@101.000"],
        6: ["mic@200.000"],
    }
    assert body["refreshing"] is True


def test_after_a_correction_the_patch_drops_a_voice_whose_only_clip_it_took(voice_state, voice_lines, monkeypatch):
    from liberational.listen import revisions

    monkeypatch.setattr(revisions, "filter_lines", lambda state, rows: [r for r in rows if r["ts_start"] != 200.0])
    patched = server._patch_voice_groups(VOICE_ANSWER, _voice_key(), _voice_key(review=2))
    assert patched is not None
    assert ([g["id"] for g in patched[1]["groups"]], patched[1]["count"]) == ([5], 1)


def test_the_patch_waits_for_the_full_answer_when_the_kept_lines_cant_be_read(voice_state, voice_lines, monkeypatch):
    from liberational import speech_log

    def unreadable():
        raise OSError("speech log unreadable")

    monkeypatch.setattr(speech_log.shared(), "all_lines", unreadable)
    assert server._patch_voice_groups(VOICE_ANSWER, _voice_key(), _voice_key(purge=2)) is None


def test_after_a_purge_who_s_this_answers_at_once_while_the_full_answer_is_worked_out(
    voice_state, voice_lines, monkeypatch
):
    from liberational import speech_log

    voice_lines.purge(speech_log.UTTERANCES, lambda e: e["ts_start"] == 100.0)
    pending, submitted = Future(), threading.Event()
    pool = create_autospec(server.ProcessPoolExecutor, instance=True)

    def submit(fn):
        submitted.set()
        return pending  # the full answer, still being worked out

    pool.submit.side_effect = submit
    monkeypatch.setattr(server, "_groups_in_worker", True)
    monkeypatch.setattr(server, "_voices_executor", pool)
    cache = server._Fresh(
        "purged-voice-test",
        server._VOICE_GROUPS.compute,
        lambda: _voice_key(purge=2),
        lambda: 0,
        patch=server._VOICE_GROUPS.patch,
    )
    cache.value, cache.keys = VOICE_ANSWER, (_voice_key(), 0)
    try:
        code, answer = cache.get()
        assert submitted.wait(5), "the full answer was not asked for"
        assert not pending.done()  # answered without it
        assert code == 200 and answer["refreshing"] is True
        assert [c["line"] for c in answer["groups"][0]["clips"]] == ["mic@101.000"]
    finally:
        refresh = next(t for t in threading.enumerate() if t.name == "fresh-purged-voice-test")
        pending.set_result(VOICE_ANSWER)
        refresh.join(5)
        assert not refresh.is_alive()


def test_voice_patch_refuses_an_answer_that_failed(voice_state):
    assert server._patch_voice_groups((500, {"error": "x"}), _voice_key(), _voice_key(mark=2)) is None


@pytest.mark.parametrize(
    ("aside", "clips"),
    [
        pytest.param(["mic@100.000"], {5: ["mic@101.000"], 6: ["mic@200.000"]}, id="by-key"),
        pytest.param([101.0], {5: ["mic@100.000"], 6: ["mic@200.000"]}, id="by-millisecond-as-before-247"),
        pytest.param(["mic@200.000"], {5: ["mic@100.000", "mic@101.000"]}, id="its-last-clip-takes-the-row"),
    ],
)
def test_voice_patch_drops_a_clip_answered_nobody(voice_state, aside, clips):
    """ "Nobody" on a clip puts its line aside; the patched answer must not play it again."""
    _, skips = voice_state
    skips["line"] = aside
    patched = server._patch_voice_groups(VOICE_ANSWER, _voice_key(), _voice_key(skips=2))
    assert patched is not None
    _, body = patched
    assert {g["id"]: [c["line"] for c in g["clips"]] for g in body["groups"]} == clips
    assert body["count"] == len(clips)


def test_voice_patch_drops_a_clip_marked_multiple_speakers_or_given_to_someone(voice_state):
    """Marking "Multiple speakers" on a clip clears it immediately rather than waiting until the list is rebuilt."""
    from liberational.people import registry

    reg, _ = voice_state
    reg.detach(registry.utterance(100.0, "mic"), "multiple speakers", "test")
    patched = server._patch_voice_groups(VOICE_ANSWER, _voice_key(), _voice_key(mark=2))
    assert patched is not None
    assert {g["id"]: [c["line"] for c in g["clips"]] for g in patched[1]["groups"]} == {
        5: ["mic@101.000"],
        6: ["mic@200.000"],
    }
    reg.claim(registry.utterance(200.0, "mic"), "Alice", registry.Basis.OWNER_TAG, "test")
    patched = server._patch_voice_groups(VOICE_ANSWER, _voice_key(), _voice_key(mark=3))
    assert patched is not None
    assert [g["id"] for g in patched[1]["groups"]] == [5], "its only clip was Alice's: the voice goes"


def test_voice_patch_waits_for_the_full_answer_when_the_names_cant_be_read(voice_state, monkeypatch):
    def unreadable(state, sids):
        raise OSError("event log unreadable")

    monkeypatch.setattr(
        server.registry, "voice_names", create_autospec(server.registry.voice_names, side_effect=unreadable)
    )
    assert server._patch_voice_groups(VOICE_ANSWER, _voice_key(), _voice_key(mark=2)) is None


@pytest.mark.parametrize("edit", ["nobody", "multiple", "assigned"])
def test_voice_clip_correction_applies_while_worker_recompute_is_blocked(voice_state, monkeypatch, edit):
    from liberational.people import registry

    reg, skips = voice_state
    corrections = {
        "nobody": lambda: skips["line"].append("mic@100.000"),
        "multiple": lambda: reg.detach(registry.utterance(100.0, "mic"), "multiple speakers", "test"),
        "assigned": lambda: reg.claim(registry.utterance(100.0, "mic"), "Alice", registry.Basis.OWNER_TAG, "test"),
    }
    corrections[edit]()
    pending, submitted = Future(), threading.Event()
    pool = create_autospec(server.ProcessPoolExecutor, instance=True)

    def submit(fn):
        assert fn is server._unknown_voices_job
        submitted.set()
        return pending

    pool.submit.side_effect = submit
    monkeypatch.setattr(server, "_groups_in_worker", True)
    monkeypatch.setattr(server, "_voices_executor", pool)
    cache = server._Fresh(
        "blocked-voice-test",
        server._VOICE_GROUPS.compute,
        lambda: _voice_key(mark=2, skips=2),
        lambda: 0,
        patch=server._VOICE_GROUPS.patch,
    )
    cache.value, cache.keys = VOICE_ANSWER, (_voice_key(), 0)
    try:
        code, answer = cache.get()
        assert submitted.wait(5), "voice refresh did not reach its worker"
        assert not pending.done()
        assert code == 200 and answer["refreshing"] is True
        assert {g["id"]: [c["line"] for c in g["clips"]] for g in answer["groups"]} == {
            5: ["mic@101.000"],
            6: ["mic@200.000"],
        }
    finally:
        refresh = next(t for t in threading.enumerate() if t.name == "fresh-blocked-voice-test")
        pending.set_result(VOICE_ANSWER)
        refresh.join(5)
        assert not refresh.is_alive()


def test_an_answer_the_patch_brings_up_to_date_counts_as_ready():
    kept, owner, *_ = _slow_kept(lambda answer, before, now: answer if now == "b" else None)
    owner["key"] = "b"
    assert kept.ready()
    owner["key"] = "c"  # one the patch refuses
    assert not kept.ready()


def test_a_request_waits_for_a_knowledge_lock_only_briefly(monkeypatch):
    # Request threads piling up behind a knowledge lock were the 11-07 overload: they fail fast instead.
    seen = []
    monkeypatch.setattr(
        http.server.BaseHTTPRequestHandler,
        "handle_one_request",
        lambda self: seen.append(knowledge._budget(knowledge.LOCK_WAIT_S)),
    )
    server.H.__new__(server.H).handle_one_request()
    assert seen == [knowledge.REQUEST_WAIT_S]
    assert knowledge._budget(knowledge.LOCK_WAIT_S) == knowledge.LOCK_WAIT_S  # and only inside the request
