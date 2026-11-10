"""Slow dashboard views kept fresh (server._Fresh): the owner's changes show at once, new data a moment later."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import create_autospec

import pytest

from liberational.dashboard import server
from liberational.vision import faces


def view(answers: list[tuple[int, dict]]):
    owner, data = {"v": 0}, {"v": 0}
    gate = threading.Event()
    calls: list[int] = []

    def compute() -> tuple[int, dict]:
        calls.append(len(calls))
        if len(calls) > 1:
            gate.wait(5)  # a background refresh waits until the test lets it go
        return answers[min(len(calls), len(answers)) - 1]

    fresh = server._Fresh("test", compute, lambda: owner["v"], lambda: data["v"])
    return fresh, owner, data, gate, calls


def test_new_data_serves_the_kept_answer_and_refreshes_behind_it():
    fresh, _, data, gate, calls = view([(200, {"n": 1}), (200, {"n": 2})])
    assert fresh.get() == (200, {"n": 1})
    data["v"] = 1  # a line heard since
    assert fresh.get() == (200, {"n": 1}), "the kept answer, at once"
    gate.set()
    for _ in range(100):
        if fresh.value == (200, {"n": 2}):
            break
        threading.Event().wait(0.02)
    assert fresh.get() == (200, {"n": 2}) and len(calls) == 2


def test_the_owners_change_is_worked_out_at_once():
    fresh, owner, _, gate, calls = view([(200, {"n": 1}), (200, {"n": 2})])
    gate.set()
    fresh.get()
    owner["v"] = 1  # a voice named
    assert fresh.get() == (200, {"n": 2}) and len(calls) == 2


def test_a_purge_bypasses_cached_face_groups_immediately(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "INBOX_SKIPS", tmp_path / "skips.json")
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown.npz")
    epoch = {"purge": None}
    log = create_autospec(server.speech_log.SpeechLog, instance=True)
    log.newest_voice_purge.side_effect = lambda: epoch["purge"]
    monkeypatch.setattr(server.speech_log, "shared", create_autospec(server.speech_log.shared, return_value=log))
    groups = ["Face 1"]
    fresh = server._Fresh("face groups", lambda: (200, {"groups": groups.copy()}), server._faces_owner_key, lambda: 0)
    assert fresh.get() == (200, {"groups": ["Face 1"]})
    groups.clear()
    epoch["purge"] = "purge-after-ingest-delete"
    assert fresh.get() == (200, {"groups": []})


def test_a_video_only_purge_leaves_the_face_groups_key_as_it_was(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "INBOX_SKIPS", tmp_path / "skips.json")
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown.npz")
    purges = {"any": "a", "voice": "a"}
    log = create_autospec(server.speech_log.SpeechLog, instance=True)
    log.newest_purge.side_effect = lambda: purges["any"]
    log.newest_voice_purge.side_effect = lambda: purges["voice"]
    monkeypatch.setattr(server.speech_log, "shared", create_autospec(server.speech_log.shared, return_value=log))
    before = server._faces_owner_key(), server._faces_owner_key(groups=False)
    purges["any"] = "nightly-video-window"  # SpeechLog.newest_voice_purge leaves a video-only purge out
    assert (server._faces_owner_key(), server._faces_owner_key(groups=False)) == before
    purges["voice"] = purges["any"] = "a-forget"
    assert server._faces_owner_key() != before[0] and server._faces_owner_key(groups=False) != before[1]


def test_an_error_is_not_kept():
    fresh, _, _, gate, calls = view([(500, {"error": "x"}), (200, {"n": 1})])
    gate.set()
    assert fresh.get()[0] == 500
    assert fresh.get() == (200, {"n": 1}) and len(calls) == 2


def test_concurrent_cold_readers_share_one_computation(monkeypatch):
    computing = threading.Event()
    release = threading.Event()
    second_work = threading.Event()
    entries_lock = threading.Lock()
    calls = []

    def compute():
        calls.append("computed")
        computing.set()
        assert release.wait(5), "the test did not release the cold computation"
        return 200, {"n": 1}

    fresh = server._Fresh("cold", compute, lambda: 0, lambda: 0)
    work = fresh._work
    entries = 0

    def observed_work(keys):
        nonlocal entries
        with entries_lock:
            entries += 1
            if entries == 2:
                second_work.set()  # both get calls already decided the cache was cold
        return work(keys)

    monkeypatch.setattr(fresh, "_work", create_autospec(work, side_effect=observed_work))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(fresh.get)
        try:
            assert computing.wait(5)
            second = pool.submit(fresh.get)
            assert second_work.wait(5)
            assert not second.done()
        finally:
            release.set()
        assert first.result(timeout=5) == (200, {"n": 1})
        assert second.result(timeout=5) == (200, {"n": 1})
    assert calls == ["computed"]


def test_a_raised_initial_computation_can_be_retried():
    calls = []

    def compute():
        calls.append("computed")
        if len(calls) == 1:
            raise RuntimeError("view source unavailable")
        return 200, {"n": 1}

    fresh = server._Fresh("retry", compute, lambda: 0, lambda: 0)
    with pytest.raises(RuntimeError, match="view source unavailable"):
        fresh.get()
    assert fresh.get() == (200, {"n": 1})
    assert calls == ["computed", "computed"]


def test_owner_change_bypasses_old_refresh_and_keeps_its_newer_answer(monkeypatch):
    owner, data = {"v": 0}, {"v": 0}
    refreshing, release = threading.Event(), threading.Event()
    owner_computing, refreshed = threading.Event(), threading.Event()
    calls = []

    def compute():
        revision = owner["v"], data["v"]
        calls.append(revision)
        if revision == (0, 1):
            refreshing.set()
            assert release.wait(15)
        if revision[0] == 1:
            owner_computing.set()
        return 200, {"revision": revision}

    fresh = server._Fresh("owner change", compute, lambda: owner["v"], lambda: data["v"])
    refresh = fresh._refresh

    def observed_refresh(keys):
        try:
            return refresh(keys)
        finally:
            refreshed.set()  # cache publication and the busy flag have both completed

    monkeypatch.setattr(fresh, "_refresh", create_autospec(refresh, side_effect=observed_refresh))
    assert fresh.get() == (200, {"revision": (0, 0)})
    data["v"] = 1
    assert fresh.get() == (200, {"revision": (0, 0)})
    assert refreshing.wait(5)
    owner["v"] = 1
    with ThreadPoolExecutor(max_workers=1) as pool:
        corrected = pool.submit(fresh.get)
        try:
            assert owner_computing.wait(5), "an owner change waited behind the older data refresh"
            assert corrected.result(timeout=5) == (200, {"revision": (1, 1)})
        finally:
            release.set()
    assert refreshed.wait(5)
    assert fresh.get() == (200, {"revision": (1, 1)})
    assert calls == [(0, 0), (0, 1), (1, 1)]


def test_old_owner_sample_cannot_replace_a_newer_published_answer(monkeypatch):
    owner, data = {"v": 0}, {"v": 0}
    sampled, release, refreshed = threading.Event(), threading.Event(), threading.Event()
    completion = threading.local()
    calls = []

    def owner_key():
        revision = owner["v"]
        if getattr(completion, "old_owner", False):
            completion.old_owner = False
            sampled.set()  # this completion has already sampled owner 0
            assert release.wait(15)
        return revision

    def compute():
        revision = owner["v"], data["v"]
        calls.append(revision)
        completion.old_owner = revision == (0, 1)
        return 200, {"revision": revision}

    fresh = server._Fresh("publication", compute, owner_key, lambda: data["v"])
    refresh = fresh._refresh

    def observed_refresh(keys):
        try:
            return refresh(keys)
        finally:
            refreshed.set()

    monkeypatch.setattr(fresh, "_refresh", create_autospec(refresh, side_effect=observed_refresh))
    assert fresh.get() == (200, {"revision": (0, 0)})
    data["v"] = 1
    assert fresh.get() == (200, {"revision": (0, 0)})
    assert sampled.wait(5)
    owner["v"] = 1
    with ThreadPoolExecutor(max_workers=1) as pool:
        corrected = pool.submit(fresh.get)
        try:
            assert corrected.result(timeout=5) == (200, {"revision": (1, 1)})
        finally:
            release.set()
    assert refreshed.wait(5)
    assert fresh.get() == (200, {"revision": (1, 1)})
    assert calls == [(0, 0), (0, 1), (1, 1)]


def test_ready_requires_a_published_success_for_current_owner_but_allows_new_data():
    fresh, owner, data, gate, calls = view([(200, {"n": 1}), (200, {"n": 2})])
    gate.set()
    assert not fresh.ready()
    assert fresh.get() == (200, {"n": 1})
    assert fresh.ready()
    data["v"] = 1
    assert fresh.ready()
    assert calls == [0]  # readiness never starts a refresh
    owner["v"] = 1
    assert not fresh.ready()
    assert fresh.get() == (200, {"n": 2})
    assert fresh.ready()


def test_ready_rejects_success_that_was_not_published_after_owner_changed():
    owner = {"v": 0}

    def compute():
        owner["v"] += 1
        return 200, {"n": 1}

    fresh = server._Fresh("unpublished", compute, lambda: owner["v"], lambda: 0)
    assert fresh.get() == (200, {"n": 1})
    assert fresh.value is None
    assert not fresh.ready()


def test_ready_never_calls_owner_key_while_holding_the_publication_lock():
    def owner():
        assert fresh.lock.acquire(blocking=False), "owner callbacks must run outside the publication lock"
        try:
            return 0
        finally:
            fresh.lock.release()

    fresh = server._Fresh("owner callback", lambda: (200, {"n": 1}), owner, lambda: 0)
    assert fresh.get() == (200, {"n": 1})
    assert fresh.ready()


def test_failed_computation_is_not_ready():
    fresh, _, _, gate, _calls = view([(500, {"error": "unavailable"})])
    gate.set()
    assert fresh.get()[0] == 500
    assert not fresh.ready()


@pytest.mark.parametrize("failure", [False, True])
def test_cold_polls_share_background_work_and_surface_failure(failure):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def compute():
        calls.append("read")
        entered.set()
        assert release.wait(5)
        return (500, {"error": "unavailable"}) if failure else (200, {"groups": [], "count": 0})

    fresh = server._Fresh("cold polling", compute, lambda: 0, lambda: 0)
    pending = (202, {"pending": True, "groups": [], "count": None, "skipped": []})
    try:
        assert fresh.poll() == pending
        assert entered.wait(2)
        assert fresh.poll() == pending
        assert calls == ["read"]
        work = fresh.cold
        assert work is not None
    finally:
        release.set()
    assert work.result(timeout=5)[0] == (500 if failure else 200)
    expected = (500, {"error": "unavailable"}) if failure else (200, {"groups": [], "count": 0})
    assert fresh.poll() == expected
    assert calls == ["read"]


def test_warm_poll_validates_owner_instead_of_serving_a_cold_completion():
    owner = {"v": 0}
    fresh = server._Fresh("poll correction", lambda: (200, {"revision": owner["v"]}), lambda: owner["v"], lambda: 0)
    assert fresh.poll()[0] == 202
    work = fresh.cold
    assert work is not None
    assert work.result(timeout=5) == (200, {"revision": 0})
    owner["v"] = 1
    assert fresh.poll() == (200, {"revision": 1})


def test_cold_poll_reports_a_raised_source_error_and_allows_retry():
    calls = []

    def compute():
        calls.append("read")
        if len(calls) == 1:
            raise RuntimeError("source unavailable")
        return 200, {"groups": [], "count": 0}

    fresh = server._Fresh("cold failure", compute, lambda: 0, lambda: 0)
    assert fresh.poll()[0] == 202
    work = fresh.cold
    assert work is not None
    with pytest.raises(RuntimeError, match="source unavailable"):
        work.result(timeout=5)
    with pytest.raises(RuntimeError, match="source unavailable"):
        fresh.poll()
    assert fresh.poll()[0] == 202
    retry = fresh.cold
    assert retry is not None
    assert retry.result(timeout=5) == (200, {"groups": [], "count": 0})
    assert fresh.poll() == (200, {"groups": [], "count": 0})
