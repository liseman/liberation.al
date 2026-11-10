"""Review's chips: who and what was around during a day, as time spans (dashboard/presence.py)."""

import time
from concurrent.futures import Future

import pytest

from liberational import config as cfg
from liberational import memory_log
from liberational.dashboard import presence as pr
from liberational.domain.identity import Basis
from liberational.listen import revisions
from liberational.memory import presence_counts
from liberational.people import registry
from presence_fake import add_frames
from speech_fake import fake

DAY = "2026-10-23"
T0 = time.mktime(time.strptime(DAY, "%Y-%m-%d")) + 20 * 3600  # 20:00 that day


@pytest.fixture(autouse=True)
def presence_on():
    """Presence is off by default while it is rebuilt (#1808); these tests are of its chips."""
    with cfg.override({"presence": {"enabled": True}}):
        yield


@pytest.fixture
def state(tmp_path, monkeypatch, vision):
    state = tmp_path
    for name, start, end in [("Alice", T0, T0 + 60), ("alice", T0 + 65, T0 + 90), ("Alice", T0 - 86400, T0 - 86000)]:
        vision.visit(name, start, end, "cam0", best_sim=0.8, samples=5)
    for i in range(10):  # one observation every 10 s; a grounding at 0 s and 30 s, then one at 60 s
        g = T0 + 30 * (i * 10 // 30)
        items = [{"label": "Guitars", "ts": g}]  # a caption's boxes are one grounding's, so they share its time
        if g == T0:
            items.append({"label": "lamp", "ts": g})
        if i == 9:
            items.append({"label": "Television", "ts": g})
        vision.caption(T0 + 10 * i, caption="", objects=items)
    vision.caption(T0 + 1000, caption="", objects=[{"label": "tv", "ts": T0 + 1000}])
    vision.caption(T0 - 86400, caption="", objects=[{"label": "lamp", "ts": T0 - 86400}])  # the day before
    lines = [
        {"ts_start": T0 + 95, "ts_end": T0 + 97, "speaker": "Alice", "text": "hi"},
        {"ts_start": T0 + 300, "ts_end": T0 + 301, "speaker": "Speaker 4", "text": "who"},
        {"ts_start": T0 + 310, "ts_end": T0 + 311, "speaker": "unknown", "text": "?"},
        {"ts_start": T0 + 400, "ts_end": T0 + 402, "speaker": "Dana", "text": "hello"},
    ]
    fake().write(lines)
    said = [
        {"ts_start": T0 + 5, "text": "voice ready", "status": True},
        {"ts_start": T0 + 98, "ts_end": T0 + 100, "text": "Hello Alice."},
    ]
    [fake().reply(x) for x in said]
    md = (
        f"# Friday {DAY}\n\n## What happened\n"
        f"- [20:00](#t={T0})-[20:01](#t={T0 + 60}) Alice & Dana: plan the **trip** .\n"
        f"- [20:10](#t={T0 + 600}) Dana: asks about dinner.\n"
        "- no link here\n"
        f"\n## Talked about\n- Alice will book it [h1](#t={T0 + 300}).\n"
        f"\n## People present\n- Alice (seen): [20:00](#t={T0}) to [20:01](#t={T0 + 60})\n"
    )
    day_page(state, md)
    monkeypatch.setattr(cfg, "assistant_name", lambda: "Bob")
    monkeypatch.setattr(pr, "_cache", {})
    return state


def day_page(state, md, supersedes=None):
    """The day's journal page (memory/journal_pages.py), as the journal writes it: spanning the whole local day."""
    a, b = memory_log.day_bounds(DAY)
    record = {"ts_start": a, "ts_end": b, "level": "day", "period": DAY, "md": md}
    if supersedes:
        record["supersedes"] = supersedes
    return memory_log.at(state).append(memory_log.PAGES, record)


def current_page(state):
    page = memory_log.at(state).page("day", DAY)
    assert page is not None
    return page["id"]


def chips(state):
    return {c["id"]: c for c in pr.presence(DAY, state)["chips"]}


def test_rejected_speech_invalidates_an_ended_days_cached_person_chips(state):
    assert chips(state)["person:dana"]["spans"] == [[T0 + 400, T0 + 402]]
    row = next(e for e in fake().all_lines() if e.get("speaker") == "Dana")
    token = revisions.reject_lines(state, [row])
    assert "person:dana" not in chips(state)
    revisions.undo_correction(state, token)
    assert chips(state)["person:dana"]["spans"] == [[T0 + 400, T0 + 402]]


def test_people_merge_visits_and_lines_across_case_and_skip_unnamed(state):
    c = chips(state)
    assert c["person:alice"]["spans"] == [[T0, T0 + 97]]  # 60 s, 65 to 90 s, and a line at 95 s, joined
    assert c["person:dana"]["spans"] == [[T0 + 400, T0 + 402]]
    assert "person:speaker 4" not in c
    assert "person:unknown" not in c


def test_a_person_chip_says_when_each_camera_view_saw_them(state, vision):
    """#328: one lane per camera view. Speech has no camera, so a person only heard carries no "cams"."""
    vision.visit("Alice", T0 + 500, T0 + 520, "cam1", best_sim=0.8, samples=3)
    got = chips(state)
    assert got["person:alice"]["cams"] == {"cam0": [[T0, T0 + 90]], "cam1": [[T0 + 500, T0 + 520]]}
    assert "cams" not in got["person:dana"]


def test_person_chips_keep_speech_independent_of_overlapping_and_other_camera_visits(state, vision):
    fake().say({"ts_start": T0 + 20, "ts_end": T0 + 22, "speaker": "Alice", "text": "within a visit"})
    vision.visit("Alice", T0 + 500, T0 + 520, "cam1", best_sim=0.8, samples=3)
    got = chips(state)
    assert got["person:alice"]["speech"] == [[T0 + 20, T0 + 22], [T0 + 95, T0 + 97]]
    assert got["person:alice"]["cams"] == {"cam0": [[T0, T0 + 90]], "cam1": [[T0 + 500, T0 + 520]]}
    assert got["person:alice"]["spans"] == [[T0, T0 + 97], [T0 + 500, T0 + 520]]
    assert got["person:dana"]["speech"] == [[T0 + 400, T0 + 402]]


def test_faces_nobody_named_make_one_unknown_faces_chip_per_camera_view(state, vision):
    rows = [  # (ts, cam, name): unnamed on cam1 twice 5 s apart, then 25 s later; once on cam0; one named
        (T0 + 1000, "cam1", None),
        (T0 + 1005, "cam1", None),
        (T0 + 1030, "cam1", None),
        (T0 + 2000, "cam0", None),
        (T0 + 3000, "cam0", "Alice"),
        (T0 - 86400, "cam0", None),  # the day before
    ]
    for ts, cam, name in rows:
        vision.face(ts, cam, name, 0.6 if name else None)
    u = chips(state)[pr.UNKNOWN_FACES]
    assert (u["kind"], u["label"]) == ("person", "Unknown faces")
    assert u["cams"] == {"cam0": [[T0 + 2000, T0 + 2001]], "cam1": [[T0 + 1000, T0 + 1005], [T0 + 1030, T0 + 1031]]}
    assert u["spans"] == [[T0 + 1000, T0 + 1005], [T0 + 1030, T0 + 1031], [T0 + 2000, T0 + 2001]]


def test_objects_are_normalized_and_need_two_groundings(state):
    c = chips(state)
    assert c["object:guitar"]["label"] == "guitar"
    assert c["object:guitar"]["spans"] == [[T0, T0 + 90 + pr.OBJECT_PAD]]
    assert c["object:tv"]["count"] == 2  # "Television" at 90 s and "tv" at 1000 s are one kind of thing
    assert "object:lamp" not in c  # only ever in one grounding


def test_events_come_from_the_day_pages_linked_bullets_and_spoken_replies(state):
    c = chips(state)
    happened = c["event:what happened"]
    assert happened["spans"] == [[T0, T0 + 60], [T0 + 600, T0 + 600 + pr.EVENT_PAD]]
    assert happened["notes"][0] == "Alice & Dana: plan the trip."
    assert c["event:talked about"]["count"] == 1
    assert "event:people present" not in c
    assert c["event:bob spoke"]["spans"] == [[T0 + 98, T0 + 100]]  # status lines aren't speech
    assert c["event:bob spoke"]["notes"] == ["Hello Alice."]


def test_the_sections_code_writes_make_no_event_chips(state):
    md = (
        f"# Friday {DAY}\n\n## Ate\n- Dinner: cake [h2](#t={T0 + 1200}).\n"
        f"\n## People present\n- Dana (heard): [20:01](#t={T0 + 60}) to [20:02](#t={T0 + 120})\n"
        "\n## Counts\n| What | Today | Usual |\n| --- | ---: | --- |\n| Door opened | 3 | 2 to 5 |\n"
        f"\n## Evidence\n- h2: [20:20](#t={T0 + 1200}) to [20:30](#t={T0 + 1800})\n"
    )
    day_page(state, md, supersedes=current_page(state))
    events = sorted(k for k in chips(state) if k.startswith("event:"))
    assert events == ["event:ate", "event:bob spoke"]


def test_a_rewritten_day_page_replaces_the_events_at_once(state):
    assert "event:what happened" in chips(state)  # cached now
    md = f"# Friday {DAY}\n\n## Ate\n- Dinner: cake [h2](#t={T0 + 1200}).\n"
    day_page(state, md, supersedes=current_page(state))
    c = chips(state)
    assert "event:ate" in c and "event:what happened" not in c  # the current page of the day, not the cache


def test_order_and_cache(state):
    kinds = [c["kind"] for c in pr.presence(DAY, state)["chips"]]
    assert kinds == sorted(kinds, key=pr.KINDS.index)
    first = pr.presence(DAY, state)
    assert pr.presence(DAY, state) is first  # unchanged sources: the cached result
    fake().say({"ts_start": T0 + 900, "ts_end": T0 + 901, "speaker": "Erin", "text": "yo"})
    assert "person:erin" in chips(state)


def test_a_label_only_in_the_log_refreshes_the_cache(state):
    assert "person:erin" not in chips(state)
    pr.presence(DAY, state)  # cached
    registry.at(state).claim(registry.utterance(T0 + 310), "Erin", Basis.OWNER_TAG, "dashboard")
    assert "person:erin" in chips(state)


def test_missing_sources_give_no_chips(tmp_path):
    assert pr.presence(DAY, tmp_path)["chips"] == []


@pytest.mark.parametrize(
    "raw, want",
    [
        ("Sewing_Machines", "sewing machine"),
        ("boxes", "box"),
        ("glasses", "glass"),
        ("bus", "bus"),
        ("couch", "sofa"),
        ("Batteries", "battery"),
        ("", ""),
        pytest.param("TVs", "tv", id="short-plural"),
        pytest.param("tv", "tv", id="short-singular"),
        pytest.param("news", "news", id="news"),
        pytest.param("lens", "lens", id="lens"),
        pytest.param("canvas", "canvas", id="canvas"),
        pytest.param("camera lenses", "camera lens", id="lens-plural"),
    ],
)
def test_object_label(raw, want):
    assert pr.object_label(raw) == want


@pytest.mark.parametrize(
    "setting, chip, expected",
    [
        pytest.param(
            "person_gap_s",
            "person:alice",
            [[T0, T0 + 60], [T0 + 65, T0 + 90], [T0 + 95, T0 + 97]],
            id="person-gap",
        ),
        pytest.param(
            "object_gap_s",
            "object:guitar",
            [[T0 + i * 10, T0 + i * 10 + 2] for i in range(10)],
            id="object-gap",
        ),
    ],
)
def test_changed_gap_refreshes_cached_presence_without_source_changes(state, setting, chip, expected):
    before = chips(state)
    with cfg.override({"dashboard": {setting: 1.0}}):
        after = chips(state)
        assert after[chip]["spans"] == expected
        assert after[chip]["spans"] != before[chip]["spans"]
    assert chips(state)[chip]["spans"] == before[chip]["spans"]


def detector_frame(at, *, camera="cam0", positive=True):
    return {"at": float(at), "cam": camera, "boxes": [[0.1, 0.1, 0.9, 0.9, 0.9]] if positive else []}


def add(vision, frames):
    add_frames(vision, [(f["at"], f["cam"], bool(f["boxes"])) for f in frames])


def test_sampled_presence_chips_match_facade_counts_and_keep_existing_chips(state, vision):
    baseline = chips(state)
    pr._cache.clear()  # an ended day is read once; this one gains its records only now
    add(
        vision,
        [
            detector_frame(T0),
            detector_frame(T0 + 1),
            detector_frame(T0 + 2, positive=False),
            detector_frame(T0 + 13),
            detector_frame(T0, camera="cam1"),
            detector_frame(T0, camera="cam2", positive=False),
        ],
    )
    result = pr.presence(DAY, state)
    after = {chip["id"]: chip for chip in result["chips"]}
    expected = presence_counts.collect(state, start=result["start"], end=result["end"])
    assert {name: chip for name, chip in after.items() if not name.startswith("presence:")} == baseline
    assert after["presence:cam0"]["count"] == expected["cameras"]["cam0"]["count"] == 2
    assert after["presence:cam1"]["count"] == expected["cameras"]["cam1"]["count"] == 1
    assert "presence:cam2" not in after
    assert after["presence:cam0"]["spans"] == [[T0, T0 + 10], [T0 + 13, T0 + 23]]
    assert after["presence:cam0"]["total_s"] == 0
    assert after["presence:cam0"]["kind"] == "event"
    assert after["presence:cam0"]["label"] == "cam0 came into view"
    assert "Sampled presence" in after["presence:cam0"]["notes"][0]
    assert "Unknown coverage" in after["presence:cam0"]["notes"][0]
    assert "Start boundary uncertain" in after["presence:cam0"]["notes"][0]
    assert [chip["id"] for chip in result["chips"]][-2:] == ["presence:cam0", "presence:cam1"]


def test_presence_playback_context_clips_to_day_end_and_excludes_preexisting_start(state, vision):
    start, end = pr.day_bounds(DAY)
    add(
        vision,
        [
            detector_frame(start - 0.5),
            detector_frame(start),
            detector_frame(end - 2, camera="cam1"),
        ],
    )
    result = chips(state)
    assert "presence:cam0" not in result
    assert result["presence:cam1"]["spans"] == [[end - 2, end]]
    assert result["presence:cam1"]["total_s"] == 0


def test_an_unchanged_ended_day_is_answered_without_reading_anything_again(state, vision):
    """#2398: the movement read alone took 8 to 12 s a load; an ended day whose records didn't change skips it."""
    add(vision, [detector_frame(T0)])
    first = pr.presence(DAY, state)
    reads = vision.reads
    assert pr.presence(DAY, state) is first
    assert vision.reads == reads, "nothing read again"
    fake().say({"ts_start": T0 + 900, "ts_end": T0 + 901, "speaker": "Erin", "text": "yo"})
    fake().say({"ts_start": T0 + 10 * 86400, "ts_end": T0 + 10 * 86400 + 1, "speaker": "Sam", "text": "later"})
    again = pr.presence(DAY, state)
    assert again is not first, "a line of that day, written late, is news"
    assert pr.presence(DAY, state) is again
    fake().say({"ts_start": T0 + 10 * 86400 + 5, "ts_end": T0 + 10 * 86400 + 6, "speaker": "Sam", "text": "today"})
    assert pr.presence(DAY, state) is again, "a line of another day isn't"


def test_presence_cache_observes_playback_context_setting(state, vision):
    add(vision, [detector_frame(T0)])
    assert chips(state)["presence:cam0"]["spans"] == [[T0, T0 + 10]]
    with cfg.override({"presence": {"chip_context_s": 3.0}}):
        result = chips(state)["presence:cam0"]
        assert result["spans"] == [[T0, T0 + 3]]
        assert result["total_s"] == 0


def test_disabled_presence_preserves_existing_behavior_and_does_not_collect(state, monkeypatch, vision):
    add(vision, [detector_frame(T0)])
    existing = {key: value for key, value in chips(state).items() if not key.startswith("presence:")}

    def forbidden(*args, **kwargs):
        pytest.fail("disabled presence must not read detector projection")

    monkeypatch.setattr(presence_counts, "collect", forbidden)
    with cfg.override({"presence": {"enabled": False}}):
        assert chips(state) == existing


@pytest.mark.parametrize(
    "context", [0, -1, float("nan"), float("inf"), True], ids=["zero", "negative", "nan", "infinity", "boolean"]
)
def test_invalid_presence_playback_context_is_rejected(state, context):
    with (
        cfg.override({"presence": {"chip_context_s": context}}),
        pytest.raises(ValueError, match="presence.chip_context_s"),
    ):
        pr.presence(DAY, state)


def test_a_day_still_going_picks_up_new_captions_after_the_refresh_interval(state, vision, monkeypatch):
    now = [T0 + 2000]
    monkeypatch.setattr(pr.time, "time", lambda: now[0])
    assert "object:chair" not in chips(state)
    for t in (T0 + 1500, T0 + 1510):
        vision.caption(t, caption="", objects=[{"label": "chair", "ts": t}])
    assert "object:chair" not in chips(state)  # cached: no file changed
    now[0] += pr.CAPTION_REFRESH_S
    assert chips(state)["object:chair"]["count"] == 1


def test_request_stays_pending_without_waiting_and_reuses_completed_day(tmp_path, monkeypatch):
    future = Future()
    submitted = []

    class Executor:
        def submit(self, load):
            submitted.append(load)
            return future

    monkeypatch.setattr(pr, "_requests", {})
    monkeypatch.setattr(pr, "_executor", Executor())
    first = pr.for_request(DAY, tmp_path)
    assert first["pending"] and first["chips"] == []
    assert pr.for_request(DAY, tmp_path)["pending"]
    assert len(submitted) == 1
    result = {"day": DAY, "chips": [{"id": "object:mug"}]}
    future.set_result(result)
    assert pr.for_request(DAY, tmp_path) is result
    assert len(submitted) == 1


def test_a_day_load_runs_in_a_process_of_its_own(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTINUUM_SOCKET", str(tmp_path / "no-broker.sock"))  # the child never finds a real one
    monkeypatch.setattr(pr, "_requests", {})
    monkeypatch.setattr(pr, "_executor", None)
    try:
        assert pr.for_request(DAY, tmp_path)["pending"]
        with pr._request_lock:
            [job] = pr._requests.values()
        result = job.future.result(timeout=120)
        assert result["day"] == DAY
        assert result["error"] if not result["pending"] else not result.get("error")
    finally:
        executor = pr._executor
        if executor is not None:
            executor.shutdown(wait=True)


def test_a_day_load_whose_process_died_is_an_error_and_the_next_poll_starts_a_new_one(tmp_path, monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    pools = []

    class Executor:
        def submit(self, load):
            future = Future()
            future.set_exception(BrokenProcessPool("the day load's process died"))
            return future

    def made():
        pools.append(Executor())
        return pools[-1]

    monkeypatch.setattr(pr, "_requests", {})
    monkeypatch.setattr(pr, "_executor", None)
    monkeypatch.setattr(pr, "ProcessPoolExecutor", lambda **_kw: made())
    assert pr.for_request(DAY, tmp_path)["pending"]
    second = pr.for_request(DAY, tmp_path)  # the job died: reported, and resubmitted to a new process
    assert (second["error"], second["pending"]) == ("BrokenProcessPool", False)
    assert len(pools) == 2


def test_the_idle_warm_keeps_an_answer_a_request_would_refresh(tmp_path, monkeypatch):
    clock = [1000.0]
    submitted = []

    class Executor:
        def submit(self, load):
            future = Future()
            submitted.append(future)
            return future

    monkeypatch.setattr(pr.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pr, "_requests", {})
    monkeypatch.setattr(pr, "_executor", Executor())
    pr.for_request(DAY, tmp_path)
    result = {"day": DAY, "chips": [{"id": "object:mug"}]}
    submitted[0].set_result(result)  # done at 1000.0
    clock[0] += 120.0
    assert pr.for_request(DAY, tmp_path, max_age_s=pr.IDLE_WARM_S) is result
    assert len(submitted) == 1  # nobody asked: two minutes old is fresh enough
    assert pr.for_request(DAY, tmp_path) is result  # a visit gets it at once
    assert len(submitted) == 2  # and starts the fresh one its next poll gets
    submitted[1].set_result(result)
    clock[0] += pr.IDLE_WARM_S
    pr.for_request(DAY, tmp_path, max_age_s=pr.IDLE_WARM_S)
    assert len(submitted) == 3  # idle for IDLE_WARM_S: warmed again


def test_a_process_that_died_while_idle_is_replaced_at_the_next_submit(tmp_path, monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    class Broken:
        def submit(self, load):
            raise BrokenProcessPool("the idle worker was killed")

    started = []

    class Fresh:
        def submit(self, load):
            future = Future()
            started.append(load)
            return future

    monkeypatch.setattr(pr, "_requests", {})
    monkeypatch.setattr(pr, "_executor", Broken())
    monkeypatch.setattr(pr, "ProcessPoolExecutor", lambda **_kw: Fresh())
    assert pr.for_request(DAY, tmp_path)["pending"]
    assert len(started) == 1
    assert isinstance(pr._executor, Fresh)


def test_background_job_failure_is_an_error_not_an_empty_success(tmp_path, monkeypatch, caplog):
    pending = Future()
    submitted = []

    class Executor:
        def submit(self, load):
            submitted.append(load)
            return pending

    def fail(day, state):
        raise ValueError("synthetic corrupt detector record")

    monkeypatch.setattr(pr, "_requests", {})
    monkeypatch.setattr(pr, "_executor", Executor())
    monkeypatch.setattr(pr, "presence", fail)
    pr.for_request(DAY, tmp_path)
    pending.set_result(submitted[0]())
    result = pr.for_request(DAY, tmp_path)
    assert result["error"] == "ValueError" and result["pending"] is False
    assert "synthetic corrupt detector record" in caplog.text


@pytest.mark.parametrize("kind", ["missing", "closed", "arena", "unavailable", "timeout"])
def test_broker_restart_is_pending_and_next_poll_retries_without_waiting_a_minute(tmp_path, monkeypatch, kind):
    import continuum_py

    error = {
        "missing": FileNotFoundError("broker socket missing"),
        "closed": continuum_py.ConnectionClosedError("broker closed"),
        "arena": continuum_py.NotReadyError("arena starting"),
        "unavailable": continuum_py.BrokerError("Unavailable", "broker starting"),
        "timeout": TimeoutError("broker request deadline"),
    }[kind]
    submitted = []
    futures = []

    class Executor:
        def submit(self, load):
            future = Future()
            submitted.append(load)
            futures.append(future)
            return future

    def unavailable(day, state):
        raise error

    monkeypatch.setattr(pr, "_requests", {})
    monkeypatch.setattr(pr, "_executor", Executor())
    monkeypatch.setattr(pr, "presence", unavailable)
    assert pr.for_request(DAY, tmp_path)["pending"]
    futures[0].set_result(submitted[0]())
    waiting = pr.for_request(DAY, tmp_path)
    assert waiting["pending"] and not waiting.get("error")
    assert len(submitted) == 2
    assert pr.for_request(DAY, tmp_path)["pending"]
    assert len(submitted) == 2  # The retry is still running: never enqueue an overlapping day load.
    result = {"day": DAY, "chips": [{"id": "object:mug"}]}
    monkeypatch.setattr(pr, "presence", lambda day, state: result)
    futures[1].set_result(submitted[1]())
    assert pr.for_request(DAY, tmp_path) == result


def test_persistent_error_remains_visible_while_next_poll_retries(tmp_path, monkeypatch):
    submitted = []
    futures = []

    class Executor:
        def submit(self, load):
            future = Future()
            submitted.append(load)
            futures.append(future)
            return future

    monkeypatch.setattr(pr, "_requests", {})
    monkeypatch.setattr(pr, "_executor", Executor())
    pr.for_request(DAY, tmp_path)
    failure = {"day": DAY, "chips": [], "partial": True, "pending": False, "error": "ValueError"}
    futures[0].set_result(failure)
    assert pr.for_request(DAY, tmp_path) == failure
    assert len(submitted) == 2
    result = {"day": DAY, "chips": [{"id": "object:mug"}]}
    futures[1].set_result(result)
    assert pr.for_request(DAY, tmp_path) == result


def test_long_day_job_stays_fresh_from_completion_not_submission(tmp_path, monkeypatch):
    future = Future()
    submitted = []
    clock = [0.0]

    class Executor:
        def submit(self, load):
            submitted.append(load)
            return future

    monkeypatch.setattr(pr, "_requests", {})
    monkeypatch.setattr(pr, "_executor", Executor())
    monkeypatch.setattr(pr.time, "monotonic", lambda: clock[0])
    pr.for_request(DAY, tmp_path)
    clock[0] = 90.0
    future.set_result({"day": DAY, "chips": []})
    assert pr.for_request(DAY, tmp_path) == {"day": DAY, "chips": []}
    assert len(submitted) == 1


def test_reviews_answer_leaves_out_what_the_page_never_reads():
    """#2398: movement's tracks and the chips' evidence were 25 MB a day, sent to a page that reads neither."""
    full = {
        "day": DAY,
        "chips": [{"id": "movement:visible:cam0", "spans": [[1, 2]], "source_ids": ["a"], "evidence": [{"k": 1}]}],
        "movement": {
            "available": True,
            "cameras": {
                "cam0": {"available": True, "entry_count": 2, "exit_count": 1, "tracks": [1] * 9, "unknown": [2]}
            },
        },
    }
    page = pr.for_page(full)
    assert page["chips"] == [{"id": "movement:visible:cam0", "spans": [[1, 2]]}]
    assert page["movement"] == {
        "available": True,
        "cameras": {"cam0": {"available": True, "entry_count": 2, "exit_count": 1}},
    }
    assert full["chips"][0]["source_ids"] == ["a"] and full["movement"]["cameras"]["cam0"]["tracks"], (
        "presence()'s own answer is left whole"
    )
    assert pr.for_page({"day": DAY, "chips": [], "pending": True}) == {"day": DAY, "chips": [], "pending": True}
