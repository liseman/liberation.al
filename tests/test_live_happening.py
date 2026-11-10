"""Live's "What's happening" (dashboard/happening.py): arrivals and departures from the camera views,
finished conversations told as one named line, never first person, and a cursor of when each row was written, so a
tab polling since=updated gets every row, however late it lands. Synthetic visits and transcript lines with a fake
model in a temp state dir; never the real data/."""

import http.client
import json
import threading
import urllib.parse

import pytest

from liberational import config as cfg
from liberational.dashboard import happening
from liberational.memory import plain_text
from speech_fake import fake
from voices import write_voices

NOW = 1_793_200_000.0


def line(ts, text, sid=None, dur=3.0):
    e = {"ts_start": ts, "ts_end": ts + dur, "text": text}
    if sid is not None:
        e["speaker_id"] = sid
    return e


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Kara (voice 1) and Luca (voice 2), Olive the owner (voice 3); no visits or lines yet."""
    write_voices(tmp_path, {1: "Kara", 2: "Luca", 3: "Olive"})
    monkeypatch.setattr(happening, "_built", {})
    monkeypatch.setattr(happening, "_settled", {})
    monkeypatch.setattr(happening, "_rows", {})
    cfgs = {
        "owner": {"name": "Olive"},
        "paths": {"state": str(tmp_path)},
        "vision": {"camera_names": ["cam0=front camera"]},
    }
    with cfg.override(cfgs):
        yield tmp_path


@pytest.fixture
def rewrites(state):
    """make(replies) -> (the texts the model was given, a named-lines cache whose model answers from replies, and
    in the first person, always dropped, for anything else)."""

    def make(replies):
        calls = []

        def rewrite(text):
            calls.append(text)
            return replies.get(text, json.dumps({"text": "I talked with Kara", "topics": []}))

        return calls, plain_text.Rewrites(state, rewrite, background=False, checker=plain_text.check_named)

    return make


def visits(vision, *rows):
    for name, cam, start, end in rows:
        vision.visit(name, start, end, cam, best_sim=0.8, samples=5)


def talk(state, *lines):
    for e in lines:
        fake().say(e)


def read(state, rw, since=None, now=NOW, limit=50):
    happening._built.clear()
    return happening.happening(state, since, limit, now=now, rewrites=rw)


def texts(out):
    return [r["text"] for r in out["items"]]


def test_people_come_and_go_on_the_camera_views(state, rewrites, vision):
    visits(vision, ("Kara", "cam0", NOW - 3000, NOW - 2400), ("Luca", "cam1", NOW - 100, NOW - 10))
    _, rw = rewrites({})
    out = read(state, rw)
    assert [(r["kind"], r["text"], r["cam"], r["camera_name"]) for r in out["items"]] == [
        ("arrival", "Kara came in", "cam0", "front camera"),
        ("departure", "Kara left", "cam0", "front camera"),
        ("arrival", "Luca came in", "cam1", None),  # seen under LEAVE_S ago: still here
    ]
    assert out["items"][0]["people"] == ["Kara"]


def test_someone_still_in_the_room_never_leaves(state, rewrites, vision):
    visits(vision, ("Kara", "cam0", NOW - 900, NOW - 60))  # seen a minute ago
    _, rw = rewrites({})
    assert texts(read(state, rw)) == ["Kara came in"]
    visits(vision, ("Kara", "cam0", NOW - 50, NOW - 10))  # her next visit, a few seconds' gap later
    assert texts(read(state, rw)) == ["Kara came in"]  # one stay, no "left" and back


def test_someone_unseen_for_long_enough_has_left(state, rewrites, vision):
    visits(vision, ("Kara", "cam0", NOW - 900, NOW - happening.LEAVE_S - 1))
    _, rw = rewrites({})
    assert texts(read(state, rw)) == ["Kara came in", "Kara left"]


def test_a_face_nobody_named_is_someone_not_yet_named(state, rewrites, vision):
    vision.face(NOW - 60, "cam0", None, None, cluster="Face 3")
    _, rw = rewrites({})
    (row,) = read(state, rw)["items"]
    assert (row["text"], row["people"]) == ("Someone not yet named came in", [])


def test_a_finished_conversation_is_one_named_line_with_strangers_as_someone(state, rewrites):
    talk(
        state,
        line(NOW - 150, "Is the couch new?", sid=1),  # over a minute and a half ago: finished
        line(NOW - 145, "Since Monday.", sid=2),
        line(NOW - 140, "Nice.", sid=9),  # a voice nobody has named
    )
    src = "Kara: Is the couch new?\nLuca: Since Monday.\nsomeone: Nice."
    calls, rw = rewrites({src: json.dumps({"text": "Kara asked Luca about the couch", "topics": []})})
    assert read(state, rw)["items"] == []  # being written: a moment before the plain line
    rw.drain()
    out = read(state, rw)
    assert [(r["kind"], r["text"], r["people"]) for r in out["items"]] == [
        ("talk", "Kara asked Luca about the couch", ["Kara", "Luca"])
    ]
    assert calls == [src]


def test_a_conversation_still_going_waits(state, rewrites):
    talk(state, line(NOW - 30, "Still talking.", sid=1))
    calls, rw = rewrites({})
    assert read(state, rw)["items"] == [] and calls == []


def test_the_owner_is_named_and_a_line_the_model_cant_tell_says_who_talked(state, rewrites):
    talk(state, line(NOW - 600, "Hello.", sid=1), line(NOW - 598, "Hi.", sid=3))
    _, rw = rewrites({})  # it answers in the first person, twice: dropped
    read(state, rw)
    rw.drain()
    assert texts(read(state, rw)) == ["Kara and Olive talked"]


def test_a_tab_polling_gets_every_row_however_late_it_lands(state, rewrites, vision):
    src = "Kara: Hi."
    _, rw = rewrites({src: json.dumps({"text": "Kara said hi", "topics": []})})
    talk(state, line(NOW - 400, "Hi.", sid=1))
    first = read(state, rw)  # the line isn't written, but its hold is over: the plain line goes out
    assert texts(first) == ["Kara talked"]
    rw.drain()
    visits(vision, ("Kara", "cam0", NOW - 410, NOW - 30))  # her visit closes only now
    later = read(state, rw, since=first["updated"], now=NOW + 10)
    assert [(r["id"], r["text"]) for r in later["items"]] == [
        (f"a:person:kara:{NOW - 410:.0f}", "Kara came in"),  # before the talk row in time, and still given out
        (f"t:{NOW - 400:.3f}", "Kara said hi"),  # the same row, now with the model's line: the page replaces it
    ]
    assert later["updated"] > first["updated"]
    quiet = read(state, rw, since=later["updated"], now=NOW + 20)
    assert (quiet["items"], quiet["updated"]) == ([], later["updated"])  # nothing new: the cursor stays


def test_a_tab_far_behind_catches_up_without_skipping(state, rewrites, vision):
    _, rw = rewrites({})
    for k in range(12):
        visits(vision, (f"P{k}", "cam0", NOW - 3000 + 100 * k, NOW - 2990 + 100 * k))
    first = read(state, rw, since=0.0, limit=10)
    second = read(state, rw, since=first["updated"], limit=10)
    third = read(state, rw, since=second["updated"], limit=10)
    got = [r["id"] for out in (first, second, third) for r in out["items"]]
    assert len(got) == 24 and len(set(got)) == 24  # every arrival and departure, each once
    assert third["items"][-1]["id"].startswith("d:person:p11")


def windows_read(monkeypatch):
    """The [start, end) windows happening asks sightings.faces for."""
    asked = []
    real = happening.sightings.faces

    def faces(start, end, *args, **kwargs):
        asked.append((start, end))
        return real(start, end, *args, **kwargs)

    monkeypatch.setattr(happening.sightings, "faces", faces)
    return asked


def unnamed(state, rw, now):
    happening._built.clear()
    return [r["text"] for r in happening.happening(state, None, 50, now=now, rewrites=rw)["items"]]


def test_older_face_sightings_are_kept_between_builds_and_only_the_recent_part_is_read(
    state, rewrites, vision, monkeypatch
):
    _, rw = rewrites({})
    vision.face(NOW - 3000, "cam0", None, None, cluster="Face 3")  # settled: long before SETTLED_S
    asked = windows_read(monkeypatch)
    assert unnamed(state, rw, NOW) == ["Someone not yet named came in", "Someone not yet named left"]
    split = NOW - happening.SETTLED_S
    assert asked == [(NOW - happening.LOOKBACK_S - happening.EDGE_S, split), (split, NOW)]
    asked.clear()
    vision.face(NOW + 5, "cam1", None, None, cluster="Face 4")  # new, in the recent part
    later = unnamed(state, rw, NOW + 10)
    assert later.count("Someone not yet named came in") == 2
    assert asked == [(split, NOW + 10)]  # the seven hours behind it weren't read again


@pytest.mark.parametrize(
    "change",
    [
        # A forget in the segment store moves its deletion revision.
        pytest.param(
            lambda monkeypatch: monkeypatch.setattr(happening, "_segment_revision", lambda state: 99), id="a-purge"
        ),
        pytest.param(None, id="the-refresh-interval"),
    ],
)
def test_the_kept_face_sightings_are_read_again_after_a_purge_or_the_refresh_interval(
    state, rewrites, vision, monkeypatch, change
):
    _, rw = rewrites({})
    asked = windows_read(monkeypatch)
    unnamed(state, rw, NOW)
    asked.clear()
    vision.face(NOW - 3000, "cam0", None, None, cluster="Face 3")  # landed late, in the settled part
    when = NOW + 10
    if change is None:
        when = NOW + happening.SETTLED_REFRESH_S
    else:
        change(monkeypatch)
    assert "Someone not yet named came in" in unnamed(state, rw, when)
    assert asked[0] == (when - happening.LOOKBACK_S - happening.EDGE_S, when - happening.SETTLED_S)


def test_the_first_poll_is_the_newest_rows_and_the_cursor_now(state, rewrites, vision):
    _, rw = rewrites({})
    for k in range(5):
        visits(vision, (f"P{k}", "cam0", NOW - 3000 + 100 * k, NOW - 2990 + 100 * k))
    out = read(state, rw, limit=2)
    assert [r["id"] for r in out["items"]] == [f"a:person:p4:{NOW - 2600:.0f}", f"d:person:p4:{NOW - 2590:.0f}"]
    assert read(state, rw, since=out["updated"])["items"] == []  # everything up to now is behind the cursor


def test_the_log_is_built_once_for_many_polls(state, rewrites, monkeypatch):
    builds = []
    real = happening._build
    monkeypatch.setattr(happening, "_build", lambda *a: builds.append(1) or real(*a))
    _, rw = rewrites({})
    happening._built.clear()
    for k in range(3):
        happening.happening(state, None, 50, now=NOW + k, rewrites=rw)
    happening.happening(state, None, 50, now=NOW + happening.BUILD_S + 1, rewrites=rw)
    happening.happening(state, None, 50, now=NOW + happening.BUILD_S + 2, rewrites=rw)
    settle(state)
    assert len(builds) == 2


def settle(state):
    """Wait for the background build of a state, if one is running."""
    with happening._lock:
        thread = happening._building.get(str(state))
    if thread is not None:
        thread.join(timeout=30)
        assert not thread.is_alive()


def row(rid, text):
    return {
        "id": rid,
        "ts": NOW - 20,
        "kind": "arrival",
        "text": text,
        "people": [],
        "cam": "cam0",
        "camera_name": None,
    }


def test_a_poll_past_build_s_answers_from_the_last_build_while_the_next_one_runs(state, rewrites, monkeypatch):
    _, rw = rewrites({})
    started, release = threading.Event(), threading.Event()
    builds = iter([[row("a:kara", "Kara came in")], [row("a:kara", "Kara came in"), row("a:luca", "Luca came in")]])

    def build(*_args):
        rows = next(builds)
        if len(rows) == 2:
            started.set()
            assert release.wait(10)
        return rows

    monkeypatch.setattr(happening, "_build", build)
    first = happening.happening(state, None, 50, now=NOW, rewrites=rw)
    assert texts(first) == ["Kara came in"]
    try:
        held = happening.happening(state, first["updated"], 50, now=NOW + happening.BUILD_S, rewrites=rw)
        assert started.wait(10)
        assert (held["items"], held["updated"]) == ([], first["updated"])  # not waiting for the build
    finally:
        release.set()
    settle(state)
    landed = happening.happening(state, first["updated"], 50, now=NOW + happening.BUILD_S + 1, rewrites=rw)
    assert texts(landed) == ["Luca came in"]


def test_a_failed_background_build_keeps_the_last_one(state, rewrites, monkeypatch):
    _, rw = rewrites({})
    monkeypatch.setattr(happening, "_build", lambda *_args: [row("a:kara", "Kara came in")])
    first = happening.happening(state, None, 50, now=NOW, rewrites=rw)

    def broken(*_args):
        raise RuntimeError("broker gone")

    monkeypatch.setattr(happening, "_build", broken)
    happening.happening(state, None, 50, now=NOW + happening.BUILD_S, rewrites=rw)
    settle(state)
    after = happening.happening(state, None, 50, now=NOW + happening.BUILD_S + 1, rewrites=rw)
    assert (texts(after), after["updated"]) == (texts(first), first["updated"])
    settle(state)  # the retry this poll started


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        pytest.param("Kara asked Luca about the couch", True, id="named"),
        pytest.param("Someone asked Kara about dinner", True, id="someone-and-a-name"),
        pytest.param("Someone said the door was open", True, id="someone-said"),
        pytest.param("I asked Kara about dinner", False, id="first-person"),
        pytest.param("Kara told Luca I was late", False, id="i-after-a-name"),
        pytest.param("She asked Luca about the couch", False, id="a-pronoun-first"),
        pytest.param("Speaker 4 asked Kara about dinner", False, id="an-automatic-label"),
        pytest.param("Speaker_4 asked Kara about dinner", False, id="an-automatic-label-run-together"),
        pytest.param("The owner told Kara about the trip", False, id="the-owner"),
        pytest.param("Kara told you about the trip", False, id="you"),
    ],
)
def test_named_lines_start_with_who_and_never_say_i(text, ok):
    assert (plain_text.check_named(json.dumps({"text": text}))[0] is not None) == ok


def test_the_route_answers_and_refuses_a_bad_since(state, rewrites, monkeypatch, serve):
    from liberational.dashboard import context, server

    monkeypatch.setattr(context, "STATE", state)
    _, rw = rewrites({})
    monkeypatch.setitem(happening._talk_lines, str(state), rw)  # never the model
    base = urllib.parse.urlsplit(serve(server.H))

    def get(query):
        c = http.client.HTTPConnection("127.0.0.1", base.port, timeout=10)
        try:
            c.request("GET", "/live/happening?" + query)
            r = c.getresponse()
            return r.status, json.loads(r.read())
        finally:
            c.close()

    assert get("")[0] == 200
    assert get("since=1793000000&limit=5")[1]["updated"] < 1793000000  # a cursor past the end starts afresh
    assert get("since=soon")[0] == 400
    assert get("since=nan")[0] == 400
    assert get("limit=many")[0] == 400


def test_a_cursor_past_the_end_gets_the_log_afresh(state, rewrites, vision):
    _, rw = rewrites({})
    visits(vision, ("Kara", "cam0", NOW - 900, NOW - 400))
    first = read(state, rw)
    stale = read(state, rw, since=NOW)  # a timestamp cursor from before the log moved to stream positions
    assert (texts(stale), stale["updated"]) == (texts(first), first["updated"])


def test_a_cursor_from_before_a_restart_gets_the_rows_afresh(state, rewrites, monkeypatch):
    _, rw = rewrites({})
    builds = iter([[row("a:kara", "Kara came in")]] + [[row("a:kara", "Kara and Luca came in")]] * 2)
    monkeypatch.setattr(happening, "_build", lambda *_args: next(builds))
    read(state, rw)
    before = read(state, rw, now=NOW + happening.BUILD_S)  # the revision moves the cursor on
    happening._rows.clear()  # a restarted dashboard: the journal starts over, shorter than the old cursor
    after = read(state, rw, since=before["updated"], now=NOW + 60)
    assert texts(after) == ["Kara and Luca came in"]
    assert after["updated"] < before["updated"]  # the cursor started over too


@pytest.mark.parametrize(
    ("named_track", "unknown_track", "unknown_cam", "named_at", "unknown_at", "arrivals"),
    [
        ("run1:cam0:4", "run1:cam0:4", "cam0", -20, -19, 1),
        ("run1:cam0:4", "run1:cam0:5", "cam0", -20, -19, 2),
        ("run1:cam0:4", "run2:cam0:4", "cam0", -20, -19, 2),
        ("4", "4", "cam0", -40, -1, 2),
        ("4", "4", "cam1", -20, -19, 2),
        ("run1:cam0:4", "run1:cam0:4", "cam0", -19, -20, 1),
        (None, None, "cam0", -20, -19, 2),
    ],
)
def test_identity_flicker_is_not_a_second_physical_arrival(
    state,
    vision,
    monkeypatch,
    named_track,
    unknown_track,
    unknown_cam,
    named_at,
    unknown_at,
    arrivals,
):
    from liberational.dashboard import presence
    from liberational.people import transcript_view

    visits(vision, ("Kara", "cam0", NOW + named_at, NOW - 1))
    rows = [
        {"name": "Kara", "cam": "cam0", "ts": NOW + named_at, "track_id": named_track},
        {"name": None, "cam": unknown_cam, "ts": NOW + unknown_at, "track_id": unknown_track},
    ]
    monkeypatch.setattr(happening.sightings, "faces", lambda *args, **kwargs: rows)
    found = happening._comings(state, NOW - 120, NOW, transcript_view.View(state))
    assert len([row for row in found if row["kind"] == "arrival"]) == arrivals
    # The observation remains uncertain in Review; deduplicating arrival events does not assign it a person.
    assert presence.unknown_faces(state, NOW - 120, NOW) is not None


def test_obsolete_anonymous_episode_is_retracted_for_incremental_clients(state, rewrites, monkeypatch):
    row = {
        "id": "a:unknown-faces:123",
        "ts": NOW - 20,
        "kind": "arrival",
        "text": "Someone not yet named came in",
        "people": [],
        "cam": "cam0",
        "camera_name": None,
    }
    row["id"] = f"a:{happening.presence.UNKNOWN_FACES}:123"
    _, rw = rewrites({})
    monkeypatch.setattr(happening, "_build", lambda *args: [row])
    first = read(state, rw)
    monkeypatch.setattr(happening, "_build", lambda *args: [])
    next_poll = read(state, rw, since=first["updated"])
    assert next_poll["items"] == [
        {
            "id": row["id"],
            "ts": row["ts"],
            "kind": None,
            "text": None,
            "people": None,
            "cam": None,
            "camera_name": None,
            "removed": True,
        }
    ]
    assert read(state, rw)["items"] == []


def test_rows_older_than_the_window_leave_the_journal(state, rewrites, monkeypatch):
    _, rw = rewrites({})
    later = NOW + happening.LOOKBACK_S + happening.EDGE_S
    builds = iter([[row("a:kara", "Kara came in")], [{**row("a:luca", "Luca came in"), "ts": later - 20}]])
    monkeypatch.setattr(happening, "_build", lambda *_args: next(builds))
    read(state, rw)
    assert texts(read(state, rw, since=0, now=later)) == ["Luca came in"]  # Kara's arrival isn't given out again
