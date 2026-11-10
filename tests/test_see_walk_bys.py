"""Walk-bys as they happen (see/walk_bys.py), for the activity worker's "Walked past" spans (#1403): the same rules
as crossing_counts, decided once a chain has ended."""

import json
import math

import pytest

from liberational.continuum import keyed_id
from liberational.see import walk_bys as W

T = 1_800_000_000.0
LINE = {"kind": "vertical", "position": 0.5}


def body(t, *people):
    """A body frame with each (track, foot x) person in it."""
    return {
        "ts_start": t,
        "id": keyed_id(t, "walk-by-body"),
        "cam": "cam0",
        "boxes": [[x - 0.04, 0.3, x + 0.04, 0.8, 0.9] for _, x in people],
        "track_ids": [tid for tid, _ in people],
    }


def crossing(t, tid, direction="left_to_right"):
    return {
        "ts_start": t,
        "cam": "cam0",
        "id": keyed_id(t, "walk-by-crossing", tid),
        "track_id": tid,
        "direction": direction,
    }


def run(frames, crossings, *, until=None, pipeline=None):
    """Feed frames (t, [(tid, x)]) and crossings in time order (a track's crossings come as it ends), then empty
    frames to `until`; the walk-bys decided."""
    p = pipeline or W.WalkByPipeline("cam0", line=LINE)
    stream = [(t, 0, people) for t, people in frames] + [(c["ts_start"], 1, c) for c in crossings]
    out = []
    for t, kind, item in sorted(stream, key=lambda s: (s[0], s[1])):
        out += p.feed_body(body(t, *item)) if kind == 0 else p.feed_crossing(item)
    last = max(t for t, _, _ in stream)
    for k in range(1, int(((until or last + 2 * W.FINAL_S) - last) * 4) + 1):
        out += p.feed_body(body(last + k / 4))
    return out, p


def path(tid, where, t0, t1):
    """Frames every 0.25 s with `tid`'s foot at x = where(seconds since t0)."""
    return [(t0 + k / 4, [(tid, where(k / 4))]) for k in range(int((t1 - t0) * 4) + 1)]


def glide(a, b, seconds):
    return lambda t: a + (b - a) * min(1.0, t / seconds)


def test_someone_hovering_at_the_line_is_no_walk_by():
    frames = path("desk", lambda t: 0.5 + 0.08 * (-1) ** int(t), T, T + 60)
    crossings = [crossing(T + 60, "desk", "right_to_left" if k % 2 else "left_to_right") for k in range(10)]
    crossings = [{**c, "ts_start": T + 2 + 5 * k, "id": keyed_id(T + k, "hover")} for k, c in enumerate(crossings)]
    out, p = run(frames, crossings)
    assert out == [] and p.snapshot()["pending"] == [], "decided, and no span"


def test_a_clear_pass_is_one_walk_by_naming_its_crossing():
    c = crossing(T + 12, "a")
    out, _ = run(path("a", glide(0.2, 0.8, 4), T + 10, T + 14), [c])
    assert [(w["ts"], w["direction"], w["events"], w["verified"]) for w in out] == [
        (T + 12, "left_to_right", [c["id"]], True)
    ]


def test_back_and_forth_while_staying_in_view_is_one_walk_by_with_every_crossing():
    crossings = [crossing(T + 1 + 3.1 * k, "a", "left_to_right" if k % 2 == 0 else "right_to_left") for k in range(9)]
    out, _ = run(path("a", lambda t: 0.5 + 0.3 * math.sin(t), T, T + 30), crossings)
    (walk,) = out
    assert (walk["crossings"], walk["events"]) == (9, [c["id"] for c in crossings])


def test_a_broken_track_that_restarted_where_it_ended_is_one_walk_by():
    frames = path("a", glide(0.2, 0.49, 2), T + 10, T + 12) + path("b", glide(0.51, 0.8, 2), T + 12.5, T + 14.5)
    out, _ = run(frames, [crossing(T + 12.6, "b")])
    assert [w["tracks"] for w in out] == [["a", "b"]]


def test_a_joined_walk_by_names_the_body_records_of_both_halves():
    frames = path("a", glide(0.2, 0.49, 2), T + 10, T + 12) + path("b", glide(0.51, 0.8, 2), T + 12.5, T + 14.5)
    c = crossing(T + 12.6, "b")
    _, p = run(frames, [c], until=T + 15)
    left = keyed_id(T + 10, "walk-by-body")  # only track a saw the left side, and it made no crossing
    assert left in p.state_parents(), "the checkpoint depends on it too"
    (walk,) = [w for w in p.feed_body(body(T + 30))]
    assert left in walk["sources"] and keyed_id(T + 14.5, "walk-by-body") in walk["sources"]
    assert walk["events"] == [c["id"]]


def test_tracks_that_never_cross_age_out_and_leave_room_for_new_people(monkeypatch):
    monkeypatch.setattr(W, "MAX_TRACKS", 3)
    frames = [(T + k, [(f"by{k}", 0.1)]) for k in range(3)]  # three people who never reach the line
    frames += path("a", glide(0.2, 0.8, 4), T + 10 + W.KEEP_S, T + 14 + W.KEEP_S)
    out, p = run(frames, [crossing(T + 12 + W.KEEP_S, "a")])
    assert [w["verified"] for w in out] == [True], "a was followed, so its pass was checked"
    assert "by0" not in p.snapshot()["tracks"]


def test_two_people_walking_past_together_are_two():
    frames = [(T + 10 + k / 4, [("a", 0.2 + 0.15 * k / 4), ("b", 0.25 + 0.15 * k / 4)]) for k in range(17)]
    out, _ = run(frames, [crossing(T + 12, "a"), crossing(T + 11.75, "b")])
    assert sorted(w["tracks"][0] for w in out) == ["a", "b"]


def test_a_walk_by_waits_until_its_chain_has_ended():
    frames = path("a", glide(0.2, 0.8, 4), T + 10, T + 14)
    out, p = run(frames, [crossing(T + 12, "a")], until=T + 14 + W.FINAL_S - 0.5)
    assert out == [] and len(p.snapshot()["pending"]) == 1
    assert len(p.feed_body(body(T + 14 + W.FINAL_S))) == 1


def test_a_checkpoint_taken_mid_walk_decides_the_same():
    frames = path("a", glide(0.2, 0.8, 4), T + 10, T + 14)
    c = crossing(T + 12, "a")
    _, p = run(frames, [c], until=T + 15)
    again = W.WalkByPipeline("cam0", state=json.loads(json.dumps(p.snapshot())))
    assert again.state_parents() == p.state_parents() and c["id"] in again.state_parents()
    assert [w["events"] for w in again.feed_body(body(T + 30))] == [[c["id"]]]
    assert [w["events"] for w in p.feed_body(body(T + 30))] == [[c["id"]]]


def test_stopping_decides_every_waiting_chain():
    frames = path("a", glide(0.2, 0.8, 4), T + 10, T + 14)
    _, p = run(frames, [crossing(T + 12, "a")], until=T + 15)
    assert len(p.flush()) == 1 and p.snapshot()["pending"] == []


def test_a_full_queue_decides_its_oldest_early_and_never_drops_one(monkeypatch):
    monkeypatch.setattr(W, "MAX_PENDING", 3)
    p = W.WalkByPipeline("cam0", line=LINE)
    out = []
    for k in range(5):
        out += p.feed_crossing(crossing(T + k, f"t{k}"))  # nothing seen of them: each counts once, unfollowed
    assert len(p.snapshot()["pending"]) <= 3
    assert len(out) + len(p.flush()) == 5


def test_a_snapshot_this_version_could_not_have_made_is_refused():
    state = W.WalkByPipeline("cam0", line=LINE).snapshot()
    with pytest.raises(ValueError, match="walk-by"):
        W.WalkByPipeline("cam1", state=state)
    with pytest.raises(ValueError, match="walk-by"):
        W.WalkByPipeline("cam0", state={**state, "pending": [{"id": "x"}]})
