"""People on a screen (#2468): which boxes a TV, laptop, or screen zone holds (see/screens.py), screen zones
(see/zones.py), and a camera's screen zones read again as they change (see/actions_worker.py)."""

import threading
from pathlib import Path

import numpy as np
import pytest

from liberational import config
from liberational.see import actions_worker, screens, zones

TV = np.array([[0.0, 0.1, 0.6, 0.95, 0.9, 62]], np.float32)  # a big TV on the left of the frame
SQUARE = ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))


@pytest.mark.parametrize(
    ("box", "share"),
    [
        pytest.param((0.1, 0.2, 0.3, 0.9), 1.0, id="all-inside"),
        pytest.param((0.5, 0.2, 0.7, 0.9), 0.55, id="half-out"),  # 0.5 to 0.61 (the padded edge) of 0.5 to 0.7
        pytest.param((0.7, 0.2, 0.9, 0.9), 0.0, id="beside-it"),
        pytest.param((0.3, 0.3, 0.3, 0.9), 0.0, id="no-area"),
    ],
)
def test_how_much_of_a_box_a_screen_holds(box, share):
    assert screens.in_box(box, TV[0]) == pytest.approx(share)


def test_how_much_of_a_box_a_zone_holds_by_its_sample_points():
    assert screens.in_polygon((0.1, 0.1, 0.4, 0.4), SQUARE) == 1.0
    assert screens.in_polygon((0.3, 0.1, 0.7, 0.4), SQUARE) == pytest.approx(10 / 25)  # 2 of 5 columns inside
    assert screens.in_polygon((0.6, 0.6, 0.9, 0.9), SQUARE) == 0.0


def test_a_box_is_on_a_screen_when_mostly_inside_a_screen_box_or_a_screen_zone():
    boxes = np.array(
        [
            [0.1, 0.2, 0.3, 0.9, 0.9],  # inside the TV
            [0.5, 0.2, 0.7, 0.9, 0.9],  # half out of it: someone in front of it
            [0.7, 0.05, 0.8, 0.3, 0.9],  # inside the zone only
            [0.7, 0.6, 0.9, 0.9, 0.9],  # nowhere
        ],
        np.float32,
    )
    zone = ((0.65, 0.0), (0.95, 0.0), (0.95, 0.4), (0.65, 0.4))
    assert screens.on_screen(boxes, TV, [zone]).tolist() == [True, False, True, False]
    assert screens.on_screen(boxes, np.zeros((0, 6), np.float32)).tolist() == [False] * 4


def test_a_box_nearly_as_big_as_the_screen_is_someone_in_front_of_it():
    # The back of a real head in front of a laptop, from a camera behind: inside its box, but 0.66 of its size.
    laptop = np.array([[0.555, 0.412, 0.636, 0.553, 0.8, 63]], np.float32)
    head = np.array([[0.578, 0.469, 0.644, 0.577, 0.77]], np.float32)
    assert screens.in_box(head[0], laptop[0]) >= screens.SCREEN_INSIDE
    assert screens.share(head[0], laptop[0]) > screens.SCREEN_SHARE_MAX
    assert screens.on_screen(head, laptop).tolist() == [False]
    # Someone the TV shows, a quarter of its size.
    tv = np.array([[0.54, 0.086, 0.738, 0.296, 0.9, 62]], np.float32)
    shown = np.array([[0.58, 0.123, 0.657, 0.267, 0.7]], np.float32)
    assert screens.on_screen(shown, tv).tolist() == [True]
    # A zone drawn as a screen holds anyone inside it, however big.
    assert screens.on_screen(
        head, np.zeros((0, 6), np.float32), [((0.5, 0.4), (0.7, 0.4), (0.7, 0.6), (0.5, 0.6))]
    ).tolist() == [True]


R = screens.RELEASE_NS


def test_a_track_born_on_a_screen_stays_a_screen_track_until_it_has_been_out_of_it_a_while():
    tracks = screens.ScreenTracks()
    assert tracks.update({0: 1, 1: 2}, np.array([True, False]), 0) == {1}
    assert tracks.update({0: 1, 1: 2}, np.array([True, True]), 1) == {1}, "a real track that moves in front stays real"
    assert tracks.update({0: 1}, np.array([False]), 2) == {1}, "one frame out is not leaving"
    assert tracks.update({0: 1}, np.array([True]), 3) == {1}, "back in: the time outside starts again"
    assert tracks.update({0: 1}, np.array([False]), 4) == {1}
    assert tracks.update({0: 1}, np.array([False]), 4 + R) == set(), "out for RELEASE_NS: real from now on"
    assert tracks.released_at(1) == 4
    assert tracks.update({0: 1}, np.array([True]), 5 + R) == set()
    counts = tracks.take()
    assert (counts.detections, counts.inside, counts.masked, counts.born, counts.released) == (9, 5, 5, 1, 1)
    assert tracks.take() == screens.Counts()


def test_frames_with_no_screen_known_are_no_evidence_of_leaving_one():
    tracks = screens.ScreenTracks()
    tracks.update({0: 1}, np.array([True]), 0)
    assert tracks.update({0: 1}, np.array([False]), 1, known=False) == {1}
    assert tracks.update({0: 1}, np.array([False]), 1 + 5 * R, known=False) == {1}


def test_a_screen_the_detector_misses_for_a_moment_is_held():
    held = screens.HeldScreens()
    tv = np.array([[0.1, 0.1, 0.5, 0.5, 0.9, 62]], np.float32)
    none = np.zeros((0, 6), np.float32)
    assert len(held.update(tv, 0)) == 1
    assert len(held.update(none, screens.SCREEN_HOLD_NS - 1)) == 1
    moved = tv + np.array([[0.01, 0, 0.01, 0, 0, 0]], np.float32)
    assert held.update(moved, screens.SCREEN_HOLD_NS)[:, 0].tolist() == pytest.approx([0.11]), "the same screen"
    assert len(held.update(none, 2 * screens.SCREEN_HOLD_NS)) == 0


def test_an_ended_or_gapped_screen_track_is_forgotten():
    tracks = screens.ScreenTracks()
    tracks.update({0: 1, 1: 2}, np.array([True, True]), 0)
    tracks.end(1)
    assert not tracks.is_screen(1) and tracks.is_screen(2)
    tracks.clear()
    assert not tracks.is_screen(2)


def test_a_screen_zone_round_trips_and_is_left_out_of_counting():
    area = {"name": "Couch", "polygon": [list(p) for p in SQUARE]}
    screen = {"name": "Projector wall", "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 0.5], [0.5, 0.5]], "kind": "screen"}
    parsed = zones.parse_zones([area, screen])
    assert [z.kind for z in parsed] == ["area", "screen"]
    assert "kind" not in parsed[0].as_dict(), "a plain zone is stored as before"
    assert zones.parse_zone(parsed[1].as_dict()) == parsed[1]
    assert zones.counted(parsed) == [parsed[0]]
    definition = zones.Definition("def-1", "cam0", parsed, 1.0)
    assert zones.screen_polygons(definition) == [parsed[1].polygon]
    assert zones.screen_polygons(None) == []
    with pytest.raises(ValueError, match="kind"):
        zones.parse_zone({**area, "kind": "tv"})


def test_a_camera_whose_only_zone_is_a_screen_counts_nobody_in_it():
    screen = zones.parse_zone({"name": "TV", "polygon": [list(p) for p in SQUARE], "kind": "screen"})
    pipeline = zones.ZonePipeline("cam0")
    pipeline.define(zones.Definition("def-1", "cam0", (screen,), 1.0))
    frame = {"id": "f1", "ts_start": 10.0, "boxes": [[0.1, 0.1, 0.3, 0.4]], "track_ids": [7], "ok": True}
    assert pipeline.frame("person", frame, ["person"]) == []


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_cameras_screen_zones_are_read_again_once_a_minute_and_kept_through_a_failed_read(monkeypatch, tmp_path):
    screen = zones.parse_zone({"name": "TV", "polygon": [list(p) for p in SQUARE], "kind": "screen"})
    sets = [{"cam0": zones.Definition("def-1", "cam0", (screen,), 1.0)}, {}, RuntimeError("store down")]
    reads = []

    def current(state):
        reads.append(state)
        got = sets.pop(0)
        if isinstance(got, Exception):
            raise got
        return got

    monkeypatch.setattr(zones, "current", current)
    clock = Clock()
    read = actions_worker.ScreenZones("cam0", tmp_path, clock)
    assert read() == [screen.polygon]
    clock.now = actions_worker.SCREEN_ZONES_S - 1
    assert read() == [screen.polygon] and len(reads) == 1
    clock.now = actions_worker.SCREEN_ZONES_S
    assert read() == [], "the zone was taken away"
    clock.now = 2 * actions_worker.SCREEN_ZONES_S
    assert read() == [] and len(reads) == 3, "a failed read keeps the last zones"


# ---- whether a screen is on (#2468) ----

S = screens.SAMPLE_NS
TV_BOX = (0.54, 0.086, 0.738, 0.296)


def picture(seed: int) -> np.ndarray:
    """A screen's patch: a different picture for each seed, the same one for the same seed."""
    return np.random.default_rng(seed).integers(0, 256, (60, 90, 3), dtype=np.uint8)


def test_a_still_screen_is_never_on():
    activity = screens.ScreenActivity()
    for i in range(20):
        assert not activity.sample(i * S, [(TV_BOX, picture(1))])
    assert activity.spans() == []


def test_a_moving_picture_is_on_from_its_third_change_and_stays_on_through_a_short_pause(monkeypatch):
    monkeypatch.setattr(screens, "MIN_ON_NS", 0)
    activity = screens.ScreenActivity()
    on = [activity.sample(i * S, [(TV_BOX, picture(i))]) for i in range(8)]
    assert on == [False, False, False, True, True, True, True, True]  # changes come from the 2nd sample on
    assert activity.spans() == [(3 * S, 7 * S)]
    for i in range(8, 8 + 10):  # 5 s still: under OFF_NS, the span stays open
        activity.sample(i * S, [(TV_BOX, picture(7))])
    assert activity.closed == [] and activity.open_ns == (3 * S, 10 * S)


def test_the_span_ends_once_the_screen_is_still_or_gone_for_off_ns(monkeypatch):
    monkeypatch.setattr(screens, "MIN_ON_NS", 0)
    activity = screens.ScreenActivity()
    for i in range(8):
        activity.sample(i * S, [(TV_BOX, picture(i))])
    last_on = 7 * S
    i = 8
    while i * S - last_on < screens.OFF_NS:
        activity.sample(i * S, [])  # the screen isn't seen
        i += 1
    activity.sample(i * S, [])
    assert activity.spans() == [(3 * S, last_on)]
    assert activity.open_ns is None


def test_a_gap_ends_the_span_at_its_last_moving_sample(monkeypatch):
    monkeypatch.setattr(screens, "MIN_ON_NS", 0)
    activity = screens.ScreenActivity()
    for i in range(8):
        activity.sample(i * S, [(TV_BOX, picture(i))])
    activity.gap()
    assert activity.spans() == [(3 * S, 7 * S)]


class WallClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_the_spans_file_merges_a_growing_span_keeps_a_day_and_says_what_was_on_when(tmp_path):
    path = tmp_path / "screens_on.json"
    clock = WallClock(20_000.0)
    on = screens.ScreensOn(path, clock)
    on.add("cam0", [(int(10_000e9), int(10_060e9))])
    on.add("cam0", [(int(10_000e9), int(10_120e9))])  # the same span, still open, a minute longer
    assert screens.ScreensOn(path).cams == {"cam0": [[10_000.0, 10_060.0]]}, "written at most every WRITE_S"
    clock.now += screens.WRITE_S
    on.add("living-room", [(int(19_000e9), int(19_030e9))])
    assert screens.ScreensOn(path).cams == {"cam0": [[10_000.0, 10_120.0]], "living-room": [[19_000.0, 19_030.0]]}
    assert screens.on_during(path, 10_100.0, 10_110.0)
    assert screens.on_during(path, 10_123.0, 10_130.0), "give or take pad_s"
    assert not screens.on_during(path, 20_000.0, 20_010.0)
    clock.now = 10_120.0 + screens.KEEP_S + 1  # a day later, the first span is dropped, whichever camera adds
    on.add("living-room", [(int(clock.now * 1e9), int(clock.now * 1e9) + 1)])
    assert "cam0" not in screens.ScreensOn(path).cams


def test_a_missing_or_broken_spans_file_says_nothing_was_on(tmp_path):
    assert not screens.on_during(tmp_path / "none.json", 0, 1)
    (tmp_path / "bad.json").write_text("{not json")
    assert not screens.on_during(tmp_path / "bad.json", 0, 1)


def test_the_cells_someone_stands_in_front_of_are_covered():
    screen = (0.0, 0.0, 1.0, 1.0)
    assert not screens.covered(screen, []).any()
    left_half = screens.covered(screen, [(0.0, 0.0, 0.5, 1.0)])
    assert left_half.mean() == pytest.approx(0.5)
    assert left_half[:, 0].all() and not left_half[:, -1].any()


def test_someone_moving_in_front_of_a_still_screen_does_not_turn_it_on():
    activity = screens.ScreenActivity()
    still = picture(1)
    for i in range(12):
        patch = still.copy()
        patch[:, : 45 if i % 2 else 30] = 255 if i % 2 else 0  # a person passing the left half, back and forth
        assert not activity.sample(i * S, [(TV_BOX, patch)], [(0.54, 0.0, 0.64, 1.0)])


def test_a_screen_mostly_hidden_is_skipped_not_called_off_or_on():
    activity = screens.ScreenActivity()
    for i in range(8):
        activity.sample(i * S, [(TV_BOX, picture(i))])
    assert activity.open_ns is not None
    hidden_until = (0.0, 0.0, 1.0, 1.0)  # someone right in front of the camera
    for i in range(8, 12):
        activity.sample(i * S, [(TV_BOX, picture(100 + i))], [hidden_until])
    assert activity.screens[0].moved == [True] * screens.WINDOW, "the hidden samples added nothing"


def test_a_burst_shorter_than_a_show_is_not_reported():
    activity = screens.ScreenActivity()
    for i in range(8):  # 4 s of moving picture
        activity.sample(i * S, [(TV_BOX, picture(i))])
    assert activity.open_ns is not None
    assert activity.spans() == []


def test_only_screen_zones_are_watched_for_a_moving_picture():
    """A detected TV may be a computer monitor (Nov 6: cam0's desk monitor, #2669): only a drawn screen zone counts."""
    zone = ((0.0, 0.0), (0.3, 0.05), (0.2, 0.3))
    np.testing.assert_allclose(screens.playing([zone])[:, :4], [[0.0, 0.0, 0.3, 0.3]], atol=1e-6)
    assert screens.playing([]).shape == (0, 6)


def test_two_cameras_writing_at_once_both_keep_their_spans(tmp_path, monkeypatch):
    # #3093: the first camera's write stalls; the second's, made after with both cameras' spans, must not be
    # replaced by the first's older map.
    path = tmp_path / "screens_on.json"
    on = screens.ScreensOn(path, WallClock(20_000.0))
    real_write, writing = Path.write_text, threading.Event()

    def slow_first_write(self: Path, body: str) -> int:
        if not writing.is_set():
            writing.set()
            threading.Event().wait(0.3)  # long enough for the other camera's write to land first, unlocked
        return real_write(self, body)

    monkeypatch.setattr(Path, "write_text", slow_first_write)
    first = threading.Thread(target=on.add, args=("cam0", [(int(19_000e9), int(19_030e9))], True))
    first.start()
    assert writing.wait(5)
    on.add("cam1", [(int(19_100e9), int(19_130e9))], flush=True)
    first.join(5)
    monkeypatch.undo()
    assert screens.ScreensOn(path).cams == {"cam0": [[19_000.0, 19_030.0]], "cam1": [[19_100.0, 19_130.0]]}


def test_cameras_starting_at_once_share_one_spans_writer(tmp_path, monkeypatch):
    # #3093: two camera threads that both find no writer yet must not each make one.
    made, second = [], threading.Event()

    def slow_writer(path: Path) -> object:
        made.append(path)
        if len(made) > 1:
            second.set()
        else:
            second.wait(0.3)  # an unlocked second thread gets here meanwhile
        return object()

    monkeypatch.setattr(actions_worker, "_screens_on", None)
    monkeypatch.setattr(screens, "ScreensOn", slow_writer)
    got: list[object] = []
    threads = [threading.Thread(target=lambda: got.append(actions_worker.screens_on())) for _ in range(2)]
    with config.override({"paths": {"state": str(tmp_path)}}):
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
    assert len(made) == 1 and len(got) == 2 and got[0] is got[1]
