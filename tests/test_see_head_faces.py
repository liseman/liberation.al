"""Faces from tracked heads: when a track is looked at, what it's named, and what is written."""

import numpy as np
import pytest

from liberational import segment_log
from liberational.see import head_faces, subjects
from liberational.see.faces import Face
from liberational.see.head_faces import NS, HeadFaces, head_box


@pytest.fixture(autouse=True)
def recognition_provenance(monkeypatch):
    monkeypatch.setattr("liberational.vision.faces.recognition_model", lambda: "test-adaface-model")


W, H = 960, 540


def joints(points, conf=0.9):
    """A 26-joint pose with the head joints (nose, eyes, ears) at `points`, and the rest unplaced."""
    j = np.full((26, 3), [-1.0, -1.0, 0.0])
    for i, (x, y) in enumerate(points):
        j[i] = [x, y, conf]
    return j


HEAD = joints([(500, 100), (490, 92), (510, 92), (480, 96), (520, 96)])  # a 40 px spread


def test_head_box_frames_the_head_joints_and_extends_toward_the_chin():
    box = head_box(HEAD, W, H)
    assert box is not None
    x1, y1, x2, y2 = box
    assert (x2 - x1, y2 - y1) == (104, 104)  # 2.6 times the 40 px spread
    assert x1 < 480 and x2 > 520
    assert (y1 + y2) / 2 > 96  # below the joints' mean


@pytest.mark.parametrize(
    "pose",
    [
        pytest.param(joints([(500, 100)]), id="one-joint"),
        pytest.param(joints([(500, 100), (502, 101)]), id="too-small"),
        pytest.param(joints([(500, 100), (490, 92), (510, 92)], conf=0.1), id="unsure"),
    ],
)
def test_no_head_box_without_enough_of_a_head(pose):
    assert head_box(pose, W, H) is None


def test_head_box_stays_inside_the_frame():
    box = head_box(joints([(5, 5), (25, 5), (5, 25)]), W, H)
    assert box is not None
    x1, y1, x2, y2 = box
    assert x1 == 0 and y1 == 0 and x2 <= W and y2 <= H


class Script:
    """A detector that finds the next scripted faces in each patch it's given."""

    def __init__(self, *found):
        self.found, self.patches = list(found), []

    def detect(self, frame):
        self.patches.append(frame.shape)
        return self.found.pop(0) if self.found else []


class Names:
    """A matcher over a gallery of one axis per person ({axis: name}), with the live gate's shape: the best person
    must reach 0.45 and beat the next by 0.10. An unknown face's group is its strongest axis."""

    def __init__(self, names):
        self.names = names

    def match(self, emb):
        emb = np.asarray(emb, np.float64) / np.linalg.norm(emb)
        sims = sorted(((float(emb[k]), name) for k, name in self.names.items()), reverse=True)
        best, name = sims[0]
        runner = sims[1][0] if len(sims) > 1 else -1.0
        return (name if best >= 0.45 and best - runner >= 0.10 else None), best

    def unknown(self, emb):
        return f"Face {int(np.argmax(emb))}"


class Written:
    def __init__(self):
        self.rows = {}

    def append(self, stream, cam, records):
        self.rows.setdefault(stream, []).extend((cam, r) for r in records)


STRAIGHT_ON = np.array([[25, 30], [45, 30], [35, 42], [27, 55], [43, 55]], np.float64)
# The nose at an eye, half the eyes' distance off their middle: a profile (or a back of a head's stray landmarks)
SIDE_ON = np.array([[25, 30], [45, 30], [45, 42], [27, 55], [43, 55]], np.float64)


def face(person, box=(10.0, 12.0, 60.0, 70.0), emb=None, norm=None, kps=STRAIGHT_ON):
    if emb is None:
        emb = np.zeros(512, np.float32)
        emb[person] = 1.0
    return Face(bbox=box, det=0.9, kps=kps, emb=emb, norm=norm)


def side(person, norm=14.0, emb=None):
    """The side or back of `person`'s head: readable (#2355), but no face to the camera (#3155)."""
    return face(person, emb=emb, norm=norm, kps=SIDE_ON)


def stage(detector, names, written):
    return HeadFaces("cam0", detector, Names(names), written, sharpness=lambda patch, box: 100.0)


def crop(box):
    x1, y1, x2, y2 = box
    return np.zeros((y2 - y1, x2 - x1, 3), np.uint8)


def test_each_track_is_looked_at_once_a_second_and_its_match_is_written_in_frame_terms():
    detector, written = Script([face(1)], [face(1)]), Written()
    faces = stage(detector, {1: "Alice"}, written)
    t0 = 1_800_000_000 * NS
    for i in range(31):  # one second and a frame at 30 fps
        faces.feed(crop, W, H, t0 + i * NS // 30, f"frame-{i}", [(7, HEAD)])
    faces.flush(t0 + NS)
    assert len(detector.patches) == 2, "the first frame and the one a second later"
    (cam, first), (_, second) = written.rows[segment_log.FACE_MATCHES]
    assert cam == "cam0"
    assert (first["track_id"], first["frame_id"], first["name"]) == ("7", "frame-0", "Alice")
    assert second["frame_id"] == "frame-30"
    x1, y1, _, _ = head_box(HEAD, W, H) or (0, 0, 0, 0)
    assert first["bbox"] == pytest.approx([(x1 + 10) / W, (y1 + 12) / H, (x1 + 60) / W, (y1 + 70) / H], abs=1e-5)
    (_, embedding), _ = written.rows[segment_log.FACE_EMBEDDINGS]
    assert (embedding["match_id"], embedding["name"]) == (first["id"], "Alice")
    assert embedding["model"] == "test-adaface-model"


def test_a_head_with_no_face_writes_nothing_and_waits_for_its_next_turn():
    detector, written = Script([], [face(1)]), Written()
    faces = stage(detector, {1: "Alice"}, written)
    t0 = 1_800_000_000 * NS
    faces.feed(crop, W, H, t0, "back-of-head", [(7, HEAD)])
    faces.feed(crop, W, H, t0 + NS // 2, "half-a-second", [(7, HEAD)])
    faces.feed(crop, W, H, t0 + NS, "a-second", [(7, HEAD)])
    faces.flush(t0 + NS)
    assert [r["frame_id"] for _, r in written.rows[segment_log.FACE_MATCHES]] == ["a-second"]


def test_a_track_keeps_its_name_through_a_bad_look():
    looks = [[face(1)], [face(1)], [face(9)], [face(1)]]
    written = Written()
    faces = stage(Script(*looks), {1: "Alice"}, written)
    t0 = 1_800_000_000 * NS
    for i in range(4):
        faces.feed(crop, W, H, t0 + i * NS, f"f{i}", [(7, HEAD)])
    faces.flush(t0 + 4 * NS)
    names = [r["name"] for _, r in written.rows[segment_log.FACE_MATCHES]]
    assert names == ["Alice", "Alice", "Alice", "Alice"], "one stranger look doesn't rename the track"
    stranger = written.rows[segment_log.FACE_MATCHES][2][1]
    assert stranger["cluster"] is None, "the track is named, so the look joins no unknown group"
    assert stranger["sim"] == 0.0, "the look keeps its own similarity, for its quality"


def weak(person, k):
    """A turned or blurred look of `person`: 0.4 along their axis, the rest along an axis of its own."""
    emb = np.zeros(512, np.float32)
    emb[person], emb[100 + k] = 0.4, np.sqrt(1 - 0.4**2)
    return face(person, emb=emb)


class Counting(Names):
    """Names that records each face it's asked to group."""

    def __init__(self, names):
        super().__init__(names)
        self.grouped = []

    def unknown(self, emb):
        self.grouped.append(np.asarray(emb))
        return super().unknown(emb)


def run_track(looks, names, seconds):
    """`looks` a second apart on track 7; (the face.match rows, the matcher)."""
    written, matcher = Written(), Counting(names)
    faces = HeadFaces("cam0", Script(*looks), matcher, written, sharpness=lambda patch, box: 100.0)
    t0 = 1_800_000_000 * NS
    for i in range(seconds):
        faces.feed(crop, W, H, t0 + i * NS, f"f{i}", [(7, HEAD)])
    faces.flush(t0 + seconds * NS)
    return [r for _, r in written.rows[segment_log.FACE_MATCHES]], matcher


def test_a_track_that_becomes_a_named_person_never_makes_an_unnamed_group():
    # #3001: its first look matched nobody, but the track was Alice all along; no "Face N" waits to be named for it
    rows, matcher = run_track([[weak(1, k)] for k in range(4)], {1: "Alice", 2: "Ann"}, 4)
    assert [r["name"] for r in rows] == [None, "Alice", "Alice", "Alice"]
    assert [r["cluster"] for r in rows] == [None] * 4 and matcher.grouped == []


def test_a_strangers_first_looks_get_no_group_until_the_track_has_enough():
    rows, matcher = run_track([[face(5)] for _ in range(4)], {1: "Alice", 2: "Ann"}, 4)
    first = head_faces.GROUP_AFTER_LOOKS - 1
    assert [r["cluster"] for r in rows] == [None] * first + ["Face 5"] * (4 - first)
    assert len(matcher.grouped) == 1


def near(person, k=300, w=0.42):
    """A look `w` like `person`, short of the 0.45 gate, the rest along axis `k` (shared, so the mean stays there)."""
    emb = np.zeros(512, np.float32)
    emb[person], emb[k] = w, np.sqrt(1 - w**2)
    return face(person, emb=emb)


def test_a_track_close_to_naming_someone_gets_no_group_before_it_is_named():
    # #3048: three looks 0.42 like Alice name no one, but the track is Alice, as her fourth look shows
    rows, matcher = run_track([[near(1)]] * 3 + [[face(1)]], {1: "Alice", 2: "Ann"}, 4)
    assert [r["name"] for r in rows] == [None, None, None, "Alice"]
    assert [r["cluster"] for r in rows] == [None] * 4 and matcher.grouped == []


def test_a_track_that_stays_close_to_someone_but_unnamed_is_grouped_at_the_hold_limit():
    # A lookalike: every look 0.42 like Alice, never named; it still gets a group, once the track has had its chance
    rows, matcher = run_track([[near(1)]] * 9, {1: "Alice", 2: "Ann"}, 9)
    held = head_faces.GROUP_HOLD_LOOKS - 1
    assert [r["cluster"] for r in rows] == [None] * held + ["Face 300"] * (9 - held)
    assert len(matcher.grouped) == 1


@pytest.mark.parametrize(
    ("look", "grouped"),
    [
        pytest.param(face(5), True, id="straight-on"),
        pytest.param(face(5, norm=19.0, kps=SIDE_ON), True, id="no-frontal-landmarks-but-a-face-norm"),
        pytest.param(side(5, norm=14.0), False, id="side-or-back-of-head"),
        pytest.param(side(5, norm=None), False, id="side-no-norm-given"),
    ],
)
def test_only_looks_that_show_the_face_count_toward_an_unnamed_group(look, grouped):
    # #3155: nine looks of a stranger, past the hold limit; the side or back of a head never makes a group
    rows, matcher = run_track([[look]] * 9, {1: "Alice", 2: "Ann"}, 9)
    first = head_faces.GROUP_AFTER_LOOKS - 1
    expected = [None] * first + ["Face 5"] * (9 - first) if grouped else [None] * 9
    assert [r["cluster"] for r in rows] == expected
    assert bool(matcher.grouped) is grouped


def test_a_side_look_joins_no_group_and_the_group_comes_from_the_face_looks():
    # Two looks at the back of a stranger's head (nearest axis 6), then their face three times (axis 5), then the
    # back again: the group waits for three face looks, is picked from them alone, and the last look doesn't join it.
    looks = [[side(6)], [side(6)], [face(5)], [face(5)], [face(5)], [side(6)]]
    rows, matcher = run_track(looks, {1: "Alice", 2: "Ann"}, 6)
    assert [r["cluster"] for r in rows] == [None, None, None, None, "Face 5", None]
    assert [int(np.argmax(e)) for e in matcher.grouped] == [5]


def test_a_side_look_still_names_the_person_it_matches():
    rows, matcher = run_track([[side(1)]] * 3, {1: "Alice", 2: "Ann"}, 3)
    assert [r["name"] for r in rows] == ["Alice"] * 3 and matcher.grouped == []


def test_a_strangers_track_stays_in_one_group_however_its_looks_vary():
    # Each look alone is nearest a group of its own (its unrelated part); the track's mean is the stranger's
    rows, matcher = run_track([[weak(5, k)] for k in range(8)], {1: "Alice", 2: "Ann"}, 8)
    assert {r["cluster"] for r in rows} - {None} == {"Face 5"}
    assert all(int(np.argmax(e)) == 5 for e in matcher.grouped), "grouped by the mean, never by one look"


def test_a_track_of_looks_too_weak_to_name_alone_is_named_by_their_mean():
    # Each look is 0.4 like Alice, short of the 0.45 gate; their mean grows toward Alice as the unrelated parts cancel.
    looks = [[weak(1, k)] for k in range(3)]
    written = Written()
    faces = stage(Script(*looks), {1: "Alice", 2: "Ann"}, written)
    t0 = 1_800_000_000 * NS
    for i in range(3):
        faces.feed(crop, W, H, t0 + i * NS, f"f{i}", [(7, HEAD)])
    faces.flush(t0 + 3 * NS)
    assert [r["name"] for _, r in written.rows[segment_log.FACE_MATCHES]] == [None, "Alice", "Alice"]


def test_a_named_persons_looks_become_one_visit_with_its_best_shots():
    written = Written()
    faces = stage(Script(*[[face(1)]] * 3), {1: "Alice"}, written)
    t0 = 1_800_000_000 * NS
    for i in range(3):
        faces.feed(crop, W, H, t0 + i * NS, f"f{i}", [(7, HEAD)])
    faces.flush(None)  # stopping closes the open visit
    ((cam, visit),) = written.rows[segment_log.FACE_VISITS]
    assert (cam, visit["name"], visit["samples"]) == ("cam0", "Alice", 3)
    assert (visit["ts_start"], visit["ts_end"]) == (t0 / NS, t0 / NS + 2)
    matches = {r["id"] for _, r in written.rows[segment_log.FACE_MATCHES]}
    assert {s["detection"] for s in visit["shots"]} <= matches


def test_an_ended_track_starts_its_mean_over():
    written = Written()
    faces = stage(Script([face(1)], [face(2)]), {1: "Alice", 2: "Ann"}, written)
    t0 = 1_800_000_000 * NS
    faces.feed(crop, W, H, t0, "f0", [(7, HEAD)])
    faces.end(7)
    faces.feed(crop, W, H, t0 + NS // 10, "f1", [(7, HEAD)])  # looked at again at once, as a new track
    faces.flush(t0 + NS)
    assert [r["name"] for _, r in written.rows[segment_log.FACE_MATCHES]] == ["Alice", "Ann"]


class Roster:
    def person(self, name):
        return {"Alice": "0190a5b8-0000-7000-8000-00000000000a", "Bob": "0190a5b8-0000-7000-8000-00000000000b"}.get(
            name
        )


def test_an_embedding_is_keyed_by_its_person_or_else_its_face_group():
    written = Written()
    faces = HeadFaces(
        "cam0",
        Script([face(1)], *([face(9)] for _ in range(head_faces.GROUP_AFTER_LOOKS))),
        Names({1: "Alice"}),
        written,
        sharpness=lambda p, b: 100.0,
        people=Roster(),
    )
    t0 = 1_800_000_000 * NS
    faces.feed(crop, W, H, t0, "f0", [(7, HEAD)])
    for i in range(head_faces.GROUP_AFTER_LOOKS):  # someone else, unnamed: grouped once the track has enough looks
        faces.feed(crop, W, H, t0 + i * NS, f"f{i}", [(8, HEAD)])
    faces.flush(t0 + head_faces.GROUP_AFTER_LOOKS * NS)
    (_, named), *_, (_, unnamed) = written.rows[segment_log.FACE_EMBEDDINGS]
    assert (named["person"], named["group"]) == ("0190a5b8-0000-7000-8000-00000000000a", None)
    assert unnamed["person"] is None
    assert unnamed["group"] == subjects.group("Face 9")


def test_every_frame_after_a_look_puts_the_named_head_in_the_live_boxes():
    live = []
    faces = HeadFaces(
        "cam0", Script([], [face(1)]), Names({1: "Alice"}), Written(), sharpness=lambda p, b: 100.0, live=live.append
    )
    t0 = 1_800_000_000 * NS
    faces.feed(crop, W, H, t0, "f0", [(7, HEAD)], frame_seq=10)  # back of the head: no face, nothing live
    faces.feed(crop, W, H, t0 + NS, "f1", [(7, HEAD)], frame_seq=40)  # named now
    faces.feed(crop, W, H, t0 + NS + NS // 30, "f2", [(7, HEAD)], frame_seq=41)  # no look due, still a box
    assert [(r["frame_id"], r["frame_seq"], r["start_ns"]) for r in live] == [
        ("f1", 40, t0 + NS),
        ("f2", 41, t0 + NS + NS // 30),
    ]
    (box,) = live[1]["faces"]
    x1, y1, x2, y2 = head_box(HEAD, W, H) or (0, 0, 0, 0)
    assert box["bbox"] == pytest.approx([x1 / W, y1 / H, x2 / W, y2 / H], abs=1e-5)
    assert (box["name"], box["track_id"], live[1]["cam"]) == ("Alice", "7", "cam0")


def test_a_track_is_the_person_the_mean_of_its_whole_track_matches():
    # Alice twice, Bob three times, then a stranger three times: the mean is still most like Bob.
    looks = [[face(1)], [face(1)], [face(2)], [face(2)], [face(2)], [face(9)], [face(9)], [face(9)]]
    faces = HeadFaces(
        "cam0", Script(*looks), Names({1: "Alice", 2: "Bob"}), Written(), sharpness=lambda p, b: 100.0, people=Roster()
    )
    t0 = 1_800_000_000 * NS
    assert faces.who(7) == (None, None)
    for i in range(len(looks)):
        faces.feed(crop, W, H, t0 + i * NS, f"f{i}", [(7, HEAD)])
    assert faces.who(7) == ("0190a5b8-0000-7000-8000-00000000000b", "Bob")


def test_a_track_whose_faces_matched_no_one_is_no_one():
    faces = stage(Script([face(9)], [face(9)]), {1: "Alice"}, Written())
    t0 = 1_800_000_000 * NS
    for i in range(2):
        faces.feed(crop, W, H, t0 + i * NS, f"f{i}", [(7, HEAD)])
    assert faces.who(7) == (None, None)


def test_each_look_hands_its_square_face_patch_to_the_thumbnail_writer_by_match_id():
    detector, written, cut, saved = Script([face(1)]), Written(), [], []

    def cropping(box):
        cut.append(box)
        return crop(box)

    faces = HeadFaces(
        "cam0",
        detector,
        Names({1: "Alice"}),
        written,
        sharpness=lambda p, b: 100.0,
        shots=lambda sid, p: saved.append((sid, p.shape)),
    )
    faces.feed(cropping, W, H, 1_800_000_000 * NS, "frame-0", [(7, HEAD)])
    faces.flush(None)
    ((_, match),) = written.rows[segment_log.FACE_MATCHES]
    x1, y1, _, _ = head_box(HEAD, W, H) or (0, 0, 0, 0)
    # The face is 50 x 58 at (x1 + 10, y1 + 12): a square half again its longer side (87), around its middle.
    side = 87
    left, top = int(x1 + 35 - side / 2), int(y1 + 41 - side / 2)
    assert cut[-1] == (left, top, left + side, top + side)
    assert saved == [(match["id"], (side, side, 3))]


def test_a_face_past_the_frames_edge_is_written_inside_it():
    detector, written = Script([face(1, box=(10.0, 12.0, 700.0, 70.0))]), Written()
    faces = stage(detector, {1: "Alice"}, written)
    faces.feed(crop, W, H, 1_800_000_000 * NS, "frame-0", [(7, HEAD)])
    faces.flush(None)
    ((_, match),) = written.rows[segment_log.FACE_MATCHES]
    assert match["bbox"][2] == 1.0 and all(0 <= v <= 1 for v in match["bbox"])


@pytest.mark.parametrize("norm", [pytest.param(17.5, id="with-norm"), pytest.param(None, id="recognizer-gave-none")])
def test_each_look_writes_what_its_face_is_made_of_beside_its_match(norm):
    # A head half turned: the ears placed, the nose and eyes less so (#2355).
    back = joints([(500, 100), (490, 92), (510, 92), (480, 96), (520, 96)])
    back[:3, 2] = [0.45, 0.31, 0.3]
    detector, written = Script([face(1, norm=norm)]), Written()
    faces = stage(detector, {1: "Alice"}, written)
    t0 = 1_800_000_000 * NS
    faces.feed(crop, W, H, t0, "frame-0", [(7, back)])
    faces.flush(t0)
    ((_, match),) = written.rows[segment_log.FACE_MATCHES]
    ((cam, signal),) = written.rows[segment_log.FACE_SIGNALS]
    assert (cam, signal["match_id"], signal["ts_start"]) == ("cam0", match["id"], match["ts_start"])
    assert signal["head"] == pytest.approx([0.45, 0.31, 0.3, 0.9, 0.9])
    assert signal["frontal"] == pytest.approx(1.0)  # the fake's landmarks are straight on
    assert signal["norm"] == norm


@pytest.mark.parametrize(
    ("nose", "norm"),
    [
        pytest.param(0.9, 9.9, id="turned-away-low-norm"),
        pytest.param(0.39, 20.0, id="blur-low-nose"),
    ],
)
def test_a_look_that_hardly_reads_as_a_face_is_set_aside_with_only_its_signal(nose, norm):
    # The track's first look is a clear Alice; the next one doesn't read as a face (#2355). It must not name the
    # track, start a face group, become a photo, or leave an embedding: only its signal is written.
    seen = joints([(500, 100), (490, 92), (510, 92), (480, 96), (520, 96)])
    seen[0, 2] = nose
    detector, written, saved = Script([face(1, norm=20.0)], [face(9, norm=norm)]), Written(), []
    faces = HeadFaces(
        "cam0",
        detector,
        Names({1: "Alice"}),
        written,
        sharpness=lambda p, b: 100.0,
        shots=lambda sid, p: saved.append(sid),
    )
    t0 = 1_800_000_000 * NS
    faces.feed(crop, W, H, t0, "f0", [(7, HEAD)])
    faces.feed(crop, W, H, t0 + NS, "f1", [(7, seen)])
    faces.flush(None)
    ((_, match),) = written.rows[segment_log.FACE_MATCHES]
    ((_, embedding),) = written.rows[segment_log.FACE_EMBEDDINGS]
    (_, clear), (_, aside) = written.rows[segment_log.FACE_SIGNALS]
    assert match["frame_id"] == "f0" and embedding["match_id"] == match["id"] and saved == [match["id"]]
    assert clear["match_id"] == match["id"]
    assert aside["match_id"] != match["id"] and aside["norm"] == norm and aside["head"][0] == pytest.approx(nose)
    assert faces.who(7)[1] == "Alice"  # a look of someone else's axis would have pulled the mean off her


def test_a_look_at_the_cutoffs_is_kept():
    edge = joints([(500, 100), (490, 92), (510, 92), (480, 96), (520, 96)])
    edge[0, 2] = 0.4
    written = Written()
    faces = stage(Script([face(1, norm=10.0)]), {1: "Alice"}, written)
    faces.feed(crop, W, H, 1_800_000_000 * NS, "f0", [(7, edge)])
    faces.flush(None)
    ((_, match),) = written.rows[segment_log.FACE_MATCHES]
    assert match["name"] == "Alice"


class Learning(Names):
    """Names that records what each ended track would teach the gallery (#3013)."""

    def __init__(self, names):
        super().__init__(names)
        self.taught = []

    def learn(self, looks, names, source):
        self.taught.append((source, len(looks), list(names)))
        return 0


@pytest.mark.parametrize("ending", ["end", "gap"])
def test_a_track_that_ends_offers_its_looks_and_their_own_matches_to_the_gallery(ending):
    matcher = Learning({1: "Alice"})
    faces = HeadFaces("cam0", Script([face(1)], [face(9)], [face(1)]), matcher, Written(), sharpness=lambda p, b: 100.0)
    t0 = 1_800_000_000 * NS
    for i in range(3):
        faces.feed(crop, W, H, t0 + i * NS, f"f{i}", [(7, HEAD)])
    faces.end(7) if ending == "end" else faces.gap()
    assert matcher.taught == [("cam0:7", 3, ["Alice", None, "Alice"])]


def test_the_side_or_back_of_a_head_is_never_offered_to_the_gallery():
    # #3155: the track is Alice, but only its two face looks may teach; the side look's own match still counts
    matcher = Learning({1: "Alice"})
    faces = HeadFaces("cam0", Script([face(1)], [side(1)], [face(1)]), matcher, Written(), sharpness=lambda p, b: 100.0)
    t0 = 1_800_000_000 * NS
    for i in range(3):
        faces.feed(crop, W, H, t0 + i * NS, f"f{i}", [(7, HEAD)])
    faces.end(7)
    assert matcher.taught == [("cam0:7", 2, ["Alice", "Alice", "Alice"])]
