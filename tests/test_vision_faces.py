"""The vision loop runs without the optional faces extra instead of crashing at startup, and the faces command line
says so when it can't read an image instead of handing None to the face model."""

import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from liberational import config as cfg
from liberational import model_runtime
from liberational.vision import faces


def test_enroll_skips_an_image_it_cannot_read(tmp_path):
    with patch.object(faces, "embed", autospec=True) as embed, pytest.raises(SystemExit, match="no faces found"):
        faces.enroll("Alice", [str(tmp_path / "missing.jpg")])
    embed.assert_not_called()


def test_who_reports_an_image_it_cannot_read(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["faces", "who", str(tmp_path / "missing.jpg")])
    with patch.object(faces, "identify", autospec=True) as identify, pytest.raises(SystemExit, match="can't read"):
        faces.main()
    identify.assert_not_called()


def unit(k):
    v = np.zeros(512, np.float32)
    v[k] = 1
    return v


@pytest.mark.parametrize(
    ("counts", "kept"),
    [
        pytest.param([5, 5, 5], [2, 3, 4], id="all-equal-evicts-the-oldest"),
        pytest.param([5, 2, 5], [1, 3, 4], id="evicts-the-least-seen"),
        pytest.param([3, 1, 1], [1, 3, 4], id="tie-goes-to-the-oldest"),
    ],
)
def test_a_new_stranger_in_a_full_store_keeps_its_face_number(tmp_path, monkeypatch, counts, kept):
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown_clusters.npz")
    monkeypatch.setattr(faces, "UNK_MAX", 3)
    faces._unk_save([1, 2, 3], np.stack([unit(0), unit(1), unit(2)]), counts, 4)
    assert [faces.group_unknown(unit(3)) for _ in range(3)] == [4, 4, 4]
    ids, _, seen, nxt = faces._unk_load()
    assert sorted(int(i) for i in ids) == kept
    assert seen[[int(i) for i in ids].index(4)] == 3
    assert nxt == 5


DAY = 86_400.0


@pytest.fixture
def full(tmp_path, monkeypatch):
    """A store of UNK_MAX 3, its state dir in tmp_path, and a clock to move: (clock, state)."""
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown_clusters.npz")
    monkeypatch.setattr(faces, "UNK_MAX", 3)
    clock = [100 * DAY]
    monkeypatch.setattr(faces.time, "time", lambda: clock[0])
    state = tmp_path / "state"
    state.mkdir()
    with cfg.override({"paths": {"state": str(state)}}):
        yield clock, state


def stored():
    return sorted(int(i) for i in faces._unk_load()[0])


def test_a_full_store_of_old_groups_keeps_a_new_stranger_seen_once(full):
    # Ranked by faces alone, days-old groups seen twice outlived every new stranger's group, seen once
    faces._unk_save([1, 2], np.stack([unit(0), unit(1)]), [2, 2], 3)  # from before the store dated groups
    assert faces.group_unknown(unit(2)) == 3
    assert faces.group_unknown(unit(3)) == 4
    assert stored() == [2, 3, 4]


def test_room_comes_from_the_group_people_no_longer_lists_then_one_seen_once_then_the_oldest_seen(full):
    clock, _ = full
    faces.group_unknown(unit(0))  # group 1, seen 3 times, last 8 days ago: past People's 7 days
    faces.group_unknown(unit(0))
    faces.group_unknown(unit(0))
    clock[0] += 6 * DAY
    faces.group_unknown(unit(1))  # group 2, seen 3 times, last 2 days ago
    faces.group_unknown(unit(1))
    faces.group_unknown(unit(1))
    clock[0] += 1 * DAY
    faces.group_unknown(unit(2))  # group 3, seen once, a day ago
    clock[0] += 1 * DAY
    assert faces.group_unknown(unit(3)) == 4
    assert stored() == [2, 3, 4]
    assert faces.group_unknown(unit(4)) == 5  # 3 and 4 were seen once: the older goes
    assert stored() == [2, 4, 5]
    faces.group_unknown(unit(4))  # 5 is seen again, so 4 is the one seen once
    clock[0] += 1
    assert faces.group_unknown(unit(5)) == 6
    assert stored() == [2, 5, 6]
    faces.group_unknown(unit(5))
    clock[0] += 1
    assert faces.group_unknown(unit(6)) == 7  # all are listed now: the one seen longest ago goes
    assert stored() == [5, 6, 7]


def put_aside(state):
    (state / "inbox_skips.json").write_text(json.dumps({"face": [1], "voice": [], "line": []}))


def took_a_shot_out(state):
    assert faces.recenter(1, np.stack([unit(0)]), 1)


def undid_its_name(state):
    assert faces.drop_groups([1]) == 1  # named, it left the store
    faces.unname_cluster(1, unit(0), 2)


@pytest.mark.parametrize("touch", [put_aside, took_a_shot_out, undid_its_name], ids=lambda f: f.__name__)
def test_a_group_the_owner_touched_is_never_evicted(full, touch):
    _, state = full
    faces._unk_save([1, 2], np.stack([unit(0), unit(1)]), [2, 2], 3)  # 1 would go first
    touch(state)
    faces.group_unknown(unit(2))
    faces.group_unknown(unit(3))
    faces.group_unknown(unit(4))
    assert stored() == [1, 4, 5]


def test_a_group_marked_not_someone_is_never_evicted(full, monkeypatch):
    from liberational.people import face_nots

    monkeypatch.setattr(face_nots, "read", lambda state: {1: {"person-1"}})
    faces._unk_save([1, 2], np.stack([unit(0), unit(1)]), [2, 2], 3)
    faces.group_unknown(unit(2))
    faces.group_unknown(unit(3))
    assert stored() == [1, 3, 4]


def test_nothing_is_evicted_while_the_owners_marks_cannot_be_read(full):
    _, state = full
    (state / "inbox_skips.json").write_text("{not json")
    faces._unk_save([1, 2, 3], np.stack([unit(0), unit(1), unit(2)]), [2, 2, 2], 4)
    assert faces.group_unknown(unit(3)) == 4
    assert stored() == [1, 2, 3, 4]


def blend(a, b, w):
    """A unit vector at cosine about w from unit(a) and the rest toward unit(b)."""
    v = w * unit(a) + np.sqrt(1 - w * w) * unit(b)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown_clusters.npz")


def test_a_group_keeps_its_first_face_so_it_cannot_drift_to_another_person(store):
    # Person A, then faces that are each a little closer to person B: a running mean followed them and took B too.
    first = faces.group_unknown(unit(0))
    drifting = [faces.group_unknown(blend(0, 1, w)) for w in (0.7, 0.6, 0.55)]
    assert drifting == [first] * 3
    assert faces.group_unknown(unit(1)) != first  # B starts a group of their own
    ids, vecs, seen, _ = faces._unk_load()
    i = [int(x) for x in ids].index(first)
    assert float(vecs[i] @ unit(0)) == pytest.approx(1.0)
    assert seen[i] == 4


@pytest.mark.parametrize(
    ("similarity", "joins"),
    [pytest.param(0.47, True, id="above-the-join"), pytest.param(0.43, False, id="below-the-join")],
)
def test_a_face_joins_only_above_the_unknown_join(store, similarity, joins):
    first = faces.group_unknown(unit(0))
    assert bool(faces.group_unknown(blend(0, 1, similarity)) == first) is joins


def test_a_group_is_dated_only_by_a_face_starting_or_joining_it(store, monkeypatch):
    # People counts only groups seen lately, so the store dates each group by its faces, never by a save
    clock = [1_000.0]
    monkeypatch.setattr(faces.time, "time", lambda: clock[0])
    faces._unk_save([1, 2], np.stack([unit(0), unit(1)]), [2, 2], 3)  # groups from before: no face has dated them
    assert faces.unknown_last_seen() == {}
    clock[0] = 5_000.0
    assert faces.group_unknown(unit(0)) == 1  # joins group 1
    new = faces.group_unknown(unit(2))  # starts a group
    assert faces.unknown_last_seen() == {1: 5_000.0, new: 5_000.0}


def test_a_store_stamped_at_its_first_save_counts_none_of_those_times(store):
    # Only a later face dates a group, not its first save's time
    np.savez(
        faces.UNK_DB,
        ids=np.array([1, 2, 3]),
        vecs=np.stack([unit(0), unit(1), unit(2)]),
        counts=np.array([2, 2, 2]),
        next=np.int64(4),
        sums=np.stack([unit(0), unit(1), unit(2)]),
        last_seen=np.array([1_000.0, 1_000.0, 1_200.0]),
    )
    assert faces.unknown_last_seen() == {3: 1_200.0}


# Folding near-duplicate groups: one stranger split across groups becomes one, and every old number still reads
# as the group it went into.


@pytest.fixture
def folding(tmp_path, monkeypatch):
    """A group store and a state dir in tmp_path: the state dir."""
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown_clusters.npz")
    state = tmp_path / "state"
    state.mkdir()
    with cfg.override({"paths": {"state": str(state)}}):
        yield state


def look(vision, frame, group):
    """Record a face.match of unnamed group `group` in frame `frame`, a minute ago."""
    vision.add("face.match", ts_start=time.time() - 60, cam="cam0", frame_id=frame, cluster=f"Face {group}", name=None)


def test_near_duplicates_fold_into_the_oldest_and_their_numbers_stay_aliases(folding):
    clock = time.time()
    # 3 and 5 are one stranger (0.8 alike), 9 is someone else, 4 is like 3 too but put aside on People.
    faces._unk_save(
        [5, 3, 9, 4],
        np.stack([blend(0, 1, 0.8), unit(0), unit(2), blend(0, 3, 0.9)]),
        [2, 4, 3, 6],
        10,
        last_seen={5: clock, 3: clock - 3600},
    )
    (folding / "inbox_skips.json").write_text(json.dumps({"face": [4], "voice": [], "line": []}))
    assert faces.fold_unknown() == [(5, 3, pytest.approx(0.8))]
    assert faces.unknown_groups() == [{"id": 3, "count": 6}, {"id": 4, "count": 6}, {"id": 9, "count": 3}]
    assert faces.group_aliases() == {5: 3}
    assert [faces.resolve_group(g) for g in (5, 3, 9)] == [3, 3, 9]
    assert faces.unknown_last_seen()[3] == pytest.approx(clock)  # the newer of the two
    assert faces._unk_load()[3] == 10  # numbers already handed out are never reused


@pytest.mark.parametrize(
    ("similarity", "folds"),
    [pytest.param(0.72, True, id="above-the-fold"), pytest.param(0.68, False, id="below-the-fold")],
)
def test_groups_fold_only_above_the_fold_similarity(folding, similarity, folds):
    faces._unk_save([1, 2], np.stack([unit(0), blend(0, 1, similarity)]), [1, 1], 3)
    assert bool(faces.fold_unknown()) is folds
    assert len(faces.unknown_groups()) == (1 if folds else 2)


def test_a_dry_run_folds_nothing(folding):
    faces._unk_save([1, 2], np.stack([unit(0), blend(0, 1, 0.8)]), [1, 1], 3)
    assert [(a, b) for a, b, _ in faces.fold_unknown(apply=False)] == [(2, 1)]
    assert len(faces.unknown_groups()) == 2
    assert faces.group_aliases() == {}


def test_a_group_the_owner_touched_never_folds_or_takes_another(folding):
    faces._unk_save([1, 2, 3], np.stack([unit(0), blend(0, 1, 0.8), unit(4)]), [1, 1, 1], 4)
    assert faces.recenter(1, np.stack([unit(0)]), 0)  # the owner took a shot out of 1
    assert faces.fold_unknown() == []
    assert sorted(g["id"] for g in faces.unknown_groups()) == [1, 2, 3]


def test_two_groups_seen_in_one_frame_never_fold(folding, vision):
    faces._unk_save([1, 2, 3], np.stack([unit(0), blend(0, 1, 0.8), blend(0, 2, 0.75)]), [1, 1, 1], 4)
    look(vision, "frame-a", 1)
    look(vision, "frame-a", 2)
    look(vision, "frame-b", 3)
    assert [(a, b) for a, b, _ in faces.fold_unknown()] == [(3, 1)]  # 2 was beside 1: two people


def test_a_group_seen_beside_one_folded_in_never_joins_it(folding, vision):
    # 2 folds into 1; 3 is like 1 too, but was seen beside 2, so it is someone else
    faces._unk_save([1, 2, 3], np.stack([unit(0), blend(0, 1, 0.9), blend(0, 2, 0.8)]), [1, 1, 1], 4)
    look(vision, "frame-a", 2)
    look(vision, "frame-a", 3)
    assert [(a, b) for a, b, _ in faces.fold_unknown()] == [(2, 1)]


def test_a_stranger_split_across_two_groups_folds_once_their_centroids_meet(folding):
    # Two first faces 0.31 apart start separate groups; faces between them join
    # the first, whose centroid moves toward the second until they are one stranger.
    second = blend(0, 1, 0.31)
    between = unit(0) * 1.05 + second
    between = (between / np.linalg.norm(between)).astype(np.float32)
    assert faces.group_unknown(unit(0)) == 1
    assert faces.group_unknown(second) == 2
    assert [faces.group_unknown(between) for _ in range(2)] == [1, 1]
    assert sorted(g["id"] for g in faces.unknown_groups()) == [1, 2]  # centroids 0.66 alike: not yet
    assert [faces.group_unknown(between) for _ in range(2)] == [1, 1]
    assert faces.unknown_groups() == [{"id": 1, "count": 6}]
    assert faces.group_aliases() == {2: 1}
    assert faces.group_unknown(second) == 1  # the second's faces come to the one group now


def test_a_join_never_folds_the_owners_group(folding):
    second = blend(0, 1, 0.31)
    between = unit(0) * 1.05 + second
    between = (between / np.linalg.norm(between)).astype(np.float32)
    faces.group_unknown(unit(0))
    faces.group_unknown(second)
    (folding / "inbox_skips.json").write_text(json.dumps({"face": [2], "voice": [], "line": []}))
    for _ in range(4):
        faces.group_unknown(between)
    assert sorted(g["id"] for g in faces.unknown_groups()) == [1, 2]


def test_a_join_folds_nothing_while_the_owners_marks_cannot_be_read(folding):
    second = blend(0, 1, 0.31)
    between = unit(0) * 1.05 + second
    between = (between / np.linalg.norm(between)).astype(np.float32)
    faces.group_unknown(unit(0))
    faces.group_unknown(second)
    (folding / "inbox_skips.json").write_text("{not json")
    for _ in range(4):
        faces.group_unknown(between)
    assert sorted(g["id"] for g in faces.unknown_groups()) == [1, 2]


def test_an_alias_outlives_its_group_being_named_and_follows_a_later_fold(folding):
    faces._unk_save([1, 2, 3], np.stack([unit(0), unit(1), unit(2)]), [1, 1, 1], 4, aliases={3: 2})
    faces._unk_save([1, 2], np.stack([unit(0), unit(1)]), [1, 1], 4, aliases={2: 1})  # then 2 went into 1
    assert faces.group_aliases() == {3: 1, 2: 1}
    faces.drop_groups([1])  # as naming it takes it out of the store
    assert faces.resolve_group(3) == 1


def test_a_regroup_keeps_the_merged_numbers_as_aliases(folding):
    faces._unk_save([3, 5], np.stack([unit(0), blend(0, 1, 0.6)]), [1, 1], 6)
    assert [(a, b) for a, b, _ in faces.regroup_unknown()] == [(5, 3)]
    assert faces.group_aliases() == {5: 3}


def test_face_inventory_uses_the_actual_provider_and_device_after_prepare(tmp_path, monkeypatch):
    class Detection:
        taskname = "detection"
        session = SimpleNamespace(
            get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            get_provider_options=lambda: {"CUDAExecutionProvider": {"device_id": "1"}, "CPUExecutionProvider": {}},
        )

        def prepare(self, **kwargs):
            assert kwargs == {"ctx_id": 0, "input_size": (640, 640), "det_thresh": 0.5}

    class Recognition:
        session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

        def __init__(self, path, execution_providers):
            assert path == tmp_path / "recognition.onnx"
            assert execution_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    detector = tmp_path / "detector.onnx"
    recognizer = tmp_path / "recognition.onnx"
    detector.touch()
    recognizer.touch()
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "insightface.model_zoo", SimpleNamespace(get_model=lambda *a, **kw: Detection()))
    monkeypatch.setattr(faces, "AdaFaceRecognition", Recognition)
    monkeypatch.setattr(faces, "detector_path", lambda: detector)
    monkeypatch.setattr(faces, "recognition_path", lambda: recognizer)
    monkeypatch.setattr(faces, "recognition_model", lambda: "test-adaface-model")
    monkeypatch.setattr(faces, "providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(faces, "_app", None)
    with (
        cfg.override(
            {"paths": {"state": str(tmp_path)}, "models": {"insightface": {"detector": "scrfd/detector.onnx"}}}
        ),
        model_runtime.reporting("see-faces"),
    ):
        loaded = faces.app()
        assert faces.app() is loaded
        (report,) = (tmp_path / "model_loads").glob("*.json")
        models = json.loads(report.read_text())["models"]
        assert {(model["key"], model["name"], model["device"]) for model in models} == {
            ("models.insightface.detector", "scrfd/detector.onnx", "GPU 1"),
            ("models.adaface.recognition", "test-adaface-model", "CPU"),
        }


# AdaFace upstream face_alignment/mtcnn_pytorch/src/align_trans.py's square112 reference points,
# rounded to the float32 precision used by the retained InsightFace five-point aligner.
LANDMARKS = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


@pytest.fixture
def recognition(tmp_path, monkeypatch):
    pytest.importorskip("insightface")
    output = np.zeros((1, 512), np.float32)
    output[0, :2] = [3, 4]
    feeds = []

    class Session:
        def __init__(self, path, providers):
            assert providers == ["CPUExecutionProvider"]

        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map={"face_input": faces.ADAFACE_INPUT_FORMAT})

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 3, 112, 112], type="tensor(float)")]

        def run(self, names, feed):
            feeds.append(feed["input"])
            return [output]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    path = tmp_path / "face.onnx"
    path.touch()
    return faces.AdaFaceRecognition(path, ["CPUExecutionProvider"]), output, feeds


def test_adaface_uses_selected_official_cvlface_rgb_transform(recognition):
    model, _, feeds = recognition
    # The selected HF model card uses torchvision ToTensor + Normalize(.5, .5) on RGB PIL input.
    # Hand-derived expected RGB values after converting this OpenCV BGR pixel: +1, -1/255, -1.
    # This deliberately differs from the older original AdaFace BGR checkpoint.
    image = np.full((112, 112, 3), [0, 127, 255], dtype=np.uint8)
    face = SimpleNamespace(kps=LANDMARKS)
    embedding = model.get(image, face)
    assert feeds[0].shape == (1, 3, 112, 112)
    assert feeds[0].dtype == np.float32
    np.testing.assert_allclose(feeds[0][0, :, 50, 50], [1, -1 / 255, -1], atol=1e-7)
    np.testing.assert_allclose(embedding[:3], [0.6, 0.8, 0], atol=1e-7)
    assert face.embedding is embedding


def test_adaface_keeps_upstream_five_point_alignment_when_the_face_is_translated(recognition):
    model, _, feeds = recognition
    image = np.zeros((140, 140, 3), np.uint8)
    image[7:119, 10:122] = [0, 127, 255]
    model.get(image, SimpleNamespace(kps=LANDMARKS + [10, 7]))
    np.testing.assert_allclose(feeds[0][0, :, 0, 0], [1, -1 / 255, -1], atol=1e-7)
    np.testing.assert_allclose(feeds[0][0, :, 111, 111], [1, -1 / 255, -1], atol=1e-7)


@pytest.mark.parametrize("bad", [np.zeros((1, 512)), np.full((1, 512), np.nan), np.ones((1, 256))])
def test_adaface_rejects_invalid_vectors_instead_of_writing_them(recognition, bad):
    model, _, _ = recognition
    model.session.run = lambda *a: [bad]
    with pytest.raises(ValueError, match="embedding"):
        model.get(np.zeros((112, 112, 3), np.uint8), SimpleNamespace(kps=LANDMARKS))


def test_missing_recognition_file_does_not_load_the_detector(tmp_path, monkeypatch):
    detector = tmp_path / "det.onnx"
    detector.touch()
    monkeypatch.setattr(faces, "detector_path", lambda: detector)
    monkeypatch.setattr(faces, "recognition_path", lambda: tmp_path / "missing.onnx")
    monkeypatch.setattr(faces, "_app", None)
    with pytest.raises(FileNotFoundError, match="configured models"):
        faces.app()


def test_adaface_rejects_unmarked_or_older_bgr_artifacts(recognition, tmp_path, monkeypatch):
    model, _, _ = recognition
    monkeypatch.setattr(type(model.session), "get_modelmeta", lambda self: SimpleNamespace(custom_metadata_map={}))
    with pytest.raises(ValueError, match="RGB export metadata"):
        faces.AdaFaceRecognition(tmp_path / "face.onnx", ["CPUExecutionProvider"])


def test_scrfd_adapter_keeps_five_points_and_unit_embedding_for_head_faces(recognition, monkeypatch):
    from liberational.see.faces import InsightFaceDetector

    model, _, _ = recognition

    class Detection:
        def detect(self, image, max_num, metric):
            assert max_num == 0 and metric == "default"
            return np.array([[0, 0, 112, 112, 0.9]], dtype=np.float32), LANDMARKS[None]

    pipeline = faces.FacePipeline(Detection(), model)
    monkeypatch.setattr(faces, "app", lambda: pipeline)
    result = InsightFaceDetector().detect(np.full((112, 112, 3), [0, 127, 255], np.uint8))
    assert len(result) == 1
    assert result[0].bbox == (0, 0, 112, 112)
    assert result[0].det == pytest.approx(0.9)
    np.testing.assert_array_equal(result[0].kps, LANDMARKS)
    np.testing.assert_allclose(result[0].emb[:3], [0.6, 0.8, 0], atol=1e-7)
    assert result[0].norm == pytest.approx(5.0), "AdaFace's raw norm, before the unit vector"


def tagged(*people, faces=3):
    """Owner-tagged groups for held_out: one per person given, its centroid along the axis given (person, axis)."""
    return {gid: (person, np.stack([unit(axis)] * faces), unit(axis)) for gid, (person, axis) in enumerate(people, 1)}


@pytest.mark.parametrize(
    ("groups", "status", "counts"),
    [
        pytest.param([("Alice", 0)] * 3 + [("Bob", 1)] * 3, "on", (6, 0, 0), id="each-group-named-from-the-others"),
        pytest.param([("Alice", 0)] * 5, "unchecked", (5, 0, 0), id="too-few-groups-to-judge"),
        pytest.param([("Alice", 0)] * 6, "unchecked", (6, 0, 0), id="one-person-nothing-can-be-wrong"),
        pytest.param(
            [("Alice", 0)] * 3 + [("Bob", 1)] * 2 + [("Alice", 1)], "off", (3, 1, 2), id="too-many-wrong"
        ),  # the mis-tagged group is named Bob, and Bob's own groups tie between him and it: unknown
        pytest.param(
            [("Alice", 0), ("Alice", 2), ("Alice", 4), ("Bob", 1), ("Bob", 3), ("Bob", 5)],
            "off",
            (0, 0, 6),
            id="too-few-decided",
        ),
    ],
)
def test_the_held_out_check_names_each_tagged_group_from_the_others(groups, status, counts):
    check = faces.held_out(tagged(*groups))
    assert check["status"] == status
    assert (check["right"], check["wrong"], check["unknown"]) == counts


def test_the_held_out_check_names_a_mis_tagged_group():
    check = faces.held_out(tagged(*[("Alice", 0)] * 3, *[("Bob", 1)] * 3, ("Alice", 1)))
    assert [(s["gid"], s["tagged"], s["looks_like"]) for s in check["suspects"]] == [(7, "Alice", "Bob")]


@pytest.mark.parametrize(
    ("saved", "trusted"),
    [
        pytest.param(None, True, id="no-check-yet"),
        pytest.param('{"status": "unchecked"}', True, id="unchecked"),
        pytest.param('{"status": "on"}', True, id="on"),
        pytest.param('{"status": "off"}', False, id="off"),
        pytest.param("not json", True, id="unreadable"),
    ],
)
def test_naming_is_trusted_unless_the_saved_check_failed(tmp_path, monkeypatch, saved, trusted):
    monkeypatch.setattr(faces, "CHECK", tmp_path / "check.json")
    if saved is not None:
        (tmp_path / "check.json").write_text(saved)
    assert faces.trusted() is trusted


# ---- learning from confidently named tracks


@pytest.fixture
def named(tmp_path, monkeypatch):
    """A gallery of Alice (axis 0) and Bob (axis 1), the owner's rows, and no learned ones yet."""
    monkeypatch.setattr(faces, "DB", tmp_path / "db.npz")
    monkeypatch.setattr(faces, "LEARNED_DB", tmp_path / "learned.npz")
    monkeypatch.setattr(faces, "NOTS_DB", tmp_path / "nots.npz")
    monkeypatch.setattr(faces, "CHECK", tmp_path / "check.json")
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown_clusters.npz")
    faces.save_db(["Alice", "Bob"], np.stack([unit(0), unit(1)]))
    with cfg.override({"faces": {"learn": True}}):
        yield


def views(person, k0, n, w=0.8):
    """`n` looks of `person`: w along their axis, the rest along an axis of each look's own."""
    return [blend(person, 100 + k0 + k, w) for k in range(n)]


def test_a_confident_track_teaches_new_views_of_its_person_and_says_so_without_a_name(named, caplog):
    caplog.set_level("INFO")
    added = faces.learn_from_track(views(0, 0, 5), [None] * 5, "cam0:7")
    assert added == faces.LEARN_PER_TRACK
    learned = faces.load_learned()
    assert learned["names"] == ["Alice"] * added and learned["sources"] == ["cam0:7"] * added
    assert faces.load_db()[0] == ["Alice", "Bob"]  # the owner's rows are as they were
    assert len(faces.gallery()[0]) == 2 + added
    # One line for the track, with the count: a line per view repeated the same text.
    (line,) = [r.getMessage() for r in caplog.records if "face row" in r.getMessage()]
    assert line.startswith(f"learned {added} face rows for person ") and "Alice" not in line


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("unsure", id="mean-short-of-learn-sim"),
        pytest.param("someone-else", id="a-look-matched-someone-else"),
        pytest.param("off", id="setting-off"),
        pytest.param("untrusted", id="gallery-not-trusted"),
    ],
)
def test_a_track_teaches_nothing_unless_it_is_sure_and_learning_is_on(named, case):
    looks, names = views(0, 0, 5), [None] * 5
    if case == "unsure":
        looks = [blend(0, 100, 0.45)] * 5  # the same 0.45-like-Alice look: a mean short of LEARN_SIM
    if case == "someone-else":
        names[2] = "Bob"
    if case == "untrusted":
        faces.CHECK.write_text(json.dumps({"status": "off"}))
    with cfg.override({"faces": {"learn": case != "off"}}):
        assert faces.learn_from_track(looks, names, "cam0:7") == 0
    assert faces.load_learned()["names"] == []


def test_only_the_owners_rows_vouch_for_a_track_so_learned_rows_cannot_pull_a_lookalike_in(named):
    # A learned row along axis 5 is Alice's; a track along it is sure by the gallery, but not by the owner's rows
    faces._save_learned(
        {
            "names": ["Alice"],
            "embs": unit(5)[None],
            "sources": ["cam0:1"],
            "ts": np.array([0.0]),
            "sims": np.array([0.9]),
            "margins": np.array([0.5]),
        }
    )
    looks = [blend(5, 100 + k, 0.9) for k in range(5)]
    assert faces.learn_from_track(looks, [None] * 5, "cam0:7") == 0


def test_a_view_that_looks_like_someone_else_is_not_learned(named):
    like_bob = 0.8 * unit(0) + 0.45 * unit(1) + 0.4 * unit(300)  # Alice, but 0.45 like Bob's row
    like_bob = (like_bob / np.linalg.norm(like_bob)).astype(np.float32)
    looks = [*views(0, 0, 4), like_bob]
    faces.learn_from_track(looks, [None] * 5, "cam0:7")
    learned = faces.load_learned()
    assert all(float(e @ unit(1)) <= faces.LEARN_OTHERS for e in learned["embs"])


@pytest.mark.parametrize(
    ("group_like", "learned"),
    [
        pytest.param(0.78, False, id="group-nearly-as-like-as-alice"),
        pytest.param(0.65, True, id="group-clearly-less-like"),
    ],
)
def test_a_view_nearly_as_like_an_unnamed_group_as_its_person_is_not_learned(named, group_like, learned):
    # Five looks each 0.80 like Alice; an unnamed group's face is `group_like` like the first look alone
    looks = views(0, 0, 5)
    group = group_like * looks[0] + np.sqrt(1 - group_like**2) * unit(400)
    faces._unk_save([1], group[None].astype(np.float32), [1], 2)
    assert faces.learn_from_track(looks, [None] * 5, "cam0:7") == faces.LEARN_PER_TRACK
    kept = faces.load_learned()["embs"]
    assert bool((kept @ looks[0] > 0.999).any()) is learned


def test_learned_views_stay_within_their_budget_and_never_push_out_the_owners(named):
    for t in range(20):
        faces.learn_from_track(views(0, 10 * t, 5), [None] * 5, f"cam0:{t}")
    learned = faces.load_learned()
    assert learned["names"].count("Alice") == faces.LEARN_MAX_PER_NAME
    assert faces.load_db()[0] == ["Alice", "Bob"]


def test_renaming_or_forgetting_a_person_moves_or_drops_their_learned_views_and_unlearn_removes_them(named):
    faces.learn_from_track(views(0, 0, 5), [None] * 5, "cam0:7")
    faces.learn_from_track(views(1, 50, 5), [None] * 5, "porch:3")
    faces.rename("alice", "Alicia")
    assert sorted(set(faces.load_learned()["names"])) == ["Alicia", "Bob"]
    faces.forget("Bob")
    assert set(faces.load_learned()["names"]) == {"Alicia"}
    assert faces.unlearn() == faces.LEARN_PER_TRACK and faces.load_learned()["names"] == []


def learned_alice_at(axis):
    faces._save_learned(
        {
            "names": ["Alice"],
            "embs": unit(axis)[None],
            "sources": ["cam0:1"],
            "ts": np.array([0.0]),
            "sims": np.array([0.9]),
            "margins": np.array([0.5]),
        }
    )


@pytest.mark.parametrize(
    ("face", "named_as"),
    [
        pytest.param(lambda: blend(5, 9, 0.5), None, id="only-a-learned-row-under-the-bar"),
        pytest.param(lambda: blend(5, 9, 0.6), "Alice", id="only-a-learned-row-over-the-bar"),
        pytest.param(lambda: blend(0, 9, 0.5), "Alice", id="the-owners-row-names-it"),
    ],
)
def test_a_name_only_learned_rows_give_needs_a_learned_row_at_the_bar(named, face, named_as):
    # Learned views widen a person; alone they name a face only when one is LEARNED_BAR alike
    learned_alice_at(5)
    assert faces.match(face())[0] == named_as


def test_autoname_plans_by_the_same_bar(named):
    learned_alice_at(5)
    names, embs, owners = faces.gallery_rows()
    reps = np.stack([blend(5, 9, 0.5), blend(5, 9, 0.6)])
    assert faces._plan_names(names, embs, [1, 2], [3, 3], reps, set(), owners) == [
        (2, "Alice", pytest.approx(0.6, abs=1e-3))
    ]


# ---- the one-time cleanup of groups and learned rows from the side or back of a head


def test_dropping_groups_keeps_the_rest_their_members_and_the_next_number(tmp_path, monkeypatch):
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown_clusters.npz")
    faces._unk_save([1, 2, 3], np.stack([unit(0), unit(1), unit(2)]), [4, 5, 6], 4)
    assert faces.drop_groups([2, 9]) == 1
    ids, _, seen, nxt = faces._unk_load()
    assert ([int(i) for i in ids], seen, nxt) == ([1, 3], [4, 6], 4)
    assert faces.group_unknown(unit(7)) == 4, "a new stranger never takes a dropped number"


def test_unlearning_views_removes_only_those_learned_rows_and_never_the_owners(named):
    added = faces.learn_from_track(views(0, 0, 5), [None] * 5, "cam0:7")
    learned = faces.load_learned()
    assert added == 2
    side = learned["embs"][0]
    assert faces.unlearn_views([side * 3.0, unit(0)]) == 1  # any length; the owner's row of Alice is no learned row
    assert faces.load_learned()["embs"] == pytest.approx(learned["embs"][1:])
    assert faces.load_db()[0] == ["Alice", "Bob"]


# ---- the owner's negatives: "not this person" holds the matcher off that face


def dana(k, w=0.6):
    """A look of Dana, who is `w` like Alice's row (axis 0) and mostly a face of her own (axis 200)."""
    v = w * unit(0) + 0.7 * unit(200) + np.sqrt(max(0.0, 1 - w * w - 0.49)) * unit(300 + k)
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_a_face_the_owner_said_is_not_alice_keeps_its_next_look_from_being_named_alice(named):
    held_out = dana(1, w=0.58)
    assert faces.match(held_out)[0] == "Alice"  # a lookalike the gallery names Alice
    assert faces.add_not("Alice", dana(0), "group:7")
    assert faces.match(held_out) == (None, pytest.approx(0.58, abs=1e-3))
    assert faces.match(blend(0, 9, 0.8))[0] == "Alice", "Alice herself still clears her negative by the margin"
    assert faces.remove_not("Alice", "group:7")
    assert faces.match(held_out)[0] == "Alice"


def test_negatives_are_kept_apart_bounded_and_not_added_twice(named, monkeypatch):
    monkeypatch.setattr(faces, "NOT_MAX_PER_NAME", 2)
    assert faces.add_not("Alice", dana(0), "shot:a")
    assert not faces.add_not("Alice", dana(0), "shot:b"), "the same face again adds nothing"
    assert faces.add_not("Alice", unit(400), "shot:c")
    assert faces.add_not("Alice", unit(401), "shot:d")
    names, _, keys = faces.load_nots()
    assert names == ["Alice", "Alice"] and "shot:d" in keys
    assert faces.load_db()[0] == ["Alice", "Bob"] and faces.load_learned()["names"] == []


def test_a_negative_follows_a_rename_and_goes_when_the_person_is_forgotten(named):
    faces.add_not("Alice", dana(0), "group:7")
    faces.rename("alice", "Alicia")
    assert faces.load_nots()[0] == ["Alicia"]
    assert faces.match(dana(1, w=0.58))[0] is None
    faces.forget("Alicia")
    assert faces.load_nots()[0] == []


def test_naming_a_group_someone_takes_back_their_not_for_it(named):
    faces._unk_save([7], dana(0)[None], [1], 8)
    group = faces.group_vector(7)
    assert group is not None and float(group @ dana(0)) == pytest.approx(1.0)
    faces.add_not("Alice", group, "group:7")
    with patch.object(faces, "_asserted", autospec=True, return_value=True):
        faces.name_cluster(7, "Alice")
    assert faces.load_nots()[0] == []


# ---- a reference photo attached to a person teaches their face


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "media.jpg"
    assert cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))
    return path


def faces_found(*vecs):
    return [(np.array([0, 0, 4, 4]), v) for v in vecs]


def test_a_one_face_reference_photo_teaches_the_person_so_a_held_out_look_is_named(named, photo):
    dana_photo, dana_later = blend(3, 200, 1.0), blend(3, 201, 0.7)
    assert faces.match(dana_later)[0] is None
    with patch.object(faces, "embed", autospec=True, return_value=faces_found(dana_photo)):
        added = faces.enroll_reference("Dana", photo)
    assert added["result"] == "added" and added["dropped"] is None
    assert faces.match(dana_later)[0] == "Dana"
    with patch.object(faces, "embed", autospec=True, return_value=faces_found(dana_photo)):
        assert faces.enroll_reference("Dana", photo)["result"] == "same", "the same photo again adds nothing"
    assert faces.counts()["Dana"] == 1
    assert faces.remove_row("Dana", added["vec"], added["dropped"])  # Remove reference
    assert faces.match(dana_later)[0] is None


@pytest.mark.parametrize(
    ("found", "result"),
    [pytest.param((), "none", id="no-face"), pytest.param((unit(3), unit(4)), "several", id="group-photo")],
)
def test_a_reference_photo_without_exactly_one_face_never_trains(named, photo, found, result):
    with patch.object(faces, "embed", autospec=True, return_value=faces_found(*found)):
        assert faces.enroll_reference("Dana", photo) == {"faces": len(found), "result": result}
    assert faces.load_db()[0] == ["Alice", "Bob"]
