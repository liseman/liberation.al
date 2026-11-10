"""The face files (#411): faces/db.npz and unknown_clusters.npz are written whole and moved into place, and every
read-modify-write holds a lock across processes, so a torn file, a crash mid-write, or two writers at once (vision,
identity, the dashboard) never lose a face. Temp files and made-up faces only; never the real data/."""

import json
import os
import subprocess
import sys
import threading
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from liberational.vision import faces

E = np.eye(512, dtype=np.float32)


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(faces, "DB", tmp_path / "db.npz")
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown_clusters.npz")
    return tmp_path


def test_a_save_that_dies_midway_leaves_the_old_file_whole(files, monkeypatch):
    faces.save_db(["Kara"], E[:1])
    real = np.savez

    def dies(f, **arrays):
        real(f, **arrays)
        f.flush()
        f.truncate(40)  # half written, as a crash or a full disk leaves it
        raise OSError("disk full")

    monkeypatch.setattr(faces.np, "savez", dies)
    with pytest.raises(OSError, match="disk full"):
        faces.save_db(["Kara", "Bob"], E[:2])
    monkeypatch.setattr(faces.np, "savez", real)
    assert faces.load_db()[0] == ["Kara"]
    assert [p.name for p in files.iterdir() if p.suffix == ".npz"] == ["db.npz"]  # no temp file left behind


ADD_ROWS = """
import pathlib, sys, time, numpy as np
from liberational.vision import faces
faces.DB = pathlib.Path(sys.argv[1])
go = pathlib.Path(sys.argv[4])
while not go.exists():  # both start writing at the same moment
    time.sleep(0.001)
for k in range(int(sys.argv[3])):
    faces.add_row(sys.argv[2] + str(k), np.eye(512, dtype=np.float32)[k % 512])
"""


def test_two_processes_adding_faces_at_once_lose_none(files):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    go, n = files / "go", 150
    procs = [
        subprocess.Popen([sys.executable, "-c", ADD_ROWS, str(faces.DB), who, str(n), str(go)], env=env)
        for who in ("a", "b")
    ]
    go.touch()
    assert [p.wait(120) for p in procs] == [0, 0]
    names = faces.load_db()[0]
    assert sorted(names) == sorted([f"{w}{k}" for w in "ab" for k in range(n)])  # without the lock, rows vanish


def test_threads_grouping_unknown_faces_lose_none(files):
    def see(base):
        for k in range(20):
            faces.group_unknown(E[base + k])

    workers = [threading.Thread(target=see, args=(b,)) for b in (0, 100)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(60)
    assert len(faces.unknown_groups()) == 40


def test_a_write_inside_another_takes_the_lock_once(files):
    faces._unk_save([3], E[3:4], [2], 4)

    def nested():
        with faces.editing(), faces.editing_unknown():  # a caller holding both, then name_cluster takes them again
            faces.name_cluster(3, "Dana")

    t = threading.Thread(target=nested, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive()  # it would hang here if the lock weren't reentrant
    assert faces.load_db()[0] == ["Dana"]
    assert faces.unknown_groups() == []


def test_a_briefly_broken_file_is_read_again(files, monkeypatch):
    faces.save_db(["Kara"], E[:1])
    real, calls = np.load, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise zipfile.BadZipFile("Bad CRC-32 for file 'names.npy'")  # as the dashboard read one mid-write
        return real(*a, **k)

    monkeypatch.setattr(faces.np, "load", flaky)
    monkeypatch.setattr(faces, "READ_WAIT_S", 0.0)
    assert faces.load_db()[0] == ["Kara"]
    assert len(calls) == 2


def test_a_file_that_stays_broken_raises(files, monkeypatch):
    faces.UNK_DB.write_bytes(b"PK\x03\x04 not a whole zip")
    monkeypatch.setattr(faces, "READ_WAIT_S", 0.0)
    with pytest.raises(zipfile.BadZipFile):
        faces.unknown_groups()


def test_an_added_duplicate_view_does_not_evict_the_only_distinct_enrolled_view(files):
    faces.save_db(["Alice"] * faces.MAX_PER_NAME, np.vstack([E[0], np.tile(E[1], (faces.MAX_PER_NAME - 1, 1))]))
    dropped = faces.add_row("Alice", E[1])
    assert faces.match(E[0]) == ("Alice", pytest.approx(1.0))
    assert dropped == pytest.approx(E[1].tolist())
    names, vectors = faces.load_db()
    assert len(names) == len(vectors) == faces.MAX_PER_NAME


@pytest.fixture
def recovery(files, monkeypatch):
    from liberational.domain.identity import Basis

    faces.save_db(["Alice", "Dana"], E[:2])
    active = {1}
    reg = SimpleNamespace(
        person_of=lambda ref: SimpleNamespace(
            person="alice" if int(ref.id.split(":")[-1]) in active else None,
            standing=[SimpleNamespace(basis=Basis.OWNER_TAG)],
        ),
        person=lambda pid: SimpleNamespace(display_name="Alice"),
        person_id=lambda name: "alice" if name == "Alice" else None,
    )
    monkeypatch.setattr(faces.registry, "at", lambda state: reg)

    def write(op, *, undone=False):
        (files / "people_undo.json").write_text(json.dumps([{"ops": [op], "undone": undone}]))

    op = {
        "op": "face_group",
        "gid": 1,
        "name": "Alice",
        "vec": E[2].tolist(),
        "model": faces.recognition_model(),
    }
    return write, op, active


def test_naming_a_duplicate_group_keeps_the_distinct_enrollment(files):
    faces.save_db(["Alice"] * faces.MAX_PER_NAME, np.vstack([E[0], np.tile(E[1], (faces.MAX_PER_NAME - 1, 1))]))
    faces._unk_save([3], E[1:2], [2], 4)
    enrolled = {}
    faces._name_cluster_files(3, "Alice", enrolled)
    assert faces.match(E[0]) == ("Alice", pytest.approx(1.0))
    assert enrolled["model"] == faces.recognition_model()
    assert faces.unknown_groups() == []


def test_merging_face_banks_retains_a_unique_old_view(files):
    faces.save_db(
        ["Alice"] * faces.MAX_PER_NAME + ["Dana"] * faces.MAX_PER_NAME,
        np.vstack([E[0], np.tile(E[1], (faces.MAX_PER_NAME - 1, 1)), np.tile(E[2], (faces.MAX_PER_NAME, 1))]),
    )
    assert faces.rename("Alice", "Dana") == faces.MAX_PER_NAME
    assert faces.match(E[0]) == ("Dana", pytest.approx(1.0))
    assert faces.counts() == {"Dana": faces.MAX_PER_NAME}


def near(sim: float, axis: int, toward: int = 0) -> np.ndarray:
    """A unit face whose cosine with E[toward] is `sim`, leaning along E[axis]."""
    return (sim * E[toward] + np.sqrt(1 - sim * sim) * E[axis]).astype(np.float32)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param([("Alice", 0.5), ("Bob", 0.3)], "Alice", id="clear-leader"),
        pytest.param([("Alice", 0.5), ("Bob", 0.45)], None, id="too-close-to-a-second-person"),
        pytest.param([("Alice", 0.5), ("Alice", 0.48)], "Alice", id="own-rows-are-no-rival"),
        pytest.param([("Alice", 0.38)], None, id="below-the-gate"),
        pytest.param([("Alice", 0.5)], "Alice", id="adaface-centroid-level"),
    ],
)
def test_a_face_is_named_only_when_one_person_clearly_leads(files, rows, expected):
    faces.save_db([n for n, _ in rows], np.stack([near(s, k + 1) for k, (_, s) in enumerate(rows)]))
    assert faces.match(E[0])[0] == expected


def test_regroup_merges_split_groups_into_the_oldest_and_keeps_put_aside_ones(files):
    # Face 3 is the anchor; 5 and 4 look like it, 9 is someone else. 4 was put aside, so it stays out of it.
    faces._unk_save([5, 3, 9, 4], np.stack([near(0.6, 1), E[0], E[2], near(0.7, 3)]), [2, 4, 3, 6], 10)
    merges = faces.regroup_unknown(keep=[4])
    assert merges == [(5, 3, pytest.approx(0.6))]
    assert faces.unknown_groups() == [{"id": 3, "count": 6}, {"id": 4, "count": 6}, {"id": 9, "count": 3}]
    assert faces._unk_load()[3] == 10  # numbers already handed out are never reused


def test_regroup_joins_anchors_only_so_merges_never_chain(files):
    # 6 is too far from 3 to join it, so it is its own anchor; 8 is close to 6 but not to 3.
    b = near(0.3, 1)
    c = (0.6 * b + 0.8 * E[2]).astype(np.float32)
    faces._unk_save([3, 6, 8], np.stack([E[0], b, c]), [1, 1, 1], 9)
    assert [(m[0], m[1]) for m in faces.regroup_unknown()] == [(8, 6)]


def test_a_regroup_dry_run_changes_nothing(files):
    faces._unk_save([3, 5], np.stack([E[0], near(0.6, 1)]), [1, 1], 6)
    assert [(m[0], m[1]) for m in faces.regroup_unknown(apply=False)] == [(5, 3)]
    assert faces.unknown_groups() == [{"id": 3, "count": 1}, {"id": 5, "count": 1}]


def tilt(a: float) -> np.ndarray:
    """A unit face at cosine `a` with E[0], leaning along E[1]."""
    return near(a, 1)


@pytest.fixture
def three(files):
    """Face 1: its first face along E[0], then two more at 0.5 toward E[1], so its centroid leans to E[1]."""
    gid = faces.group_unknown(E[0])
    assert [faces.group_unknown(tilt(0.5)) for _ in range(2)] == [gid, gid]
    return gid


def test_a_face_joins_by_the_group_centroid_once_it_has_three_members(three):
    # 0.30 like the first face (under the join) but 0.85 like the members' centroid.
    assert faces.group_unknown(tilt(0.3)) == three


def test_the_first_face_still_guards_the_centroid(three):
    # 0.65 like the centroid, but nothing like the first face: it starts a group of its own.
    assert faces.group_unknown(E[1]) != three


def test_a_tag_enrolls_the_group_centroid(three):
    faces.name_cluster(three, "Dana")
    names, rows = faces.load_db()
    centroid = (E[0] + 2 * tilt(0.5)) / np.linalg.norm(E[0] + 2 * tilt(0.5))
    assert names == ["Dana"]
    assert float(rows[0] @ centroid) == pytest.approx(1.0, abs=1e-5)


def test_a_writer_that_does_not_track_members_keeps_them(three):
    ids, vecs, counts, nxt = faces._unk_load()
    faces._unk_save(ids, vecs, [c + 1 for c in counts], nxt)  # as the dashboard's count edits write
    assert faces.group_unknown(tilt(0.3)) == three
