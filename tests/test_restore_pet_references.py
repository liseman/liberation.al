"""tools/restore_pet_references.py: which saved photo a stranded label is restored from, and which of
the detector's animals in it is the track's."""

import importlib.util
import pathlib

import numpy as np
import pytest

from liberational.people import face_shots, pet_shots

_spec = importlib.util.spec_from_file_location(
    "restore_pet_references", pathlib.Path(__file__).resolve().parents[1] / "tools" / "restore_pet_references.py"
)
assert _spec and _spec.loader
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_a_tracks_photo_is_its_newest_revision_and_never_another_tracks(tmp_path):
    folder = face_shots.directory(tmp_path, "run:cat:1", pet_shots.KIND)
    folder.mkdir(parents=True)
    for name in [
        "run:cat:1-{v}-r2.jpg",
        "run:cat:1-{v}-r10.jpg",
        "run:cat:1-{v}-grid80-r11.jpg",
        "run:cat:12-{v}-r12.jpg",
    ]:
        (folder / name.format(v=face_shots.VERSION)).write_bytes(b"jpg")
    assert tool.photo(tmp_path, "run:cat:1").name == f"run:cat:1-{face_shots.VERSION}-r10.jpg"
    assert tool.photo(tmp_path, "run:cat:2") is None


@pytest.mark.parametrize(
    ("animals", "expected"),
    [
        pytest.param(
            [[0.0, 0.0, 0.2, 0.2, 0.99, 15], [0.1, 0.3, 0.9, 0.7, 0.5, 15]], [0.1, 0.3, 0.9, 0.7], id="central"
        ),
        pytest.param([], None, id="none-found"),
    ],
)
def test_the_photos_own_animal_is_the_one_nearest_its_centre(animals, expected):
    found = tool.central(np.array(animals, np.float32).reshape(-1, 6))
    assert found is None if expected is None else np.allclose(found, expected)
