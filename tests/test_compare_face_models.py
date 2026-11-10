"""The lightweight comparison keeps manually reviewed relations and reports real cosines, without labels as truth."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "compare_face_models", Path(__file__).parents[1] / "tools/compare_face_models.py"
)
assert _spec and _spec.loader
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)


def test_pair_scores_preserve_uncertain_judgments_and_do_not_claim_identity_accuracy():
    a, b, c = np.zeros((3, 512), np.float32)
    a[0] = 5
    b[:2] = [3, 4]
    c[1] = -1
    pairs = [{"a": "a", "b": "b", "relation": "same"}, {"a": "b", "b": "c", "relation": "uncertain"}]
    result = comparison.score_pairs({"a": a, "b": b, "c": c}, pairs)
    assert result[0] == {**pairs[0], "cosine": pytest.approx(0.6)}
    assert result[1] == {**pairs[1], "cosine": pytest.approx(-0.8)}


def test_manifest_resolves_scratch_files_and_rejects_pairs_without_a_visual_judgment(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps({"images": {"a": "a.jpg", "b": "b.jpg"}, "pairs": [{"a": "a", "b": "b", "relation": "uncertain"}]})
    )
    images, pairs = comparison.manifest(path)
    assert images == {"a": tmp_path / "a.jpg", "b": tmp_path / "b.jpg"}
    assert pairs == [{"a": "a", "b": "b", "relation": "uncertain"}]
    path.write_text(
        json.dumps(
            {"images": {"a": "a.jpg", "b": "b.jpg"}, "pairs": [{"a": "a", "b": "b", "relation": "tag says Alice"}]}
        )
    )
    with pytest.raises(ValueError, match="visual judgment"):
        comparison.manifest(path)
