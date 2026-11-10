"""Source-only artifact safeguards; no checkpoint download, network, model construction or export."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("export_adaface", Path(__file__).parents[1] / "tools/export_adaface.py")
assert _spec and _spec.loader
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)


def test_export_refuses_unverified_artifacts_before_importing_model_code(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("raise AssertionError('unverified code must never execute')")
    with pytest.raises(ValueError, match="checksum"):
        exporter.export(source, tmp_path / "missing-weights.pt", tmp_path / "output.onnx")
    assert not (tmp_path / "output.onnx").exists()


def test_checked_accepts_the_requested_digest_and_rejects_a_changed_file(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"source")
    expected = hashlib.sha256(b"source").hexdigest()
    exporter.checked(path, expected)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        exporter.checked(path, expected)


def test_only_official_network_prefix_is_removed_without_dropping_unknown_weights():
    assert exporter.network_state({"net.input_layer.0.weight": 1}) == {"input_layer.0.weight": 1}
    with pytest.raises(ValueError, match="net-prefixed"):
        exporter.network_state({"model.input_layer.0.weight": 1})
    with pytest.raises(ValueError, match="net-prefixed"):
        exporter.network_state({"net.input_layer.0.weight": 1, "classifier.weight": 2})
