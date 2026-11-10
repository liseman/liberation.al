"""Draft export never scores questions or overwrites confirmed/manual labels."""

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from liberational import config as cfg
from liberational import memory_log
from liberational.memory import ask_history

_spec = importlib.util.spec_from_file_location(
    "export_ask_history", Path(__file__).resolve().parents[1] / "tools/export_ask_history.py"
)
assert _spec is not None and _spec.loader is not None
export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export)


@pytest.fixture
def labels(tmp_path):
    with cfg.override({"paths": {"measurement": str(tmp_path / "measurement")}}):
        yield tmp_path / "measurement"


def test_stdout_exports_draft_canonical_moments_and_reports_skips_without_writing_files(
    memory, tmp_path, monkeypatch, capsys
):
    history = ask_history.History(tmp_path)
    evidence = memory.append(memory_log.JOURNAL, {"ts_start": 1000.0, "level": 30, "text": "A made-up event"})
    exchange = history.start("What happened?", surface="review", day="2023-11-14", scope="history", created_at=1100.0)
    history.finish(
        exchange,
        {
            "answer": "An event [J1].",
            "evidence": [{"label": "J1", "ts": 1000.0, "source": "journal", "interpretation": evidence}],
        },
        started_at=1105.0,
        finished_at=1110.0,
        status="completed",
    )
    history.start("Not scorable", surface="review", day="2023-11-14", scope="history", created_at=1200.0)
    monkeypatch.setattr(export.ask_history, "at", lambda state=None: history)
    assert export.main(["--state-dir", str(tmp_path), "--surface", "review", "--day", "2023-11-14"]) == 0
    captured = capsys.readouterr()
    row = json.loads(captured.out)
    assert row["id"] == exchange
    assert row["draft"] is True
    assert row["asked_at"] == 1105.0
    assert row["evidence"] == [{"at": 1000.0, "within_s": 1, "source": "journal"}]
    assert "skipped 1" in captured.err
    assert list(tmp_path.iterdir()) == []


def test_output_preserves_confirmed_and_manual_rows_updates_matching_drafts_and_appends_new(labels):
    labels.mkdir()
    target = labels / "retrieval.jsonl"
    old = [
        {"id": "confirmed", "question": "Sam checked", "draft": False, "answer_includes": ["yes"]},
        {"id": "manual", "question": "Manual label"},
        {"id": "draft", "question": "Old draft", "draft": True},
    ]
    target.write_text("".join(json.dumps(row) + "\n" for row in old))
    drafts = [
        {"id": "confirmed", "question": "Changed", "draft": True},
        {"id": "manual", "question": "Changed", "draft": True},
        {"id": "draft", "question": "Updated draft", "draft": True},
        {"id": "new", "question": "New draft", "draft": True},
    ]
    assert export.write_drafts(target, drafts) == (1, 1, 2)
    assert [json.loads(line) for line in target.read_text().splitlines()] == [old[0], old[1], drafts[2], drafts[3]]


def test_invalid_existing_jsonl_is_preserved_without_creating_temporary_output(labels):
    labels.mkdir()
    target = labels / "retrieval.jsonl"
    original = '{"id":"manual"}\nnot json\n'
    target.write_text(original)
    with pytest.raises(ValueError, match="invalid JSON on line 2"):
        export.write_drafts(target, [{"id": "new", "draft": True}])
    assert target.read_text() == original
    assert list(labels.iterdir()) == [target]


def test_failed_atomic_replace_keeps_existing_labels_and_removes_temporary_file(labels, monkeypatch):
    labels.mkdir()
    target = labels / "retrieval.jsonl"
    original = '{"id":"manual","question":"Keep"}\n'
    target.write_text(original)
    replace = create_autospec(export.os.replace, side_effect=OSError("disk failed"))
    monkeypatch.setattr(export.os, "replace", replace)
    with pytest.raises(OSError, match="disk failed"):
        export.write_drafts(target, [{"id": "new", "draft": True}])
    assert target.read_text() == original
    assert list(labels.iterdir()) == [target]


@pytest.mark.parametrize("relative", ["../outside.jsonl", "voice_tags.json", "nested/../../outside.jsonl"])
def test_output_must_be_jsonl_inside_the_configured_measurement_runtime_directory(labels, relative):
    with pytest.raises(ValueError, match="paths.measurement"):
        export.output_path(labels / relative)


def test_symlink_escape_is_rejected(labels, tmp_path):
    labels.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (labels / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="paths.measurement"):
        export.output_path(labels / "link" / "retrieval.jsonl")


def test_source_directory_remains_forbidden_even_if_configured_as_measurement(monkeypatch):
    with (
        cfg.override({"paths": {"measurement": str(cfg.ROOT / "src")}}),
        pytest.raises(ValueError, match="source directories"),
    ):
        export.output_path(cfg.ROOT / "src" / "retrieval.jsonl")


def test_tracked_file_is_refused_before_read_or_replace(monkeypatch):
    run = create_autospec(
        subprocess.run, return_value=subprocess.CompletedProcess([], 0, "data/measurement/retrieval.jsonl", "")
    )
    monkeypatch.setattr(export.subprocess, "run", run)
    with (
        cfg.override({"paths": {"measurement": str(cfg.ROOT / "data" / "measurement")}}),
        pytest.raises(ValueError, match="tracked file"),
    ):
        export.output_path(cfg.ROOT / "data" / "measurement" / "retrieval.jsonl")


def test_cli_reports_storage_failure_and_does_not_claim_an_export(monkeypatch, capsys):
    at = create_autospec(ask_history.at, side_effect=ConnectionError("broker offline"))
    monkeypatch.setattr(export.ask_history, "at", at)
    assert export.main(["--stdout"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "broker offline" in captured.err
