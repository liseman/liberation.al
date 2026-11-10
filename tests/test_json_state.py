"""The shared JSON state file helpers (liberational/json_state.py)."""

import json
import os
import stat

import pytest
from hypothesis import given
from hypothesis import strategies as st

from liberational import json_state

JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=10,
)


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


@given(data=JSON, indent=st.none() | st.just(1), sort_keys=st.booleans(), ensure_ascii=st.booleans())
def test_what_is_written_is_read_back(tmp_path_factory, data, indent, sort_keys, ensure_ascii):
    path = tmp_path_factory.mktemp("state") / "a.json"
    json_state.write(path, data, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    assert json_state.read(path, "missing") == data


@pytest.mark.parametrize("content", [None, "", "{half", b"\xff\xfe"], ids=["missing", "empty", "cut-off", "not-text"])
def test_a_missing_or_broken_file_reads_as_the_default(tmp_path, content):
    path = tmp_path / "a.json"
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        path.write_text(content)
    assert json_state.read(path, {"fresh": True}) == {"fresh": True}
    assert json_state.read(path) is None


def test_a_write_makes_the_directory_and_leaves_only_the_file(tmp_path):
    path = tmp_path / "new" / "deeper" / "a.json"
    json_state.write(path, {"n": 1})
    json_state.write(path, {"n": 2})
    assert json.loads(path.read_text()) == {"n": 2}
    assert [p.name for p in path.parent.iterdir()] == ["a.json"]


@pytest.mark.parametrize(("private", "expected"), [(False, 0o644), (True, 0o600)])
def test_a_file_is_owner_only_when_private_and_readable_otherwise(tmp_path, private, expected):
    path = tmp_path / "a.json"
    old = os.umask(0o077)  # a strict umask must not hide a public file, nor a loose one open a private file
    try:
        json_state.write(path, [1], private=private)
    finally:
        os.umask(old)
    assert mode(path) == expected


def test_data_json_cant_hold_is_refused_before_the_old_file_is_touched(tmp_path):
    path = tmp_path / "a.json"
    json_state.write(path, {"kept": True})
    with pytest.raises(TypeError):
        json_state.write(path, {"bad": object()})
    assert json_state.read(path) == {"kept": True}
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_a_failed_replace_keeps_the_old_file_and_removes_the_temporary_one(tmp_path, monkeypatch):
    path = tmp_path / "a.json"
    json_state.write(path, {"kept": True})

    def full(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", full)
    with pytest.raises(OSError, match="No space left"):
        json_state.write(path, {"new": True})
    assert json_state.read(path) == {"kept": True}
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_a_durable_write_reaches_the_disk_before_it_replaces_the_file(tmp_path, monkeypatch):
    path = tmp_path / "a.json"
    events = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: events.append("fsync") or real_fsync(fd))
    monkeypatch.setattr(os, "replace", lambda src, dst: events.append("replace") or real_replace(src, dst))

    json_state.write(path, {"n": 1})
    assert events == ["replace"]

    events.clear()
    json_state.write(path, {"n": 2}, durable=True)
    assert events == ["fsync", "replace"]


def test_write_text_replaces_a_file_that_isnt_one_json_value(tmp_path):
    path = tmp_path / "lines.jsonl"
    json_state.write_text(path, '{"a": 1}\n{"a": 2}\n')
    assert path.read_text().splitlines() == ['{"a": 1}', '{"a": 2}']
