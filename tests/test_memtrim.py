"""Returning freed memory to the system (memtrim.py, #2850)."""

import os
import threading
from unittest.mock import create_autospec

import pytest

from liberational import memtrim, segment_log
from liberational.dashboard import presence


@pytest.fixture
def calls(monkeypatch):
    """The sizes malloc_trim is called with, in place of glibc's, with no process trimming yet."""
    got = []
    monkeypatch.setattr(memtrim, "_malloc_trim", lambda: lambda pad: got.append(pad) or 1)
    monkeypatch.setattr(memtrim, "_started", set())
    return got


def test_a_trim_asks_glibc_to_return_everything_it_can(calls):
    assert memtrim.trim() is True
    assert calls == [0]


def test_without_glibc_nothing_is_trimmed_or_started(monkeypatch):
    monkeypatch.setattr(memtrim, "_malloc_trim", lambda: None)
    monkeypatch.setattr(memtrim, "_started", set())
    assert memtrim.trim() is False
    assert memtrim.start() is False
    assert memtrim._started == set()


def test_a_process_starts_one_trim_thread_however_often_asked(calls):
    before = sum(t.name == "malloc-trim" for t in threading.enumerate())
    assert memtrim.start(every_s=3600) is True  # an hour: the thread never trims during the test
    assert memtrim.start(every_s=3600) is False
    assert memtrim._started == {os.getpid()}
    assert sum(t.name == "malloc-trim" for t in threading.enumerate()) == before + 1


def test_reviews_day_history_worker_trims_and_keeps_its_hours(monkeypatch):
    start = create_autospec(memtrim.start)
    keep = create_autospec(segment_log.keep_hours)
    monkeypatch.setattr(presence.multiprocessing, "parent_process", lambda: object())  # as in its worker process
    monkeypatch.setattr(presence.cfg.load, "cache_clear", lambda: None)
    monkeypatch.setattr(memtrim, "start", start)
    monkeypatch.setattr(segment_log, "keep_hours", keep)
    monkeypatch.setattr(presence, "presence", lambda day, state: {"chips": []})
    presence._load("2026-11-07", "/nowhere")
    start.assert_called_once_with()
    keep.assert_called_once_with()


def test_a_day_load_in_the_dashboards_own_process_starts_no_trim(monkeypatch):
    start = create_autospec(memtrim.start)
    monkeypatch.setattr(presence.multiprocessing, "parent_process", lambda: None)  # tests and other importers
    monkeypatch.setattr(memtrim, "start", start)
    monkeypatch.setattr(presence, "presence", lambda day, state: {"chips": []})
    presence._load("2026-11-07", "/nowhere")
    start.assert_not_called()
