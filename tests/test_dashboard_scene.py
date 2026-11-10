"""Live's captions (live_status.scene_entries): one read of the vision log serves every open Live feed for SCENE_POLL_S."""

import time

import pytest

from liberational.dashboard import live_status


@pytest.fixture
def reads(vision, monkeypatch):
    """Counts the vision log's scene reads, and lets a test move the dashboard's clock."""
    calls = []
    plain = vision.scene

    def scene(now, *args, **kwargs):
        calls.append(now)
        return plain(now, *args, **kwargs)

    monkeypatch.setattr(vision, "scene", scene)
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    vision.caption(time.time() - 5, caption="a person at the desk")
    return calls, clock


def test_feeds_asking_within_the_poll_share_one_read(reads):
    calls, clock = reads
    first = live_status.scene_entries()
    assert [e["caption"] for e in first] == ["a person at the desk"]
    clock[0] += live_status.SCENE_POLL_S - 0.1
    assert live_status.scene_entries() == first
    assert len(calls) == 1


def test_the_next_poll_reads_again_and_sees_new_captions(reads, vision):
    calls, clock = reads
    live_status.scene_entries()
    vision.caption(time.time() - 1, caption="nobody")
    clock[0] += live_status.SCENE_POLL_S
    assert [e["caption"] for e in live_status.scene_entries()] == ["a person at the desk", "nobody"]
    assert len(calls) == 2


def test_a_failed_read_is_not_kept(reads, vision, monkeypatch):
    calls, _ = reads

    def down(now, *args, **kwargs):
        raise OSError("broker down")

    monkeypatch.setattr(vision, "scene", down)
    assert live_status.scene_entries() == []
    monkeypatch.setattr(vision, "scene", lambda now, *a, **k: calls.append(now) or [{"id": "x", "caption": "back"}])
    assert live_status.scene_entries() == [{"id": "x", "caption": "back"}]
