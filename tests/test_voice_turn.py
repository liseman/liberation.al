"""Bounded background admission against the broker's tracked voice turn."""

import asyncio
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from liberational import config as cfg
from liberational import continuum_link, voice_turn

continuum_py = pytest.importorskip("continuum_py")


@pytest.mark.parametrize("turn", ["Listening", "Thinking", "Speaking"])
def test_busy_reads_tracked_active_turn_without_consuming_events(monkeypatch, turn):
    link = create_autospec(continuum_link.BrokerLink, instance=True)
    link.call.side_effect = lambda fn, **kwargs: turn
    monkeypatch.setattr(voice_turn, "_link", lambda: link)
    assert voice_turn.busy()


def test_absent_broker_does_not_sleep(monkeypatch):
    monkeypatch.setattr(voice_turn, "_link", lambda: None)
    sleep = create_autospec(voice_turn.time.sleep)
    monkeypatch.setattr(voice_turn.time, "sleep", sleep)
    voice_turn.wait_quiet()
    sleep.assert_not_called()


@pytest.mark.parametrize("turn", ["Idle", None])
def test_idle_or_disconnected_does_not_hold(monkeypatch, turn):
    link = create_autospec(continuum_link.BrokerLink, instance=True)
    link.call.return_value = turn
    monkeypatch.setattr(voice_turn, "_link", lambda: link)
    assert not voice_turn.busy()


def test_broker_error_does_not_hold(monkeypatch):
    link = create_autospec(continuum_link.BrokerLink, instance=True)
    link.call.side_effect = continuum_py.ConnectionClosedError("closed")
    monkeypatch.setattr(voice_turn, "_link", lambda: link)
    assert not voice_turn.busy()


def test_wait_stops_as_soon_as_turn_finishes(monkeypatch):
    turns = iter([True, True, False])
    monkeypatch.setattr(voice_turn, "busy", lambda **kwargs: next(turns))
    sleeps = []
    monkeypatch.setattr(voice_turn.time, "sleep", sleeps.append)
    voice_turn.wait_quiet(1)
    assert sleeps == [0.1, 0.1]


def test_dead_voice_reaches_configured_cap(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(voice_turn, "busy", lambda **kwargs: True)
    monkeypatch.setattr(voice_turn.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(voice_turn.time, "sleep", sleep)
    with cfg.override({"voice": {"background_max_wait_s": 0.25}}):
        voice_turn.wait_quiet()
    assert clock.now == pytest.approx(0.25)
    assert sleeps == pytest.approx([0.1, 0.1, 0.05])


@pytest.mark.parametrize("limit", [-1, float("inf"), float("nan")])
def test_unbounded_or_negative_wait_is_rejected(limit):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        voice_turn.wait_quiet(limit)


class EventClient:
    voice_turn = "Idle"

    def __init__(self):
        self.events = asyncio.Queue()
        self.waiting = asyncio.Event()
        self.cancelled = False

    async def next_event(self):
        if self.events.empty():
            self.waiting.set()
        try:
            return await self.events.get()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def test_gate_discards_only_its_events_and_cancels_old_reader_on_reconnect(monkeypatch, tmp_path):
    link = voice_turn._GateLink(tmp_path / "unused.sock")
    current = SimpleNamespace(client=None)

    async def connected(self):
        return current.client

    monkeypatch.setattr(continuum_link.BrokerLink, "connected", connected)

    async def check():
        first, second = EventClient(), EventClient()
        first.events.put_nowait(("SystemStateChanged", b"Thinking"))
        first.events.put_nowait(("SystemStateChanged", b"Idle"))
        current.client = first
        assert await link.connected() is first
        await asyncio.wait_for(first.waiting.wait(), 1)
        assert first.events.empty()
        assert await link.connected() is first
        assert not first.cancelled
        current.client = second
        second.events.put_nowait(("SystemStateChanged", b"Speaking"))
        assert await link.connected() is second
        await asyncio.wait_for(second.waiting.wait(), 1)
        assert first.cancelled
        assert second.events.empty()
        return second

    try:
        second = link.run(check())
    finally:
        link.close()
    assert second.cancelled


def test_connection_read_cannot_outlast_short_wait_cap(monkeypatch):
    link = create_autospec(continuum_link.BrokerLink, instance=True)
    link.call.return_value = "Idle"
    monkeypatch.setattr(voice_turn, "_link", lambda: link)
    voice_turn.wait_quiet(0.01)
    assert 0 < link.call.call_args.kwargs["timeout"] <= 0.01
