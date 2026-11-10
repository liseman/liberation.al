import asyncio

from liberational import speech_log
from liberational.dashboard import mic_levels

NOW = 1_800_000_000.0


def test_activity_uses_fresh_latest_gate_state_per_microphone(tmp_path):
    store = tmp_path / "segments"
    for device, rows in {
        "mic-a": [(True, NOW - 1)],
        "mic-b": [(False, NOW - 1)],
        "stale": [(False, NOW - 4)],
        "future": [(False, NOW + 1)],
    }.items():
        writer = speech_log.Writer(device, store)
        for state, at in rows:
            writer.write(speech_log.ACTIVITY, {"ts_start": at, "device": device, "in_speech": state})
        writer.close()
    # an older state written later must not replace the newer one
    late = speech_log.Writer("mic-a-late", store)
    late.write(speech_log.ACTIVITY, {"ts_start": NOW - 2, "device": "mic-a", "in_speech": False})
    late.close()
    assert asyncio.run(mic_levels.activity(NOW, store)) == {"mic-a": True, "mic-b": False}


def test_activity_is_unknown_when_the_listen_worker_has_never_published(tmp_path):
    assert asyncio.run(mic_levels.activity(NOW, tmp_path / "segments")) == {}


class _Reader:
    def read_record(self) -> tuple[int, bytes] | None:
        return None  # no new audio this window


class _Sub:
    def __init__(self):
        self.rings = [{"lease": {"ring_stride": mic_levels.STAMP.size + 2}}]
        self.pending = [{"position": 1}, {"position": 2}]  # batch notices the meters never look at
        self.delivered = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.pending:
            await asyncio.Event().wait()  # live: nothing more until the meters stop
        self.delivered += 1
        return self.pending.pop(0)

    async def unsubscribe(self):
        pass


class _Client:
    """A broker connection that records which calls it gets."""

    def __init__(self, name, options):
        self.name, self.options, self.calls, self.subs = name, options, [], []

    async def close(self):
        pass

    async def list_streams(self):
        self.calls.append("list_streams")
        return [{"name": "mic0.audio", "shape_name": "PCM_S16LEx1"}]

    async def subscribe(self, streams):
        self.calls.append("subscribe")
        self.subs.append(_Sub())
        return self.subs[-1]

    def ring_reader(self, lease):
        self.calls.append("ring_reader")
        return _Reader()

    async def query(self, *args, **kwargs):
        self.calls.append("query")
        return {"items": []}


def test_the_audio_subscription_has_a_connection_of_its_own(monkeypatch, tmp_path):
    """The meters never consume their audio subscriptions' events, and a full subscription queue blocks that
    connection's socket reader, so its requests time out. Subscriptions and their rings are on one connection;
    every request is on another (the gate state is read from the segment store, not the broker)."""
    import continuum_py

    made: list[_Client] = []

    async def connect(socket, client_name="python", **kwargs):
        made.append(_Client(client_name, kwargs))
        return made[-1]

    monkeypatch.setattr(continuum_py.Client, "connect", staticmethod(connect))
    sent = []
    asyncio.run(mic_levels.stream(tmp_path / "broker.sock", sent.append, lambda: len(sent) >= 2))
    assert len(sent) == 2 and "error" not in sent[0]
    subscribing = [c for c in made if "subscribe" in c.calls]
    querying = [c for c in made if "list_streams" in c.calls]
    assert len(subscribing) == 1 and len(querying) == 1
    assert subscribing[0] is not querying[0]
    assert set(subscribing[0].calls) == {"subscribe", "ring_reader"}
    # Its subscription events are drained, and its policy holds a backlog rather than closing the connection
    # (the default would end the ring leases the meters read).
    assert subscribing[0].options.get("subscription_overflow") == "backpressure"
    assert querying[0].options.get("subscription_overflow", "close") == "close"
    assert [sub.delivered for sub in subscribing[0].subs] == [2]
