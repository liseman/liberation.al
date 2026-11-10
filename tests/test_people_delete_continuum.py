import asyncio
import uuid
from pathlib import Path

import continuum_py
import pytest

from liberational.dashboard import people_writes


async def note(client: continuum_py.Client, stream_id: int, about: str, at_ns: int) -> str:
    pid = str(uuid.uuid7())
    header = continuum_py.encode_header(
        {
            "id": pid,
            "stream_id": stream_id,
            "start_ns": at_ns,
            "end_ns": at_ns,
            "layer": "Layer2Decision",
            "format": "Text",
            "shape_name": "notes",
            "parents": [],
            "subjects": [about],
        }
    )
    client.ingest(header, inline=b"{}")
    return pid


def test_their_payloads_are_purged_from_continuum_and_others_stay(
    broker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dana, alice = str(uuid.uuid7()), str(uuid.uuid7())

    async def setup() -> int:
        async with await continuum_py.Client.connect(str(broker), client_name="t") as c:
            sid = (await c.register_stream("notes", kind="derived", format="Text", shape_name="notes", slab_class=0))[
                "stream_id"
            ]
            await note(c, sid, dana, 1_000)
            await note(c, sid, alice, 2_000)
            for _ in range(100):
                if len((await c.query("notes", 0, 10_000))["items"]) == 2:
                    break
                await asyncio.sleep(0.02)
            return sid

    async def left() -> list[list[str]]:
        async with await continuum_py.Client.connect(str(broker), client_name="t") as c:
            items = (await c.query("notes", 0, 10_000))["items"]
            return [continuum_py.decode_header(it["header"])["subjects"] for it in items]

    asyncio.run(setup())
    monkeypatch.setenv("CONTINUUM_SOCKET", str(broker))
    assert people_writes._purge_continuum({dana}) == (1, 0)
    assert asyncio.run(left()) == [[alice]]


def test_with_no_broker_running_there_is_nothing_to_purge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTINUUM_SOCKET", str(tmp_path / "missing.sock"))
    assert people_writes._purge_continuum({str(uuid.uuid7())}) == (0, 0)


def test_a_purge_that_fails_is_counted_and_the_rest_still_go(broker: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTINUUM_SOCKET", str(broker))
    # Not a UUID, so the broker refuses it; the valid one is still asked.
    assert people_writes._purge_continuum({"not-an-id", str(uuid.uuid7())}) == (0, 1)
