"""Registers ingested files on Continuum, through the calls the v1 importers used (the audio
archive import's, and `pack-import-vision`'s `recordings`): `register_v1_index` for FLAC on `<stream>.audio_v1`, and
`register_v1_video_index` for H.264 MP4 on `<stream>.video_v1`. Each file is registered in place: the broker reads
an `index.jsonl` naming it by absolute path with its start, written to a temporary directory of its own, so nothing
beside the media is written. Both calls are idempotent, so registering a file again returns the same chunk.

Only a registered worker commits chunks: each connection registers as `ingest-<pid>`.
"""

import asyncio
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, Protocol

from continuum_py import BrokerError, Client

from liberational import continuum_link, logs
from liberational.ingest.store import Item

LOG = logs.get("ingest")


class Refused(Exception):
    """The broker didn't register a file, with its reason (a layout the stream doesn't take, a span the catalog
    already holds something else for, or the broker being down)."""


class Registrar(Protocol):
    """Where converted files go: a chunk id per registered file, or Refused."""

    def video(self, stream: str, path: Path, start: float, fps: float) -> str: ...

    def audio(self, stream: str, path: Path, start: float) -> str: ...


def _index_line(path: Path, start: float, **extra: object) -> str:
    return json.dumps({"file": str(path.resolve()), "start": start, **extra}) + "\n"


class BrokerRegistrar:
    """Registers on the broker at `socket` (the pack's own, `continuum_link.socket_path()`, by default). One file at a
    time per process: the dashboard's worker and a poll in the same process share the lock."""

    _lock = threading.Lock()

    def __init__(self, socket: Path | None = None) -> None:
        self.socket = socket

    def _run[T](self, work: Coroutine[Any, Any, T]) -> T:
        with self._lock:
            return asyncio.run(work)

    def purge_item(self, item: Item) -> str | None:
        """Purge this item's registered chunks and their descendants through the broker's existing cascade. What the
        vision workers made of them is segment rows (#1755), which the broker's purge doesn't reach."""

        ids = {str(o["chunk_id"]) for o in item.outputs}
        if not ids:
            return None

        async def go() -> str | None:
            async with await self._client() as client:
                key = hashlib.sha256(f"{item.id}:{','.join(sorted(ids))}".encode()).hexdigest()
                # A response can be lost after the catalog accepted the purge. Recover its receipt instead
                # of issuing an empty second purge and incorrectly declaring the first one's keepers done.
                after = None
                while True:
                    page = await client.list_purges(after=after, limit=256)
                    for purge in page:
                        if purge.get("reason") != "IngestDeleted":
                            continue
                        saved = json.loads(purge.get("detail") or "{}")
                        if isinstance(saved, dict) and saved.get("key") == key:
                            return str(purge["purge_id"])
                    if len(page) < 256:
                        break
                    after = str(page[-1]["purge_id"])
                detail = json.dumps({"source": "ingest", "key": key}, separators=(",", ":"))
                result = await client.purge(ids=sorted(ids), reason="IngestDeleted", detail=detail)
                return str(result["purge_id"])

        return self._run(go())

    def purge_finished(self, purge_id: str) -> bool:
        """Whether every keeper (including vector storage) acknowledged this purge."""

        async def go() -> bool:
            async with await self._client() as client:
                result = await client.get_purge(purge_id)
                return bool(result.get("completed_ns"))

        return self._run(go())

    async def _client(self) -> Client:
        sock = self.socket or continuum_link.socket_path()
        try:
            client = await Client.connect(str(sock), client_name="ingest")
        except (OSError, BrokerError) as e:
            raise Refused(f"the Continuum broker isn't reachable at {sock}: {e}") from e
        try:
            await client.register_worker(f"ingest-{os.getpid()}", [])
        except BrokerError as e:
            if "already registered" not in str(e):
                await client.close()
                raise
        return client

    def video(self, stream: str, path: Path, start: float, fps: float) -> str:
        async def go() -> str:
            async with await self._client() as client:
                with tempfile.TemporaryDirectory(prefix="ingest-video-") as tmp:
                    (Path(tmp) / "index.jsonl").write_text(_index_line(path, start, cam=stream, fps=fps))
                    try:
                        entries = await client.register_v1_video_index(stream, tmp)
                    except BrokerError as e:
                        raise Refused(str(e)) from e
            for e in entries:
                if e.get("chunk_id"):
                    return str(e["chunk_id"])
                raise Refused(str(e.get("skipped") or "not registered"))
            raise Refused("the broker registered nothing")

        return self._run(go())

    def audio(self, stream: str, path: Path, start: float) -> str:
        async def go() -> str:
            async with await self._client() as client:
                with tempfile.TemporaryDirectory(prefix="ingest-audio-") as tmp:
                    (Path(tmp) / "index.jsonl").write_text(_index_line(path, start))
                    try:
                        entries = await client.register_v1_index(f"{stream}.audio_v1", tmp)
                    except BrokerError as e:
                        raise Refused(str(e)) from e
            for e in entries:
                if e.get("registered") and e.get("chunk_id"):
                    return str(e["chunk_id"])
                raise Refused(str(e.get("skipped") or "not registered"))
            raise Refused("the broker registered nothing")

        return self._run(go())
