"""Yield background model calls to the broker's voice turn, with a bounded fail-open wait.

One connection per process/socket tracks Welcome and broadcasts in the SDK reader without consuming worker
events. Foreground voice, Ask, and Live calls bypass this gate: they may be the work keeping the turn active.
A request already submitted to a model server cannot be recalled by this admission check.
"""

import asyncio
import math
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from liberational import config as cfg
from liberational import continuum_link

if TYPE_CHECKING:
    from continuum_py import Client


class _GateLink(continuum_link.BrokerLink):
    """A dedicated state observer that discards its own unused pushed messages.

    SDK state tracking does not consume events. Drain this connection's event queue continuously so a long-lived
    background process does not retain every past voice transition. Other workers' event queues stay untouched.
    """

    def __init__(self, socket: Path) -> None:
        self._drain_lock = asyncio.Lock()
        self._drain_client: Client | None = None
        self._drain_task: asyncio.Task[None] | None = None
        super().__init__(socket, client_name="background-voice-gate", timeout=0.5)

    async def _stop_drain(self) -> None:
        if self._drain_task is not None:
            self._drain_task.cancel()
            await asyncio.gather(self._drain_task, return_exceptions=True)
            self._drain_task = None
        self._drain_client = None

    async def connected(self) -> Client:
        client = await super().connected()
        async with self._drain_lock:
            if client is not self._drain_client:
                await self._stop_drain()
                self._drain_client = client
                self._drain_task = asyncio.create_task(_discard_events(client))
        return client

    def close(self) -> None:
        async def stop() -> None:
            async with self._drain_lock:
                await self._stop_drain()

        try:
            self.run(stop())
        finally:
            super().close()


async def _discard_events(client: Client) -> None:
    import continuum_py

    try:
        while await client.next_event() is not None:
            pass
    except OSError, continuum_py.ConnectionClosedError, continuum_py.ContinuumError:
        pass  # the next admission read reconnects; this connection has no other event consumer


_LINKS: dict[Path, _GateLink] = {}
_LOCK = threading.Lock()


def _link() -> continuum_link.BrokerLink | None:
    socket = continuum_link.socket_path()
    if not socket.exists():
        return None
    with _LOCK:
        if socket not in _LINKS:
            _LINKS[socket] = _GateLink(socket)
        return _LINKS[socket]


async def _read(client: Client) -> str | None:
    import continuum_py

    turn = client.voice_turn
    if turn is None:
        raise continuum_py.ConnectionClosedError("voice state connection closed")
    return turn


def busy(*, timeout_s: float = 0.5) -> bool:
    """Whether the broker currently reports Listening, Thinking, or Speaking. Unavailable means no hold."""
    import continuum_py

    try:
        link = _link()
        return link is not None and link.call(_read, timeout=timeout_s) in {"Listening", "Thinking", "Speaking"}
    except OSError, TimeoutError, continuum_py.ConnectionClosedError, continuum_py.BrokerError:
        return False


def wait_quiet(max_wait_s: float | None = None) -> None:
    """Wait for Idle, a disconnect, or the finite wait cap. With no broker, return immediately."""
    limit = float(cfg.get("voice.background_max_wait_s") if max_wait_s is None else max_wait_s)
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("voice background wait must be finite and nonnegative")
    deadline = time.monotonic() + limit
    while (remaining := deadline - time.monotonic()) > 0:
        if not busy(timeout_s=min(0.5, remaining)):
            return
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def close_all() -> None:
    """Close shared gate connections, for shutdown and tests that replace the broker."""
    with _LOCK:
        for link in _LINKS.values():
            link.close()
        _LINKS.clear()
