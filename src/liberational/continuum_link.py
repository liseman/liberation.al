"""A broker connection for synchronous code: one Continuum client on a background event loop, shared by every
call, made on first use and made again after the broker restarts.

The ported readers (`vision_log`, and `speech_log` in time) wrap their async reads in it, so the dashboard and the
journal, which are threaded, call the broker the way they used to read a file.

It also says when the broker is away: a restart is expected, so every service reports it the same way,
one WARN when the broker goes (`OUTAGE.lost`) and one INFO when it's back (`OUTAGE.back`), instead of a traceback per
failed call. `broker_outage` tells those errors from others, `failed` logs a caught error (its traceback only when it
is not an outage), and `run_service` runs a service's main coroutine so an outage that ends it is one line and an exit
code the service manager restarts it on (EXIT_OUTAGE), not a crash dump. `connected` is a service's client for that
coroutine: it closes the client at the end without a traceback when the broker went first.
"""

import asyncio
import contextlib
import contextvars
import logging
import os
import sys
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import continuum_py
from continuum_py import Client

from liberational import config as cfg
from liberational import logs

LOG = logs.get("continuum-link")


def socket_path() -> Path:
    """The broker's socket: `CONTINUUM_SOCKET`, or `continuum/broker.sock` under the pack's state directory."""
    env = os.environ.get("CONTINUUM_SOCKET")
    return Path(env) if env else cfg.path("paths.state") / "continuum" / "broker.sock"


#: The exit code of a service the broker's going away stopped (EX_TEMPFAIL): try again later.
EXIT_OUTAGE = 75
#: While the broker stays away, a reminder this often, so a quiet log never hides a service stuck waiting.
REMIND_S = 300.0


def broker_outage(error: BaseException | None) -> bool:
    """Whether an error (or one it was raised from) is the broker being away, as during a restart: its socket missing
    or refusing connections, the connection closed, or no reply in time. A missing file that names a path, or any
    other timeout, is not one."""
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, continuum_py.ConnectionClosedError):
            return True
        # The SDK's socket errors come from Rust: "No such file or directory (os error 2)", with no filename.
        socket_error = FileNotFoundError | ConnectionRefusedError | ConnectionResetError | BrokenPipeError
        if isinstance(error, socket_error) and error.filename is None and "os error" in str(error):
            return True
        if isinstance(error, TimeoutError) and "no reply from the broker" in str(error):
            return True
        error = error.__cause__ or error.__context__
    return False


class Outage:
    """Whether this process has lost the broker: one WARN when it goes, one INFO when it's back, whichever call
    notices first, and a reminder every REMIND_S while it stays away (thread-safe)."""

    def __init__(self, log: logging.Logger) -> None:
        self.log = log
        self._lock = threading.Lock()
        self._since: float | None = None
        self._reminded = 0.0

    @property
    def down(self) -> bool:
        return self._since is not None

    def lost(self, error: BaseException) -> None:
        now = time.monotonic()
        with self._lock:
            if self._since is None:
                self._since = self._reminded = now
                first = True
            elif now - self._reminded >= REMIND_S:
                self._reminded, first = now, False
            else:
                return
            away = now - self._since
        if first:
            self.log.warning("lost the broker (%s: %s); waiting for it", type(error).__name__, error)
        else:
            self.log.warning("still no broker after %.0f s (%s: %s)", away, type(error).__name__, error)

    def back(self) -> None:
        if self._since is None:  # the common case, without the lock
            return
        with self._lock:
            if self._since is None:
                return
            away, self._since = time.monotonic() - self._since, None
        self.log.info("the broker is back after %.0f s", away)


OUTAGE = Outage(LOG)


def failed(log: logging.Logger, message: str, error: BaseException, level: int = logging.WARNING) -> None:
    """A caught error: `OUTAGE.lost` when it's the broker being away (one WARN per outage, no traceback), else
    `message` at `level` with its traceback."""
    if broker_outage(error):
        OUTAGE.lost(error)
    else:
        log.log(level, message, exc_info=error)


def exit_on_outage(main: Callable[[], None]) -> None:
    """`main()` for a service: an outage that ends it is one WARN and SystemExit(EXIT_OUTAGE), which the service
    manager restarts it on, rather than a traceback. Any other error is raised as it is."""
    try:
        main()
    except Exception as error:
        if not broker_outage(error):
            raise
        OUTAGE.lost(error)
        raise SystemExit(EXIT_OUTAGE) from None


def run_service(main: Coroutine[Any, Any, None]) -> None:
    """`asyncio.run(main)` through `exit_on_outage`."""
    exit_on_outage(lambda: asyncio.run(main))


async def close_quietly(client: Client) -> None:
    """Closes a client at teardown. A broker that's already gone (it stopped or restarted first) leaves nothing to
    close: that's `OUTAGE.lost`, one WARN, not an error. Any other error is raised."""
    try:
        await client.close()
    except Exception as error:
        if not broker_outage(error):
            raise
        OUTAGE.lost(error)


@contextlib.asynccontextmanager
async def connected(socket: Path | None = None, *, client_name: str) -> AsyncGenerator[Client]:
    """A service's broker client for the length of the block, closed after it.

    A block that ends on its own (stopped) closes the client with `close_quietly`. A block that raised (a stream
    reader that ended, say) closes it plainly: if the close fails because the broker is gone, that outage is raised
    from the block's error, so `run_service` reports the broker leaving (one WARN, EXIT_OUTAGE) rather than the
    reader that noticed it first."""
    client = await Client.connect(str(socket or socket_path()), client_name=client_name)
    try:
        yield client
    except BaseException:
        await client.close()
        raise
    await close_quietly(client)


#: `run` and `call`'s default wait: the link's own `timeout`.
LINK_TIMEOUT = -1.0
#: How long a call waits out a busy broker (its writer queue full; `until_not_busy`) before the refusal is raised.
BUSY_WAIT_S = 30.0


def busy(e: BaseException) -> bool:
    """Whether a broker error is the broker saying it is busy (its writer queue is full; try again shortly)."""
    return (
        isinstance(e, continuum_py.BrokerError)
        and len(e.args) > 1
        and e.args[0] == "Unavailable"
        and "busy" in str(e.args[1])
    )


async def until_not_busy[T](call: Callable[[], Awaitable[T]], wait_s: float = BUSY_WAIT_S) -> T:
    """`await call()`, tried again while the broker says it is busy, for up to `wait_s` seconds (then the refusal is
    raised). A registration or a small write made while a producer is flooding the writer queue gets through once
    the queue drains, instead of failing a service's start."""
    deadline = asyncio.get_running_loop().time() + wait_s
    pause = 0.05
    while True:
        try:
            return await call()
        except continuum_py.BrokerError as e:
            if not busy(e) or asyncio.get_running_loop().time() >= deadline:
                raise
        await asyncio.sleep(pause)
        pause = min(pause * 2, 1.0)


async def until_streams_ready[T](call: Callable[[], Awaitable[T]], wait_s: float) -> T:
    """Retry only an initial NotFound subscription while its producer starts, within one deadline.

    The returned subscription is followed normally. Errors after it was established are not retried here.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_s
    pause = 0.1
    waiting = False
    last_error: continuum_py.BrokerError | None = None
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(f"input streams were not ready within {wait_s:g} s") from last_error
        limit = asyncio.timeout_at(deadline)
        try:
            async with limit:
                result = await call()
            if waiting:
                LOG.info("input streams are ready")
            return result
        except TimeoutError as error:
            if limit.expired():
                raise TimeoutError(f"input streams were not ready within {wait_s:g} s") from error
            raise
        except continuum_py.BrokerError as error:
            if not error.args or error.args[0] != "NotFound":
                raise
            last_error = error
            if not waiting:
                LOG.info("waiting for input streams: %s", error)
                waiting = True
        await asyncio.sleep(min(pause, max(0.0, deadline - loop.time())))
        pause = min(pause * 2, 1.0)


# Who asked, inside the link's loop: the loop runs every caller's work on its own thread, so a cost found there (a
# decode, say) would otherwise not say whose it is.
CALLER: contextvars.ContextVar[str] = contextvars.ContextVar("continuum_link_caller", default="?")
_PLUMBING = ("continuum_link", "vision_log", "speech_log", "memory_log", "read", "asyncio", "concurrent")


def caller() -> str:
    """The calling thread's name and the first function outside the log modules that called in: "thread module.fn"."""
    frame = sys._getframe(1)
    for _ in range(24):
        if frame is None:
            break
        module = str(frame.f_globals.get("__name__", "")).removeprefix("liberational.")
        if not module.startswith(_PLUMBING):
            return f"{threading.current_thread().name} {module}.{frame.f_code.co_name}"
        frame = frame.f_back
    return threading.current_thread().name


async def _labelled[T](coro: Coroutine[Any, Any, T], label: str) -> T:
    CALLER.set(label)  # this task's own context: run_coroutine_threadsafe starts each call in a task of its own
    return await coro


class BrokerLink:
    """One broker client on a background event loop. `call` runs a coroutine against the client and blocks until it
    finishes (or `timeout` seconds pass); a connection the broker closed is made again once and the call retried."""

    def __init__(self, socket: Path | None = None, *, client_name: str, timeout: float = 10.0) -> None:
        self.socket = socket or socket_path()
        self.client_name = client_name
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name=client_name, daemon=True)
        self._thread.start()
        self._client: Client | None = None
        self._lock = asyncio.Lock()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The background loop, for a caller that streams from it (a subscription)."""
        return self._loop

    def run[T](self, coro: Awaitable[T], timeout: float | None = LINK_TIMEOUT) -> T:
        """Runs a coroutine on the background loop and waits for it: `timeout` seconds (`LINK_TIMEOUT`, the default,
        is the link's own), or for as long as it takes with None. A wait that runs out cancels the coroutine, so
        nothing goes on behind the caller's back."""
        future: Future[T] = asyncio.run_coroutine_threadsafe(_labelled(_as_coroutine(coro), caller()), self._loop)
        try:
            return future.result(self.timeout if timeout == LINK_TIMEOUT else timeout)
        except TimeoutError:
            future.cancel()
            raise

    async def connected(self) -> Client:
        """The client, connecting first if there's none."""
        async with self._lock:
            if self._client is None:
                self._client = await Client.connect(str(self.socket), client_name=self.client_name)
            return self._client

    def call[T](self, fn: Callable[[Client], Awaitable[T]], timeout: float | None = LINK_TIMEOUT) -> T:
        """`fn(client)` on the background loop, retried once on a fresh connection if the broker had closed it;
        `timeout` as for `run`."""

        async def attempt() -> T:
            try:
                result = await fn(await self.connected())
            except continuum_py.ConnectionClosedError:
                async with self._lock:
                    self._client = None
                result = await fn(await self.connected())
            OUTAGE.back()
            return result

        return self.run(attempt(), timeout)

    def close(self) -> None:
        """Closes the connection and stops the background loop."""

        async def shut() -> None:
            if self._client is not None:
                await self._client.close()

        try:
            self.run(shut())
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


async def _as_coroutine[T](aw: Awaitable[T]) -> T:
    return await aw
