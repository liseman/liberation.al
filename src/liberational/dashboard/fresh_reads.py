"""`_Fresh`: a slow read answered from the last answer while a new one is worked out in the background, with a
"starting" answer until the first exists (moved out of the server, #61). The unnamed face and voice groups use it.
"""

import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from liberational import logs, voice_turn
from liberational.dashboard import photo_files
from liberational.dashboard.context import Reply, log


class _Fresh:
    """A slow view's last answer, kept. Its key has two parts: what the owner changes (names, skips, merges), and what
    comes in on its own (new lines, new faces). When the owner's part changed, it's worked out again at once, so their
    change shows on the next read; when only the data did, the last answer is served at once and worked out again in
    the background for the next one. Matching cold reads share in-flight work, and an older completion cannot
    replace a newer published result. Live and People poll these, and working one out takes a second or more.

    With `patch`, an owner change it can apply to the last answer is served at once too: patch(answer, owner key it was
    worked out for, owner key now) returns the answer brought up to date as far as is cheap, or None when it can't be
    (then the read waits for the full answer, as without one). The full answer is worked out in the background
    meanwhile, and published even if the owner changed again while it ran, since the next read patches it
    again."""

    def __init__(
        self,
        name: str,
        compute: Callable[[], tuple[int, dict[str, Any]]],
        owner: Callable[[], Any],
        data: Callable[[], Any],
        patch: Callable[[Reply, Any, Any], Reply | None] | None = None,
    ) -> None:
        self.name, self.compute, self.owner, self.data, self.patch = name, compute, owner, data, patch
        self.lock = threading.Lock()
        self.pending: list[tuple[tuple[Any, Any], Future[Reply]]] = []
        self.next_work = 0
        self.published_work = 0
        self.value: tuple[int, dict[str, Any]] | None = None
        self.keys: tuple[Any, Any] | None = None
        self.busy = False
        self.cold: Future[Reply] | None = None

    def poll(self) -> Reply:
        """Admit a cold HTTP read without waiting for imports, owner keys, or the shared computation.

        Internal readers still use blocking get() so startup readiness is checked. Once published, normal
        owner/purge validation stays synchronous: pending never serves an unvalidated former answer.
        """
        with self.lock:
            warm = self.value is not None
            if warm and self.cold is not None and self.cold.done():
                self.cold = None
            if not warm:
                if self.cold is not None and self.cold.done():
                    completed, self.cold = self.cold, None
                    result = completed.result()
                    if result[0] != 200:
                        return result
                if self.cold is None:
                    self.cold = Future()
                    threading.Thread(
                        target=self._cold, args=(self.cold,), daemon=True, name=f"cold-{self.name}"
                    ).start()
        if warm:
            return self.get()
        return 202, {"pending": True, "groups": [], "count": None, "skipped": []}

    def _cold(self, future: Future[Reply]) -> None:
        try:
            photo_files._low_priority(19)
            voice_turn.wait_quiet()
            future.set_result(self.get())
        except BaseException as error:
            future.set_exception(error)

    def get(self) -> tuple[int, dict[str, Any]]:
        keys = (self.owner(), self.data())
        with self.lock:
            value, kept = self.value, self.keys
            if value is not None and kept is not None and kept[0] == keys[0]:
                if kept[1] != keys[1]:
                    self._behind(keys)
                return value
        if value is not None and kept is not None and self.patch is not None:
            patched = self.patch(value, kept[0], keys[0])
            if patched is not None:
                with self.lock:
                    self._behind(keys)
                return patched
        return self._work(keys)

    def _behind(self, keys: tuple[Any, Any]) -> None:
        """Work the answer out again in the background, unless that's under way already. Called holding the lock."""
        if not self.busy:
            self.busy = True
            threading.Thread(target=self._refresh, args=(keys,), daemon=True, name=f"fresh-{self.name}").start()

    def ready(self) -> bool:
        """Whether a successful snapshot was published for the current owner (data may refresh behind it), or one the
        patch brings up to it."""
        owner = self.owner()
        with self.lock:
            value, kept = self.value, self.keys
        if value is None or value[0] != 200 or kept is None:
            return False
        return kept[0] == owner or (self.patch is not None and self.patch(value, kept[0], owner) is not None)

    def _work(self, keys: tuple[Any, Any]) -> Reply:
        # Coalesce matching reads, but let an owner correction bypass an older background refresh.
        work = 0
        with self.lock:
            if self.value is not None and self.keys == keys:
                return self.value
            future = next((future for pending, future in self.pending if pending == keys), None)
            owns_read = future is None
            if future is None:
                future = Future()
                self.pending.append((keys, future))
                self.next_work += 1
                work = self.next_work
        if not owns_read:
            return future.result()
        try:
            value = self.compute()
            owner = self.owner()
            with self.lock:
                # An older owner snapshot must not replace a correction that completed while it ran. With a patch, an
                # answer whose owner moved on while it ran is still published under the keys it was worked out for:
                # the next read patches it, where without one the read would wait for yet another full answer.
                if value[0] == 200 and (owner == keys[0] or self.patch is not None) and work >= self.published_work:
                    self.value, self.keys = value, keys
                    self.published_work = work
            future.set_result(value)
            return value
        except BaseException as error:
            future.set_exception(error)
            raise
        finally:
            with self.lock:
                self.pending.remove((keys, future))

    def _refresh(self, keys: tuple[Any, Any]) -> None:
        try:
            self._work(keys)
        except Exception as e:
            log(f"{self.name}: not refreshed:", repr(e), level=logs.WARN)
        finally:
            with self.lock:
                self.busy = False
