"""Whether Ollama answers, shared by every caller of it.

When Ollama isn't running, each caller used to log its own connection error on every try (the vision loop once a
second). Instead a call goes through `call(model)`:

- the first failure logs one clear line with the fix, then calls fail fast with `Down` and Ollama is tried again
  only every `ollama.retry_s`, with a reminder logged at most every `ollama.log_every_s`;
- the first call that works again logs "Ollama is back";
- a model Ollama doesn't have gets its own line naming the model and the pull command, rate limited the same way.

`Down` and `ModelMissing` are already logged, so a caller can skip logging them again. `status()` is the same check
for the dashboard: whether Ollama answers, and which configured models it lacks.
"""

import contextlib
import dataclasses
import re
import threading
import time
import urllib.error
from collections.abc import Callable, Generator
from typing import Any

import requests

from liberational import config as cfg
from liberational import logs

# The context the cameras' model (models.ollama.vlm) is always asked for: by the vision loop's warm-up, captions, and
# grounding, and by identity's badge check. Ollama reloads a model whose num_ctx changes, and one asked with none
# loads at the model's whole 262k context (25.6 GB, 16 s) before the next call reloads it.
VLM_NUM_CTX = 4096

# The context every embedding call (memory/embedding.py) asks for, for the same reason: asked with none, Ollama loads
# qwen3-embedding:4b at its own default (32k to 40k tokens, 7.8 to 9 GB of GPU memory). The longest journal entry on
# the Linux machine (6,373 entries) is 3,164 characters, 680 tokens; questions are a sentence. 2048 leaves three
# times that, and it also matches the runner's batch size (-b 2048), which an embedded text must fit. Ollama cuts a
# longer text to fit (truncate), so an entry over it is embedded by its start, not refused.
EMBEDDING_NUM_CTX = 2048

START = "start it (e.g. systemctl enable --now ollama)"
TAGS_CACHE_S = 300.0


def pull(model: str) -> str:
    """How to install a missing model: a -t<N> tag is made from its base by the models task (fetch_models.py)."""
    if re.fullmatch(r".+-t[1-9][0-9]*", model):
        return f"make it with `mise run models` (it's {model.rsplit('-t', 1)[0]} with fewer CPU threads)"
    return f"pull it with `ollama pull {model}`"


class Unavailable(RuntimeError):
    """Ollama can't serve this call now. The reason is already in the log."""


class Down(Unavailable):
    """Ollama isn't answering at all."""


class Busy(Unavailable):
    """Ollama answers but its queue is full (HTTP 503)."""


class ModelMissing(Unavailable):
    """Ollama answers but doesn't have the model."""

    def __init__(self, model: str) -> None:
        super().__init__(f"Ollama has no model {model}: {pull(model)}")
        self.model = model


def kind(err: BaseException) -> str | None:
    """ "down" for an error that means nothing is listening, "missing" for Ollama's 404 for a model it doesn't have,
    None for anything else (a timeout is a busy Ollama, not a missing one)."""
    if isinstance(err, requests.Timeout | TimeoutError):
        return None
    if isinstance(err, requests.HTTPError | urllib.error.HTTPError):
        code = err.code if isinstance(err, urllib.error.HTTPError) else getattr(err.response, "status_code", None)
        return "missing" if code == 404 else None
    if isinstance(err, urllib.error.URLError):
        return None if isinstance(err.reason, TimeoutError) else "down"
    if isinstance(err, requests.ConnectionError | ConnectionError):
        return "down"
    return None


_LOG = logs.get("ollama")


def _for(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


@dataclasses.dataclass
class Outage:
    since: float
    next_try: float  # calls before this fail fast without trying Ollama
    logged: float


class Health:
    """One process's view of Ollama. Thread-safe: the dashboard calls it from request threads."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] | None = None) -> None:
        # Down and missing are WARN, back is INFO. A log given here (a test's) gets every line.
        self.clock, self.lock = clock, threading.Lock()
        self.log: Callable[[str], None] = log or _LOG.info
        self.warn: Callable[[str], None] = log or _LOG.warning
        self.down: Outage | None = None
        self.missing: dict[str, Outage] = {}

    def before(self, model: str) -> None:
        """Raise Down or ModelMissing while backing off. The first call after the wait is let through as the retry,
        and the wait restarts for everyone else."""
        now, retry = self.clock(), float(cfg.get("ollama.retry_s"))
        with self.lock:
            if (o := self.down) is not None:
                if now < o.next_try:
                    raise Down(self.down_text(now))
                o.next_try = now + retry
            if (o := self.missing.get(model)) is not None:
                if now < o.next_try:
                    raise ModelMissing(model)
                o.next_try = now + retry

    def failed(self, err: BaseException, model: str) -> Unavailable | None:
        """Note a failed call. Returns the Unavailable to raise instead, or None when the error isn't about whether
        Ollama or the model is there (the caller handles it as before)."""
        what = kind(err)
        if what is None or (what == "missing" and not model):  # a 404 for a call naming no model isn't about one
            return None
        now, retry, every = self.clock(), float(cfg.get("ollama.retry_s")), float(cfg.get("ollama.log_every_s"))
        with self.lock:
            if what == "down":
                if self.down is None:
                    self.down = Outage(now, now + retry, now)
                    self.warn(f"Ollama isn't running at {cfg.ollama_url()}: {START}. Trying again every {retry:g} s.")
                else:
                    self.down.next_try = now + retry
                    if now - self.down.logged >= every:
                        self.down.logged = now
                        self.warn(self.down_text(now))
                return Down(self.down_text(now))
            if self.down is not None:  # a 404 is an answer: the server is up, only this model is missing
                self.log(f"Ollama is back at {cfg.ollama_url()} (down {_for(now - self.down.since)}).")
                self.down = None
            if (o := self.missing.get(model)) is None:
                self.missing[model] = Outage(now, now + retry, now)
                self.warn(f"Ollama has no model {model}: {pull(model)}.")
            else:
                o.next_try = now + retry
                if now - o.logged >= every:
                    o.logged = now
                    self.warn(f"Ollama still has no model {model}: {pull(model)}.")
            return ModelMissing(model)

    def ok(self, model: str) -> None:
        now = self.clock()
        with self.lock:
            if self.down is not None:
                self.log(f"Ollama is back at {cfg.ollama_url()} (down {_for(now - self.down.since)}).")
                self.down = None
            if self.missing.pop(model, None) is not None:
                self.log(f"Ollama has {model} now.")

    def down_text(self, now: float) -> str:
        since = self.down.since if self.down else now
        return f"Ollama isn't running at {cfg.ollama_url()} (down {_for(now - since)}): {START}"


HEALTH = Health()


@dataclasses.dataclass
class _TagsCache:
    url: str
    checked_at: float
    models: list[dict[str, Any]]


_TAGS_CACHE_LOCK = threading.Lock()
_TAGS_CACHE: _TagsCache | None = None


# Called with the model before every request `call` wraps, when set: memory/research.py sets it in the research
# process so each of its model calls waits while instant Ask is answering.
gate: Callable[[str], object] | None = None


@contextlib.contextmanager
def call(model: str, health: Health | None = None) -> Generator[None]:
    """Wrap one Ollama request for model. Raises Down or ModelMissing (already logged) instead of the connection
    error or 404, fails fast while backing off, and logs recovery. Other errors pass through unchanged."""
    if gate is not None:
        gate(model)
    h = health or HEALTH
    h.before(model)
    try:
        yield
    except Exception as e:
        if (u := h.failed(e, model)) is None:
            raise
        raise u from e
    h.ok(model)


def configured() -> list[str]:
    """The models.ollama models, as `mise run models` pulls them. Each role is read by its own key, so a PACK_
    environment override counts, as it does for the callers."""
    return sorted({m for role in cfg.get("models.ollama") or {} if (m := cfg.get(f"models.ollama.{role}"))})


def _get(url: str, timeout: float) -> Any:
    with requests.Session() as s:
        s.trust_env = False  # no proxies: Ollama is local
        return s.get(url, timeout=timeout, allow_redirects=False)


def tags(
    get: Callable[[str, float], Any] | None = None,
    timeout: float = 1.5,
    *,
    max_age_s: float = TAGS_CACHE_S,
    refresh: bool = False,
) -> list[dict[str, Any]] | None:
    """The installed models from /api/tags, cached for max_age_s while Ollama answers."""
    global _TAGS_CACHE
    use_cache = get is None
    url = cfg.ollama_url()
    now = time.time()
    if use_cache and not refresh and max_age_s > 0:
        with _TAGS_CACHE_LOCK:
            cached = _TAGS_CACHE
            if cached is not None and cached.url == url and now - cached.checked_at < max_age_s:
                return cached.models
    try:
        response = (get or _get)(f"{url}/api/tags", timeout)
    except Exception:
        if refresh:
            with _TAGS_CACHE_LOCK:
                _TAGS_CACHE = None
        raise
    try:
        response.raise_for_status()
    except Exception:
        if refresh:
            with _TAGS_CACHE_LOCK:
                _TAGS_CACHE = None
        raise
    models = response.json().get("models", [])
    if use_cache:
        with _TAGS_CACHE_LOCK:
            _TAGS_CACHE = _TagsCache(url=url, checked_at=now, models=models)
    return models


def status(
    get: Callable[[str, float], Any] | None = None,
    timeout: float = 1.5,
    *,
    refresh: bool = False,
    max_age_s: float = TAGS_CACHE_S,
) -> dict[str, Any]:
    """For the dashboard: whether Ollama answers at ollama.url, and which configured models it doesn't have. message
    is what to tell the owner, None when all is well. An answer slower than `timeout` comes from a running Ollama
    that took the connection, so it counts as up, with its models unchecked this time: a busy Ollama took over 1.5 s
    to list them 13% of the time on the Linux machine, and each read as down."""
    url = cfg.ollama_url()
    try:
        have = {
            n
            for m in (tags(get, timeout, max_age_s=max_age_s, refresh=refresh) or [])
            for n in (m.get("name"), m.get("model"))
            if n
        }
    except requests.exceptions.ReadTimeout as e:
        return {"up": True, "missing": [], "message": None, "error": str(e)[:120]}
    except Exception as e:
        return {"up": False, "missing": [], "message": f"Ollama isn't running at {url}: {START}", "error": str(e)[:120]}
    missing = [m for m in configured() if m not in have and f"{m}:latest" not in have]
    message = "; ".join(f"Model missing: {m} ({pull(m)})" for m in missing) or None
    return {"up": True, "missing": missing, "message": message, "error": None}
