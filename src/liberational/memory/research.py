"""Research: an Ask question taken slowly, as a background job whose report lands in the Inbox.

Ask answers at once from the live setup. Research takes the same question with more search rounds, a longer answer
timeout, and no deadline, and runs one job at a time in `pack-research` at idle CPU and IO priority. Each finished job
is a report in the dashboard's Inbox: the answer with its citations, the question, how long it took, and what it
looked at (the searches and the sources behind the answer), read or unread.

Jobs are files under `paths.state/research/` (`<id>.json`, written whole and renamed into place), with a `<id>.read`
marker once the owner has opened the report. Every change happens under one advisory lock, so the dashboard (submit,
read, delete) and the worker (claim, heartbeat, finish) never overwrite each other. A job left running by a worker
that stopped goes back to the queue once; interrupted again, it fails, so one bad question can't wedge the queue.

Where its final answer runs (the GPU decision): by default research uses Ask's model and context on Ask's own Ollama (`ask.*`). A
server that keeps one model loaded (ask-ollama on the Linux machine) then never swaps Ask's model out for research's, so
instant Ask stays loaded. `[research]` can name another model, context, or server; a different model on Ask's one-model
server would unload Ask's. At the end of a job (including a failed answer), Research reloads the normal Ask model and
context when it used a different load on that same dedicated server. A failed reload is logged without discarding the
report. Before every model call the worker waits while an instant Ask question is queued or running (the busy markers
`hold()` leaves in `paths.state/ask/busy/`). A research answer already being written when one arrives is stopped and
asked again after it (ask.YIELD), since the two would share the GPU and slow the instant one down; a search round's call
is short and runs to its end. Thinking Research answers are capped at MAX_REPLY_TOKENS and half of the configured
context, whichever is smaller. Planning and ranking use an explicitly configured recall.model or the already-resident
conversational model at its shared context, keeping non-thinking JSON calls away from a thinking-only final-answer
model. Planner history drops older evidence snippets when necessary; its search decisions and retained final-answer
evidence are not discarded.

Research's final answer needs a server other than the voice loop's (`ollama.url`): the long reasoning call on the
shared one would slow voice replies, so without another server it isn't offered.
"""

import contextlib
import fcntl
import json
import os
import re
import signal
import threading
import time
import uuid
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Any

from liberational import config as cfg
from liberational import json_state, logs, ollama, ollama_health, voice_turn

LOG = logs.get("research")
MAX_QUESTION = 1200
MAX_QUEUED = 10
MAX_KEPT = 200  # finished reports kept; older ones go when a new job is submitted
MAX_EVIDENCE = 60  # cited sources kept in a report
HEARTBEAT_S = 30.0
STALE_S = 120.0  # a running job whose heartbeat is older than this was left by a worker that stopped
MAX_ATTEMPTS = 2
POLL_S = 2.0
BUSY_STALE_S = 600.0  # an instant marker older than this was left by a process that died
YIELD_MAX_S = 900.0  # never wait longer than this for instant Ask before carrying on
ANSWER_TIMEOUT_S = 600.0  # one model answer; Ask's own is 90 s
MAX_REPLY_TOKENS = 8192  # reasoning plus final text; at most half the configured context
LOOP_BUDGET_S = 1800.0  # the whole search loop; Ask's own is 80 s
ID = re.compile(r"[0-9a-f]{32}")


def root(state: Path | None = None) -> Path:
    return Path(state or cfg.path("paths.state")) / "research"


def busy_dir(state: Path | None = None) -> Path:
    return Path(state or cfg.path("paths.state")) / "ask" / "busy"


# Where research may run.


def server() -> str:
    """The Ollama research's calls go to: research.ollama_url, else Ask's."""
    return str(cfg.get("research.ollama_url") or cfg.get("ask.ollama_url") or "").rstrip("/")


def unavailable() -> str | None:
    """Why research can't run on this machine, or None when it can."""
    if not cfg.get("research.enabled"):
        return "Research is turned off in Settings."
    url = server()
    if not url or url == str(cfg.get("ollama.url") or "").rstrip("/"):
        return (
            "Research needs Ask's own Ollama (the ask-ollama service and ask.ollama_url), so it never slows the "
            "voice replies."
        )
    return None


def settings() -> dict[str, Any]:
    """The config research runs under: Ask's model, context, and server unless [research] names its own."""
    return {
        "ask": {
            "model": cfg.get("research.model") or cfg.get("ask.model") or "",
            "num_ctx": int(cfg.get("research.num_ctx") or cfg.get("ask.num_ctx") or 0),
            "ollama_url": server(),
            "think": bool(cfg.get("research.think")),
        },
        # A thinking-only answer model cannot serve the loop's small non-thinking JSON calls. Reuse the
        # already-resident conversational model unless the owner explicitly chose another search model.
        "recall": {
            "enabled": True,
            "max_rounds": int(cfg.get("research.max_rounds")),
            "model": cfg.get("recall.model") or cfg.get("models.ollama.llm"),
        },
    }


# Instant Ask leaves a marker from when a question is queued until it is answered; research waits on it.


def hold(name: str, state: Path | None = None) -> Path | None:
    """Mark an instant Ask question as waiting or running, until release(); the marker, or None when it can't be
    written (research then only won't wait for it)."""
    try:
        folder = busy_dir(state)
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / f"{os.getpid()}-{name}"
        marker.write_text(str(time.time()))
        return marker
    except OSError:
        return None


def release(marker: Path | None) -> None:
    if marker is not None:
        with contextlib.suppress(OSError):
            marker.unlink()


@contextlib.contextmanager
def instant(state: Path | None = None) -> Generator[None]:
    """Around an instant Ask question: research holds its next model call until this is done."""
    marker = hold(uuid.uuid4().hex, state)
    try:
        yield
    finally:
        release(marker)


def _alive(marker: Path) -> bool:
    """Whether the process that left marker (named <pid>-...) is still running."""
    try:
        os.kill(int(marker.name.split("-", 1)[0]), 0)
    except ValueError:
        return True  # not a pid: only its age can tell
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def instant_running(state: Path | None = None) -> bool:
    now = time.time()
    try:
        return any(now - m.stat().st_mtime < BUSY_STALE_S and _alive(m) for m in busy_dir(state).iterdir())
    except OSError:
        return False


def yield_to_instant(state: Path | None = None, sleep: Any = time.sleep) -> float:
    """Wait while instant Ask is answering (up to YIELD_MAX_S); the seconds waited."""
    waited = 0.0
    while instant_running(state) and waited < YIELD_MAX_S:
        sleep(0.5)
        waited += 0.5
    return waited


# The store.


def _answer_status(result: object) -> str | None:
    """A finished report's answer status, as Ask wrote it ("ok", "partial"), else None."""
    status = result.get("status") if isinstance(result, dict) else None
    return status if isinstance(status, str) and status else None


class Store:
    """The research jobs under one state directory."""

    def __init__(self, state: Path | None = None) -> None:
        self.dir = root(state)

    @contextlib.contextmanager
    def _locked(self) -> Generator[None]:
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / ".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _path(self, ident: str) -> Path:
        if not ID.fullmatch(ident):
            raise LookupError("no such research job")
        return self.dir / f"{ident}.json"

    def _read(self, ident: str) -> dict[str, Any] | None:
        try:
            job = json.loads(self._path(ident).read_text())
        except OSError, ValueError:
            return None
        if not isinstance(job, dict):
            return None
        job["read"] = (self.dir / f"{ident}.read").exists()
        return job

    def _write(self, job: Mapping[str, Any]) -> None:
        saved = {k: v for k, v in job.items() if k != "read"}
        json_state.write(self._path(str(job["id"])), saved, ensure_ascii=False)

    def _all(self) -> list[dict[str, Any]]:
        jobs = [j for p in self.dir.glob("*.json") if (j := self._read(p.stem)) is not None]
        return sorted(jobs, key=lambda j: j["created_at"], reverse=True)

    def submit(self, question: object, day: object = None, now: float | None = None) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Ask a question to research.")
        question = " ".join(question.split())
        if len(question) > MAX_QUESTION:
            raise ValueError(f"Keep the question under {MAX_QUESTION} characters.")
        if day is not None:
            from liberational.memory import ask

            day = ask.validate_day(day)
        with self._locked():
            jobs = self._all()
            if sum(1 for j in jobs if j["status"] in ("queued", "running")) >= MAX_QUEUED:
                raise ValueError(f"{MAX_QUEUED} research questions are already waiting; try again when one is done.")
            for old in [j for j in jobs if j["status"] in ("done", "failed")][MAX_KEPT - 1 :]:
                self._remove(old["id"])
            job = {
                "id": uuid.uuid7().hex,
                "question": question,
                "day": day,
                "status": "queued",
                "created_at": time.time() if now is None else now,
                "started_at": None,
                "finished_at": None,
                "attempts": 0,
            }
            self._write(job)
            return {**job, "read": False}

    def jobs(self) -> list[dict[str, Any]]:
        """Every job, newest first, without the full report but with how its answer came out (`answer_status`: "ok",
        "partial", ...; None before one), so the Inbox list can say a report is partial without opening it."""
        with self._locked():
            return [
                {
                    **{k: v for k, v in j.items() if k not in ("result", "worker")},
                    "answer_status": _answer_status(j.get("result")),
                }
                for j in self._all()
            ]

    def get(self, ident: str) -> dict[str, Any] | None:
        with self._locked():
            job = self._read(ident)
            return None if job is None else {k: v for k, v in job.items() if k != "worker"}

    def mark(self, ident: str, read: bool) -> bool:
        with self._locked():
            if self._read(ident) is None:
                return False
            marker = self.dir / f"{ident}.read"
            if read:
                marker.touch()
            else:
                marker.unlink(missing_ok=True)
            return True

    def _remove(self, ident: str) -> None:
        for suffix in (".json", ".read"):
            (self.dir / f"{ident}{suffix}").unlink(missing_ok=True)

    def delete(self, ident: str) -> bool:
        with self._locked():
            if self._read(ident) is None:
                return False
            self._remove(ident)
            return True

    # The worker's side.

    def claim(self, worker: str, now: float | None = None) -> dict[str, Any] | None:
        """The next job to run (oldest first), marked running by `worker`. A job a stopped worker left running goes
        back to the queue, or fails once it has been interrupted MAX_ATTEMPTS times."""
        at = time.time() if now is None else now
        with self._locked():
            jobs = sorted(self._all(), key=lambda j: j["created_at"])
            for job in jobs:
                if job["status"] == "running" and at - float(job.get("heartbeat") or 0) > STALE_S:
                    if job["attempts"] >= MAX_ATTEMPTS:
                        job.update(
                            status="failed",
                            finished_at=at,
                            error="Research was interrupted twice, so it stopped trying this question.",
                        )
                    else:
                        job.update(status="queued")
                    self._write(job)
            for job in jobs:
                if job["status"] == "queued":
                    job.update(
                        status="running",
                        started_at=at,
                        heartbeat=at,
                        attempts=job["attempts"] + 1,
                        worker=worker,
                    )
                    self._write(job)
                    return job
            return None

    def beat(self, ident: str, worker: str, now: float | None = None) -> None:
        with self._locked():
            job = self._read(ident)
            if job is not None and job["status"] == "running" and job.get("worker") == worker:
                job["heartbeat"] = time.time() if now is None else now
                self._write(job)

    def finish(self, ident: str, worker: str, result: Mapping[str, Any], now: float | None = None) -> bool:
        """Store a finished job's report; False when the job was deleted (or taken over) meanwhile."""
        at = time.time() if now is None else now
        with self._locked():
            job = self._read(ident)
            if job is None or job["status"] != "running" or job.get("worker") != worker:
                return False
            ok = result.get("status") in ("ok", "partial")
            job.update(
                status="done" if ok else "failed",
                finished_at=at,
                took_s=round(at - float(job["started_at"]), 1),
                result=report(result),
            )
            if not ok:
                job["error"] = str(result.get("answer") or "Research failed.")
            self._write(job)
            return True


def report(result: Mapping[str, Any]) -> dict[str, Any]:
    """What the Inbox shows of a finished job: the answer, its cited sources, and what it looked at."""
    evidence = [e for e in (result.get("evidence") or []) if isinstance(e, dict)]
    answer = str(result.get("answer") or "")
    cited = set(re.findall(r"\[([A-Z]+\d+)\]", answer))
    shown = [e for e in evidence if e.get("id") in cited] or evidence
    sources: dict[str, int] = {}
    for e in evidence:
        kind = str(e.get("source") or "other")
        sources[kind] = sources.get(kind, 0) + 1
    rounds = [r for r in (result.get("rounds") or []) if isinstance(r, dict)]
    return {
        "status": result.get("status"),
        "answer": answer,
        "model": result.get("model"),
        "warnings": list(result.get("warnings") or []),
        "evidence": shown[:MAX_EVIDENCE],
        "looked_at": {
            "sources": sources,
            "evidence": len(evidence),
            "searches": [
                {"tool": r.get("tool"), "args": r.get("args"), "found": r.get("found")}
                for r in rounds
                if r.get("decision") == "search"
            ],
            "window": (result.get("query") or {}).get("window"),
        },
    }


# Running a job.


@contextlib.contextmanager
def deep() -> Generator[None]:
    """Research's own limits, in this process only: more search time, a longer answer timeout, a reply cap, and
    yielding to instant Ask (the gate before each call, and stopping an answer in flight)."""
    from liberational.memory import ask, recall

    saved = (
        recall.LOOP_BUDGET_S,
        ask.ANSWER_TIMEOUT_S,
        ask.MAX_REPLY_TOKENS,
        ask.ANSWER_SAMPLING,
        ask.YIELD,
        ollama_health.gate,
    )
    recall.LOOP_BUDGET_S, ask.ANSWER_TIMEOUT_S, ask.MAX_REPLY_TOKENS = LOOP_BUDGET_S, ANSWER_TIMEOUT_S, MAX_REPLY_TOKENS
    # Qwen's thinking checkpoint recommends sampling to avoid repetitive reasoning; instant Ask stays greedy.
    # https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507#best-practices
    ask.ANSWER_SAMPLING = (
        {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0} if cfg.get("research.think") else {}
    )
    ask.YIELD = instant_running

    def before_call(model: str) -> None:
        yield_to_instant()
        if model == cfg.get("models.ollama.llm"):
            voice_turn.wait_quiet()

    ollama_health.gate = before_call
    try:
        with cfg.override(settings()):
            yield
    finally:
        (
            recall.LOOP_BUDGET_S,
            ask.ANSWER_TIMEOUT_S,
            ask.MAX_REPLY_TOKENS,
            ask.ANSWER_SAMPLING,
            ask.YIELD,
            ollama_health.gate,
        ) = saved


def _restore_ask(url: str, model: str, num_ctx: int) -> None:
    """Restore the interactive load after a different background answer displaced it."""
    from liberational.memory import ask

    try:
        ollama.load(model, num_ctx=num_ctx, keep_alive=ask.keep_alive(model, num_ctx), url=url, local=True)
    except Exception as error:
        # A completed report remains useful even if reloading Ask fails. The next Ask can retry its own load.
        LOG.warning("Could not restore Ask's model after Research: %s", type(error).__name__)


def run(job: Mapping[str, Any], state: Path | None = None) -> dict[str, Any]:
    """One research job, restoring a displaced interactive model before taking another job."""
    from liberational.memory import ask

    model, num_ctx = ask.local_settings()
    url = ask.ollama_url()
    restore = False
    try:
        with deep():
            restore = url != cfg.ollama_url() and ask.ollama_url() == url and ask.local_settings() != (model, num_ctx)
            return ask.execute(str(job["question"]), "local", state, day=job.get("day"), scope="history")
    finally:
        # deep() has restored normal Ask settings and the previous gate, even when the answer raised.
        if restore:
            _restore_ask(url, model, num_ctx)


def work(stop: threading.Event, state: Path | None = None) -> None:
    """Run queued jobs one at a time until `stop` is set."""
    store = Store(state)
    worker = f"{os.uname().nodename}:{os.getpid()}"
    while not stop.is_set():
        if unavailable() is not None:
            stop.wait(30)
            continue
        job = store.claim(worker)
        if job is None:
            stop.wait(POLL_S)
            continue
        LOG.info(f"research job {job['id'][:8]} started")
        beating = threading.Event()

        def heartbeat(ident: str = job["id"], done: threading.Event = beating) -> None:
            while not done.wait(HEARTBEAT_S):
                try:
                    store.beat(ident, worker)
                except OSError:
                    LOG.warning("could not record a research heartbeat", exc_info=True)

        threading.Thread(target=heartbeat, daemon=True, name="research-heartbeat").start()
        try:
            result = run(job, state)
        except Exception as exc:
            LOG.exception("research job failed")
            result = {"status": "error", "answer": f"Research failed ({type(exc).__name__}).", "evidence": []}
        finally:
            beating.set()
        if store.finish(job["id"], worker, result):
            LOG.info(f"research job {job['id'][:8]} finished: {result.get('status')}")
        else:
            LOG.info(f"research job {job['id'][:8]} was deleted while it ran; its report is dropped")


def lower_priority() -> None:
    """Idle CPU and IO priority for this process (and the model calls it waits on are another process's)."""
    with contextlib.suppress(OSError):
        os.nice(19)
    try:
        import psutil

        if hasattr(psutil, "IOPRIO_CLASS_IDLE"):
            psutil.Process().ionice(psutil.IOPRIO_CLASS_IDLE)
    except Exception:
        LOG.debug("could not lower research IO priority", exc_info=True)


def main() -> None:
    """`pack-research`: run queued research questions one at a time, at background priority."""
    lower_priority()
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    LOG.info("research worker started")
    work(stop)
