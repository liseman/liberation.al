"""The Ingest tab's API (frontend/src/dashboard/ingest/): photos, video, and audio from uploads, links, and sources,
through the ingest pipeline (liberational/ingest/, docs/INGEST.md).

GET  /ingest/api/items        {items: [the newest 100, newest first, each with `analysis`: what reading it found (see
                              analysis())], sources: [{name, label, ongoing, configured,
                              message, seen}], jobs: [{id, what, status, error, item}], control, wyze: {cameras,
                              rtsp_url, segment_s, email, password, api_id, api_key (each: set or not)}}
POST /ingest/api/upload?name= the file itself as the body (up to ingest.max_mb): kept, then converted and registered
                              in the background. {item, new}; a duplicate's item is the one already there
POST /ingest/api/link         {"url"}: fetched and ingested in the background. {job}
POST /ingest/api/retry        {"id"}: process an item again. {job}
POST /ingest/api/delete       {"id"}: cancel, purge this item and its analyses, and remove stored files.
                              {ok, deleted, pending}; pending cleanup is retried with the same request
POST /ingest/api/sources/<name>/poll   one pass over a source now. {job}
GET  /ingest/api/google_photos/albums  {albums: [{id, title}]}
GET  /ingest/api/google_photos/photos?album=  {items: [{key, name, taken_at, url, kind, thumb}]} (the library's
                              newest without an album)
POST /ingest/api/google_photos/login   opens the sign-in window on this machine's screen. This machine only
POST /ingest/api/google_photos/import  {"items": [as listed]}: downloaded and ingested in the background. {job}

Wyze's settings (ingest.wyze_*) are saved through POST /config, as Settings saves them; secrets are never sent back,
only whether each is set. Writes are same-origin only, as uploads to People are; the sign-in window opens only for
a request from this machine. Work runs on one background thread, one job at a time, so conversions never pile up.
"""

import json
import queue
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import logs
from liberational.ingest import pipeline as ingest_pipeline
from liberational.ingest import runner
from liberational.ingest import sources as src
from liberational.ingest.sources.google_photos import GooglePhotosSource
from liberational.ingest.sources.link import LinkError, LinkSource, check_url
from liberational.ingest.sources.wyze import credentials_set

if TYPE_CHECKING:
    from liberational.dashboard.server import H

LOG = logs.get("dashboard-ingest")
PREFIX = "/ingest/api/"
MAX_JSON = 1 << 20
KEEP_JOBS = 50


@dataclass
class Job:
    id: int
    what: str
    status: str = "queued"  # queued, running, done, failed
    error: str = ""
    item: int | None = None
    at: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "what": self.what, "status": self.status, "error": self.error, "item": self.item}


class Worker:
    """One background thread running jobs in order; the newest KEEP_JOBS are kept for the tab to show."""

    def __init__(self) -> None:
        self.jobs: list[Job] = []
        self.queue: queue.Queue[tuple[Job, Callable[[], int | None]]] = queue.Queue()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.next_id = 1

    def submit(self, what: str, work: Callable[[], int | None]) -> Job:
        with self.lock:
            job = Job(self.next_id, what)
            self.next_id += 1
            self.jobs = [*self.jobs, job][-KEEP_JOBS:]
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._run, name="ingest", daemon=True)
                self.thread.start()
        self.queue.put((job, work))
        return job

    def _run(self) -> None:
        while True:
            job, work = self.queue.get()
            job.status = "running"
            try:
                job.item = work()
                job.status = "done"
            except Exception as e:
                LOG.warning(f"{job.what}: {e}")
                job.status, job.error = "failed", str(e) or type(e).__name__
            self.queue.task_done()

    def recent(self) -> list[dict[str, Any]]:
        with self.lock:
            return [j.public() for j in reversed(self.jobs)]


WORKER = Worker()
_pipeline: ingest_pipeline.Pipeline | None = None
_pipeline_lock = threading.Lock()


def pipeline() -> ingest_pipeline.Pipeline:
    """The pack's ingest pipeline, made on first use. At that point it also picks up items a restart cut off."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = ingest_pipeline.default()
            for item in _pipeline.store.pending():
                WORKER.submit(f"resume {item.name}", lambda i=item.id: _process(i))
        return _pipeline


def _process(item_id: int) -> int:
    return pipeline().process(item_id).id


def sources_view() -> list[dict[str, Any]]:
    out = []
    store = pipeline().store
    for name in src.SOURCES:
        s = src.make(name)
        st = s.status()
        out.append(
            {
                "name": name,
                "label": s.label,
                "ongoing": s.ongoing,
                "configured": st.configured,
                "message": st.message,
                "blocked": st.blocked,
                "later": st.later,
                "seen": store.seen_count(name),
            }
        )
    return out


def wyze_view() -> dict[str, Any]:
    """Wyze's settings for the tab's form: the plain ones as they are, the credentials only as set or not."""
    out: dict[str, Any] = {
        "cameras": [str(c) for c in cfg.get("ingest.wyze_cameras")],
        "rtsp_url": str(cfg.get("ingest.wyze_rtsp_url")),
        "segment_s": int(cfg.get("ingest.wyze_segment_s")),
    }
    out.update(credentials_set())
    return out


ANALYZED = "ingest-analyzed.json"  # analyze.Ledger's file, beside the store


def _analyzed() -> dict[str, Any]:
    """The analysis ledger's rows by item id, read as they are (it is replaced whole, so a read never sees half)."""
    path = pipeline().store.path.parent / ANALYZED
    try:
        rows = json.loads(path.read_text())
    except OSError, ValueError:
        return {}
    return rows if isinstance(rows, dict) else {}


def analysis(item: Any, rows: dict[str, Any]) -> dict[str, Any] | None:
    """What reading an item found, for its row's "what came of it": None while it's still being brought in;
    {"state": "waiting"} before its pass; "retrying"; "failed" with the last error; or "done" with how many frames
    were read, captions written, and lines heard. Faces in brought-in media aren't read yet."""
    if item.status != "done" or not item.outputs:
        return None
    row = rows.get(str(item.id))
    if not isinstance(row, dict):
        return {"state": "waiting"}
    state = row.get("state") or "done"
    if state == "retry":
        return {"state": "retrying"}
    if state == "failed":
        errors = row.get("errors") or []
        return {"state": "failed", "error": str(errors[-1]) if errors else None}
    return {
        "state": "done",
        "frames": int(row.get("frames") or 0),
        "captions": int(row.get("captions") or 0),
        "lines": int(row.get("lines") or 0),
    }


def view(control: bool) -> dict[str, Any]:
    rows = _analyzed()
    return {
        "items": [{**i.public(), "analysis": analysis(i, rows)} for i in pipeline().store.recent(100)],
        "deletions": pipeline().store.deletions(),
        "sources": sources_view(),
        "jobs": WORKER.recent(),
        "control": control,
        "wyze": wyze_view(),
    }


def upload(handler: H, name: str) -> tuple[int, dict[str, Any]]:
    """Streams the request body to disk (never all in memory), adds it, and queues its processing."""
    max_bytes = int(cfg.get("ingest.max_mb")) << 20
    try:
        length = int(handler.headers.get("Content-Length") or 0)
    except ValueError:
        length = -1
    if not 0 < length <= max_bytes:
        handler.close_connection = True
        return 413, {"error": f"Choose a nonempty file up to {max_bytes >> 20} MB."}
    p = pipeline()
    into = runner.incoming(p)
    dest = into / (Path(name).name or "upload")
    left = length
    try:
        handler.connection.settimeout(60)
        with dest.open("wb") as f:
            while left:
                block = handler.rfile.read(min(left, 1 << 20))
                if not block:
                    raise ValueError("Upload interrupted. Please retry.")
                f.write(block)
                left -= len(block)
        item, new = p.add(ingest_pipeline.Incoming(dest, "upload", dest.name))
    except (ValueError, OSError) as e:
        handler.close_connection = True
        return 400, {"error": str(e)}
    finally:
        runner.cleanup(into)
    if new:
        WORKER.submit(f"upload {item.name}", lambda: _process(item.id))
    return 200, {"item": item.public(), "new": new}


def _link_job(url: str) -> int:
    return runner.ingest_link(pipeline(), url, LinkSource.from_config()).id


def _poll_job(name: str) -> int | None:
    result = runner.poll_source(pipeline(), runner.ongoing([name], record=False)[0])
    if result.not_configured:
        raise src.NotConfigured(result.not_configured)
    if result.error:
        raise RuntimeError(result.error)
    return None


def _import_job(items: list[src.RemoteItem]) -> int | None:
    p = pipeline()
    gp = GooglePhotosSource.from_config()
    into = runner.incoming(p)
    last = None
    try:
        for item, path in gp.fetch_many(items, into):
            last = p.ingest(ingest_pipeline.Incoming(path, gp.name, item.name, key=item.key, taken_at=item.taken_at)).id
    finally:
        runner.cleanup(into)
    return last


def remote_items(raw: object) -> list[src.RemoteItem]:
    """The items the tab sends back to import, as RemoteItems; ValueError for anything else."""
    if not isinstance(raw, list) or not raw or len(raw) > 500:
        raise ValueError("send {items: [...]} with 1 to 500 items")
    out = []
    for r in raw:
        if not isinstance(r, dict) or not all(isinstance(r.get(k), str) and r.get(k) for k in ("key", "name", "url")):
            raise ValueError("each item needs key, name, and url")
        taken = r.get("taken_at")
        out.append(
            src.RemoteItem(
                key=r["key"],
                name=Path(r["name"]).name,
                url=r["url"],
                taken_at=float(taken) if isinstance(taken, int | float) else None,
            )
        )
    return out


def _same_origin(handler: H) -> bool:
    origin, host = handler.headers.get("Origin"), handler.headers.get("Host") or ""
    return not origin or origin in (f"http://{host}", f"https://{host}")


def _read_json(handler: H) -> Any:
    n = int(handler.headers.get("Content-Length") or 0)
    if not 0 <= n <= MAX_JSON:
        raise ValueError("too large")
    return json.loads(handler.rfile.read(n) or b"{}") if n else {}


def handle(handler: H, log: Callable[..., None] = print) -> bool:
    """Serve /ingest/api/*; False for any other path."""
    path = handler.path.split("?")[0]
    if not path.startswith(PREFIX):
        return False
    route = path.removeprefix(PREFIX)
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(handler.path).query)
    if handler.command in ("GET", "HEAD"):
        try:
            if route == "items":
                return _reply(handler, 200, view(handler._can_control()))
            if route == "google_photos/albums":
                return _reply(handler, 200, {"albums": GooglePhotosSource.from_config().albums()})
            if route == "google_photos/photos":
                album = (query.get("album") or [None])[0] or None
                found = GooglePhotosSource.from_config().photos(album, scrolls=4)
                return _reply(handler, 200, {"items": [i.public() for i in found]})
        except src.NotConfigured as e:
            return _reply(handler, 409, {"error": str(e)})
        except ValueError as e:
            return _reply(handler, 400, {"error": str(e)})
        except Exception:
            if route not in ("google_photos/albums", "google_photos/photos"):
                raise
            LOG.warning("Google Photos listing failed", exc_info=True)
            return _reply(handler, 503, {"error": "Google Photos is unavailable. Please retry."})
        return _reply(handler, 404, {"error": "not found"})
    if handler.command != "POST":
        return False
    if not _same_origin(handler):
        return _reply(handler, 403, {"error": "same-origin requests only"})
    if route == "upload":
        name = (query.get("name") or ["upload"])[0]
        code, body = upload(handler, name)
        return _reply(handler, code, body)
    if handler.headers.get_content_type() != "application/json":
        return _reply(handler, 403, {"error": "same-origin JSON only"})
    try:
        body = _read_json(handler)
        if not isinstance(body, dict):
            raise ValueError("send a JSON object")
    except ValueError:
        return _reply(handler, 400, {"error": "bad json"})
    try:
        if route == "link":
            url = check_url(str(body.get("url") or ""))
            return _reply(handler, 202, {"job": WORKER.submit(f"link {url}", lambda: _link_job(url)).public()})
        if route == "retry":
            item_id = body.get("id")
            if (
                not isinstance(item_id, int)
                or pipeline().store.get(item_id) is None
                or pipeline().store.removed(item_id)
            ):
                return _reply(handler, 404, {"error": "no such item"})
            return _reply(handler, 202, {"job": WORKER.submit(f"retry {item_id}", lambda: _process(item_id)).public()})
        if route == "delete":
            from liberational.ingest.analyze import Ledger
            from liberational.ingest.broker import BrokerRegistrar

            item_id = body.get("id")
            if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 1:
                return _reply(handler, 400, {"error": "send an item id"})
            p = pipeline()
            if body.get("poll") is True and not p.store.removed(item_id):
                return _reply(handler, 404, {"error": "no pending deletion"})
            try:
                result = p.delete(item_id, Ledger(p.store.path.parent / ANALYZED), BrokerRegistrar(), wait_s=0)
            except ingest_pipeline.UnsafeIngestPath as error:
                LOG.error("item %s: %s", item_id, error)
                p.store.deletion_progress(item_id, detail=str(error), blocked=True)
                return _reply(handler, 409, {"error": str(error), "permanent": True})
            except Exception:
                LOG.exception("item %s: deletion cleanup failed", item_id)
                p.store.deletion_progress(item_id, detail="Cleanup failed. Retry to finish deleting this item.")
                return _reply(handler, 503, {"error": "Cleanup failed. Retry to finish deleting this item."})
            return _reply(handler, 200, result)
        if route.startswith("sources/") and route.endswith("/poll"):
            name = route.removeprefix("sources/").removesuffix("/poll")
            if name not in src.SOURCES:
                return _reply(handler, 404, {"error": f"no source called {name}"})
            return _reply(handler, 202, {"job": WORKER.submit(f"poll {name}", lambda: _poll_job(name)).public()})
        if route == "google_photos/login":
            if not handler._can_control():
                return _reply(handler, 403, {"error": "the sign-in window opens only from this machine"})
            GooglePhotosSource.from_config().login()
            log("ingest: opened the Google Photos sign-in window")
            return _reply(handler, 202, {"ok": True})
        if route == "google_photos/import":
            items = remote_items(body.get("items"))
            what = f"Google Photos: {len(items)} items"
            return _reply(handler, 202, {"job": WORKER.submit(what, lambda: _import_job(items)).public()})
    except LinkError as e:
        return _reply(handler, 400, {"error": str(e)})
    except ValueError as e:
        return _reply(handler, 400, {"error": str(e)})
    return _reply(handler, 404, {"error": "not found"})


def _reply(handler: H, code: int, body: dict[str, Any]) -> bool:
    handler._json(body, code)
    return True
