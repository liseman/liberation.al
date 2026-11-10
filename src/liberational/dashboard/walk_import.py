"""Known › Places' walk import (#176): a walkthrough dropped on a place's panel is uploaded, checked, and imported
into that place in the background, as `pack-walkthrough <folder> --place <id>` does by hand.

    GET  /places/walk                    {job, max_mb, video}: the newest import (null with none), the largest upload
                                         taken (ingest.max_mb), and whether this machine takes a plain video
    POST /places/walk/upload?place=&name=
                                         the file itself as the body: the zip Stray Scanner exports, or a plain video
                                         (.mp4, .mov, .m4v; CUT3R's path, so not on macOS). Streamed to disk, checked,
                                         then imported in the background. {job} (202)
    POST /places/walk/retry              run a failed import again from the files it kept. {job} (202)
    POST /places/walk/dismiss            forget a finished import and remove what it kept. {job: null}

A job is {id, place, name, kind ("stray" or "video"), status, detail, error, result, retry, started, finished}.
`status` goes uploading, extracting, importing, then done or failed; `detail` is the importer's newest log line;
`result` is what pack-walkthrough printed ({place, revision, registered, meant_for, ...}: `meant_for` is there when
the walk didn't register and was stored as a place of its own); `retry` says a failed import kept its files.

One import at a time: an upload while another is uploading, extracting, or importing is refused (409) before its body
is read. The writes are teaching writes (`_can_teach`), same-origin, with a body type no cross-site page can send
without a preflight.

What is checked before anything is imported (`check_export`): the zip opens; it holds exactly one recording (a folder,
or the zip's root, with `rgb.mp4`, `odometry.csv`, `depth/*.png`, and intrinsics: `camera_matrix.csv` or per-frame
`fx` columns in the odometry), the layout spatial/stray.py reads; no entry is a link, absolute, or climbs out with
`..`; and its files' stated sizes fit the disk and `MAX_EXTRACTED`. Extraction (`extract`) writes only the files the
import reads (the video, the two CSVs, depth, and confidence), each to a path checked to be inside the job's folder.

Files (`paths.state/walk_imports/`): `job.json` (the newest job, rewritten on each change, so a page reload or a
dashboard restart still shows it) and `<job id>/` (the upload, then the extracted `<recording>/`). The rule for what
is kept:

- the zip is removed as soon as it's extracted or refused;
- the extracted recording (or the video) is removed when its import succeeds: the place has its cloud, plan, and
  splat by then, and the phone still has the recording;
- a failed import keeps it, so Try again needs no second upload, until the import is dismissed or the next upload
  starts, which removes it;
- an import cut off by a dashboard restart reads as failed and keeps its files the same way.

The import itself is `pack-walkthrough`'s `main` in a child process, niced (its model runs nice themselves further),
so the dashboard holds none of its memory and a crash there is a failed job, not a dead dashboard.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import logs
from liberational.dashboard import routes
from liberational.memory import places

if TYPE_CHECKING:
    from liberational.dashboard.server import H

LOG = logs.get("dashboard-walk-import")
Reply = tuple[int, dict[str, Any]]

ROOT = "walk_imports"  # under paths.state
UPLOAD_TYPE = "application/octet-stream"  # not a type a cross-site page can send without a preflight
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v")
ACTIVE = ("uploading", "extracting", "importing")
MAX_EXTRACTED = 16 << 30  # a 5 minute export is 1 to 3 GB; past this it isn't a recording
MAX_MEMBERS = 200_000  # two PNGs a frame at 60 frames a second is 36,000 for 5 minutes
DISK_MARGIN = 1 << 30  # free space left over after the upload and its extraction
NICE = 10
TIMEOUT_S = 3 * 3600.0  # an import is minutes; one still going after this is stuck
#: What a Stray Scanner export must hold, and what each is, for the refusal's message.
PIECES = {
    "rgb.mp4": "rgb.mp4 (the video)",
    "odometry.csv": "odometry.csv (the camera's poses)",
    "depth": "depth/ (the LiDAR depth frames)",
    "intrinsics": "camera_matrix.csv (the camera's intrinsics)",
}
_LOG_LINE = re.compile(r"^\S+ \[[^\]]+\] [A-Z]+ (.*)$")  # logs.Formatter's `<stamp> [tag] LEVEL message`
_RESULT_LINE = re.compile(r"^(\w+): (.*)$")  # pack-walkthrough's `key: value`
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class Refused(ValueError):
    """An upload that isn't a walk this can import; the message is for the page."""


@dataclass
class Job:
    id: str
    place: str
    name: str
    kind: str  # "stray" or "video"
    status: str = "uploading"
    detail: str = ""
    error: str = ""
    result: dict[str, str] = field(default_factory=dict)
    source: str = ""  # the extracted recording or the video, relative to the job's folder; "" before there is one
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def public(self, root: Path) -> dict[str, Any]:
        kept = bool(self.source) and (root / self.id / self.source).exists()
        out = {k: v for k, v in asdict(self).items() if k != "source"}
        return {**out, "retry": self.status == "failed" and kept}


@dataclass(frozen=True)
class Export:
    """A checked Stray Scanner zip: the recording's name, its folder inside the zip ("" at the root, else ending in
    "/"), and the entries the import reads."""

    name: str
    prefix: str
    members: tuple[zipfile.ZipInfo, ...]


def safe_name(name: str, fallback: str = "walk") -> str:
    """A file or folder name made only of letters, digits, dots, dashes, and underscores, with no leading dot."""
    return _UNSAFE_NAME.sub("-", name).strip("-.")[:64] or fallback


def _unsafe(info: zipfile.ZipInfo) -> str | None:
    """Why a zip entry can't be trusted with a path, or None: a link, an absolute path, or one that climbs out."""
    name = info.filename
    if stat.S_ISLNK(info.external_attr >> 16):
        return "is a link"
    if name.startswith(("/", "\\")) or "\\" in name or re.match(r"^[A-Za-z]:", name):
        return "has an absolute path"
    if ".." in PurePosixPath(name).parts:
        return "points outside the folder"
    return None


def _wanted(rel: str) -> bool:
    """Whether the import reads this file of an export (a path inside the recording's folder)."""
    if rel in ("rgb.mp4", "odometry.csv", "camera_matrix.csv"):
        return True
    parts = rel.split("/")
    return len(parts) == 2 and parts[0] in ("depth", "confidence") and parts[1].endswith(".png")


def check_export(path: Path, fallback_name: str = "walk") -> Export:
    """A zip checked to be one Stray Scanner export (the module docstring says what that means), or `Refused` with
    what is wrong. Only the zip's directory and its odometry header are read."""
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        raise Refused("That file isn't a zip. Drop the zip Stray Scanner exports.") from e
    with archive:
        infos = [i for i in archive.infolist() if not i.is_dir() and not i.filename.startswith("__MACOSX/")]
        if len(infos) > MAX_MEMBERS:
            raise Refused(f"That zip has {len(infos):,} files, too many to be one recording.")
        for info in infos:
            why = _unsafe(info)
            if why is not None:
                raise Refused(f"That zip isn't safe to open: an entry {why}.")
        names = {i.filename for i in infos}

        def pieces(prefix: str) -> set[str]:
            found = {p for p in ("rgb.mp4", "odometry.csv") if prefix + p in names}
            if any(n.startswith(prefix + "depth/") and n.endswith(".png") for n in names):
                found.add("depth")
            if prefix + "camera_matrix.csv" in names:
                found.add("intrinsics")
            return found

        # A recording's folder is wherever one of its two named files is: the zip's root, or a folder in it.
        folders = sorted(
            {n[: -len(p)] for n in names for p in ("rgb.mp4", "odometry.csv") if PurePosixPath(n).name == p}
        )
        if not folders:
            raise Refused(
                "That zip isn't a Stray Scanner export: it has no rgb.mp4 and no odometry.csv. "
                "Export the recording from the app and drop that zip."
            )
        if len(folders) > 1:
            raise Refused("That zip holds more than one recording. Drop one recording's zip at a time.")
        prefix = folders[0]
        found = pieces(prefix)
        if "intrinsics" not in found and "odometry.csv" in found:
            with archive.open(prefix + "odometry.csv") as f:
                header = f.readline(4096).decode("utf-8", "replace")
            if "fx" in [h.strip() for h in header.split(",")]:
                found.add("intrinsics")
        missing = [said for piece, said in PIECES.items() if piece not in found]
        if missing:
            raise Refused(f"That zip isn't a whole Stray Scanner export. It is missing: {'; '.join(missing)}.")
        members = tuple(i for i in infos if i.filename.startswith(prefix) and _wanted(i.filename[len(prefix) :]))
        size = sum(i.file_size for i in members)
        if size > MAX_EXTRACTED:
            raise Refused(
                f"That recording unpacks to {size / 2**30:.1f} GB, more than the {MAX_EXTRACTED >> 30} GB taken."
            )
        inner = PurePosixPath(prefix).name if prefix else ""
        return Export(safe_name(inner or fallback_name), prefix, members)


def extract(path: Path, export: Export, into: Path) -> Path:
    """The export's files written under `into/<export.name>/`, each streamed and each at a path checked to be inside
    that folder; returns the folder. `OSError` when the disk can't hold it."""
    folder = into / export.name
    folder.mkdir(parents=True, exist_ok=True)
    inside = folder.resolve()
    need = sum(i.file_size for i in export.members)
    if shutil.disk_usage(into).free < need + DISK_MARGIN:
        raise OSError(f"not enough free disk space to unpack {need / 2**30:.1f} GB")
    with zipfile.ZipFile(path) as archive:
        for info in export.members:
            dest = (folder / info.filename[len(export.prefix) :]).resolve()
            if not dest.is_relative_to(inside):  # check_export refused these; never trust one check
                raise Refused("That zip isn't safe to open: an entry points outside the folder.")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
    return folder


def command(source: Path, place: str) -> list[str]:
    """pack-walkthrough on `source` into `place`, niced, in this interpreter."""
    run = [sys.executable, "-c", "from liberational.spatial.walkthrough import main; main()", str(source)]
    nice = shutil.which("nice")
    return [*([nice, "-n", str(NICE)] if nice else []), *run, "--place", place]


def run_import(source: Path, place: str, progress: Callable[[str], None]) -> dict[str, str]:
    """Run the import as a child process and return what it printed (`key: value` lines); its log lines go to
    `progress` as they come. `RuntimeError` with its own message when it fails."""
    with subprocess.Popen(
        command(source, place),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        errors="replace",
    ) as child:
        killer = threading.Timer(TIMEOUT_S, child.kill)
        killer.daemon = True
        killer.start()
        result: dict[str, str] = {}
        last = ""
        try:
            for raw in child.stdout or ():
                line = raw.strip()
                if not line:
                    continue
                last = line
                if (m := _LOG_LINE.match(line)) is not None:
                    progress(m[1])
                elif (m := _RESULT_LINE.match(line)) is not None:
                    result[m[1]] = m[2]
            code = child.wait()
        finally:
            killer.cancel()
    if code != 0:
        raise RuntimeError(last.removeprefix("pack-walkthrough: ") or f"the import stopped (exit {code})")
    if not result.get("place"):
        raise RuntimeError("the import finished without saying which place it wrote")
    return result


class Importer:
    """One state directory's walk imports: the newest job, kept in `job.json`, and the thread running it."""

    def __init__(self, state: Path) -> None:
        self.root = state / ROOT
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.job = self._load()

    def _load(self) -> Job | None:
        try:
            raw = json.loads((self.root / "job.json").read_text())
            job = Job(**raw)
        except OSError, ValueError, TypeError:
            return None
        if job.status in ACTIVE:  # nothing is running it any more: the dashboard restarted
            job.status, job.finished = "failed", time.time()
            job.error = "The dashboard restarted during this import."
            self._save(job)
        return job

    def _save(self, job: Job | None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "job.json"
        if job is None:
            path.unlink(missing_ok=True)
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(job)))
        os.replace(tmp, path)

    def view(self) -> dict[str, Any] | None:
        with self.lock:
            return self.job.public(self.root) if self.job is not None else None

    def _set(self, job: Job, **change: Any) -> None:
        with self.lock:
            for k, v in change.items():
                setattr(job, k, v)
            if self.job is job:
                self._save(job)

    def _clear_files(self, keep: str | None = None) -> None:
        """Remove every job folder but `keep`'s."""
        for child in self.root.iterdir() if self.root.is_dir() else ():
            if child.is_dir() and child.name != keep:
                shutil.rmtree(child, ignore_errors=True)

    def begin(self, place: str, name: str, kind: str) -> Job | None:
        """Take the one slot for a new upload (clearing the last job's files), or None while another is active."""
        with self.lock:
            if self.job is not None and self.job.status in ACTIVE:
                return None
            job = Job(uuid.uuid4().hex, place, name, kind)
            self.job = job
            self._clear_files()
            (self.root / job.id).mkdir(parents=True, exist_ok=True)
            self._save(job)
            return job

    def refuse(self, job: Job) -> None:
        """Forget an upload that was refused, with its files."""
        with self.lock:
            shutil.rmtree(self.root / job.id, ignore_errors=True)
            if self.job is job:
                self.job = None
                self._save(None)

    def start(self, job: Job, work: Callable[[Job], Path]) -> None:
        """Run `work` (which readies the job's source and returns it) and then the import, on a thread."""
        self.thread = threading.Thread(target=self._run, args=(job, work), name="walk-import", daemon=True)
        self.thread.start()

    def _run(self, job: Job, ready: Callable[[Job], Path]) -> None:
        folder = self.root / job.id
        try:
            source = ready(job)
            self._set(job, status="importing", detail="Starting the import", source=str(source.relative_to(folder)))
            LOG.info("importing the walk %s into %s", job.name, job.place)
            result = run_import(source, job.place, lambda line: self._set(job, detail=line[:300]))
        except Exception as e:  # the job says what went wrong, whatever it was
            LOG.warning("walk import of %s failed: %s", job.name, e)
            self._set(job, status="failed", error=str(e) or type(e).__name__, finished=time.time())
            return
        shutil.rmtree(folder, ignore_errors=True)  # the place has what it needs
        LOG.info("walk %s imported: place %s revision %s", job.name, result.get("place"), result.get("revision"))
        self._set(job, status="done", detail="", result=result, finished=time.time())

    def unpack(self, upload: Path, export: Export) -> Callable[[Job], Path]:
        """The work that unpacks a checked zip, then removes it."""

        def work(job: Job) -> Path:
            self._set(job, status="extracting", detail="Unpacking the recording")
            try:
                return extract(upload, export, upload.parent)
            except Exception:
                shutil.rmtree(upload.parent, ignore_errors=True)  # a half-unpacked recording is no use to a retry
                raise
            finally:
                upload.unlink(missing_ok=True)

        return work

    def retry(self) -> Job | None:
        """Run a failed import again from the files it kept; None when there is nothing to run."""
        with self.lock:
            job = self.job
            if job is None or job.status != "failed" or not job.source:
                return None
            source = self.root / job.id / job.source
            if not source.exists():
                return None
            job.status, job.error, job.detail, job.finished = "importing", "", "Starting the import", None
            job.started = time.time()
            self._save(job)
        self.start(job, lambda _: source)
        return job

    def dismiss(self) -> bool:
        """Forget a finished job and remove what it kept; False while one is active."""
        with self.lock:
            if self.job is not None and self.job.status in ACTIVE:
                return False
            self.job = None
            self._clear_files()
            self._save(None)
            return True


_HELD: dict[Path, Importer] = {}
_HELD_LOCK = threading.Lock()


def importer(state: Path) -> Importer:
    with _HELD_LOCK:
        if state not in _HELD:
            _HELD[state] = Importer(state)
        return _HELD[state]


def takes_video() -> bool:
    """Whether a plain video can be imported here: CUT3R's path needs CUDA, so not on macOS (pack-walkthrough)."""
    return sys.platform != "darwin"


def max_bytes() -> int:
    return int(cfg.get("ingest.max_mb")) << 20


def view(state: Path) -> Reply:
    """GET /places/walk."""
    return 200, {"job": importer(state).view(), "max_mb": max_bytes() >> 20, "video": takes_video()}


def _receive(chunks: Iterable[bytes], dest: Path) -> None:
    with dest.open("wb") as f:
        for block in chunks:
            f.write(block)


def _body(handler: H, length: int) -> Iterable[bytes]:
    """The request's body in blocks, never all in memory."""
    handler.connection.settimeout(60)
    left = length
    while left:
        block = handler.rfile.read(min(left, 1 << 20))
        if not block:
            raise ValueError("Upload interrupted. Please retry.")
        left -= len(block)
        yield block


def upload(handler: H, state: Path, place: str, name: str) -> Reply:
    """POST /places/walk/upload: every refusal that needs no body comes first, with the body left unread; then the
    body is streamed to the job's folder, a zip is checked, and the import starts."""
    name = Path(name).name or "walk"
    suffix = Path(name).suffix.lower()
    kind = "stray" if suffix == ".zip" else "video" if suffix in VIDEO_SUFFIXES else None
    limit = max_bytes()
    try:
        length = int(handler.headers.get("Content-Length") or 0)
    except ValueError:
        length = -1
    refusal: Reply | None = None
    if places.get(state, place) is None:
        refusal = 404, {"error": "no such place"}
    elif kind is None:
        refusal = 415, {"error": "Drop the zip Stray Scanner exports, or a video of the walk (.mp4, .mov, or .m4v)."}
    elif kind == "video" and not takes_video():
        refusal = 415, {"error": "A plain video can't be imported on a Mac. Drop a Stray Scanner export's zip."}
    elif not 0 < length <= limit:
        refusal = 413, {"error": f"Choose a nonempty file up to {limit >> 20} MB."}
    if refusal is not None or kind is None:
        handler.close_connection = True
        return refusal or (400, {"error": "bad upload"})
    held = importer(state)
    held.root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(held.root).free < length + DISK_MARGIN:
        handler.close_connection = True
        return 507, {"error": "Not enough free disk space for that upload."}
    job = held.begin(place, name, kind)
    if job is None:
        handler.close_connection = True
        return 409, {"error": "Another walk is being imported. Wait for it to finish, then drop this one."}
    folder = held.root / job.id
    dest = folder / ("upload.zip" if kind == "stray" else safe_name(name, "walk" + suffix))
    try:
        _receive(_body(handler, length), dest)
        if kind == "stray":
            work = held.unpack(dest, check_export(dest, Path(name).stem))
        else:
            work = lambda _: dest
    except Refused as e:
        held.refuse(job)
        return 400, {"error": str(e)}
    except (ValueError, OSError) as e:
        handler.close_connection = True
        held.refuse(job)
        return 400, {"error": str(e)}
    held.start(job, work)
    return 202, {"job": job.public(held.root)}


# ---- routes (dashboard/routes.py) ----


@routes.get("/places/walk")
def _get_places_walk(h: H) -> None:
    code, reply = view(h.state)
    return h._json(reply, code)


def _may_write(h: H, content_type: str) -> Reply | None:
    """Why a write is refused before its body is read, or None."""
    origin, host = h.headers.get("Origin"), h.headers.get("Host") or ""
    if h.headers.get_content_type() != content_type or (origin and origin not in (f"http://{host}", f"https://{host}")):
        return 403, {"error": "same-origin requests only"}
    if not h._can_teach(content_type):
        return 403, {"error": "a walk can only be imported from this machine (Remote teaching is off)"}
    return None


@routes.post_raw("/places/walk/upload")
def _post_places_walk_upload(h: H) -> None:
    refused = _may_write(h, UPLOAD_TYPE)
    if refused is not None:
        h.close_connection = True  # its body is never read
        return h._json(refused[1], refused[0])
    q = h._query()
    code, reply = upload(h, h.state, q.get("place", ""), q.get("name", ""))
    return h._json(reply, code)


@routes.post("/places/walk/retry")
def _post_places_walk_retry(h: H, body: Any) -> None:
    refused = _may_write(h, "application/json")
    if refused is not None:
        return h._json(refused[1], refused[0])
    held = importer(h.state)
    job = held.retry()
    if job is None:
        return h._json({"error": "There is no failed import with its files kept to try again."}, 409)
    return h._json({"job": job.public(held.root)}, 202)


@routes.post("/places/walk/dismiss")
def _post_places_walk_dismiss(h: H, body: Any) -> None:
    refused = _may_write(h, "application/json")
    if refused is not None:
        return h._json(refused[1], refused[0])
    if not importer(h.state).dismiss():
        return h._json({"error": "That import is still running."}, 409)
    return h._json({"job": None})
