"""The People list as one read (split from the People reads, #120): everyone, worked out in a worker process, kept
until its sources change, and answered from the last read while a new one is made (`people_view`). No routes: the
list, the cards, and the writes all read people through this.
"""

import copy
import multiprocessing
import os
import pathlib
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from types import EllipsisType
from typing import Any

from liberational import config as cfg
from liberational import logs, memtrim, speech_log, vision_log
from liberational.dashboard import (
    context,
    inbox_state,
    known_voices,
    people_list,
    photo_files,
    transcript_lines,
    unnamed_groups,
)
from liberational.dashboard.context import log
from liberational.people import face_shots, registry, transcript_view

try:
    import numpy as np
except ImportError:
    np = None


def mtime(p: pathlib.Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return 0


class Starting(Exception):
    """A route can't answer yet but will shortly (the people list's first read): the handler answers 503 with
    `starting`, which the frontend waits out."""


PEOPLE_WAIT_S = 2.0  # a /people request waits this long for its read, then answers from the last one


# While only speech moved the key, the last roster read is refreshed at most this often. A read is about a
# second of Python holding the GIL, and every request after a speech commit started one (every 30 to 60 s). A name,
# a contact, a skip, or a purge still reads at once: it moves `fresh`.
PEOPLE_REFRESH_S = 120.0


class _PeopleReads:
    """Share only overlapping roster computations. A read runs on its own thread, so a request that can't wait for
    it (a cold start's 7 to 30 s) answers from the last roster read, marked `stale`, or raises Starting when
    there is none yet; the read goes on and the next request has it.

    `fresh` is the part of the key a person's own change moves (a name given, a contact added). While it matches the
    last read, a request answers from that read at once and one read at a time refreshes it in the background (a
    later request starts the next), for new speech and for what the key doesn't cover, such as new unknown faces.
    Speech moves the key every few seconds, and relabeling the transcript for every request made each one wait.
    A changed `fresh` still waits for its own read, so an edit shows in the answer to it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pending: dict[tuple[Any, ...], Future[dict[str, Any]]] = {}
        self.last: dict[str, Any] | None = None
        self.last_fresh: tuple[Any, ...] | None = None
        self.cold_read: Future[dict[str, Any]] | None = None  # the shared read before any has finished (`cold`)
        self.last_at = 0.0  # time.monotonic() when the last read finished

    def get(
        self,
        key: tuple[Any, ...],
        read: Callable[[], dict[str, Any]],
        wait_s: float | None = None,
        fresh: tuple[Any, ...] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            last = self.last if fresh is not None and self.last_fresh == fresh else None
            future = self.pending.get(key)
            recent = last is not None and time.monotonic() - self.last_at < PEOPLE_REFRESH_S
            if future is None and (last is None or not self.pending) and not recent:
                future = self.pending[key] = Future()
                threading.Thread(
                    target=self._read, args=(key, fresh, future, read), name="people-read", daemon=True
                ).start()
        if last is not None:
            return copy.deepcopy(last)
        assert future is not None  # only a reusable last read skips starting one
        try:
            return copy.deepcopy(future.result(wait_s))
        except TimeoutError:
            if future.done():
                raise  # the read itself timed out (a broker timeout): not ours to hide
            with self.lock:
                last = self.last
            if last is None:
                raise Starting("The people list is still loading.") from None
            return {**copy.deepcopy(last), "stale": True}

    def peek(self, fresh: tuple[Any, ...]) -> dict[str, Any] | None:
        """The last read while `fresh` still matches it, starting no refresh: for a page that shows one row of it (a
        person's), which the list's own requests keep fresh. Shared, not copied: read it, never change it."""
        with self.lock:
            return self.last if self.last is not None and self.last_fresh == fresh else None

    def cold(self, read: Callable[[], dict[str, Any]], wait_s: float) -> dict[str, Any] | None:
        """Before any read has finished (a cold start), answer without working out the key: building it
        reads the registry, which the start's warm-ups keep busy for seconds, so "still loading" took 6 to 10 s.
        One background `read` is shared; it answers when it finishes within `wait_s`, else Starting. None once a
        read has finished: the key's normal path applies from then on."""
        with self.lock:
            if self.last is not None:
                return None
            if self.cold_read is None or self.cold_read.done() and self.cold_read.exception() is not None:
                future: Future[dict[str, Any]] = Future()
                self.cold_read = future

                def run() -> None:
                    try:
                        future.set_result(read())
                    except BaseException as error:
                        future.set_exception(error)

                threading.Thread(target=run, name="people-cold", daemon=True).start()
            future = self.cold_read
        try:
            return copy.deepcopy(future.result(wait_s))
        except TimeoutError:
            if future.done():
                raise  # the read itself timed out (a broker timeout): not ours to hide
            raise Starting("The people list is still loading.") from None

    def _read(
        self,
        key: tuple[Any, ...],
        fresh: tuple[Any, ...] | None,
        future: Future[dict[str, Any]],
        read: Callable[[], dict[str, Any]],
    ) -> None:
        # Out of `pending` before the result is set: a request that sees it done must not join a finished read.
        try:
            value = read()
        except BaseException as error:
            with self.lock:
                del self.pending[key]
            future.set_exception(error)
        else:
            with self.lock:
                del self.pending[key]
                self.last, self.last_fresh, self.last_at = value, fresh, time.monotonic()
            future.set_result(value)


_PEOPLE_READS = _PeopleReads()


def people_view(wait_s: float | None | EllipsisType = ..., *, cold: bool = False) -> dict[str, Any]:
    """GET /people. While nobody's names, contacts, labels, or skips have changed and nothing was purged since the
    last read, it answers from that read and refreshes it in the background; otherwise it waits up to `wait_s`
    (PEOPLE_WAIT_S when left out, None waits it out) for a read of the changed inputs, then answers from the last one
    (_PeopleReads). With `cold` (the plain GET only), before any read has finished it answers without working out
    the key (`_PeopleReads.cold`); a change's own answer (a name, a skip, a removal) never does, so it can't be the
    roster from before the change."""
    wait = PEOPLE_WAIT_S if isinstance(wait_s, EllipsisType) else wait_s
    if cold and wait is not None and (got := _PEOPLE_READS.cold(lambda: people_view(wait_s=None), wait)) is not None:
        return got
    commits, purge = speech_log.shared().mark()
    fresh = _people_fresh(purge)
    return _PEOPLE_READS.get((*fresh, commits), _people_view_worked_out, wait, fresh)


_people_executor: ProcessPoolExecutor | None = None


_people_lock = threading.Lock()


def _people_view_job() -> dict[str, Any]:
    """The people list's full read, in its worker process (_people_view_worked_out)."""
    if multiprocessing.parent_process() is not None:
        cfg.load.cache_clear()  # a setting saved since: the dashboard rereads its pack.toml, and so does this
        context.low_priority(10)
        memtrim.start()  # what a refresh freed goes back to the system
    return _people_view()


def _people_view_worked_out() -> dict[str, Any]:
    """The people list's full read (_people_view), worked out in one process of its own, as the face and
    voice groups are. It relabels the retained speech for the roster's counts and the unnamed voices' clips: 1.6 to 4 s
    of pure Python every PEOPLE_REFRESH_S while a page polls /people, holding the interpreter lock every request
    needs. _PeopleReads stays in the dashboard, so a person's own change still waits for its own read. A worker that
    died, or ran past GROUPS_WORKER_TIMEOUT_S, is replaced on the next read, and this one fails as a read does."""
    global _people_executor
    if not unnamed_groups._groups_in_worker:
        return _people_view()
    with _people_lock:
        if _people_executor is None:
            _people_executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        executor = _people_executor
    try:
        return executor.submit(_people_view_job).result(timeout=unnamed_groups.GROUPS_WORKER_TIMEOUT_S)
    except (BrokenProcessPool, TimeoutError) as error:
        with _people_lock:
            if _people_executor is executor:
                _people_executor = None
        for process in list((getattr(executor, "_processes", None) or {}).values()):
            process.terminate()  # one still running past the timeout
        executor.shutdown(wait=False, cancel_futures=True)
        log("people: the people list worker stopped:", repr(error), level=logs.WARN)
        raise RuntimeError("The people list could not be read.") from error


def _people_fresh(purge: object) -> tuple[Any, ...]:
    """The part of the people list's key a person's own change moves (_PeopleReads)."""
    # A purge is fresh too: what someone deleted never comes back from the last read.
    return (
        unnamed_groups._voices_key(),
        unnamed_groups._faces_owner_key(groups=False),
        purge,
        str(people_list.CONTACTS_FILE),  # which state this is: a different one never answers from another's read
        mtime(people_list.CONTACTS_FILE),
        mtime(people_list.PEOPLE_ADDED),
        mtime(inbox_state.INBOX_SKIPS),
        cfg.owner_name(),
        cfg.owner_label(),
        cfg.assistant_name(),
    )


def _people_view() -> dict[str, Any]:
    """GET /people: the roster, plus voices and faces nobody has named yet (skipped ones left out)."""
    raw, view = speech_log.retained(), transcript_lines._view()
    skips = inbox_state.inbox_skips()
    sp = known_voices._speakers()["speakers"]
    # Only voices with a line left to play: the backfill and refine move lines to named people, which can leave an
    # unnamed voice with nothing behind it.
    audible = unnamed_groups.speakers_with_audio(raw, view)
    voices = [
        {"id": s["id"], "label": s["label"], "n": s.get("n", 0), "last_seen": s.get("last_seen")}
        for s in sp
        if people_list.VOICE_GROUP.fullmatch((s.get("label") or "").strip())
        and s["id"] in audible
        and s["id"] not in skips["voice"]
    ]
    try:
        faces = [
            {**g, "label": f"Face {g['id']}"}
            for g in people_list._faces_mod().unknown_groups()
            if g["count"] >= 2 and g["id"] not in skips["face"]
        ][:24]
    except Exception:
        faces = []
    voices.sort(key=lambda v: -(v["last_seen"] or 0))
    _photos_from_scene()
    try:
        from liberational.people import face_photos

        face_photos.refresh_async()  # photos for unnamed faces cut from camera frames, in the background
    except Exception as ex:
        log("face photos: not refreshed:", logs.reason(ex), level=logs.WARN)
    for f in faces:
        f["photo"] = _photo_url(photo_files._unknown_photo(f["id"]), f"/faces/unknown/{f['id']}.jpg")
    clips, total = unnamed_groups.voice_clips(voices, raw=raw, view=view)
    return {
        "people": people_list.people_roster(raw, view, sp),
        "unknown": {"voices": voices, "faces": faces, "voice_clips": clips, "voice_clip_total": total},
    }


MIN_PHOTO_PX = 24  # smaller faces (frame pixels) are usually photos on a wall, not someone in the room


def _photo_url(path: pathlib.Path, url: str) -> str | None:
    try:
        return f"{url}?v={path.stat().st_mtime_ns}"
    except OSError:
        return None


def _frame_box(
    frame: Any, bbox_norm: Sequence[float], metadata: dict[str, Any] | None = None
) -> tuple[float, float, float, float]:
    """A box given as shares of the frame, in the frame's pixels."""
    h, w = frame.shape[:2]
    from liberational.see.frames import source_box_to_decoded

    inside = face_shots.in_frame(bbox_norm)  # a face at the frame's edge has a box past it
    if inside is None:
        raise ValueError("The box is outside the frame")
    box = source_box_to_decoded(inside, metadata) if metadata is not None else inside
    if box is None:
        raise ValueError("The source frame geometry is unavailable")
    x1, y1, x2, y2 = (float(v) for v in box)
    return x1 * w, y1 * h, x2 * w, y2 * h


def _crop(cam: str, ts: float, bbox_norm: Sequence[float], dest: pathlib.Path, size: int = 192) -> bool:
    """Square face crop of a face found on `cam` at `ts` (its box as shares of the frame), cut from that frame on the
    vision log and saved as a JPEG. False when the frame is gone, its geometry is unknown, or the face is too small."""
    from PIL import Image

    try:
        evidence = vision_log.at(context.STATE).frame_evidence(cam, ts)
        frame = evidence.image if evidence is not None else None
    except Exception as e:
        log("faces: the frame couldn't be read:", e, level=logs.WARN)
        return False
    if evidence is None or frame is None:
        return False
    try:
        x1, y1, x2, y2 = _frame_box(frame, bbox_norm, evidence.metadata)
    except ValueError:
        return False  # the box can't be placed on this frame

    side = min(x2 - x1, y2 - y1)
    if side < MIN_PHOTO_PX:
        return False
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box = (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))
    img = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))  # ty: ignore[unresolved-attribute]  # numpy is a declared dependency
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name("." + dest.name)
    img.crop(box).resize((size, size), Image.Resampling.LANCZOS).save(tmp, "JPEG", quality=88)
    os.replace(tmp, dest)
    return True


_scene_photos = {"t": 0.0}


def _living_name(view: transcript_view.View, name: str) -> bool:
    """Is `name` what a living person is called now, or a name the registry never had (a face tagged before the
    backfill)? False for the old name of someone renamed, and for someone forgotten."""
    if (person := view.person_at(name)) is not None:
        return registry.norm(person.display_name) == registry.norm(name)
    return not view.forgotten_at(name)


def _photos_from_scene() -> None:
    """Crops for faces in view now: named people with no photo yet, and unnamed groups (refreshed each minute). Only
    faces found in the last 30 seconds count, so a face named for someone deleted or renamed since gets no photo."""
    if time.time() - _scene_photos["t"] < 10:
        return
    _scene_photos["t"] = time.time()
    now, done, view = time.time(), set(), None
    for f in reversed(people_list._live_faces(30, now)):
        n, bbox = f["label"].strip(), f.get("bbox_norm")
        if not bbox or people_list._norm(n) in done or people_list._norm(n) == "unknown":
            continue
        done.add(people_list._norm(n))
        cam = str(f.get("cam") or "")
        try:
            if people_list.FACE_GROUP.fullmatch(people_list._norm(n)):
                p = photo_files._unknown_photo(people_list._norm(n).split()[1])
                if not p.exists() or now - p.stat().st_mtime > 60:
                    _crop(cam, f["ts"], bbox, p)
            elif not photo_files._photos(n):
                view = view or transcript_lines._view()
                if _living_name(view, n):
                    _crop(cam, f["ts"], bbox, photo_files._photo_dir(n) / f"{int(now * 1000)}.jpg")
        except OSError as ex:
            log("people: photo crop failed:", ex, level=logs.ERROR)
