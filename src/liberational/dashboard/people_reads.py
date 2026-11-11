"""The People tab's reads: the list of people, the Inbox's unnamed list, and why a line or a face has its name
(moved out of the server, #61). A person's summary and card are in people_cards.py.

GET /people                 everyone, with the cards most likely opened next read ahead
GET /people/references      a person's reference media, and GET /people/reference-media/<id> one file
GET /inbox/unnamed          the faces, voices, and lines waiting for a name
GET /identity/why           why a line or a recorded face has its name
GET /health                 the dashboard is up, and how far the People warm-up has got
"""

import copy
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from liberational import speech_log, voice_turn
from liberational.dashboard import (
    context,
    devreload,
    inbox_state,
    known_voices,
    live_status,
    people_cards,
    people_list,
    photo_files,
    roster_read,
    routes,
    transcript_lines,
    unnamed_groups,
    voice_clones,
)
from liberational.dashboard.context import LOG, SSE_CLIENTS
from liberational.people import face_shots, registry
from liberational.see.faces import VISIT_GAP_S
from liberational.vision import views

if TYPE_CHECKING:
    from liberational.dashboard.server import H

UNNAMED_FACES, UNNAMED_VOICES, UNNAMED_VIDEOS = 24, 12, 12  # how many of each the unnamed list offers


NEAR_WINDOW_S = 3600.0  # how far either side of a line its neighbours are read: a copy is seconds from it


def lines_near(e: dict[str, Any], n: int = 200) -> list[dict[str, Any]]:
    """The lines stored on each side of e in the transcript, where another mic's copy of it is: written a moment
    apart live, or together by a re-transcription. At most n each side, from an hour around it: a copy of a line is
    seconds from it, so there is no need to walk from the newest line down to an old one."""
    ts = e.get("ts_start")
    if not isinstance(ts, int | float):
        return []
    key = registry.key_of(e)
    window = transcript_lines.heard_lines(float(ts) - NEAR_WINDOW_S, float(ts) + NEAR_WINDOW_S)
    at = next((i for i, x in enumerate(window) if registry.key_of(x) == key), None)
    if at is None:
        return []
    older = list(reversed(window[:at]))[:n]  # nearest first, as the walk from the newest line gave them
    newer = window[at + 1 :][-n:]  # the n nearest of those after it, in time order
    return [*newer, *older]


def unknown_groups_view() -> tuple[int, dict[str, Any]]:
    """(code, reply) for GET /faces/unknown/groups: Live's "Who's this?" rows (vision/face_guess.py), best guess
    first, how many, and the groups put aside (so Live leaves them out when they come into view). Kept (_Fresh): new
    sightings come in within GROUPS_REFRESH_S, the owner's own changes at once."""
    return unnamed_groups._ready_face_groups(*unnamed_groups._FACE_GROUPS.get())


def unknown_voices_view() -> tuple[int, dict[str, Any]]:
    """(code, reply) for GET /voices/unknown/groups: the unnamed voices waiting for a name as "Who's this?" rows
    (people/voice_guess.py), best guess first, each with its most useful clips to play, and how many. The voices
    /people lists as unknown.voices (unnamed, a line left to play, not put aside) that have a clip worth playing, the
    24 heard most recently. Guessed against the named voices that are people, not voice clones made from one uploaded
    recording (the live matcher leaves those out too). Live polls this, and it reads the whole transcript,
    so the rows are kept (_Fresh): a line heard since is taken in in the background, the owner's own changes at
    once."""
    return unnamed_groups._VOICE_GROUPS.get()


CARD_WARM = 8  # cards read ahead when the people list is served: the first this many people in it


def _voice_letters(i: int) -> str:
    """0 -> "A", 25 -> "Z", 26 -> "AA": the letter an unnamed voice is called by in the pickers."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(65 + r) + out
    return out


def why_name_view(q: Mapping[str, str]) -> tuple[dict[str, Any], int]:
    """(reply, status) for GET /identity/why: ?line=<key> (or ?ts_start=) for a transcript line, ?shot=<sighting id>
    for a recorded face (dashboard/why_name.py)."""
    from liberational.dashboard import why_name

    if (shot := q.get("shot")) is not None:
        if not photo_files.SIGHTING_ID.fullmatch(shot):
            return {"error": "shot is a sighting id"}, 400
        row = why_name.shot_row(context.STATE, shot)
        if row is None:
            return {"error": "no such shot"}, 404
        reg = None
        if people_list._has_log():
            with registry.safely("why this name"):
                reg = registry.at(context.STATE)
        return why_name.for_shot(row, reg), 200
    if q.get("line") is None and q.get("ts_start") is None:
        return {"error": "send line (a line key) or shot (a sighting id)"}, 400
    try:
        e = _stored_line(q.get("ts_start"), q.get("line"))
    except AmbiguousLine as ex:
        return {"error": str(ex)}, 409
    except TypeError, ValueError:
        return {"error": "ts_start is a time"}, 400
    if e is None:
        return {"error": "no such line"}, 404
    return why_name.for_line(e, transcript_lines._view(), lines_near(e)), 200


def _newest_face(f: Mapping[str, Any]) -> tuple[float, int]:
    """Sort key for the unnamed faces: newest seen first, then most seen; never seen (no shot left) last."""
    seen = f.get("last_seen")
    return -(float(seen) if isinstance(seen, int | float) else 0.0), -int(f.get("count") or 0)


def _newest_lines() -> dict[int, dict[str, Any]]:
    """{voice id: its newest line with audio, as read through the event log}, from one backward read of the transcript
    (what speakers_with_audio and latest_for_speaker would take one read per voice for)."""
    out: dict[int, dict[str, Any]] = {}
    # Share the retained speech cache and resolve voices in one batch, rather than one registry call per line.
    for e in reversed(
        transcript_lines._view().individual_lines(e for e in speech_log.retained() if transcript_lines.has_audio(e))
    ):
        if transcript_lines.has_audio(e) and (sid := e.get("speaker_id")) is not None and sid not in out:
            out[sid] = e
    return out


def people_unnamed() -> dict[str, Any]:
    """GET /inbox/unnamed: what nobody is named for yet, for People's "pick what's theirs" (contract at the top).

    Failed face history leaves face groups and voice samples usable, with an `incomplete` message for Retry.
    """
    from liberational.people import face_photos
    from liberational.vision import sightings

    skips = inbox_state.inbox_skips()
    try:
        groups = [
            g for g in people_list._faces_mod().unknown_groups() if g["count"] >= 2 and g["id"] not in skips["face"]
        ]
    except Exception:
        groups = []
    # Both readers need the complete month, including named faces that exclude mixed camera runs.
    # Current names and corrections are projected afresh for each request.
    now = time.time()
    incomplete = None
    try:
        rows = sightings.faces(now - sightings.SHOT_DAYS * 86400, now + 1, state=context.STATE)
    except Exception as ex:
        LOG.warning("people: unnamed face history failed: %s", repr(ex))
        rows = []
        incomplete = "Could not load face photos and video clips."
    try:
        recent = face_photos.recent_by_group(state=context.STATE, rows=rows)
    except Exception as ex:
        LOG.warning("people: unnamed face shots failed: %s", repr(ex))
        recent = {}
    faces, named, shots_pending, shots_unavailable, saved_shots = (
        [],
        views.names(),
        0,
        0,
        face_shots.saved(context.STATE, revision=photo_files._shot_revision()),
    )
    for g in groups:
        seen = recent.get(g["id"], [])
        ready, count, gone = people_list._ready_faces(seen, saved_shots)
        shots_pending += count
        shots_unavailable += gone
        photos = [f"/faces/unknown/{g['id']}.jpg"] if photo_files._unknown_photo(g["id"]).exists() else []
        photos += [f"/faces/shot/{s['id']}.jpg" for s in ready]
        faces.append(
            {
                "id": g["id"],
                "photos": photos[:4],
                "count": g["count"],
                "last_seen": seen[0]["ts"] if seen else None,
                "cam": seen[0]["cam"] if seen else None,
                "camera_name": named.get(seen[0]["cam"], seen[0]["cam"]) if seen else None,
                **({"shots_pending": count} if count else {}),
                **({"shots_unavailable": gone} if gone else {}),
            }
        )
    faces.sort(key=_newest_face)
    newest = _newest_lines()
    unnamed = [
        s
        for s in known_voices._speakers()["speakers"]
        if people_list.VOICE_GROUP.fullmatch((s.get("label") or "").strip())
        and s["id"] in newest
        and s["id"] not in skips["voice"]
    ]
    unnamed.sort(key=lambda s: -(s.get("last_seen") or 0))
    voices = []
    for i, s in enumerate(unnamed[:UNNAMED_VOICES]):
        e = newest[s["id"]]
        sample = None
        if e and isinstance(e.get("ts_start"), int | float):
            sample = {"line": registry.key_of(e), "ts_start": e["ts_start"], "text": (e.get("text") or "").strip()}
        voices.append(
            {
                "id": s["id"],
                "label": f"Unnamed voice {_voice_letters(i)}",
                "lines": s.get("n", 0),
                "last_heard": s.get("last_seen"),
                "sample": sample,
            }
        )
    try:
        clips = sightings.unnamed_clips(VISIT_GAP_S, UNNAMED_VIDEOS, context.STATE, rows=rows)
    except Exception as ex:
        LOG.warning("people: unnamed clips failed: %s", repr(ex))
        clips = []
    videos = [
        {
            "id": k["id"],
            "cam": k["cam"],
            "camera_name": named.get(k["cam"], k["cam"]),
            "start": k["start"],
            "end": k["end"],
            "duration_s": round(k["end"] - k["start"], 1),
            "thumb": f"/faces/shot/{k['thumb']}.jpg" if k["thumb"] is not None and k["thumb"] in saved_shots else None,
            "what": people_cards._what(k["cam"], k["start"], k["end"]),
        }
        for k in clips
    ]
    videos, videos_pending, videos_unavailable = people_cards._ready_videos(videos)
    result = {"faces": faces[:UNNAMED_FACES], "voices": voices, "videos": videos}
    if videos_pending:
        result["videos_pending"] = videos_pending
    if videos_unavailable:
        result["videos_unavailable"] = videos_unavailable
    if shots_pending:
        result["shots_pending"] = shots_pending
    if shots_unavailable:
        result["shots_unavailable"] = shots_unavailable
    return {**result, "incomplete": incomplete} if incomplete else result


class AmbiguousLine(LookupError):
    """A time alone that names lines from more than one microphone: the caller has to send the line's key."""


AMBIGUOUS = "more than one line starts then; send its line"


def _stored_line(ts: object = None, line: object = None) -> dict[str, Any] | None:
    """A transcript line as stored: by its key (`line`, registry.key_of), else by its ts_start to the millisecond.
    Raises AmbiguousLine when the time alone matches lines from two microphones, and ValueError for a bad time.
    Reads the moments around it (transcript_lines.line_window), not the whole history."""
    if isinstance(line, str) and line:
        return transcript_lines.line_at(key=line)
    hits = transcript_lines.line_window(float(ts))  # ty: ignore[invalid-argument-type]  # a bad value is the caller's ValueError or TypeError
    if len({registry.key_of(e) for e in hits}) > 1:
        raise AmbiguousLine(AMBIGUOUS)
    return hits[0] if hits else None


@routes.get_when(
    "/people/references, /people/reference-media/<id>",
    lambda p: p == "/people/references" or p.startswith("/people/reference-media/"),
)
def _get_people_references(h: H) -> None:
    p = h.path.split("?")[0]
    from liberational.people import references

    try:
        if p == "/people/references":
            return h._json(
                {
                    "references": references.listing(
                        context.STATE,
                        h._query().get("person", ""),
                    )
                }
            )
        token = p.removeprefix("/people/reference-media/")
        meta = references.metadata(context.STATE, token)
        return h._media(
            references.entry(context.STATE, token) / meta["file"], meta["mime"], {"X-Content-Type-Options": "nosniff"}
        )
    except (ValueError, OSError) as exc:
        return h._json({"error": str(exc)}, 400)


# not under /people/, where it would be the page of someone named "unnamed"
@routes.get("/inbox/unnamed")
def _get_inbox_unnamed(h: H) -> None:
    return h._json(people_unnamed())


@routes.get("/people")
def _get_people(h: H) -> None:
    view = roster_read.people_view(cold=True)
    people_cards._CARD_READS.warm([str(person["name"]) for person in view.get("people", [])[:CARD_WARM]])
    return h._json(view)


# why a line or a recorded face has its name; read only
@routes.get("/identity/why")
def _get_identity_why(h: H) -> None:
    return h._json(*why_name_view(h._query()))


@routes.get("/health")
def _get_health(h: H) -> None:
    return h._json(
        {
            "ok": True,
            "ts": time.time(),
            "clients": SSE_CLIENTS[0],
            "boot": devreload.BOOT,
            "people_warmup": _people_warm_status(),
        }
    )


_PEOPLE_WARM_LOCK = threading.Lock()


_PEOPLE_WARMED = threading.Event()  # the people warm-up finished (done or incomplete)


_PEOPLE_WARM_STATUS: dict[str, Any] = {"state": "pending", "stage": None, "elapsed_s": None, "errors": []}


def _people_warm_status() -> dict[str, Any]:
    """Cheap readiness metadata for /health, without triggering any data or model read."""
    with _PEOPLE_WARM_LOCK:
        return copy.deepcopy(_PEOPLE_WARM_STATUS)


def _set_people_warm(
    state: str, stage: str | None, *, elapsed_s: float | None = None, errors: Sequence[str] = ()
) -> None:
    with _PEOPLE_WARM_LOCK:
        _PEOPLE_WARM_STATUS.update(state=state, stage=stage, elapsed_s=elapsed_s, errors=list(errors))


def _warm_people() -> None:
    """Warm first-page dependencies after serving starts, at low priority and through the bounded voice gate.

    Header inputs go first, then roster/clone/voice dependencies and raw inbox history. Face groups go last so their
    short refresh interval starts near completion. A successful read alone is not ready: raw windows must have
    published, and groups must have published for their current owner. Retry an unpublished snapshot once, then
    report incomplete rather than claim the first page is warm.
    This only warms readers/imports; it never builds a model or keeps a permanent roster response.
    """
    context.low_priority(19)
    started = time.monotonic()
    errors: list[str] = []
    stages: tuple[tuple[str, Callable[[], Any], Callable[[], bool] | None], ...] = (
        ("microphones", live_status.mic_status, None),
        ("scene", live_status.scene_entries, None),
        ("cameras", live_status.camera_status, None),
        ("roster", lambda: roster_read.people_view(wait_s=None), None),
        ("clones", voice_clones.clones_list, None),
        ("voice groups", unknown_voices_view, unnamed_groups._VOICE_GROUPS.ready),
        ("face groups", unknown_groups_view, unnamed_groups._FACE_GROUPS.ready),
    )
    for name, read, ready in stages:
        try:
            for attempt in range(2):
                _set_people_warm("waiting", name)
                voice_turn.wait_quiet()
                _set_people_warm("running", name)
                result = read()
                if isinstance(result, tuple) and result[0] != 200:
                    errors.append(f"{name}: HTTP {result[0]}")
                    break
                if isinstance(result, dict) and result.get("error"):
                    errors.append(f"{name}: unavailable")
                    break
                if result is not False and (ready is None or ready()):
                    break
                if attempt == 1:
                    errors.append(f"{name}: snapshot not published")
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}")
    elapsed = time.monotonic() - started
    _set_people_warm("incomplete" if errors else "done", None, elapsed_s=round(elapsed, 3), errors=errors)
    _PEOPLE_WARMED.set()
    if errors:
        LOG.warning("people: warm-up incomplete: %s", "; ".join(errors))
    else:
        LOG.info(f"people: warmed in {elapsed:.1f} s")
