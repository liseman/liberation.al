"""A person's card and summary (split from the People reads, #120): their best photos, recent video and voice, where
they are now, and what's remembered about them, each card kept and refreshed in the background (`_CardReads`).

GET /people/<name>       a person's summary
GET /people/<name>/card  a person's contact card
"""

import copy
import datetime as dt
import re
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Collection
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import logs, speech_log
from liberational.dashboard import (
    call_log,
    context,
    people_list,
    photo_files,
    roster_read,
    routes,
    transcript_lines,
    unnamed_groups,
)
from liberational.dashboard.context import log
from liberational.dashboard.unnamed_groups import CARD_SHOTS, CARD_VIDEOS, CARD_VOICES
from liberational.memory import remembered
from liberational.people import card_prefs, face_shots, registry, transcript_view, video_thumbs
from liberational.vision import views

if TYPE_CHECKING:
    from liberational.dashboard.server import H


MEMORY_FILE = context.STATE / "memory.jsonl"


def _ready_videos(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    revision = photo_files._shot_revision()
    saved = video_thumbs.saved(context.STATE, revision)
    ready, pending, unavailable = [], 0, 0
    for row in rows:
        status, thumb = unnamed_groups._video_status(row, revision, saved)
        if status == "ready":
            ready.append({**row, "thumb": thumb})
        pending += status == "pending"
        unavailable += status == "unavailable"
    return ready, pending, unavailable


def _roster_row(name: str) -> dict[str, Any] | None:
    """Their row of the people list, from the list's last read: opening a person reads the roster GET /people already
    answered from, the way a pet's page reads the Pets list, instead of building a roster of its own (2 to 10 s on a
    busy machine), and starts no refresh of it. After a change to names or contacts, it waits for the list's
    own read of the change (people_view); before any read has finished, it builds one."""
    last = roster_read._PEOPLE_READS.peek(roster_read._people_fresh(speech_log.shared().mark()[1]))
    if last is not None:
        row = next((p for p in last["people"] if people_list._norm(p["name"]) == people_list._norm(name)), None)
        return copy.deepcopy(row)
    try:
        people = roster_read.people_view()["people"]
    except roster_read.Starting:
        people = people_list.people_roster()
    return next((p for p in people if people_list._norm(p["name"]) == people_list._norm(name)), None)


def person_summary(name: str) -> dict[str, Any] | None:
    """GET /people/<name>: their row of the people list (voices, faces, number, role, last heard and seen), how many
    lines they said (in the room and on calls), and what memory notes say about them."""
    if not call_log._safe_name(name):
        return None
    me = _roster_row(name)
    who = {
        k: me[k]
        for k in (
            "voices",
            "faces",
            "number",
            "owner",
            "enrolled",
            "last_heard",
            "last_seen",
            "last_seen_pending",
            "last_seen_error",
            "role",
            "organization",
            "aliases",
        )
        if me and k in me
    }
    if not me or not me.get("lines"):
        return {"name": me["name"] if me else name, "lines": 0, **who}
    memory = [text for text in remembered.items(MEMORY_FILE) if people_list._norm(name) in text.lower()]
    return {
        **who,
        "name": me["name"],
        "lines": me["lines"],
        "room_lines": me.get("room", 0),
        "call_lines": me.get("calls", 0),
        "memory": memory[-8:],
        "note": f"Built only from transcripts and notes {cfg.assistant_name()} has. Public-web lookups are not run automatically.",
    }


def _what(cam: str, start: float, end: float, name: str | None = None) -> str | None:
    """A short label for what happens in a clip ("Sitting on a chair, hand over mouth, glasses on."), from the
    camera's caption, with no subject and no name (memory/clip_labels.py). Only a cached one: an uncached clip is
    queued for a background rewrite and gets None now, so the card never waits on the model."""
    from liberational.memory import clip_labels

    try:
        return clip_labels.at(context.STATE).get(cam, start, end, name)
    except Exception as ex:  # a label is a nicety: the card shows the time and place without one
        log("card: clip label failed:", repr(ex), level=logs.WARN)
        return None


def _review_link(name: str, ts: float) -> str:
    """Review at that moment with the person picked, in the hash Review's url.ts reads."""
    q = urllib.parse.urlencode(
        {"person": name, "day": dt.datetime.fromtimestamp(ts).date().isoformat(), "t": f"{ts:.2f}"},
        quote_via=urllib.parse.quote,
    )
    return "/review#" + q


def _recent_voice(
    name: str,
    limit: int = CARD_VOICES,
    skip: Collection[str] = (),
    *,
    raw: list[dict[str, Any]] | None = None,
    view: transcript_view.View | None = None,
    named_lines: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The person's newest room lines with words, newest first: [{line, ts_start, text, mic}]. line is the line key
    (registry.key_of); the frontend plays it by ts_start until /clip takes line keys."""
    nm, out = people_list._norm(name), []
    if named_lines is None:
        lines = raw[-people_list.ROOM_TAIL :] if raw is not None else transcript_lines.heard_tail(people_list.ROOM_TAIL)
        view = view if view is not None else transcript_lines._view()
        named_lines = view.named(lines)
    for e in reversed(named_lines):
        if (
            people_list._norm(e.get("speaker") or "") == nm
            and (e.get("text") or "").strip()
            and isinstance(e.get("ts_start"), int | float)
            and registry.key_of(e) not in skip
        ):
            device = e.get("device")
            out.append(
                {
                    "line": registry.key_of(e),
                    "ts_start": e["ts_start"],
                    "text": e["text"].strip(),
                    "mic": device if isinstance(device, str) and device.strip() else None,
                }
            )
            if len(out) >= limit:
                break
    return out


def person_card(name: str) -> dict[str, Any] | None:
    """GET /people/<name>/card: the contact card's data (contract in the docstring at the top of this file), or None
    for a name that can't be one."""
    from liberational.vision import sightings

    if not call_log._safe_name(name):
        return None
    # Share raw speech and one fresh identity snapshot across roster and media readers. Cache only raw rows:
    # retained() follows commits and purges, while a new View sees renames without waiting for new speech.
    raw, view = speech_log.retained(), transcript_lines._view()
    named_lines = view.named(raw[-people_list.ROOM_TAIL :])  # for recent voices
    saved_shots = face_shots.saved(context.STATE, revision=photo_files._shot_revision())
    me = _roster_row(name)
    name = me["name"] if me else name.strip()
    prefs = card_prefs.get(context.STATE, name)
    ready, shots_pending, shots_unavailable = people_list._ready_faces(
        [r for r in sightings.ranked(name, context.STATE) if r["id"] not in prefs["hidden"]["shot"]], saved_shots
    )
    hidden_videos = set(prefs["hidden"]["video"])
    named = views.names()
    videos = []
    for v in sightings.recent_visits(name, CARD_VIDEOS + len(hidden_videos), context.STATE):
        if _video_id(v["cam"], v["start"]) in hidden_videos or len(videos) >= CARD_VIDEOS:
            continue
        thumb = "/people/video-thumb.jpg?" + urllib.parse.urlencode(
            {"cam": v["cam"], "start": f"{v['start']:.3f}", "end": f"{v['end']:.3f}"}
        )
        videos.append(
            {
                "id": _video_id(v["cam"], v["start"]),
                "cam": v["cam"],
                "camera_name": named.get(v["cam"], v["cam"]),
                "start": v["start"],
                "end": v["end"],
                "duration_s": round(v["end"] - v["start"], 1),
                "thumb": thumb,
                "what": _what(v["cam"], v["start"], v["end"], name),
                "review": _review_link(name, v["start"]),
            }
        )
    videos, videos_pending, videos_unavailable = _ready_videos(videos)
    seen = me.get("last_seen") if me else None
    cam = me.get("last_seen_cam") if me else None
    return {
        "name": name,
        "photo": people_list.main_photo(name, ready=ready, saved_shots=saved_shots),
        "shots": [
            people_list._shot(r)
            for r in sightings.best_spread(
                name, CARD_SHOTS, context.STATE, skip=prefs["hidden"]["shot"], ready={r["id"] for r in ready}
            )
        ],
        **({"shots_pending": shots_pending} if shots_pending else {}),
        **({"shots_unavailable": shots_unavailable} if shots_unavailable else {}),
        "videos": videos,
        **({"videos_pending": videos_pending} if videos_pending else {}),
        **({"videos_unavailable": videos_unavailable} if videos_unavailable else {}),
        "voices": _recent_voice(name, skip=prefs["hidden"]["voice"], named_lines=named_lines),
        "last_seen": {"ts": seen, "cam": cam, "camera_name": me.get("last_seen_camera_name") if me else None}
        if seen
        else None,
        "here_now": me.get("here_now") if me else None,
    }


CARD_REFRESH_S = 5.0  # a card older than this is answered as is and read again in the background


CARD_PENDING_REFRESH_S = 1.0  # sooner while crops or thumbnails are still being made, so progress shows on each poll


class _CardReads:
    """The last card read per person, so opening one answers from it at once, the way the people list does
    (_PeopleReads). A card older than CARD_REFRESH_S is answered as is, marked `refreshing` so the page asks again, and
    read again in the background, one read per
    person at a time. A card is read on the request only the first time, or when its key moved: names, contacts, or a
    purge (the people list's fresh key), a forget (the segment store's revision), or card_prefs.json. Every People
    POST clears them all (CARD_EDITS), and a read that started before a clear is never kept. When GET /people is
    served, the first CARD_WARM people's cards that aren't cached under the current key are read in the background, so
    the first open is a cached one too. A cached card isn't read again ahead however old it is: its open answers from
    it, marked `refreshing`, and reads it behind (reading the first eight again every 5 minutes, whether or not
    anyone opened them, was about 2% of a core while a page polled the list)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last: dict[str, tuple[tuple[Any, ...], float, dict[str, Any]]] = {}  # norm name: (key, read at, card)
        self.reading: set[str] = set()
        self.epoch = 0  # moved by clear(): a read that began before it is dropped
        self.warming = False

    def get(self, name: str) -> dict[str, Any] | None:
        if not call_log._safe_name(name):
            return None
        k, key, now = people_list._norm(name), _card_key(), time.time()
        with self.lock:
            hit = self.last.get(k)
        if hit is not None and hit[0] == key:
            pending = hit[2].get("shots_pending") or hit[2].get("videos_pending")
            old = now - hit[1] >= (CARD_PENDING_REFRESH_S if pending else CARD_REFRESH_S)
            if old:
                self._refresh(name)
            with self.lock:
                reading = k in self.reading
            card = copy.deepcopy(hit[2])
            # An older card is answered as is, marked so the page asks again for the one read behind it (a
            # complete card polled no more, and showed the old one for the whole visit).
            return {**card, "refreshing": True} if old or reading else card
        return self._read(name)

    def _read(self, name: str) -> dict[str, Any] | None:
        with self.lock:
            epoch = self.epoch
        key, at = _card_key(), time.time()
        card = person_card(name)
        if card is not None:
            with self.lock:
                if self.epoch == epoch:
                    self.last[people_list._norm(name)] = (key, at, copy.deepcopy(card))
        return card

    def _refresh(self, name: str) -> None:
        k = people_list._norm(name)
        with self.lock:
            if k in self.reading:
                return
            self.reading.add(k)

        def run() -> None:
            try:
                self._read(name)
            except Exception as ex:
                log("card: background read failed:", repr(ex), level=logs.WARN)
            finally:
                with self.lock:
                    self.reading.discard(k)

        threading.Thread(target=run, name="card-read", daemon=True).start()

    def warm(self, names: list[str]) -> None:
        """Read the cards of these people that aren't cached under the current key, one at a time, in the
        background; a warm already running is left to finish."""
        with self.lock:
            if self.warming:
                return
            self.warming = True

        def run() -> None:
            try:
                for name in names:
                    key = _card_key()
                    with self.lock:
                        hit = self.last.get(people_list._norm(name))
                    if hit is None or hit[0] != key:
                        self._read(name)
            except Exception as ex:
                log("card: warm-up failed:", repr(ex), level=logs.WARN)
            finally:
                with self.lock:
                    self.warming = False

        threading.Thread(target=run, name="card-warm", daemon=True).start()

    def clear(self) -> None:
        with self.lock:
            self.last.clear()
            self.epoch += 1


def _card_key() -> tuple[Any, ...]:
    """What a cached card must still match to be answered from (_CardReads)."""
    return (
        roster_read._people_fresh(speech_log.shared().mark()[1]),
        photo_files._shot_revision(),
        roster_read.mtime(context.STATE / card_prefs.PREFS),
    )


_CARD_READS = _CardReads()


def _video_id(cam: str, start: float) -> str:
    """A card or picker video's id: its camera and start, "cam0@1793212345.50"."""
    return f"{cam}@{start:.2f}"


@routes.get_re(r"/people/([^/]+)/card")
def _get_people_x_card(h: H, m: re.Match[str]) -> None:
    card = _CARD_READS.get(urllib.parse.unquote(m.group(1)))
    return h._json(card or {"error": "bad name"}, 200 if card else 400)


@routes.get_last("/people/<name>", lambda p: p.startswith("/people/"))
def _get_people_summary(h: H) -> None:
    p = h.path.split("?")[0]
    from urllib.parse import unquote

    s2 = person_summary(unquote(p[len("/people/") :]))
    return h._json(s2 or {"error": "bad name"}, 200 if s2 else 400)
