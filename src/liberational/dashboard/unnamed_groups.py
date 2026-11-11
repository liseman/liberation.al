"""The unnamed faces and voices waiting for a name, grouped, each list worked out in the background and patched in
place as the owner names them (moved out of the server, #61).

GET /faces/unknown/groups             the unnamed face groups
GET /faces/unknown/<group>/sightings  one group's sightings, a page at a time
GET /voices/unknown/groups            the unnamed voice groups
GET /pets/sightings                   the pet sightings, with which photos are ready
"""

import bisect
import json
import math
import multiprocessing
import pathlib
import re
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import logs, memtrim, proc_table, speech_log
from liberational.dashboard import (
    context,
    fresh_reads,
    inbox_state,
    known_voices,
    people_list,
    photo_files,
    routes,
    transcript_lines,
    voice_clones,
)
from liberational.dashboard.context import LOG, Reply
from liberational.people import face_shots, registry, together, transcript_view, video_thumbs

if TYPE_CHECKING:
    from liberational.dashboard.server import H
    from liberational.vision.sightings import FaceBatch

CARD_SHOTS, CARD_VIDEOS, CARD_VOICES = 6, 4, 4  # the contact card: best face shots, recent video, recent voice


def speech_mark() -> int:
    """The newest commit sequence on the speech streams: a line committed after it has a higher one."""
    return speech_log.shared().mark((speech_log.UTTERANCES, speech_log.REPLIES))[0]


def speakers_with_audio(
    raw: Sequence[dict[str, Any]] | None = None, view: transcript_view.View | None = None
) -> set[int]:
    """The (merged-resolved) speaker ids that still have at least one line with audio, not counting lines marked as
    a TV's: a voice heard only from one is nobody to name."""
    view = view if view is not None else transcript_lines._view()
    lines = reversed(raw) if raw is not None else transcript_lines.iter_back()
    # Match independent line() reads: audio availability does not borrow labels from another microphone's copy.
    return {
        sid
        for e in view.individual_lines(e for e in lines if transcript_lines.has_audio(e) and not e.get("media"))
        if (sid := e.get("speaker_id")) is not None
    }


def audio_s(e: dict[str, Any]) -> float:
    """How long a line's audio is, in seconds (0 for a line without audio)."""
    if not transcript_lines.has_audio(e):
        return 0.0
    a = e["audio"]
    return max(0.0, (int(a.get("end_ns") or a["start_ns"]) - int(a["start_ns"])) / 1e9)


BAD_CAMERAS_LOGGED = 32  # the most unusable camera values _video_status says once each; the rest pass silently
_bad_cameras: set[str] = set()
_bad_cameras_lock = threading.Lock()


def _video_status(row: Mapping[str, Any], revision: str, saved: frozenset[str]) -> tuple[str, str | None]:
    """A listed video's current thumbnail readiness; a miss only queues work, never decodes on the request."""
    cam, start, end = str(row.get("cam") or ""), round(float(row["start"]), 3), round(float(row["end"]), 3)
    if not video_thumbs.CAMERA.fullmatch(cam):
        # No file can be named for it, and one such row must not fail the whole listing: no thumbnail, said once.
        with _bad_cameras_lock:
            fresh = cam not in _bad_cameras and len(_bad_cameras) < BAD_CAMERAS_LOGGED
            if fresh:
                _bad_cameras.add(cam)
        if fresh:
            LOG.warning("no video thumbnail for a row whose camera is %r: not a camera name", cam)
        return "unavailable", None
    path = video_thumbs.path(context.STATE, cam, start, revision)
    if path.name in saved and path.is_file():
        url = "/people/video-thumb.jpg?" + urllib.parse.urlencode(
            {"cam": cam, "start": f"{start:.3f}", "end": f"{end:.3f}"}
        )
        return "ready", url
    key = json.dumps([cam, start, end], separators=(",", ":"))
    return (
        "pending" if photo_files._crop_jobs.offer(context.STATE, video_thumbs.KIND, key, revision) else "unavailable"
    ), None


def _ready_face_groups(code: int, body: dict[str, Any]) -> Reply:
    """Apply current crop availability after the slower identity snapshot, so progress updates on every poll.

    A group with no saved picture and none still being cut is left out: the owner can't name a face they can't see.
    One whose every crop failed (its video is gone: a catalog reset, or the video window passed) is also dropped from
    the unnamed store (`_drop_face_groups`), so it doesn't come back. A group whose crops are still pending stays."""
    if code != 200:
        return code, body
    from liberational.vision import face_guess

    revision = photo_files._shot_revision()
    groups, pending, unavailable, saved = [], 0, 0, face_shots.saved(context.STATE, "shots_grid", revision)
    originals = face_shots.saved(context.STATE, revision=revision)
    hidden: set[int] = set()
    failed: list[dict[str, Any]] = []
    for group in body.get("groups", []):
        row = {k: v for k, v in group.items() if k != "_all_shots"}
        shots = group.get("_all_shots", group["shots"])
        ready, count, gone = people_list._ready_faces(shots, saved, grid=True)
        pending += count
        unavailable += gone
        if not ready and not count:
            hidden.add(row["id"])
            if gone:
                failed.append(row)
            continue
        detail_count = sum(
            str(shot["id"]) in originals and face_shots.path(context.STATE, str(shot["id"]), revision).is_file()
            for shot in shots
        )
        row.update(
            shots=ready[: face_guess.THUMBS], shot_count=detail_count, shots_pending=count, shots_unavailable=gone
        )
        groups.append(row)
    if failed:
        _drop_face_groups(failed, "every picture of it failed to cut")
    count_ = body.get("count")
    return code, {
        **{k: v for k, v in body.items() if not k.startswith("_")},
        "groups": groups,
        **({"count": max(0, count_ - len(hidden))} if isinstance(count_, int) else {}),
        "shots_pending": pending,
        "shots_unavailable": unavailable,
    }


def _drop_face_groups(rows: Iterable[Mapping[str, Any]], why: str) -> set[int]:
    """Remove unnamed face groups that have no picture left from the store, as the machine data they are. Never one
    the owner touched: marked not someone, a shot taken out, or a name undone (put-aside ones are never listed).
    The groups removed."""
    faces = people_list._faces_mod()
    held = faces.unknown_held()
    ids = {int(r["id"]) for r in rows if not r.get("not") and int(r["id"]) not in held}
    if ids:
        faces.drop_groups(ids)
        LOG.info("who's this: dropped %d unnamed face group(s) with no picture left: %s", len(ids), why)
    return ids


def _ready_pet_view(code: int, body: dict[str, Any]) -> Reply:
    """Keep sighting/video metadata, but mark photos so grids never request an uncached track."""
    if code != 200:
        return code, body
    from liberational.people import pet_shots

    revision = photo_files._shot_revision()
    saved = face_shots.saved(context.STATE, pet_shots.KIND + "_grid", revision)
    missing: set[str] = set()
    unavailable: set[str] = set()

    def item(row: dict[str, Any]) -> dict[str, Any]:
        track = str(row["track_id"])
        original_ready = pet_shots.path(context.STATE, track, revision).is_file()
        ready = (
            track in saved
            and face_shots.path(context.STATE, track, revision, pet_shots.KIND + "_grid").is_file()
            and original_ready
        )
        if not ready:
            (
                missing
                if photo_files._crop_jobs.offer(context.STATE, pet_shots.KIND + "_grid", track, revision)
                else unavailable
            ).add(track)
        return {**row, "photo_ready": original_ready, "thumb_ready": ready}

    groups, dropped = [], 0
    for group in body.get("groups", []):
        rows = [item(row) for row in group["sightings"]]
        if not any(r["photo_ready"] or r["thumb_ready"] or r["track_id"] in missing for r in rows):
            # No picture saved and none can still be cut (its video is gone): nobody can name it, so it isn't listed.
            # Pet groups are worked out from the sightings on every read, so there's no stored group to remove.
            dropped += 1
            continue
        groups.append(
            {
                **group,
                "sightings": rows,
                "photos_pending": sum(r["track_id"] in missing for r in rows),
                "photos_unavailable": sum(r["track_id"] in unavailable for r in rows),
            }
        )
    seen = {name: [item(row) for row in rows] for name, rows in body.get("seen", {}).items()}
    saved_videos = video_thumbs.saved(context.STATE, revision)
    videos_pending = videos_unavailable = 0
    for rows in seen.values():
        # A sighting known only from its saved photo (a label carried over from another embedding model) has no
        # camera and so no video: the card's videos are the newest sightings that have one.
        for row in [row for row in rows if row.get("cam")][:CARD_VIDEOS]:
            video = {**row, "end": max(row["end"], row["start"] + 5)}  # pet clips keep at least five seconds
            status, thumb = _video_status(video, revision, saved_videos)
            row.update(video_status=status, video_thumb=thumb)
            videos_pending += status == "pending"
            videos_unavailable += status == "unavailable"
    seen_pending = {
        name: sum(r["track_id"] in missing and not r["photo_ready"] for r in rows) for name, rows in seen.items()
    }
    seen_unavailable = {
        name: sum(r["track_id"] in unavailable and not r["photo_ready"] for r in rows) for name, rows in seen.items()
    }
    count = body.get("count")
    return code, {
        **body,
        "groups": groups,
        **({"count": max(0, count - dropped)} if isinstance(count, int) else {}),
        "seen": seen,
        "seen_pending": seen_pending,
        "seen_unavailable": seen_unavailable,
        "photos_pending": len(missing),
        "photos_unavailable": len(unavailable),
        "videos_pending": videos_pending,
        "videos_unavailable": videos_unavailable,
    }


def put_aside(e: dict[str, Any], skipped: set[float | str]) -> bool:
    """Whether People's inbox put this line aside (inbox_skips): by its key, or by its millisecond."""
    ts = e.get("ts_start")
    if not isinstance(ts, int | float):
        return False
    return round(float(ts), 3) in skipped or registry.key_of(e) in skipped


SIGHTINGS_PAGE_MAX = 200  # a face group's shots per request (GET /faces/unknown/<id>/sightings)


FACES_SHOWN = 24  # the most face groups Who's this? lists at once; the count says how many wait in all


def faces_waiting(skipped: set[float]) -> list[dict[str, int]]:
    """Unnamed face groups waiting for a name, most seen first: seen at least twice and not put aside. The same groups
    as /people's unknown.faces (people_view), kept apart from it while People is being redesigned. Raises when the
    groups can't be read."""
    return [g for g in people_list._faces_mod().unknown_groups() if g["count"] >= 2 and g["id"] not in skipped]


_named_cache: dict[str, Any] = {"key": None, "value": []}


def _named_lines(since: float) -> list[tuple[float, float, str]]:
    """The transcript's lines since `since` that name a person, as (start, end, the person's current name), sorted.
    Kept until the transcript, the voices, or the names change (_voices_key), and read from the hour before `since`
    on only, so the cache holds while the faces' oldest sighting moves: Who's this? asks every 30 s."""
    key = (_voices_key(), since // 3600)
    if _named_cache["key"] != key:
        from_ = (since // 3600) * 3600
        raw = transcript_lines.heard_lines()
        heard = [
            e
            for e in raw
            if isinstance(e.get("ts_start"), int | float) and e["ts_start"] >= from_ and not e.get("assistant")
        ]
        view = transcript_lines._view()
        _named_cache.update(
            key=key,
            value=sorted(
                (float(e["ts_start"]), float(e.get("ts_end") or e["ts_start"]), view.people[pid].display_name)
                for e, pid in view.speakers(heard)
                if pid is not None and pid in view.people and not e.get("twin_of")  # the same speech once
            ),
        )
    lines: list[tuple[float, float, str]] = _named_cache["value"]
    return lines[bisect.bisect_left(lines, (since,)) :]


def _named_sightings(
    windows: Iterable[tuple[float, float]], people: people_list._People, *, face_batch: FaceBatch | None = None
) -> list[tuple[float, str]]:
    """The recorded faces with a name seen within any of the windows, as (time, the person's current name), oldest
    first; a face whose name is now nobody's is left out. Windows that touch are read as one."""
    from liberational.vision import sightings

    merged: list[list[float]] = []
    for a, b in sorted(windows):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if not merged:
        return []
    batch = face_batch.faces([(a, b + 0.001) for a, b in merged]) if face_batch is not None else None
    if batch is None:
        rows = [
            (f["ts"], f["name"])
            for a, b in merged
            for f in sightings.faces(a, b + 0.001, state=context.STATE)
            if f["name"]
        ]
    else:
        rows = [(f["ts"], f["name"]) for f in batch if f["name"]]
    return [(float(t), n) for t, written in rows if (n := people.name(str(written)))]


def _with_together(
    rows: list[dict[str, Any]],
    pairs: Callable[[dict[str, Any], list[tuple[float, str]]], list[tuple[float, str]]],
    spans: Iterable[tuple[float, float]],
    people: people_list._People,
    pad: float = together.WINDOW_S,
    *,
    face_batch: FaceBatch | None = None,
) -> None:
    """Give each "Who's this?" row its suggestion from who was seen or heard with it (people/together.py), and
    sort a row with one among the maybes. The named faces are read once, within `pad` of the given spans: the named
    lines for faces, the waiting voices' lines for voices."""
    windows = ((a - pad, b + pad) for a, b in spans)
    seen = (
        _named_sightings(windows, people, face_batch=face_batch)
        if face_batch is not None
        else _named_sightings(windows, people)
    )
    for r in rows:
        r["together"] = together.suggest(pairs(r, seen))
        if (
            r["together"]
            and r.get("guess")
            and people_list._norm(r["guess"]["name"]) == people_list._norm(r["together"]["name"])
        ):
            r["together"]["name"] = r["guess"]["name"]  # one spelling, so Who's this? offers one Yes
    rank = {"probable": 0, "maybe": 1, "none": 2}

    def key(r: dict[str, Any]) -> tuple[int, float, float]:
        band = "maybe" if r["band"] == "none" and r.get("together") else r["band"]
        return rank[band], -r["score"] if r["band"] != "none" else 0, -(r.get("last_seen") or 0)

    rows.sort(key=key)


GROUPS_REFRESH_S = 30.0  # Live's "Who's this?" face rows take new sightings in at most this late


def _faces_owner_key(groups: bool = True) -> tuple[Any, ...]:
    """What the owner changes that the face rows show: the names and decisions (the event log), the skips, and the
    unknown face groups (merged, named, or cleared). A purge bypasses stale results immediately. Without `groups`,
    the groups are left out: the face worker also adds them and evicts the least seen, so the set moves with no
    owner change, and naming or undoing one writes the event log anyway (/people)."""
    mark = registry.at(context.STATE).mark() if registry.log_exists(context.STATE) else None
    try:
        st = inbox_state.INBOX_SKIPS.stat()
        skips = st.st_size, st.st_mtime_ns
    except OSError:
        skips = None
    # Normal sightings rewrite counts and centroids. Only membership is an owner change; otherwise an active
    # camera can discard every warmed snapshot before it publishes and defeat the 30-second data refresh.
    if not groups:
        return str(context.STATE), mark, skips, speech_log.shared().newest_voice_purge()
    from liberational.vision import face_rejects

    try:
        ids = tuple(sorted(g["id"] for g in people_list._faces_mod().unknown_groups()))
    except Exception:
        # Do not serve a former good snapshot for an unreadable database. The computation returns its existing
        # actionable HTTP error; keeping key sampling non-throwing also preserves People's other usable results.
        ids = None
    # A purge only of camera video (the nightly window, an age-out) takes no sighting or face row: left out, as
    # the voice groups' key leaves it out, so the next read is patched rather than waiting for the full answer.
    purge = speech_log.shared().newest_voice_purge()
    return str(context.STATE), mark, skips, ids, face_rejects.stamp(), purge, _face_nots_stamp()


def _face_nots_stamp() -> tuple[str, int, int] | None:
    """Where the owner's "not <person>" marks on face groups stand (people/face_nots.py): None without a store."""
    from liberational import knowledge
    from liberational.people import face_hidden, face_nots

    if not registry.knowledge_path(context.STATE).exists():
        return None
    # with what the owner ignored and blocked (people/face_hidden.py): a block hides rows with no other file changed
    marks = (face_nots.NOTS, face_hidden.IGNORES, face_hidden.BLOCKS)
    return knowledge.at(registry.knowledge_path(context.STATE)).generation(marks)


_FACE_GROUP_LAST_SEEN: dict[tuple[str, int], float] = {}


def _hide_emptied_face_groups(rows: list[dict[str, Any]], face_batch: FaceBatch | None) -> list[dict[str, Any]]:
    """Hide a known-recent group once a complete request snapshot proves its evidence is gone.

    Keep unknown/older history and incomplete snapshots. Remember positive observations after hiding a row,
    so the next refresh cannot resurrect it just because it was absent from the last response. This cache
    expires with the existing shot window and makes no extra broker reads or changes to face aggregates.
    """
    if face_batch is None or face_batch.found is None or face_batch.matched is None:
        return rows
    state = str(context.STATE)
    with _FACE_GROUPS.lock:
        for key, ts in list(_FACE_GROUP_LAST_SEEN.items()):
            if key[0] == state and ts < face_batch.start:
                del _FACE_GROUP_LAST_SEEN[key]
        for row in rows:
            ts = row.get("last_seen")
            if isinstance(ts, int | float) and face_batch.start <= ts <= face_batch.end:
                key = (state, row["id"])
                _FACE_GROUP_LAST_SEEN[key] = max(_FACE_GROUP_LAST_SEEN.get(key, ts), ts)
        known = dict(_FACE_GROUP_LAST_SEEN)
    # Use the already-loaded raw joins, not a second identity projection or a capped/quality-filtered scan.
    # Any surviving member is reason to keep the group, including one whose identity has since changed.
    detections = {r["id"] for r in face_batch.found.records}
    present = {str(r.get("cluster")) for r in face_batch.matched.records if r.get("detection") in detections}
    present |= {f"Face {gid}" for gid in _moved_into(detections)}
    return [
        row
        for row in rows
        if f"Face {row['id']}" in present
        or not (face_batch.start <= known.get((state, row["id"]), -float("inf")) <= face_batch.end)
    ]


def _hide_resolved_face_groups(rows: list[dict[str, Any]], face_batch: FaceBatch | None) -> list[dict[str, Any]]:
    """An old unnamed-gallery aggregate can outlive the naming of all its retained observations.

    Only a complete snapshot and positively named members prove a group resolved. No crop candidates alone
    proves nothing: low-confidence, missing-box, rejected, or SCAN-limited unknown evidence must stay accounted.
    Reuse the snapshot, projecting just the candidate members rather than naming the whole history twice.
    """
    if face_batch is None or face_batch.found is None or face_batch.matched is None:
        return rows
    candidates = {f"Face {row['id']}" for row in rows if not row.get("_all_shots", row.get("shots"))}
    members: dict[str, set[str]] = {}
    windows: set[tuple[float, float]] = set()
    for match in face_batch.found.records:
        cluster = str(match.get("cluster"))
        if cluster in candidates:
            members.setdefault(cluster, set()).add(str(match["id"]))
            ts = float(match["ts_start"])
            windows.add((max(face_batch.start, ts - 0.001), min(face_batch.end, ts + 0.001)))
    if not windows:
        return rows
    projected = face_batch.faces(sorted(windows))
    if projected is None:
        return rows
    named = {str(row["id"]) for row in projected if row["name"] is not None}
    resolved = {cluster for cluster, ids in members.items() if ids <= named}
    return [row for row in rows if f"Face {row['id']}" not in resolved]


def _moved_into(detections: set[Any]) -> set[int]:
    """The face groups that hold a recorded sighting the owner moved into them (vision/face_rejects.py): the
    sighting's own record still names the group it left, so a raw read doesn't find these."""
    from liberational.vision import face_rejects

    return {gid for shot, gid in face_rejects.moved().items() if shot in detections}


def _without_blocked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows whose face isn't one the owner blocked (people/face_hidden.py): every group's face against every
    block in one product, and nothing read at all while nothing is blocked."""
    from liberational.people import face_hidden

    if not rows or not face_hidden.blocks(context.STATE):
        return rows
    faces = people_list._faces_mod()
    ids, vecs, counts, _, sums = faces._unk_load_sums()
    walled = face_hidden.blocked_groups(
        context.STATE, [int(i) for i in ids], faces._representatives(vecs, counts, sums)
    )
    return [r for r in rows if int(r["id"]) not in walled]


NEW_GROUP_GRACE_S = 600.0  # a group a face started or joined this recently is kept while its sightings are written


def _drop_sourceless_face_groups(
    rows: list[dict[str, Any]], face_batch: FaceBatch | None, now: float
) -> list[dict[str, Any]]:
    """Drop the groups with no recorded sighting left in a complete shot-window snapshot (a catalog reset, or the
    video window passed): with no sighting there is no video to cut a picture from, so nobody can name them. They
    leave the store too (`_drop_face_groups`). A group a face started or joined in the last NEW_GROUP_GRACE_S stays,
    as does one with any sighting at all (a low-confidence one included), and any the owner touched."""
    if face_batch is None or face_batch.found is None or face_batch.matched is None:
        return rows
    faces = people_list._faces_mod()
    aliases = faces.group_aliases()
    detections = {r["id"] for r in face_batch.found.records}
    present: set[int] = set()
    for r in face_batch.matched.records:
        found = re.fullmatch(r"Face (\d+)", str(r.get("cluster")))
        if found and r.get("detection") in detections:
            present.add(aliases.get(int(found[1]), int(found[1])))
    present |= _moved_into(detections)
    stamped = faces.unknown_last_seen()
    candidates = [
        r
        for r in rows
        if not r.get("_all_shots", r.get("shots"))
        and int(r["id"]) not in present
        and stamped.get(int(r["id"]), -math.inf) < now - NEW_GROUP_GRACE_S
    ]
    if not candidates:
        return rows
    dropped = _drop_face_groups(candidates, "no sighting of it is left")
    return [r for r in rows if int(r["id"]) not in dropped]


def _unknown_groups() -> tuple[int, dict[str, Any]]:
    from liberational.people import face_nots
    from liberational.vision import face_guess, sightings

    skipped = set(inbox_state.inbox_skips()["face"])
    try:
        times: dict[int, list[float]] = {}
        waiting = faces_waiting(skipped)
        now = time.time()
        # One request's snapshot of the shot window, shared by the guesses and co-occurrence.
        face_batch = sightings.FaceBatch(now - sightings.SHOT_DAYS * 86400, now + 1, context.STATE) if waiting else None
        rows = face_guess.groups(
            waiting, times=times, face_batch=face_batch, all_shots=True, nots=face_nots.names(context.STATE)
        )
        rows = _hide_emptied_face_groups(rows, face_batch)
        rows = _hide_resolved_face_groups(rows, face_batch)
        rows = _drop_sourceless_face_groups(rows, face_batch, now)
        stamps = [t for ts in times.values() for t in ts]
        lines = _named_lines(min(stamps) - together.WINDOW_S) if stamps else []
        people = people_list._People()
        heard: dict[str, Any] = {}

        def pairs(r: dict[str, Any], seen: list[tuple[float, str]]) -> list[tuple[float, str]]:
            if "f" not in heard:  # set up once for every row
                heard["f"] = together.Heard(lines, seen, people.owner())
            return heard["f"](times.get(r["id"], []))

        # A named face matters only near a named line: within the window of a sighting that is itself within the
        # window of the line, so twice the window.
        _with_together(
            rows, pairs, [(s, e) for s, e, _ in lines], people, pad=2 * together.WINDOW_S, face_batch=face_batch
        )
    except Exception as ex:
        LOG.exception(f"who's this: {logs.reason(ex)}")  # a 500 is an error, with its traceback
        return 500, {"error": "Could not read the faces waiting for a name."}
    rows = [r for r in rows if not _certain(r)]
    try:
        rows = _without_blocked(rows)
    except Exception as ex:
        LOG.exception(f"who's this: {logs.reason(ex)}")
        return 500, {"error": "Could not read the faces waiting for a name."}
    # Only groups seen in the last faces.unnamed_days, by the newer of when a face last started or joined it
    # and its newest recorded sighting: the store keeps up to UNK_MAX groups however long ago they were seen. A group
    # neither dates hasn't been seen since the store began dating them, so it's older than the window. Older groups
    # stay in the store and come back when seen again.
    since = now - float(cfg.get("faces.unnamed_days")) * 86400
    stamped = people_list._faces_mod().unknown_last_seen()
    rows = [
        r
        for r in rows
        if max((t for t in (r.get("last_seen"), stamped.get(r["id"])) if t is not None), default=-math.inf) >= since
    ]
    # Count every group that is still asked about, by the same filters as the listed ones, then list the first
    # FACES_SHOWN: counting only the listed rows pinned the count at FACES_SHOWN while each name pulled the next group
    # in, and counting unfiltered groups counted resolved or certain ones.
    return 200, {
        "groups": rows[:FACES_SHOWN],
        "count": len(rows),
        "skipped": sorted(int(i) for i in skipped),
        "_counted": [r["id"] for r in rows],  # every group the count counts, so a patch can recount
    }


def _patch_face_groups(answer: Reply, before: Any, now: Any) -> Reply | None:
    """The last Who's this? face answer after an owner change, as far as is cheap to know: the groups no longer
    unknown (named, merged, or cleared) or now put aside leave the rows and the count, and `refreshing` says the full
    answer (new groups, guesses that moved with a name) is being worked out. None, so the read waits for the full
    answer, after a purge, a face rejects change (the pictures shown), or a change of state directory, or when the
    groups can't be read. A "not <person>" mark is applied too: the row lists it, and a guess it rules out goes.
    The owner key is _faces_owner_key's: (state, mark, skips, ids, rejects, purge, nots)."""
    from liberational.people import face_nots

    code, body = answer
    if code != 200 or not isinstance(before, tuple) or not isinstance(now, tuple) or len(before) != 7 or len(now) != 7:
        return None
    if now[3] is None or (before[0], before[4], before[5]) != (now[0], now[4], now[5]):
        return None
    ids = set(now[3])
    skipped = {int(i) for i in inbox_state.inbox_skips()["face"]}
    waiting = [i for i in body.get("_counted", []) if i in ids and i not in skipped]
    denied = face_nots.names(context.STATE) if before[6] != now[6] else {}

    def marked(g: dict[str, Any]) -> dict[str, Any]:
        if before[6] == now[6]:
            return g
        nots = sorted(denied.get(g["id"], set()))
        guess = g.get("guess")
        if guess and guess["name"].casefold() in {n.casefold() for n in nots}:
            return {**g, "not": nots, "guess": None, "score": 0, "band": "none"}
        return {**g, "not": nots}

    return 200, {
        **body,
        "groups": [marked(g) for g in body["groups"] if g["id"] in ids and g["id"] not in skipped],
        "count": len(waiting),
        "_counted": waiting,
        "skipped": sorted(skipped),
        "refreshing": True,
    }


GROUPS_STALE_S = 600.0  # face groups are worked out again this often even with nothing new: shots age out of the window


GROUPS_WORKER_TIMEOUT_S = 300.0  # a group worker that takes longer than this is stopped and started again


_groups_in_worker: bool = (
    False  # main() turns it on: face groups, voice groups, and the people list then read in workers
)


_groups_executor: ProcessPoolExecutor | None = None


_groups_lock = threading.Lock()


def _unknown_groups_job() -> Reply:
    """The face groups' full answer, in the worker process (_unknown_groups_worked_out)."""
    if multiprocessing.parent_process() is not None:
        cfg.load.cache_clear()  # a setting saved since: the dashboard rereads its pack.toml, and so does this
        context.low_priority(10)
        memtrim.start()  # what a refresh freed goes back to the system
    return _unknown_groups()


def _unknown_groups_worked_out() -> Reply:
    """The face groups' full answer (_unknown_groups), worked out in one process of its own. Run on a dashboard
    thread, each refresh was 1.5 to 10 s of pure Python every 30 s holding the interpreter lock every request needs:
    a 5 ms request took up to 1.5 s while it ran. The patched answers (a name, an Ignore, a not-mark)
    stay in the dashboard's process; only this moves. A worker that died, or ran past GROUPS_WORKER_TIMEOUT_S,
    is replaced on the next refresh, and this one answers the same error a failed read does."""
    global _groups_executor
    if not _groups_in_worker:
        return _unknown_groups()
    with _groups_lock:
        if _groups_executor is None:
            _groups_executor = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=proc_table.ignore_interrupt
            )
        executor = _groups_executor
    try:
        return executor.submit(_unknown_groups_job).result(timeout=GROUPS_WORKER_TIMEOUT_S)
    except (BrokenProcessPool, TimeoutError) as error:
        with _groups_lock:
            if _groups_executor is executor:
                _groups_executor = None
        for process in list((getattr(executor, "_processes", None) or {}).values()):
            process.terminate()  # one still running past the timeout
        executor.shutdown(wait=False, cancel_futures=True)
        LOG.warning("who's this: the face groups worker stopped: %s", repr(error))
        return 500, {"error": "Could not read the faces waiting for a name."}


_groups_data: dict[str, Any] = {"at": -math.inf, "key": None}


def _face_groups_data() -> tuple[Any, ...]:
    """What new data the face rows are worked out from: the unknown face groups the face worker writes (a new or
    grown group), the newest speech line (who was heard near a sighting), and the time in GROUPS_STALE_S steps. Read
    at most every GROUPS_REFRESH_S, so a refresh starts at most that often, and only when one of them moved:
    a quiet room no longer works every row out again every 30 s."""
    now = time.monotonic()
    with _groups_lock:
        if now - _groups_data["at"] < GROUPS_REFRESH_S:
            return _groups_data["key"]
    try:
        st = people_list._faces_mod().UNK_DB.stat()
        groups = st.st_mtime_ns, st.st_size
    except Exception:
        groups = None
    key = (groups, speech_log.shared().mark()[0], int(time.time() // GROUPS_STALE_S))
    with _groups_lock:
        _groups_data.update(at=now, key=key)
    return key


_FACE_GROUPS = fresh_reads._Fresh(
    "face groups",
    _unknown_groups_worked_out,
    _faces_owner_key,
    _face_groups_data,
    patch=_patch_face_groups,
)


def _certain(row: dict[str, Any]) -> bool:
    """A Who's this? row whose best guess is already 100%: nothing to ask, so it isn't listed."""
    return row.get("band") == "probable" and (row.get("score") or 0) >= 100


def _voices_key() -> tuple[Any, ...]:
    """What the owner changes that the unnamed-voice rows show: the state they're read from, the voiceprints handed
    out, the skips, the names (the event log), and the clones. Each file's (size, mtime), or None for one that isn't
    there. Purges invalidate immediately so removed audio is never served from the kept response; new lines
    are the rows' data part (_Fresh)."""

    def stamp(p: pathlib.Path) -> tuple[int, int] | None:
        try:
            st = p.stat()
            return st.st_size, st.st_mtime_ns
        except OSError:
            return None

    files = [
        context.STATE / "voiceprints.json",
        inbox_state.INBOX_SKIPS,
        context.STATE / "clones",
    ]
    # A voice named or a line reassigned in the event log changes the rows without touching a file.
    mark = registry.at(context.STATE).mark() if registry.log_exists(context.STATE) else None
    from liberational.listen import revisions

    return (
        str(context.STATE),
        *(stamp(f) for f in files),
        mark,
        # Only a purge that can take a line or its audio: a video-only one (the nightly window, an age-out of
        # cameras' video) leaves every row as it was.
        speech_log.shared().newest_voice_purge(),
        revisions.review_revision(context.STATE),
    )


def _voice_prints(ns: list[int]) -> dict[int, list[Any]]:
    """Each voiceprint's stored embeddings, from the broker (listen/voices.py prints)."""
    from liberational.listen import voices

    return voices.prints(ns)


def _with_prints(
    waiting: list[dict[str, Any]], named: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The voices with what people/voice_guess.py compares: an unnamed voice's "centroid" (the mean of its stored
    embeddings), and a named voice's too, with each embedding as one of its "refs"."""
    import numpy

    prints = _voice_prints(sorted({s["id"] for s in [*waiting, *named]}))

    def centroid(n: int) -> list[float] | None:
        vs = prints.get(n) or []
        return numpy.mean(vs, axis=0).tolist() if vs else None

    return (
        [{**s, "centroid": centroid(s["id"])} for s in waiting],
        [{**s, "centroid": centroid(s["id"]), "refs": [v.tolist() for v in prints.get(s["id"]) or []]} for s in named],
    )


def _unknown_voices() -> tuple[int, dict[str, Any]]:
    from liberational.people import voice_guess

    try:
        skipped = set(inbox_state.inbox_skips()["voice"])
        clones = voice_clones._clone_only_ids()
        raw, view = speech_log.retained(), transcript_lines._view()
        sp = known_voices._speakers()["speakers"]
        audible = speakers_with_audio(raw, view)
        waiting = [
            s
            for s in sp
            if people_list.VOICE_GROUP.fullmatch((s.get("label") or "").strip())
            and s["id"] in audible
            and s["id"] not in skipped
        ]
        named = [
            s
            for s in sp
            if not people_list.VOICE_GROUP.fullmatch((s.get("label") or "").strip())
            and (s.get("label") or "").strip().lower() not in ("unknown", MULTI)
            and s["id"] not in clones
        ]
        sims: dict[int, list[float]] = {}
        spans: dict[int, list[tuple[float, float]]] = {}
        clips, _total = voice_clips(
            [{"id": s["id"], "label": s["label"]} for s in waiting],
            per_voice=3,
            sims=sims,
            spans=spans,
            raw=raw,
            view=view,
        )
        playable = {c["id"] for c in clips}
        waiting = sorted((s for s in waiting if s["id"] in playable), key=lambda s: -(s.get("last_seen") or 0))[:24]
        waiting, named = _with_prints(waiting, named)
        rows = voice_guess.rows(waiting, named, clips, sims)
        # Named faces are read when the voices change, not as the cameras see them: suggestions for voices follow the
        # next line heard or the next name given (the cache's key), not each new sighting.
        _with_together(
            rows,
            lambda r, seen: together.seen_with(spans.get(r["id"], []), seen),
            [s for ss in spans.values() for s in ss],
            people_list._People(),
        )
    except Exception as ex:
        LOG.exception(f"who's this (voices): {logs.reason(ex)}")  # a 500 is an error, with its traceback
        return 500, {"error": "Could not read the voices waiting for a name."}
    rows = [r for r in rows if not _certain(r)]
    return 200, {"groups": rows, "count": len(rows), "skipped": sorted(int(i) for i in skipped)}


def _patch_voice_groups(answer: Reply, before: Any, now: Any) -> Reply | None:
    """The last Who's this? voice answer after an owner change, as far as is cheap to know: the voices named
    since, or put aside, leave the rows and the count, as do clips answered "Nobody", given to someone, or marked
    "Multiple speakers", and `refreshing` says the full
    answer (new voices, guesses that moved with a name) is being worked out. A new voiceprint handed out (listen writes voiceprints.json as it hears a
    new voice) or a new clone needs nothing taken out. After a purge or a transcript revision, a clip whose line is no
    longer kept (purged, or taken off by a correction) leaves too, so what was deleted never plays: the page answers at
    once rather than waiting about 25 s for the full answer. None, so the read waits for it, after a change of
    state directory, or when the names or the kept lines can't be read. The owner key is _voices_key's: (state,
    voiceprints, skips, clones, mark, purge, review revision)."""
    code, body = answer
    if code != 200 or not isinstance(before, tuple) or not isinstance(now, tuple) or len(before) != 7 or len(now) != 7:
        return None
    if before[0] != now[0]:
        return None
    alive: set[str] | None = None  # the lines still kept, when a purge or a revision may have taken some
    if (before[5], before[6]) != (now[5], now[6]):
        from liberational.listen import revisions

        try:
            kept_lines = revisions.filter_lines(context.STATE, speech_log.shared().all_lines())
        except Exception as e:
            LOG.warning("who's this (voices): lines unreadable, waiting for the full answer: %s", repr(e))
            return None
        alive = {registry.key_of(e) for e in kept_lines if isinstance(e.get("ts_start"), int | float)}
    skipped = {int(i) for i in inbox_state.inbox_skips()["voice"]}
    try:
        names = registry.voice_names(context.STATE, [g["id"] for g in body["groups"]])
    except Exception as e:
        LOG.warning("who's this (voices): names unreadable, waiting for the full answer: %s", repr(e))
        return None
    # A clip answered "Nobody" (a line put aside) leaves its row at once too, and a row left with no clip goes, as the
    # full answer leaves out a voice with nothing to play.
    lines = set(inbox_state.inbox_skips()["line"])

    def kept(clip: Mapping[str, Any]) -> bool:
        ts = clip.get("ts")
        if alive is not None and clip.get("line") not in alive:
            return False  # purged, or taken off by a correction
        return clip.get("line") not in lines and not (isinstance(ts, int | float) and round(float(ts), 3) in lines)

    # A clip whose line the owner gave someone, or took off its voice ("Multiple speakers", "unknown"), leaves too.
    taken: set[str] = set()
    shown = [(g["id"], c["line"]) for g in body["groups"] for c in g.get("clips", []) if isinstance(c.get("line"), str)]
    if shown:
        try:
            said = registry.at(context.STATE).said(
                (registry.SubjectRef(registry.SubjectKind.OBSERVATION, f"utterance:{line}"), sid) for sid, line in shown
            )
        except Exception as e:
            LOG.warning("who's this (voices): lines unreadable, waiting for the full answer: %s", repr(e))
            return None
        taken = {line for (_, line), who in zip(shown, said, strict=True) if who.detached or who.own.person is not None}
    rows = []
    for g in body["groups"]:
        if g["id"] in skipped or names.get(g["id"]):
            continue
        clips = [c for c in g.get("clips", []) if kept(c) and c.get("line") not in taken]
        if g.get("clips") and not clips:
            continue
        rows.append({**g, "clips": clips} if "clips" in g else g)
    return 200, {**body, "groups": rows, "count": len(rows), "skipped": sorted(skipped), "refreshing": True}


_voices_executor: ProcessPoolExecutor | None = None


_voices_lock = threading.Lock()


def _unknown_voices_job() -> Reply:
    """The full voice-group answer in its worker, with current settings on every refresh."""
    if multiprocessing.parent_process() is not None:
        cfg.load.cache_clear()
        context.low_priority(10)
        memtrim.start()  # what a refresh freed goes back to the system
    return _unknown_voices()


def _unknown_voices_worked_out() -> Reply:
    """Keep voice-group recomputation off the dashboard's interpreter lock.

    A separate spawned worker reads the retained speech, voiceprints, and sightings. Owner-edit patches and the
    cache's publication checks stay in the dashboard, so naming, skipping, or correcting a clip still applies
    immediately. Face work has its own worker, so a slow face refresh cannot queue this one behind it. A dead or
    timed-out worker returns the normal read error and is replaced for the next refresh.
    """
    global _voices_executor
    if not _groups_in_worker:
        return _unknown_voices()
    with _voices_lock:
        if _voices_executor is None:
            _voices_executor = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=proc_table.ignore_interrupt
            )
        executor = _voices_executor
    try:
        return executor.submit(_unknown_voices_job).result(timeout=GROUPS_WORKER_TIMEOUT_S)
    except (BrokenProcessPool, TimeoutError) as error:
        with _voices_lock:
            if _voices_executor is executor:
                _voices_executor = None
        for process in list((getattr(executor, "_processes", None) or {}).values()):
            process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        LOG.warning("who's this: the voice groups worker stopped: %s", repr(error))
        return 500, {"error": "Could not read the voices waiting for a name."}


_VOICE_GROUPS = fresh_reads._Fresh(
    "voice groups", _unknown_voices_worked_out, _voices_key, lambda: speech_mark(), patch=_patch_voice_groups
)


def voice_clips(
    voices: list[dict[str, Any]],
    per_voice: int = 8,
    sims: dict[int, list[float]] | None = None,
    spans: dict[int, list[tuple[float, float]]] | None = None,
    *,
    raw: Sequence[dict[str, Any]] | None = None,
    view: transcript_view.View | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Playable clips of unnamed voices, most useful to label first (value ranking below), and the total available.
    sims, when given, gets every playable line's match to its voice, per voice (what "Who's this?" reads to tell a
    voice whose lines don't sound alike); spans, every line's (start, end), per voice (who was seen then)."""
    want = {v["id"] for v in voices}
    if not want:
        return [], 0
    aside = set(inbox_state.inbox_skips()["line"])  # clips answered "Nobody" are not asked about again
    lines = {}
    # Another mic's copy of a line is the same speech: asked about once.
    view = view if view is not None else transcript_lines._view()
    for e in transcript_view.once(view.lines(raw[-60000:] if raw is not None else transcript_lines.heard_tail(60000))):
        sid = e.get("speaker_id")  # merges followed; None once someone labeled the line
        if sid is None:
            continue
        if put_aside(e, aside) or e.get("media"):  # answered "Nobody", or a TV's
            continue
        if spans is not None and sid in want and isinstance(e.get("ts_start"), int | float):
            spans.setdefault(sid, []).append((float(e["ts_start"]), float(e.get("ts_end") or e["ts_start"])))
        if sid in want and transcript_lines.has_audio(e) and audio_s(e) >= 1.0 and (e.get("text") or "").strip():
            lines.setdefault(sid, []).append(e)
            if sims is not None and e.get("speaker_sim"):
                sims.setdefault(sid, []).append(float(e["speaker_sim"]))
    out, total = [], 0
    label = {v["id"]: v["label"] for v in voices}
    total = sum(len(v) for v in lines.values())
    label = {v["id"]: v["label"] for v in voices}

    from liberational.listen.voices import THRESHOLD

    # Label value, not recency. A clip teaches most when the system was unsure which voice it was (its match to
    # its cluster sits near the current matching threshold) and it is long enough to hear. Clips are taken
    # round-robin across unnamed voices so no one cluster floods the list, each voice's most uncertain clips first.
    def value(e: dict[str, Any]) -> float:
        sim = float(e.get("speaker_sim") or 0.0)
        unsure = 1.0 - min(1.0, abs(sim - THRESHOLD) / 0.35)
        long_enough = min(1.0, audio_s(e) / 4.0)
        return 0.7 * unsure + 0.3 * long_enough

    ranked = {sid: sorted(es, key=value, reverse=True) for sid, es in lines.items()}
    order = sorted(ranked, key=lambda k: -len(ranked[k]))
    out = []
    for _ in range(per_voice):
        took = False
        for sid in order:
            if ranked[sid]:
                e = ranked[sid].pop(0)
                out.append(
                    {
                        "id": sid,
                        "label": label.get(sid),
                        "ts": e["ts_start"],
                        "line": registry.key_of(e),  # what /clip?line= and a skip take
                        "text": e["text"].strip(),
                        "dur": round(audio_s(e), 1),
                        "sim": round(float(e.get("speaker_sim") or 0.0), 2),
                    }
                )
                took = True
        if not took:
            break
    return out, total


MULTI = transcript_view.MULTI


@routes.get("/faces/unknown/groups")
def _get_faces_unknown_groups(h: H) -> None:
    code, o = _ready_face_groups(*_FACE_GROUPS.poll())
    return h._json(o, code)


@routes.get("/voices/unknown/groups")
def _get_voices_unknown_groups(h: H) -> None:
    code, o = _VOICE_GROUPS.poll()
    return h._json(o, code)


@routes.get_re(r"/faces/unknown/(\d{1,9})/sightings")
def _get_faces_unknown_x_sightings(h: H, m: re.Match[str]) -> None:
    from liberational.people import face_photos

    q = h._query()
    try:
        offset = max(0, int(q.get("offset", "0")))
        limit = max(1, min(SIGHTINGS_PAGE_MAX, int(q.get("limit", str(face_photos.STRIP_MAX)))))
    except ValueError:
        return h._json({"error": "bad offset or limit"}, 400)
    found = [f for f in face_photos._unnamed(context.STATE) if face_photos._group(f["cluster"]) == int(m.group(1))]
    ready, pending, unavailable = people_list._ready_faces(found)
    page = ready[offset : offset + limit]
    return h._json(
        {
            "sightings": [{"id": s["id"], "ts": s["ts"], "url": f"/faces/shot/{s['id']}.jpg"} for s in page],
            "total": len(ready),
            "shots_pending": pending,
            "shots_unavailable": unavailable,
        }
    )


@routes.get("/pets/sightings")
def _get_pets_sightings(h: H) -> None:
    from liberational.dashboard import pets

    code, reply = _ready_pet_view(*pets.sightings(context.STATE))
    return h._json(reply, code)
