"""Pack dashboard server (stdlib http.server + psutil). http://localhost:<ports.dashboard>/ (8765 by default)

This module is the HTTP plumbing: authorization, compression, caching headers, the live feed, and the slow-request
watch. Each area's module registers its own routes (dashboard/routes.py); the list below says what each answers.

GET  / and each tab's path (APP_PATHS), with Accept text/html   the dashboard app (frontend/): static/app/index.html
                            from the Vite build, filled in; the tab comes from the path. In --dev, Vite's page
                            instead, with /static/app/* passed through to it
GET  /events                server-sent events: hello, transcript, scene, speakers, ambient, vision, ping, and the
                            shell's status (privacy, services, research: dashboard/shell_status.py), then ready
GET  /scene                 state/scene.json
GET  /people/video-thumb.jpg?cam=&start=&end= prepared full-scene thumbnail; 202 queues background preparation
GET  /transcript?n=80       the last N lines heard and said (the broker's speech log)
POST /transcript/reject     {lines: [exact line keys]}: owner correction, hide the turn without deleting raw speech;
                            {ok, lines, undo} with one durable Undo token for the atomic batch
POST /transcript/media      {lines: [exact line keys]}: "This is TV": the lines stay in the transcript, marked
                            `media`, and leave people, voices, the journal, and Ask; {ok, lines, undo}
POST /voices/unknown/media  {group}: "This is TV" for every line of an unnamed voice; {ok, lines, undo}
POST /transcript/not-media  {lines: [exact line keys]}: "Not TV": these lines, and no others, are a TV's no more,
                            whoever marked them; {ok, lines, undo}, 404 when none of them is marked
POST /transcript/undo       {undo: token}: restore the rejected batch, or unmark the TV lines; {ok, lines}, 409 if
                            already undone
GET  /speakers              the voice list (people/voice_list.py), each voice labeled with its person
POST /speakers/label        {"id": 4, "name": "Sarah"}  -> runs ambient.py speakers label <id> <name>
POST /speakers/tag          {"lines": [keys], "name"} (or "ts_starts"): lines with no voice were NAME; queued in voice_tags.jsonl for
                            the microphone service, which relabels them and learns the voice
POST /speakers/me           {"id": 2}: "that was me"; names voice 2 as the owner (this machine only)
GET  /review/boxes?cam=&start=&end=  people and things one camera saw then, as boxes (at most 120 s)
GET  /audio/range?start=&end= the room's audio between two timestamps as WAV (at most 330 s), for Review under video
GET  /people                {people: [one entry per person: voices, faces, number, lines, last heard and seen],
                             unknown: {voices, faces}} (unnamed voice clusters and face groups)
POST /people/rename         {"from", "to"}: everywhere at once; to an existing name it merges
POST /people/me             {"name", "voices"?}: local explicit owner merge, confirming current voices first
POST /people/claim          {"kind": "voice"|"face", "id", "name"}: an unnamed voice or face belongs to a person;
                            a new claim returns an undo token for /people/undo
POST /people/describe       {"name", "role", "organization"}: a person's role and organization ("" clears one, a key
                            left out stays); kept in the event log, this machine only. /people/add and /people/claim
                            also take a name typed with them, like "Dana, CEO of Acme" (people/tags.py)
POST /inbox/skip            {"kind": "face"|"voice"|"line", "id"}: keep one out of People's inbox for good; {"kind": "clear"} brings all back
POST /measurement/mark      {"kind": "line"|"shot", "shown", "right", "actual"?, ...}: the name on one line or shot was right
                            or wrong, saved as an identification label under paths.measurement (dashboard/marks.py);
                            it changes no name. {"kind": "answer", "question", "asked_at", "right", "cited"?, "says"?,
                            ...}: an Ask answer was right or wrong, saved as a retrieval label. {"kind": "entries",
                            "camera", "start", "end", "events", "unsure"?, "none_seen"?, "replace"?}: one window's
                            entries and exits, saved in entries.jsonl (409 when it overlaps a labeled window). This
                            machine only
GET  /inbox/skips           {face, voice, line}: how many are put aside
POST /people/delete         {"name"}: forget who they are (face, voice, phone, sightings; recordings stay), this machine only
GET  /people/<name>/photo   newest face crop (JPEG); /photos lists them, /photos/<ms>.jpg serves one
GET  /voices/unknown/groups   {groups: [{id, label, count, last_seen, clips, guess, score, band, reason}], count,
                            skipped, refreshing?}: the unnamed voices waiting for a name, each with its most useful
                            clips and a best guess (people/voice_guess.py), best guess first; refreshing: the last
                            answer with names and Ignores applied, while the full one is worked out
GET  /faces/unknown/groups    {groups: [{id, label, count, last_seen, shots, shot_count, guess, score, band, reason}], count,
                            skipped}: the face groups waiting for a name, each with its newest shots and a best guess
                            (Live's "Who's this?", vision/face_guess.py), best guess first, and the ones put aside
                            Both group routes return 202 {pending: true, groups: [], count: null, skipped: []}
                            while their first snapshot warms; this is unknown, not an empty result. After an owner change
                            the face route serves its last answer less what was named or put aside, with
                            refreshing: true, while the full one is worked out.
GET  /faces/unknown/<id>.jpg   latest crop of an unnamed face group (refreshed while it's in view)
GET  /faces/unknown/<id>/sightings?offset=&limit=   that group's sightings, newest first, a page at a time (at most 200):
                            {"sightings": [{id, ts, url}], "total"}, each url its shot (/faces/shot/<sighting>.jpg)
POST /faces/not             {name, groups, shots, not}: these face groups are not that person (never their guess, never
                            named them automatically; not:false takes it back), and these shots are the card's own
                            Not <name>: {ok, marked, undo: [tokens for the shots]}
POST /people/photo          {name, action: main|newest|hide|show, shot}: the card's photo choices, with /pets/photo's
                            verbs: {ok, undo, card}
GET  /people/examples?name=   Training images (dashboard/face_examples.py): every face row naming matches the person
                            against, the least like the rest first: {name, count, examples: [{key, kind, source, at,
                            cam, picture, similarity}]}; each picture is /people/examples/photo?key=
POST /people/examples/remove  {name, keys}: take those rows out of the gallery: {ok, removed, count, examples}
GET  /pets/examples?pet=    a pet's training set, every sighting labeled as it: {pet, count, examples: [{track_id,
                            cam, start, source, picture, example, similarity}]}; a kept photo is
                            /pets/examples/photo?track=
POST /pets/examples/remove  {pet, track_ids}: clear those labels, mark them not the pet, and erase their example
                            vectors and photos: {ok, removed}
POST /faces/unknown/reject  {group, ids}: "Not this person": those shots leave the group for good, and its face is
                            rebuilt from the rest: {ok, removed, left, ignored}
POST /faces/unknown/split   {group, ids}: the pictures deselected on a Who's this? card leave the group before its
                            answer and wait as a group of their own: {ok, group, moved, left}
POST /faces/unknown/hide    {group, action: ignore|block, exclude}: hide the group's pictures, all but `exclude`.
                            ignore: those pictures, from now on; block: also any group whose face is at least
                            face_hidden.BLOCK_SIM like them. Kept in the knowledge store: {ok, kind, keys, hidden, left}
POST /faces/unknown/restore {kind: ignored|blocked, keys}: un-ignore pictures or unblock faces: {ok, restored}
GET  /faces/unknown/hidden  what was ignored and blocked, newest first: {ignored: [{id, url, group, at}],
                            blocked: [{id, group, at, count, shots: [{id, url}]}], ignored_count, blocked_count}
GET  /people/<name>/shots?before=&limit=   every shot of them the face worker found (vision/sightings.py), newest
                            first, one per camera per second: {total, shots: [{id, ts, cam, sim}], next}
GET  /faces/shot/<id>.jpg   a square face crop of one sighting (state/shots/, see people/face_shots.py)
GET  /people/<name>/card    the contact card: {name, photo: Shot | null, shots: [Shot] (up to 6, best first, one
                            per day until days run out), videos: [{cam, camera_name, start, end, duration_s, thumb, what,
                            review}]
                            (up to 4, newest first), voices: [{line, ts_start, text, mic}] (up to 4, newest first),
                            last_seen: {ts, cam, camera_name} | null, here_now: {cam, camera_name, since} | null}.
                            camera_name is the camera's name (vision.camera_names), or its id when it has none. Shot = {id, url, ts, cam, score};
                            id, cam, and score are null only for a saved crop (no recorded shot yet). The main photo
                            is the best-scoring shot (vision/face_quality.py); /people's "photo" is its url.
                            Each video also has "id": "<cam>@<start .2f>", for /people/correct
GET  /inbox/unnamed         what nobody is named for, each newest first: {faces: [{id, photos: [up to 4
                            urls], count, last_seen, cam, camera_name}] (up to 24), voices: [{id, label: "Unnamed voice A",
                            lines, last_heard, sample: {line, ts_start, text} | null}] (up to 12), videos: [{id, cam,
                            camera_name, start, end, duration_s, thumb, what}] (up to 12: a camera's stretch of faces nobody
                            is named for)}
POST /people/assign         {"name", "faces": [ids], "voices": [ids], "videos": [ids]}: give them all to one person (a
                            new name adds them), with one undo: {ok, name, assigned: {faces, voices, videos}, failed:
                            [{kind, id, error}], undo, people}; 400 when nothing could be given. Voices are labeled,
                            not merged, so it can be undone
POST /people/assign-many    {"items": [{"name", "faces": [ids], "voices": [ids], "videos": [ids]}]}: several
                            assignments, each to its own person, with one undo for them all: {ok, assigned: [{name,
                            faces, voices, videos}], failed: [{name, kind, id, error}], undo, people}; 400 when nothing
                            could be given
POST /people/correct        {"name", "kind": "shot"|"video"|"voice"|"track", "id", "action": "move"|"remove"|"main"|
                            "hide", "to"}: the card's per-item menu (Not <name>, Use as main photo, Hide from the
                            card), and Review's "Not <name>" on a person's box ("track": id "<cam>:<track>@<ts>", move
                            or remove only). A corrected sighting is the owner's claim on it in the people registry,
                            so it keeps that name. {ok, undo, to (a move's person), card}
POST /people/undo           {"undo": token}: put a claim, correction, or assignment back. {ok, errors, people, card | null};
                            409 when it was undone already
GET  /clones/<id>/lines      a voice's recent lines that could go into its clone's reference, newest first:
                            {lines: [{line, ts_start, text, dur_s, confirmed}]}
POST /clones/pick           {"id", "lines": [line keys]}: make that voice's clone reference from the lines picked, in that
                            order (clone_say.pick), and count each as the owner's word that it is theirs. {ok, seconds,
                            clips, ref_text}; 409 {error} when they can't (too little, or a line that isn't theirs)
GET  /review/presence?day=   people, objects, and notable events of a day as time spans (Review's chips)
GET  /journal/evidence?day=  {sources: [{label, stream, ts_start, ts_end, stage, caption, ...}]}: what each source
                            behind the day page's citations can show now: "video", "clip" or "still" (url), "audio",
                            or "expired" with a note (memory/journal_evidence.py)
GET  /journal/evidence/<day>/<file>   a promoted clip (.mp4) or still (.jpg) of a cited moment
GET  /live/happening?since=&limit=  Live's What's happening (dashboard/happening.py): {items: [{id, ts, kind:
                            arrival | departure | talk, text, people, cam, camera_name}], updated}, oldest first; since
                            is exclusive, and polling with since=updated never repeats or drops a row
GET  /journal/today[?day=]   the Journal tab's current day (memory/journal_pages.py): {day, status: settled |
                            ready | empty | pending | error, md, coverage: {through, hours, elapsed}}. The day's
                            journal page once written, else a today-so-far page built in the background (202 pending,
                            with the last one built as md, until it's ready). Today, or yesterday until its page is
                            written; 404 otherwise, 400 for a bad day
GET  /review/summary?start=&end=  what happened over that stretch, as timeline blocks (memory/review_summary.py):
                            {start, end, level, status: ready | partial | pending, blocks: [{start, end, text, tags}]}
                            from the journal at the level that fits the stretch, each line plain and never first
                            person; 400 for a bad range (at most 31 days). ?day=YYYY-MM-DD instead uses the recording
                            timezone's midnight-to-midnight bounds, including 23/25-hour clock-change days.
GET  /places                 Known › Places' places (dashboard/places.py): {places: [{id, name, named, plan (a scene),
                            plan_source, machine_newer, revision, last_visit}]}; a page load with text/html gets the app
POST /places/plan            {place, plan}: the owner's plan (a scene) as taught state, or "" to withdraw it; a teaching write, so it follows dashboard.remote_teaching; {ok, plan, plan_source}
GET  /places/plan/history?place=  the place's owner plans, oldest first ({id, at, rooms, source, withdrawn})
POST /places/plan/restore    {place, record}: make that earlier owner plan hold again, as a new record (Revert plan);
                            a teaching write like POST /places/plan; {ok, plan, plan_source}
GET  /places/scene?place=     the Places 3D view's scene: the newest revision with a map, its floor, plan
                            transform, cameras, and plan in its frame; {revision: null} with no map; with camera=,
                            one camera's own cloud; with all=1, every camera's cloud placed in the place
POST /places/scene/follow    {place, camera, use}: put a camera that was placed again by itself back on the plan's
                            spot (use "spot"), or take that back ("cloud"); the plan is not written
GET  /places/map?place=&revision=  that revision's map (PLY), with byte ranges, an ETag, and immutable caching
GET  /places/splat?place=&revision=  that revision's 3D Gaussian splat (SOG), served like its map
GET  /places/positions?place=&start=&end=[&entity=]  where people and pets were on that place's plan:
                            {place, start, end, entities, tracks: [{entity, points}], rooms: [zone spans]}, at most
                            2,000 points an entity; 400 for no place or a bad window (at most 31 days)
GET  /journal[?month=|week=|day=|hour=]  the Journal tab's column view (dashboard/journal.py): {items, page}, the
                            next column's rows (months, a month's weeks, a week's days, a day's hours, an hour's
                            events) and the period's journal page; ?entry=<id>&start=<epoch> instead gives one
                            hour's or event's evidence, resolved as /journal/evidence resolves a page's; 400 for a
                            bad date or time, 404 for an entry that isn't there
GET  /review/search?day=&q=  day-scoped local meaning search: {results: [{id, ts, end, text, kind}]}
GET  /review/days            {days: [{day, video, audio, transcript}], today}: every day with something recorded,
                            oldest first, so Review opens on one and steps between them
GET  /clip?line=..          that line (its key, registry.key_of) as GET /clip?ts_start= gives it, for one microphone's
GET  /clip?ts_start=..      that utterance as WAV (PCM16, native rate) cut from the FLAC chunk(s) with 0.4 s padding,
                            following into the next chunk when the utterance runs past the file end; &info=1 -> JSON plan
GET  /speaker/<id>/latest-clip            most recent utterance of that speaker (merged ids resolved) as WAV
GET  /speaker/<id>/latest                 ...its transcript entry as JSON
GET  /history/at?ts=..&window=30         what each camera saw nearest ts (caption, faces), from camera history;
                                          this machine only, like /camera/query, unless
                                          dashboard.remote_camera_history lets password-checked browsers in
GET  /ambient               the microphones: {ts (while any is heard), devices: {name: {name, healthy}}} (mic_status)
GET  /vision                state/vision.status.json: each camera's state (starting, on, retrying)
GET  /stats                 cpu %, memory, pack daemons (psutil), Ollama loaded models
GET  /stats/history?since=  per-service CPU, memory, and per-GPU memory, a sample each 5 s for 30 min (machine_usage.py)
GET  /audio/levels          server-sent live mic levels read from the broker's rings (dashboard/mic_levels.py), about 25/s:
                            {"ts", "levels": {mic: {"rms", "peak"}}} in dBFS; {"error"} then close when no mic records
GET  /health                {ok, ts, clients, boot}; boot changes on every restart
GET  /static/<path>         the frontend's build (static/app/): pages, and content-hashed scripts, styles, fonts
GET  /services/history?hours=24|168   bounded service lifecycle metadata or labelled five-minute samples
GET  /services              admin sidebar: every service for this platform with its state, and "control" (may switch)
POST /services              {"name": "vision", "action": "on"} -> the application starts or stops one service, or
                            (enable, disable) pack.toml's [services] choice; this machine only, and refused while the
                            application isn't running.
                            {"all": "repair"} restarts every service in the cascade's order, {"all": "off"} stops
                            every one but the dashboard
GET  /devices               source picker: this machine's cameras and microphones, which are chosen and in use
POST /devices/sensitivity   {"mic": name, "sensitivity": 0..100 or null} -> audio.mic_sensitivity; applied live
POST /devices/auto_gain     {"mic": name, "on": bool} -> audio.mic_auto_gain; applied live
GET  /research/jobs          the Inbox: research jobs and their reports (dashboard/research.py, memory/research.py);
                             POST queues one; POST .../<id>/read marks one, .../<id>/delete removes it
GET  /device-links           Settings' phones, watches, and tags: devices seen (?hours=, 24 by default) and every link
                             (dashboard/device_links.py, proximity/devices.py)
POST /device-links           {"kind", "value", "target", "name"} -> link a device to a person or name it as a thing;
                             POST /device-links/remove {"id"} retires a link; this machine only
POST /device-locations       {"kind", "value", "place", "room"} -> say a device lives in a place or a room of its plan
                             ("" for the whole place); POST /device-locations/remove {"id"}; this machine only
POST /device-access-point    {"mac", "name"} -> name a Wi-Fi access point (devices.access_points; "" clears it); this
                             machine only. /device-comings and /device-whereabouts show it as each stay's `place`, with
                             the MAC as `place_id`, and list every name set as `access_points`
GET  /zones                  every camera's owner-drawn zones (dashboard/zones.py, see/zones.py); /zones/history?cam=
                             a camera's saved versions, newest first
POST /zones                  {"cam", "zones", "base"} -> a new version of that camera's zone set; this machine only
GET  /ingest/api/...         the Ingest tab: uploads, links, sources, and Google Photos (dashboard/ingest.py)
GET  /reports/api/...        the Reports tab: report figures and the chosen layout (dashboard/reports.py)
GET  /config                settings page: every editable setting, "control", and "needs_setup" (dashboard/settings.py)
POST /config                {"values": {"owner.name": "Alice"}} -> write pack.toml and restart; this machine only
"""

import contextlib
import errno
import html
import http.server
import json
import os
import pathlib
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import Executor
from types import FrameType
from typing import Any

# `python -m` must share the canonical module with lazy imports (Objects' app page).
# Otherwise that import executes the route decorators a second time.
if __name__ == "__main__":
    sys.modules["liberational.dashboard.server"] = sys.modules[__name__]

from liberational.dashboard import people_cards, roster_read

try:
    import psutil
except ImportError:
    psutil = None
try:
    import numpy as np
    import soundfile as sf
except ImportError:
    sf = np = None
from liberational import config as cfg
from liberational import (
    continuum_link,
    knowledge,
    memtrim,
    model_runtime,
    readers,
    speech_log,
    voice_turn,
)
from liberational.dashboard import (
    assets,
    context,
    devreload,
    feed_labels,
    inbox_state,
    live_status,
    machine_stats,
    people_list,
    people_reads,
    people_writes,
    photo_files,
    routes,
    service_control,
    settings,
    shell_status,
    slow,
    speaker_lines,
    transcript_lines,
    unnamed_groups,
)
from liberational.dashboard.context import LOG, SSE_CLIENTS, SSE_LOCK
from liberational.people import (
    registry,
    transcript_view,
)
from liberational.vision import views

# Every module that registers routes (dashboard/routes.py), named once so the table is whole however the server
# is imported; this module registers its own few at its end.
routes.load(
    "liberational.dashboard.feed_labels",
    "liberational.dashboard.people_cards",
    "liberational.dashboard.roster_read",
    "liberational.dashboard.people_corrections",
    "liberational.dashboard.people_references",
    "liberational.dashboard.face_examples",
    "liberational.dashboard.activities",
    "liberational.dashboard.ask_routes",
    "liberational.dashboard.call_log",
    "liberational.dashboard.device_links",
    "liberational.dashboard.devices",
    "liberational.dashboard.fresh_reads",
    "liberational.dashboard.frontend_log",
    "liberational.dashboard.happening",
    "liberational.dashboard.inbox_state",
    "liberational.dashboard.ingest",
    "liberational.dashboard.journal",
    "liberational.dashboard.known_voices",
    "liberational.dashboard.live_status",
    "liberational.dashboard.machine_stats",
    "liberational.dashboard.marks",
    "liberational.dashboard.network_cameras",
    "liberational.dashboard.people_list",
    "liberational.dashboard.people_reads",
    "liberational.dashboard.people_writes",
    "liberational.dashboard.pets",
    "liberational.dashboard.photo_files",
    "liberational.dashboard.places",
    "liberational.dashboard.playback",
    "liberational.dashboard.presence",
    "liberational.dashboard.reports",
    "liberational.dashboard.research",
    "liberational.dashboard.review_boxes",
    "liberational.dashboard.review_reads",
    "liberational.dashboard.service_control",
    "liberational.dashboard.settings",
    "liberational.dashboard.skeletons",
    "liberational.dashboard.speaker_lines",
    "liberational.dashboard.surroundings",
    "liberational.dashboard.transcript_lines",
    "liberational.dashboard.unnamed_groups",
    "liberational.dashboard.voice_clones",
    "liberational.dashboard.zones",
)

STATIC = assets.STATIC
PURGE_POLL_S = 5.0  # how often the live feed asks whether a purge took lines it showed


def _scene_key(entries: list[dict[str, Any]]) -> tuple[Any, ...]:
    """What changes when the scene does: its newest entries' ids."""
    return tuple(e.get("id") for e in entries[-5:])


PORT = cfg.get("ports.dashboard")  # main() takes an optional port argument
APP_PATHS = (
    "/",
    "/review",
    "/people",
    "/pets",
    "/activities",  # Known › Activities; GET /activities?day= without text/html stays the JSON API
    "/devices",  # Known › Devices: phones, watches, and tags; /devices/... stay the API
    "/places",  # Known › Places: floorplans of the indoor places the Pack has been
    "/journal",  # the Journal tab; GET /journal?... without text/html stays the JSON API
    "/calls",
    "/inbox",
    "/ingest",
    "/statistics",
    "/settings",
    "/debug",
    "/skeletons",  # the skeleton playback test page, not in the nav
)  # one page app; the tab comes from the path
APP_PAGE = "index.html"  # the app (frontend/), built into static/app/
DEV = False  # main() sets it for --dev: auto-reload (dashboard/devreload.py)


def jdump(o: Any) -> bytes:
    return json.dumps(o, separators=(",", ":")).encode()


# ---- the live feed's transcript: which lines a label changed ----
AUTONAME_EVERY_S = 60.0  # how often clear face matches are named without a tag


def autoname_faces_once() -> list[dict[str, Any]]:
    """One round of automatic face naming: the waiting groups that clearly match a named person, by the live
    match gate, are named as an embedding match, each with its own undo. Put-aside groups are left alone."""
    named = people_list._faces_mod().autoname(inbox_state.inbox_skips()["face"], state=context.STATE)
    if named:
        LOG.info("faces: named automatically: %s", ", ".join(f"Face {r['gid']} as {r['name']}" for r in named))
    return named


CHECK_START_S = 120.0  # the faces' held-out check waits this long after a start


CHECK_TIMEOUT_S = 600.0  # the check's child is given this long (it takes seconds; a month of faces at idle priority)


def _run_check() -> dict[str, Any]:
    """The faces' held-out check, run in a child process at idle priority: it decodes a month of face
    embeddings, and in the dashboard's own process that held every request up for seconds every CHECK_EVERY_S. The
    child saves faces/check.json for live naming and autoname, and prints the result."""
    done = subprocess.run(
        [sys.executable, "-m", "liberational.vision.faces", "check", "--state", str(context.STATE)],
        capture_output=True,
        text=True,
        timeout=CHECK_TIMEOUT_S,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"exit {done.returncode}: {done.stderr.strip()[-500:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def check_faces_once(last: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rerun the faces' held-out check over the owner's tags, which live naming and automatic naming obey, and
    log it when its status or its suspected mis-tags change from `last`."""
    check = _run_check()
    seen = (check["status"], [s["gid"] for s in check["suspects"]])
    if last is None or seen != (last.get("status"), [s["gid"] for s in last.get("suspects", [])]):
        suspects = [
            f"Face {s['gid']} is tagged {s['tagged']} but looks like {s['looks_like']}" for s in check["suspects"]
        ]
        LOG.info(
            "faces: held-out check %s (%s right, %s wrong, %s unknown of %s tagged groups)%s%s",
            check["status"],
            check["right"],
            check["wrong"],
            check["unknown"],
            check["groups"],
            f": {check['reason']}" if check["reason"] else "",
            "".join(" " + s for s in suspects),
        )
    return check


def _autoname_faces_forever() -> None:
    """Every AUTONAME_EVERY_S: a group that forms or reaches two sightings, or a tag that teaches the gallery, can make
    a clear match. The held-out check reruns every CHECK_EVERY_S, first CHECK_START_S after a start: it decodes a
    month of face embeddings (about 4 s alone, 24 s while the start's warm-ups run), and the saved one holds
    until then. A failed round is logged and the next one tries again."""
    check: dict[str, Any] | None = None
    checked = time.monotonic() - people_list._faces_mod().CHECK_EVERY_S + CHECK_START_S
    while True:
        if time.monotonic() - checked >= people_list._faces_mod().CHECK_EVERY_S:
            try:
                check = check_faces_once(check)
            except Exception as ex:
                LOG.warning("faces: held-out check failed: %s", repr(ex))
            checked = time.monotonic()
        try:
            autoname_faces_once()
        except Exception as ex:
            LOG.warning("faces: automatic naming failed: %s", repr(ex))
        time.sleep(AUTONAME_EVERY_S)


def _warm_faces() -> threading.Thread:
    """Import vision/faces.py (and cv2) in the background at start, so the first GET /people doesn't."""

    def run() -> None:
        context.low_priority(19)
        voice_turn.wait_quiet()
        try:
            people_list._faces_mod()
        except Exception as ex:
            LOG.warning("faces helpers not preloaded: %s", ex)

    t = threading.Thread(target=run, daemon=True, name="faces-import")
    t.start()
    return t


# ----------------------------------------------------------------- http ----
LOG_POLL_S = 0.5  # how often the Debug tab's tail looks for new log lines (/logs/stream)
LOG_BATCH = 500  # lines per "lines" event


def _pw_file() -> str:
    try:
        return (context.STATE / "dash_password").read_text().strip()
    except Exception:
        return ""


DASH_PASSWORD = cfg.get("dashboard.password") or _pw_file()


def remote_teaching() -> bool:
    """dashboard.remote_teaching, read per request so a Settings change applies without a restart."""
    return bool(cfg.get("dashboard.remote_teaching"))


def remote_camera_history() -> bool:
    """dashboard.remote_camera_history, read per request so a Settings change applies without a restart."""
    return bool(cfg.get("dashboard.remote_camera_history"))


def remote_control() -> bool:
    """dashboard.remote_control, read per request so a Settings change applies without a restart."""
    return bool(cfg.get("dashboard.remote_control"))


def fill(s: str) -> str:
    """An HTML page with the assistant and owner names filled in, and NAMES for its scripts (with the cameras and
    whether dashboard.hotkeys is on)."""
    names = {"assistant": cfg.assistant_name(), "owner": cfg.owner_name()}
    cams = views.camera_ids()
    # views: each camera's name (vision.camera_names), for Review's lane labels; a page load picks up a rename
    shell = {
        **names,
        "cams": cams,
        "activeCams": views.camera_ids(active_only=True),
        "views": views.names(),
        "hotkeys": bool(cfg.get("dashboard.hotkeys")),
        "gatewayPort": int(cfg.get("ports.gateway")),  # where the page finds the camera gateway
        "needsSetup": not cfg.local_config().exists(),  # first run: the shell shows setup without asking /config
    }
    # JSON.parse retains literal __proto__ keys; a JavaScript object literal silently drops a string alias.
    shell_json = json.dumps(json.dumps(shell)).replace("<", "\\u003c")
    s = s.replace("{{names_json}}", f"JSON.parse({shell_json})")
    for k, v in names.items():
        s = s.replace("{{" + k + "}}", html.escape(v))
    return s


def vite(path: str) -> tuple[bytes, str] | None:
    """(body, content type) of `path` from Vite's dev server (ports.vite), or None when it isn't running. The Host
    it checks is its own, so the request is the one a browser on this machine would send."""
    port = int(cfg.get("ports.vite"))
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers={"Host": f"localhost:{port}"})
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.read(), r.headers.get("Content-Type") or "application/octet-stream"
    except OSError:
        return None


def app_page(name: str = APP_PAGE) -> tuple[int, bytes]:
    """(code, body) of a page of the frontend, filled in (`fill`). In --dev it comes from
    Vite, which adds its client and React's refresh preamble, and falls back to the build when Vite isn't up. A
    build older than frontend/ gets a banner saying so. With no build at all, a page that says how to make one (503:
    never a 404, so the Python side never needs Node)."""
    if DEV and (got := vite("/static/app/" + name)) is not None:
        return 200, fill(got[0].decode()).encode()
    if DEV:
        LOG.info(
            f"dev: Vite isn't answering on port {cfg.get('ports.vite')} (ports.vite), so the page is the last build"
            " (run `mise run dev`, which starts it)"
        )
    text, built = assets.built_page(name)
    if not built:
        return 503, text.encode()
    if assets.stale():
        text = text.replace("<body>", "<body>" + assets.STALE_BANNER, 1)
    return 200, fill(text).encode()


def _client_left(ex: BaseException | None) -> bool:
    """The client hung up: a reset, a broken pipe, an abort, or macOS's EPROTOTYPE from a socket mid-teardown."""
    return isinstance(ex, ConnectionError) or (isinstance(ex, OSError) and ex.errno == errno.EPROTOTYPE)


SLOW_REQUEST_S = 5.0  # a request slower than this is logged as a warning, with its route and duration


def _route(path: object) -> str:
    """A request's path without its query, for the log: a query can hold names."""
    return str(path or "?").split("?", 1)[0]


MEDIA_CHUNK = 1 << 20  # a media file is sent a MiB at a time, never read whole


def _server_time(kind: str, headers_after: float | None, done_after: float) -> float:
    """How long the server itself took over a request: for audio and video, until its headers went out, since a
    browser keeps a media response open while it plays (a 5 min segment read over a minute is not slow)."""
    if kind.startswith(("video/", "audio/")) and headers_after is not None:
        return headers_after
    return done_after


class DashboardServer(http.server.ThreadingHTTPServer):
    # Python's default listen backlog is 5: a browser opening a dozen connections at once (page assets, event stream,
    # camera mux) got some refused, which Tailscale Funnel reports as 502, and the remote page never finished
    # loading. 128 is macOS's kern.ipc.somaxconn.
    request_queue_size = 128
    daemon_threads = True

    _people_warm_started = False

    def server_close(self) -> None:
        photo_files._crop_jobs.stop()
        super().server_close()

    def service_actions(self) -> None:
        """Start warming once the serving loop is ready; never hold up startup or requests."""
        super().service_actions()
        if not self._people_warm_started:
            self._people_warm_started = True
            try:
                photo_files._crop_jobs.start(lambda: context.low_priority(19))
            except Exception as error:
                LOG.error("thumbnail worker could not start: %s", repr(error))
            try:
                threading.Thread(target=people_reads._warm_people, daemon=True, name="warm-people").start()
            except Exception as error:
                LOG.warning("people: warm-up could not start: %s", repr(error))

    def handle_error(self, request: Any, client_address: Any) -> None:
        """An error outside any one request (H handles those): one log line, not socketserver's traceback dump."""
        ex = sys.exc_info()[1]
        if _client_left(ex):
            LOG.debug(f"a client at {client_address[0]} left ({type(ex).__name__})")
        else:
            LOG.error(f"a request from {client_address[0]} failed", exc_info=ex)


class H(http.server.BaseHTTPRequestHandler):
    server_version = "pack-dashboard/1"
    _t_req: float | None = None  # when this request's line arrived (parse_request), for its duration

    def handle_one_request(self) -> None:
        """One request. A client that hangs up mid-response is one debug line, or one error line when the request
        was slow (it likely gave up waiting); any other failure is one error line naming the route, with its
        traceback, and the connection is closed. A request slower than SLOW_REQUEST_S is a warning with its
        duration, except a stream, which is long on purpose. The duration counts from when the request line
        arrived, not from when the connection opened (a browser can open one and send on it seconds later);
        while it runs, slow.py's watchdog can say what it waits on."""
        self._content_type, self._headers_at, self._t_req, t0 = "", None, None, time.monotonic()
        self._left: OSError | None = None  # the client hung up while _send wrote the reply
        try:
            with knowledge.lock_wait(knowledge.REQUEST_WAIT_S):  # fail fast, don't queue behind a lock
                super().handle_one_request()
        except Exception as ex:
            took = self._took(self._t_req or t0)
            route = f"{getattr(self, 'command', '?')} {_route(getattr(self, 'path', None))}"
            # Only a request that arrived can be given up on: an early connection closed unused is no request
            if _client_left(ex) and self._t_req is not None and took > SLOW_REQUEST_S and not self._streamed():
                LOG.error(f"{route}: the client gave up after {took:.1f} s")
            elif _client_left(ex):
                LOG.debug(f"{route}: the client left ({type(ex).__name__})")
            elif isinstance(ex, roster_read.Starting) or continuum_link.broker_outage(ex):
                if not isinstance(ex, roster_read.Starting):
                    continuum_link.OUTAGE.lost(ex)  # one WARN for the outage, not a traceback per request
                if self._headers_at is None:
                    self._starting(str(ex) if isinstance(ex, roster_read.Starting) else "The data broker is starting.")
            else:
                LOG.exception(f"{route} failed")
            self.close_connection = True
            return
        finally:
            slow.WATCH.end()
        took = self._took(self._t_req or t0)
        if took > SLOW_REQUEST_S and getattr(self, "command", None) and not self._streamed():
            route = f"{self.command} {_route(getattr(self, 'path', None))}"
            if self._left is not None and self._t_req is not None:  # it was slow, and the client didn't wait for it
                LOG.error(f"{route}: the client gave up after {took:.1f} s")
            else:
                LOG.warning(f"{route} took {took:.1f} s")

    def send_header(self, keyword: str, value: str) -> None:
        if keyword.lower() == "content-type":
            self._content_type = str(value)
        super().send_header(keyword, value)

    def _took(self, t0: float) -> float:
        at = getattr(self, "_headers_at", None)
        return _server_time(getattr(self, "_content_type", ""), None if at is None else at - t0, time.monotonic() - t0)

    def end_headers(self) -> None:
        self._headers_at = time.monotonic()
        super().end_headers()

    def _streamed(self) -> bool:
        kind = getattr(self, "_content_type", "")
        return kind.startswith(("text/event-stream", "multipart/"))

    def _long_on_purpose(self) -> bool:
        """A stream, or audio or video whose headers have gone out (the browser holds it open while it plays):
        never reported as stuck by the watchdog."""
        kind = getattr(self, "_content_type", "")
        return self._streamed() or (kind.startswith(("video/", "audio/")) and self._headers_at is not None)

    def log_message(self, format: str, *args: Any) -> None:
        pass

    @property
    def state(self) -> pathlib.Path:
        """The state directory, for a route registered by another module (dashboard/routes.py)."""
        return context.STATE

    def _local(self) -> bool:
        """A direct request from this machine, not one Tailscale proxied in."""
        return self.client_address[0] in ("127.0.0.1", "::1") and not self.headers.get("X-Forwarded-For")

    def _same_origin_json(self, content_type: str = "application/json") -> bool:
        """A JSON request (or content_type's) from a page this dashboard served: no Origin (not a browser), or one
        naming this Host over http or https (the dashboard is also served over HTTPS, as Tailscale serves it). A
        cross-site page can only send text/plain without a preflight, and its Origin names its own site."""
        origin, host = self.headers.get("Origin"), self.headers.get("Host") or ""
        return self.headers.get_content_type() == content_type and (
            not origin or origin in (f"http://{host}", f"https://{host}")
        )

    def _may_see_camera_history(self) -> bool:
        """Camera history (captions, face names): this machine, or, with dashboard.remote_camera_history on, any
        request that passed the dashboard password (_authed: every non-loopback request does). Never without one."""
        return self._local() or (bool(DASH_PASSWORD) and remote_camera_history())

    def _can_teach(self, content_type: str = "application/json") -> bool:
        """Teaching writes (names, faces, voices, pets, people and their reference photos, the Inbox, zones, transcript
        fixes): what _can_control allows, or, with dashboard.remote_teaching on, a request that
        passed the dashboard password and came as JSON (or content_type) from this dashboard's own page, so another
        site can't post it. Never without a password. Owner identity changes are not teaching and take _can_control
        alone (TEACHING_ROUTES)."""
        return self._can_control() or (
            bool(DASH_PASSWORD) and remote_teaching() and self._same_origin_json(content_type)
        )

    def _can_control(self) -> bool:
        """Local, and addressed to a loopback name, so a DNS-rebound page can't reach the service controls; or, with
        dashboard.remote_control on, a request from a device on the tailnet (_tailnet) that passed the dashboard
        password and, unless it only reads, came from this dashboard's own page. Never without a password."""
        return self._can_control_here() or (
            bool(DASH_PASSWORD) and remote_control() and self._tailnet() and self._same_origin()
        )

    def _tailnet(self) -> bool:
        """Proxied in by Tailscale Serve from a signed-in device on the tailnet, not from the internet over Funnel.
        Serve connects from loopback, drops any Tailscale-User-* headers a client sent, then names the tailnet user;
        a Funnel request gets no user and is marked Tailscale-Funnel-Request."""
        return (
            self.client_address[0] in ("127.0.0.1", "::1")
            and bool(self.headers.get("Tailscale-User-Login"))
            and not self.headers.get("Tailscale-Funnel-Request")
        )

    def _same_origin(self) -> bool:
        """A read, or a write whose Origin is absent (not a browser) or names this Host over http or https, so another
        site's page can't post it with the browser's saved password."""
        if self.command in ("GET", "HEAD"):
            return True
        origin, host = self.headers.get("Origin"), self.headers.get("Host") or ""
        return not origin or origin in (f"http://{host}", f"https://{host}")

    def _can_control_here(self) -> bool:
        """Local, and addressed to a loopback name, so a DNS-rebound page can't reach the service controls."""
        raw = self.headers.get("Host") or ""
        if not raw:
            return False
        try:
            host = settings._host(self)
        except ValueError:  # An opening IPv6 bracket without a closing one is not a host.
            return False
        # _host extracts a bracketed name even with junk after it. Keep refusing
        # those suffixes (and extra colons) rather than broadening this access check.
        suffix = raw[len(host) :]
        valid_suffix = not suffix or (suffix.startswith(":") and ":" not in suffix[1:])
        return self._local() and host in ("localhost", "127.0.0.1", "[::1]") and valid_suffix

    def _authed(self) -> bool:
        # Loopback requests (agent, local tools) skip auth; everything else (Funnel/Serve/LAN) needs HTTP Basic auth.
        if not DASH_PASSWORD or self._local():
            return True
        import base64

        h = self.headers.get("Authorization") or ""
        if h.startswith("Basic "):
            try:
                if base64.b64decode(h[6:]).decode().split(":", 1)[-1] == DASH_PASSWORD:
                    return True
            except Exception:
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="pack dashboard"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def parse_request(self) -> bool:
        ok = super().parse_request()
        if ok:  # the request line is in: the request's own time starts now
            self._t_req = time.monotonic()
            slow.WATCH.begin(f"{self.command} {_route(self.path)}", self._long_on_purpose, self._t_req)
        if ok and not self._authed():
            self.close_connection = True
            return False
        return ok

    def _send(self, code: int, ct: str, b: bytes) -> None:
        # Compress only source assets/page shells, not private API data or media ranges.
        compressible = code == 200 and (
            ct.startswith("text/html")
            or (self.path.startswith("/static/") and ct.startswith(("text/", "image/svg+xml")))
        )
        encoding = None
        if compressible and len(b) >= 512 and assets.accepts_gzip(self.headers.get("Accept-Encoding") or ""):
            packed = assets.compressed(b)
            if len(packed) < len(b):
                b, encoding = packed, "gzip"
        self.send_response(code)
        self.send_header("Content-Type", ct)
        static = code == 200 and self.path.startswith("/static/") and not DEV  # Vite's dev files change in place
        self.send_header("Cache-Control", assets.cache_control(self.path.split("?")[0]) if static else "no-store")
        if compressible:
            self.send_header("Vary", "Accept-Encoding")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.send_header("Content-Length", str(len(b)))
        # A browser that left (a reload, a closed tab) is one debug line here, never an error: a route writing its
        # reply inside its own try (GET /live/happening, /review/summary) would otherwise log it and write a 500 to
        # the dead socket.
        try:
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(b)
        except OSError as ex:
            if not _client_left(ex):
                raise
            LOG.debug(
                f"{self.command} {_route(self.path)}: the client left before the reply was written ({type(ex).__name__})"
            )
            self._left = ex  # handle_one_request names a slow request it left as the client giving up
            self.close_connection = True

    def _starting(self, reason: str) -> None:
        """503 for a route that will answer shortly: `starting` tells the frontend to wait it out and ask again."""
        self._headers_buffer = []  # drop anything a failed route began to write
        body = jdump({"ok": False, "error": f"{reason} Try again in a moment.", "starting": True})
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Retry-After", "1")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, o: Any, code: int = 200) -> None:
        self._send(code, "application/json", jdump(o))

    def _client_gone(self) -> bool:
        """Whether the client closed its side while a GET is still being worked on (a page that gave up waiting, or
        moved on to another window): its socket reads as ended. Bytes waiting there are a next request on a
        kept-alive connection, so it's still here. A long read asks this as it goes, to stop work nobody will get."""
        try:
            ready, _, _ = select.select([self.connection], [], [], 0)
            return bool(ready) and self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except BlockingIOError:
            return False
        except OSError, ValueError:  # reset, or already closed
            return True

    def _media(
        self, b: bytes | pathlib.Path, ct: str, extra: dict[str, Any] | None = None, cache: str = "no-store"
    ) -> None:
        """Whole body or a single byte range (Safari fetches media with Range requests and expects 206). A file is
        sent from disk, only the bytes asked for and a MiB at a time: a video segment is 30 to 70 MB, and a
        browser holds its range open while it plays."""
        # A file is opened before anything is sent: retention deleting it after this is fine (an open file outlives
        # its name), and one already gone is a plain 404.
        try:
            f = b.open("rb") if isinstance(b, pathlib.Path) else None
        except OSError:
            return self._json({"error": "not found"}, 404)
        with f if f is not None else contextlib.nullcontext():
            self._send_media(b, f, ct, extra, cache)

    def _send_media(
        self, b: bytes | pathlib.Path, f: Any, ct: str, extra: dict[str, Any] | None, cache: str = "no-store"
    ) -> None:
        total = len(b) if isinstance(b, bytes) else os.fstat(f.fileno()).st_size
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if m and (m.group(1) or m.group(2)):
            a = int(m.group(1)) if m.group(1) else max(0, total - int(m.group(2)))
            e = min(total - 1, int(m.group(2))) if (m.group(1) and m.group(2)) else total - 1
            if a >= total or a > e:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{total}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{e}/{total}")
        else:
            self.send_response(200)
            a, e = 0, total - 1
        self.send_header("Content-Type", ct)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", cache)
        for k, v in (extra or {}).items():
            self.send_header(k, str(v))
        self.send_header("Content-Length", str(e - a + 1))
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            if f is not None:
                f.seek(a)
                left = e - a + 1
                while left > 0 and (chunk := f.read(min(MEDIA_CHUNK, left))):
                    self.wfile.write(chunk)
                    left -= len(chunk)
                if left:  # it shrank since: the client gets a short body, and the connection mustn't carry another
                    LOG.debug(f"{self.command} {_route(self.path)}: the file ended {left} bytes early")
                    self.close_connection = True
            elif isinstance(b, bytes):
                self.wfile.write(b[a : e + 1])
        except OSError as ex:
            if not _client_left(ex):
                raise
            # The browser cancelled the range (a seek, a close, a new range): normal for video, not a failure.
            LOG.debug(
                f"{self.command} {_route(self.path)}: the client left during the media write ({type(ex).__name__})"
            )
            self.close_connection = True

    def clip(self, entry: dict[str, Any], info: bool = False) -> None:
        """A line's audio as a WAV (line_clip), or with `info` what it holds."""
        from continuum_py.read import ChunkMissing  # with the reader, like the module's other continuum_py imports

        if not transcript_lines.has_audio(entry):
            return self._json({"ok": False, "error": "no audio for that utterance"}, 404)
        try:
            data, meta = transcript_lines.line_clip(entry)
        except ChunkMissing as e:
            # The catalog names a chunk file that isn't on disk (logged once per file where it's read).
            return self._json({"ok": False, "error": f"the audio's chunk file is missing: {e.filename}"}, 404)
        except Exception as e:
            LOG.error("clip error: %s", repr(e))
            return self._json({"ok": False, "error": repr(e)[:200]}, 500)
        if info:
            return self._json(dict(meta, entry=entry, wav_bytes=len(data)))
        if not meta["frames"]:
            # Kept nowhere: still in the microphone's ring (a line of the last few seconds), or purged.
            return self._json({"ok": False, "error": "no audio kept for that line (yet)", "meta": meta}, 404)
        hdr = {"X-Clip-Dur": meta["dur_s"], "X-Clip-Sr": meta["sr"], "X-Clip-Ts-Start": entry.get("ts_start")}
        hdr["X-Clip-Speaker"] = entry.get("speaker_id")
        return self._media(data, "audio/wav", hdr)

    def do_HEAD(self) -> None:
        return self.do_GET()

    def _query(self) -> dict[str, str]:
        """The query string's values by key, decoded (the first value of a key that is given twice, a blank value as
        ""). One reader for every route: urllib.parse.parse_qs, so a `+` is the space URLSearchParams encodes it as,
        while a literal `+` sent as %2B still arrives as one (a clip's line key is looked up with its own spelling).
        Hand-splitting on & and = left `+` alone, so a camera named with a space could not be named in a query."""
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        return {k: v[0] for k, v in urllib.parse.parse_qs(query, keep_blank_values=True).items()}

    def do_GET(self) -> None:
        p = self.path.split("?")[0]
        if p in APP_PATHS and (p == "/" or "text/html" in (self.headers.get("Accept") or "")):
            # Page loads only, ahead of every module's routes: a tab whose module also serves its path as JSON
            # (GET /devices, /people, /calls) keeps it for fetch(), and a reload of the tab still gets the app.
            # The app is one page; its router reads the tab from the path.
            code, body = app_page()
            return self._send(code, "text/html; charset=utf-8", body)
        found = routes.find("GET", p)  # every other route is registered in dashboard/routes.py's table (#61)
        if found is None:
            return self._send(404, "text/plain", b"")
        route, match = found
        return route.handler(self, match) if route.kind == "pattern" else route.handler(self)

    def _stream_render(self, r: Any) -> None:
        """A clone line still being made (clone_stream.Render): a WAV of open length, each piece sent as it comes, with
        no length, so it ends when the line does. Range requests get it from the start too: it can't be skipped
        into before it exists."""
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if self.command == "HEAD":
            return
        try:
            for b in r.stream():
                self.wfile.write(b)
                self.wfile.flush()
        except OSError as ex:
            if not _client_left(ex):
                raise

    def _wav_file(self, path: str | pathlib.Path) -> None:
        """Send a wav from disk: _media streams a file a MiB at a time (a call's recording can be minutes long), sends
        Cache-Control once, and answers a missing file with its own 404."""
        return self._media(pathlib.Path(path), "audio/wav")

    def do_DELETE(self) -> None:
        found = routes.find("DELETE", self.path.split("?")[0])
        if found is None:
            return self._json({"error": "Method not supported."}, 405)
        return found[0].handler(self)

    def do_POST(self) -> None:
        # A registry write that failed (registry.recorded) wasn't done, or was undone: said, never answered as done.
        # A name, face, voice, or card change: no card is answered from before it. Cleared before (so a GET during it
        # reads afresh) and after (so a read that began during it is dropped, _CardReads.epoch).
        edit = self.path.split("?")[0].startswith(people_writes.CARD_EDITS)
        if edit:
            people_cards._CARD_READS.clear()
        try:
            self._post()
        except registry.WriteFailed as e:
            self._json({"ok": False, "error": str(e)}, 500)
        finally:
            if edit:
                people_cards._CARD_READS.clear()

    def _post(self) -> None:
        p = self.path.split("?")[0]
        if p in people_writes.TEACHING_ROUTES and not self._can_teach(
            people_writes.UPLOAD_TYPE if p == "/people/reference-upload" else "application/json"
        ):
            self.close_connection = True  # its body is never read
            code, o = people_writes.teaching_refused()
            return self._json(o, code)
        found = routes.find("POST", p)  # every other route is registered in dashboard/routes.py's table (#61)
        if found is not None and found[0].raw:
            return found[0].handler(self)  # reads its own body: an upload, raw audio, or the Objects API
        if p.startswith("/people/reference-") or p == "/people/search":
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 <= length <= 4096:
                self.close_connection = True
                return self._json({"error": "Reference request too large"}, 413)
        # All other POST endpoints: JSON body
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body: Any = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if (p.startswith("/people/reference-") or p == "/people/search") and not isinstance(body, dict):
                raise ValueError("object required")
        except Exception:
            return self._json({"ok": False, "error": "bad json"}, 400)
        if found is None:
            return self._json({"ok": False, "error": "unknown endpoint"}, 404)
        return found[0].handler(self, body)

    # -- server-sent events: one thread per client, file watch at 0.5 s
    def levels(self) -> None:
        """Stream the microphones' live levels, read from the broker's rings (dashboard/mic_levels.py), until the
        browser leaves."""
        import asyncio

        from liberational import continuum_link
        from liberational.dashboard import mic_levels

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def send(msg: dict[str, Any]) -> None:
            self.wfile.write(b"data: " + jdump(msg) + b"\n\n")
            self.wfile.flush()

        try:
            asyncio.run(mic_levels.stream(continuum_link.socket_path(), send, self._hung_up))
        except OSError:  # the browser left, or the broker isn't there
            with contextlib.suppress(OSError):
                send({"error": "The broker isn't running, so there are no live levels."})
        except Exception as e:
            LOG.warning("levels: %s", repr(e))

    def _hung_up(self) -> bool:
        """Whether the client has closed its end of the connection. A stream's client sends nothing after its request,
        so a readable socket with nothing to read means it hung up: a quiet stream would otherwise only find out at
        its next keep-alive write, holding a thread (and its log files) for up to 15 s after the tab closed."""
        try:
            if not select.select([self.connection], [], [], 0)[0]:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True
        except ValueError:  # a socket that can't peek (TLS): the keep-alive write still finds a gone client
            return False

    def log_stream(self) -> None:
        """The Debug tab's live tail of every service's log (dashboard/logtail.py): a backlog, then new lines as they
        are written, in "lines" events of [{s: service, l: text, t: when it was read, or null in the backlog}]. The
        "hello" event names the services and those with no log file yet. ?services=a,b picks services by their services.toml names; any other name is refused."""
        from liberational.dashboard import logtail

        wanted = self._query().get("services", "")
        try:
            tail = logtail.Tailer(logtail.pick([n.strip() for n in wanted.split(",")] if wanted else None))
        except ValueError as ex:
            return self._json({"error": str(ex)}, 400)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def ev(name: str, data: Any) -> None:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode())
            self.wfile.flush()

        def send(lines: list[tuple[str, str]], t: float | None) -> None:
            for i in range(0, len(lines), LOG_BATCH):  # a burst goes out in pieces the browser can draw
                ev("lines", [{"s": s, "l": text, "t": t} for s, text in lines[i : i + LOG_BATCH]])

        try:
            self.wfile.write(b"retry: 2000\n\n")
            ev("hello", {"ts": time.time(), "services": list(tail.files), "missing": tail.missing()})
            send(tail.start(), None)
            quiet = time.monotonic()
            while True:
                time.sleep(LOG_POLL_S)
                if self._hung_up():
                    break
                if got := tail.poll():
                    send(got, time.time())
                    quiet = time.monotonic()
                elif time.monotonic() - quiet > 15:
                    self.wfile.write(b": keep-alive\n\n")  # so a proxy keeps the stream open, and a gone client shows
                    self.wfile.flush()
                    quiet = time.monotonic()
        except OSError:  # the browser left
            pass
        finally:
            tail.close()

    def _shell_status(self) -> shell_status.Status:
        """This feed's shell status: what GET /devices/privacy, /services, and /research/jobs would answer this
        viewer, each read once for every feed open at the same time. Research is the owner's alone, as its route is."""
        from liberational.dashboard import privacy
        from liberational.dashboard import research as research_routes

        control = self._can_control()
        periods = shell_status.PERIOD_S
        sources: dict[str, Callable[[], Any]] = {
            "privacy": lambda: privacy.view(control),
            "services": lambda: shell_status.shared(
                ("services", control), periods["services"], lambda: service_control.services_view(control)
            ),
        }
        # As ask._owner_http decides the route: this machine, or a viewer who passed the dashboard password.
        if self._local() or bool(DASH_PASSWORD):
            sources["research"] = lambda: shell_status.shared("research", periods["research"], research_routes.view)
        return shell_status.Status(sources)

    def sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        with SSE_LOCK:
            SSE_CLIENTS[0] += 1

        def ev(name: str, data: Any) -> None:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode())
            self.wfile.flush()

        try:
            self.wfile.write(b"retry: 2000\n\n")
            mark = speaker_lines.log_mark()
            from liberational.listen import revisions

            review_mark = revisions.review_revision(context.STATE)
            _, seen = feed_labels.relabeled_tail(
                {}
            )  # before the tail, so a label given in between is sent again, not missed
            # The speech log's position before the tail: a line committed in between is sent twice rather than missed.
            speech = speech_log.shared()
            streams = (speech_log.UTTERANCES, speech_log.REPLIES)
            seq, purge = speech.mark(streams)
            entries = transcript_lines.merged_tail(80)
            ev("hello", {"ts": time.time(), "clients": SSE_CLIENTS[0]})
            voices = speaker_lines.speakers_public()
            ev("speakers", voices)
            ev("transcript", {"entries": entries, "reset": True})
            ev("scene", live_status.scene_entries())
            mics = live_status.mic_status()
            ev("ambient", mics)
            cams = live_status.camera_status()
            ev("vision", cams)
            # The shell's status (#52): pauses, the services badge, and the Inbox's jobs, sent when they change.
            status = self._shell_status()
            for kind, value in status.changed():
                ev(kind, value)
            ev("ready", {"ts": time.time()})  # the opening snapshot is complete: a kind not sent yet won't be
            # The scene is asked of the vision log every SCENE_POLL_S, and sent when its newest captions change.
            scene_at, scene_seen = time.monotonic(), _scene_key(live_status.scene_entries())
            purge_at = time.monotonic()
            last_ping = time.time()
            while True:
                time.sleep(0.5)
                voices_due = False  # a line heard or purged may change the voice list's counts
                if time.monotonic() - purge_at >= PURGE_POLL_S:
                    # A purge (a person forgotten) may have taken lines the page shows: resend the tail.
                    purge_at = time.monotonic()
                    if (now_purge := speech.newest_purge(purge)) != purge:
                        purge = now_purge
                        _, seen = feed_labels.relabeled_tail({})
                        seq = speech.mark(streams)[0]
                        ev("transcript", {"entries": transcript_lines.merged_tail(80), "reset": True})
                        voices_due = True
                new, seq = speech.since(seq, streams)
                heard_now = [e for e in new if not e.get("assistant")]
                if heard_now:
                    voices_due = True
                    heard_now = transcript_lines._view().lines(heard_now)
                    seen.update(feed_labels.label_marks(heard_now))
                    heard_now = transcript_view.once(heard_now)  # another mic's copy in the same batch shows once
                    ev("transcript", {"entries": heard_now, "reset": False})
                if replies := [e for e in new if e.get("assistant")]:  # the assistant's replies, in the same feed
                    ev("transcript", {"entries": replies, "reset": False})
                if (
                    m := speaker_lines.log_mark_shared()
                ) != mark:  # a label: send the lines it changed again, to replace the old ones
                    mark = m
                    if (review_now := revisions.review_revision(context.STATE)) != review_mark:
                        review_mark = review_now
                        _, seen = feed_labels.relabeled_tail({})
                        ev(
                            "transcript",
                            {"entries": transcript_lines.merged_tail(feed_labels.LIVE_LINES), "reset": True},
                        )
                    else:
                        changed, seen = feed_labels.relabeled_tail(seen)
                        if changed:
                            ev("transcript", {"entries": changed, "reset": False, "replace": True})
                    if (now_voices := speaker_lines.speakers_public()) != voices:  # a voice's name lives in the log
                        voices = now_voices
                        ev("speakers", voices)
                if time.monotonic() - scene_at >= live_status.SCENE_POLL_S:
                    scene_at = time.monotonic()
                    scene = live_status.scene_entries()
                    if (key := _scene_key(scene)) != scene_seen:
                        scene_seen = key
                        ev("scene", scene)
                if voices_due and (now_voices := speaker_lines.speakers_public()) != voices:
                    voices = now_voices
                    ev("speakers", voices)
                if (now_mics := live_status.mic_status()) != mics:  # cached: asked of the broker at most every 2 s
                    mics = now_mics
                    ev("ambient", mics)
                if (now_cams := live_status.camera_status())["cameras"] != cams["cameras"] or now_cams["ts"] - cams[
                    "ts"
                ] > 5:
                    cams = now_cams  # sent when it changes, and every 5 s so the page knows it's fresh
                    ev("vision", cams)
                for kind, value in status.changed():
                    ev(kind, value)
                # An event, not a comment, so pages can tell a quiet stream from a dead one.
                if time.time() - last_ping > 15:
                    last_ping = time.time()
                    ev("ping", {"ts": last_ping})
        except BrokenPipeError, ConnectionResetError, OSError:
            pass
        finally:
            with SSE_LOCK:
                SSE_CLIENTS[0] -= 1


# ---------------------------------------------------------------- routes ----
# Every route this module answers, registered in dashboard/routes.py's table (#61). The patterns and tests are
# tried in this order, after the exact paths, so a catch-all (/people/<name>) comes after the narrower routes.


@routes.get_when("/objects, /objects/...", lambda p: p == "/objects" or p.startswith("/objects/"))
def _get_objects(h: H) -> None:
    from liberational.dashboard import objects as objects_page

    objects_page.handle(h, "GET")  # the Objects page and its API, on Continuum


@routes.get("/audio/levels")
def _get_audio_levels(h: H) -> None:
    return h.levels()


@routes.get("/events")
def _get_events(h: H) -> None:
    return h.sse()


@routes.get("/logs/stream")
def _get_logs_stream(h: H) -> None:
    return h.log_stream()


@routes.get_when("/static/...", lambda p: p.startswith("/static/"))
def _get_static(h: H) -> None:
    p = h.path.split("?")[0]
    if p.startswith("/static/app/") and DEV and (got := vite(h.path)) is not None:
        # --dev: the new frontend's modules, styles, and fonts come from Vite, through this server, so the page
        # loads them from its own origin (over Tailscale too). Its hot reload talks to Vite's port directly.
        return h._send(200, got[1], got[0])
    hit = assets.read(p)
    return h._send(200, hit[1], hit[0]) if hit else h._send(404, "text/plain", b"")


@routes.post_raw_when("/objects, /objects/...", lambda p: p == "/objects" or p.startswith("/objects/"))
def _post_objects(h: H) -> None:
    from liberational.dashboard import objects as objects_page

    objects_page.handle(h, "POST")


WARM_FIRST_MAX_S = 90.0  # the later warm-ups wait at most this long for it


def _warm_devices() -> None:
    """Read Known › Devices' device list once at start, at low priority, so its day of settled hours is tallied
    (proximity/devices.py) before anyone opens the tab: the first read after a start is about 30 s, later ones about
    a second."""
    context.low_priority(19)
    try:
        from liberational.dashboard import device_links

        device_links.listing("", state=context.STATE)
    except Exception as error:  # the tab reads it itself, and says so if it can't
        LOG.warning("devices: warm-up failed: %s", repr(error))


def _after_people_warm(name: str, start: Callable[[], Any]) -> threading.Thread:
    """Start `start` once the people warm-up has finished, or WARM_FIRST_MAX_S after the start at most. The
    page waits on /people first; background warm-ups that read the same registry and history ran alongside it and
    kept its first answers waiting for the GIL and the registry."""

    def run() -> None:
        people_reads._PEOPLE_WARMED.wait(WARM_FIRST_MAX_S)
        try:
            start()
        except Exception as error:
            LOG.warning("%s %s", f"{name} could not start:", repr(error))

    thread = threading.Thread(target=run, daemon=True, name=f"after-warm-{name}")
    thread.start()
    return thread


def _stop_worker_pools() -> None:
    """Stop the dashboard's worker processes (face groups, voice groups, the people list, Review presence, and Reports'
    days) at shutdown. Each pool's queues hold named semaphores that only a normal exit unlinks: a stop that skipped
    this left the resource tracker to find 20 of them and warn on every shutdown. A worker's job is only a
    read, so one still running is stopped rather than waited for."""
    from liberational.dashboard import presence as review_presence
    from liberational.memory import reports as report_days

    pools: list[Executor | None] = [
        unnamed_groups._groups_executor,
        unnamed_groups._voices_executor,
        roster_read._people_executor,
        review_presence._executor,
        report_days._executor,
    ]
    for pool in pools:
        if pool is None:
            continue
        for process in list((getattr(pool, "_processes", None) or {}).values()):
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)


@model_runtime.reporting("dashboard")
def main() -> None:
    """pack-dashboard [PORT] [--dev]"""
    global PORT, DEV
    PORT = settings.port_arg() or PORT
    DEV = "--dev" in sys.argv[1:]
    if DEV:
        devreload.watch(LOG.info)
    srv = DashboardServer(("0.0.0.0", PORT), H)
    if not cfg.local_config().exists():
        LOG.info(f"no pack.toml yet: open http://localhost:{PORT}/ to set up the pack")
    if assets.stale():
        LOG.info(
            "the frontend build in static/app/ is older than frontend/: run `mise run frontend` (the page shows a banner)"
        )
    LOG.info(
        f"pack dashboard on http://localhost:{PORT}/{'  (dev: auto-reload on)' if DEV else ''}  (psutil {'ok' if psutil else 'MISSING'}, soundfile {'ok' if sf else 'MISSING'})"
    )
    from liberational.dashboard import presence as review_presence

    _after_people_warm("review presence", lambda: review_presence.keep_warm(context.STATE))  # today's Review answer
    from liberational.dashboard import report_health

    report_health.start(context.STATE)  # System health's samples for the Reports tab
    memtrim.start()  # what requests and reads freed goes back to the system
    threading.Thread(target=_warm_devices, name="devices-warm", daemon=True).start()  # Known › Devices
    machine_stats.SAMPLER.start()  # the Resource monitor's last half hour, always
    unnamed_groups._groups_in_worker = True  # face groups, voice groups, and the people list: a worker process each
    from liberational.memory import reports as report_days

    report_days.READ_IN_WORKER = True  # and Reports' days
    readers.warm()  # which services read which settings, for Settings' save bar
    _warm_faces()
    _after_people_warm(
        "faces autoname",
        lambda: threading.Thread(target=_autoname_faces_forever, daemon=True, name="faces-autoname").start(),
    )
    slow.watch(LOG.warning, LOG.info)  # what a slow request waits on, and each route's CPU

    def stop(signum: int, frame: FrameType | None) -> None:
        # A stop (SIGTERM) ends the serving loop and exits normally, so the worker pools are closed and their
        # semaphores unlinked; Python's default for SIGTERM ends the process without either. shutdown() waits
        # for the loop to end, so it runs on a thread of its own, not in this handler on the loop's thread.
        threading.Thread(target=srv.shutdown, name="dashboard-stop", daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass  # Ctrl-C in a terminal (pack-demo's, or a dashboard run by hand) stops it as quietly as SIGTERM does
    finally:
        srv.server_close()
        _stop_worker_pools()


if __name__ == "__main__":
    main()
