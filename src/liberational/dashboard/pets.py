"""Small local-control routes for owner-named pets, backed by native Continuum profile assertions."""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import continuum_py

from liberational import logs
from liberational.people import pets

LOG = logs.get("pets")

if TYPE_CHECKING:
    from liberational.people.pet_recognition import Sighting


HERE_NOW_S = 30  # a pet is here now when a camera saw it in the last 30 s, as a person is (server.HERE_NOW_S)
SINCE_LOOKBACK_S = 1800  # how far back "since" looks for the start of that visit


def here(state: Path, now: float | None = None) -> dict[str, dict[str, Any]]:
    """{pet id: {cam, place, since, track}} for each named pet a camera saw in the last HERE_NOW_S seconds (#2606), by the
    same naming as Review's boxes: an owner label on the track, else recognition by appearance. The camera is the one
    that saw it last, and track its newest track there (a photo of it now); "since" is when that camera's run of its tracks began, with no gap longer than HERE_NOW_S. A
    failure says nobody: Here now is extra, and the roster still answers."""
    from liberational import segment_log
    from liberational.dashboard import review_boxes
    from liberational.vision import views

    now = time.time() if now is None else now
    out: dict[str, dict[str, Any]] = {}
    try:
        latest: dict[
            str, tuple[float, str, set[str], str]
        ] = {}  # pet id: (last seen, camera, its tracks there, newest)
        for cam in views.camera_ids(active_only=True):
            for box in review_boxes.pet_boxes(state, cam, now - HERE_NOW_S, now + 1):
                pet = box.get("pet_id")
                if not pet:
                    continue
                seen, shown, tracks, newest = latest.get(pet, (0.0, cam, set(), box["track"]))
                if box["ts"] >= seen:
                    seen, shown, tracks, newest = box["ts"], cam, tracks if shown == cam else set(), box["track"]
                if shown == cam:
                    tracks.add(box["track"])
                latest[pet] = (seen, shown, tracks, newest)
        if not latest:
            return out
        log = segment_log.Log(segment_log.root(state))
        named = views.names()
        for pet, (seen, cam, tracks, newest) in latest.items():
            times = sorted(
                (
                    float(r["ts_start"])
                    for r in log.range(segment_log.ANIMAL_DETECTIONS, now - SINCE_LOOKBACK_S, now + 1, cameras=[cam])
                    if r.get("track_id") in tracks
                ),
                reverse=True,
            )
            since = seen
            for ts in times:
                if since - ts > HERE_NOW_S:
                    break
                since = min(since, ts)
            out[pet] = {"cam": cam, "place": named.get(cam, cam), "since": since, "track": newest}
    except Exception as error:
        LOG.warning("which pets are here now couldn't be read: %s", logs.reason(error))
        return {}
    return out


def listing(state: Path) -> tuple[int, dict[str, Any]]:
    """GET /pets, with `here` (#2606): the named pets a camera sees now. Unavailable broker evidence stays an error,
    never an empty roster."""
    try:
        listing = pets.at(state).listing()
        return 200, {"pets": listing, "species": pets.offered(listing), "here": here(state)}
    except (continuum_py.ContinuumError, TimeoutError, RuntimeError) as error:
        LOG.warning("pet profiles unavailable: %s", logs.reason(error))
        return 503, {"error": "Pet profiles are unavailable. Retry when the broker is ready."}


def save(body: object, control: bool, state: Path) -> tuple[int, dict[str, Any]]:
    """POST /pets/save {id?, name, species, sex?}: add or rename one stable pet subject; sex is optional."""
    if not control:
        return 403, {"ok": False, "error": "pets can only be named on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    try:
        pet = pets.at(state).save(body.get("name"), body.get("species"), body.get("id"), body.get("sex"))
        return 200, {"ok": True, "pet": pet, "pets": pets.at(state).listing()}
    except ValueError as error:
        return 400, {"ok": False, "error": str(error)}
    except LookupError as error:
        return 404, {"ok": False, "error": str(error)}
    except (continuum_py.ContinuumError, TimeoutError, RuntimeError) as error:
        LOG.warning("pet profile save unavailable: %s", logs.reason(error))
        return 503, {"ok": False, "error": "Pet profiles are unavailable. Reload before trying again."}


def remove(body: object, control: bool, state: Path) -> tuple[int, dict[str, Any]]:
    """POST /pets/remove: retire the owner binding while keeping recorded observations."""
    if not control:
        return 403, {"ok": False, "error": "pets can only be changed on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    try:
        pets.at(state).remove(body.get("id"))
        return 200, {"ok": True, "pets": pets.at(state).listing()}
    except ValueError as error:
        return 400, {"ok": False, "error": str(error)}
    except LookupError as error:
        return 404, {"ok": False, "error": str(error)}
    except (continuum_py.ContinuumError, TimeoutError, RuntimeError) as error:
        LOG.warning("pet profile removal unavailable: %s", logs.reason(error))
        return 503, {"ok": False, "error": "Pet profiles are unavailable. Reload before trying again."}


def sightings(state: Path) -> tuple[int, dict[str, Any]]:
    """GET /pets/sightings: animal sightings grouped by appearance, labels, and each species' held-out check."""
    from liberational.people import pet_recognition

    try:
        return 200, pet_recognition.view(state, pets.at(state).listing())
    except (continuum_py.ContinuumError, TimeoutError, RuntimeError) as error:
        LOG.warning("pet sightings unavailable: %s", logs.reason(error))
        return 503, {"error": "Pet sightings are unavailable. Retry when the broker is ready."}


def card(body: object, control: bool, state: Path) -> tuple[int, dict[str, Any]]:
    """POST /pets/photo {id, track, action} (#2363): a pet card's photo choices. `main` makes the sighting (its track)
    the pet's photo, `newest` goes back to its newest sighting (no track), and `hide` and `show` take a sighting off
    its card or put it back. Answers the roster, as /pets/save does."""
    if not control:
        return 403, {"ok": False, "error": "pets can only be changed on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    action, registry = body.get("action"), pets.at(state)
    try:
        if action == "main":
            pet = registry.set_photo(body.get("id"), body.get("track"))
        elif action == "newest":
            pet = registry.set_photo(body.get("id"), None)
        elif action in ("hide", "show"):
            pet = registry.hide(body.get("id"), body.get("track"), action == "hide")
        else:
            return 400, {"ok": False, "error": 'action is "main", "newest", "hide", or "show"'}
        return 200, {"ok": True, "pet": pet, "pets": registry.listing()}
    except ValueError as error:
        return 400, {"ok": False, "error": str(error)}
    except LookupError as error:
        return 404, {"ok": False, "error": str(error)}
    except (continuum_py.ContinuumError, TimeoutError, RuntimeError) as error:
        LOG.warning("pet card choice unavailable: %s", logs.reason(error))
        return 503, {"ok": False, "error": "Pet profiles are unavailable. Reload before trying again."}


def label(body: object, control: bool, state: Path) -> tuple[int, dict[str, Any]]:
    """POST /pets/sightings/label {track_ids, label}: a named pet's id, "other", or null to clear."""
    from liberational.people import pet_recognition

    if not control:
        return 403, {"ok": False, "error": "pets can only be changed on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    value = body.get("label")
    if value is not None and not isinstance(value, str):
        return 400, {"ok": False, "error": 'label is a pet id, "other", or null'}
    try:
        count = pet_recognition.label(state, body.get("track_ids"), value, pets.at(state).listing())  # ty: ignore[invalid-argument-type]
        return 200, {"ok": True, "labeled": count}
    except ValueError as error:
        return 400, {"ok": False, "error": str(error)}
    except LookupError as error:
        return 404, {"ok": False, "error": str(error)}
    except (continuum_py.ContinuumError, TimeoutError, RuntimeError) as error:
        LOG.warning("pet sighting label unavailable: %s", logs.reason(error))
        return 503, {"ok": False, "error": "Pet sightings are unavailable. Reload before trying again."}


def deny(body: object, control: bool, state: Path) -> tuple[int, dict[str, Any]]:
    """POST /pets/sightings/not {track_ids, pet, not} (#2364): these sightings are not `pet` (a named pet's id), or,
    with `not` false, take that back. Recognition never names or suggests that pet for them again."""
    from liberational.people import pet_recognition

    if not control:
        return 403, {"ok": False, "error": "pets can only be changed on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    flag = body.get("not", True)
    if not isinstance(flag, bool):
        return 400, {"ok": False, "error": "not is true or false"}
    try:
        count = pet_recognition.mark_not(
            state,
            body.get("track_ids"),  # ty: ignore[invalid-argument-type]
            body.get("pet"),  # ty: ignore[invalid-argument-type]
            pets.at(state).listing(),
            active=flag,
        )
        return 200, {"ok": True, "marked": count}
    except ValueError as error:
        return 400, {"ok": False, "error": str(error)}
    except LookupError as error:
        return 404, {"ok": False, "error": str(error)}
    except (continuum_py.ContinuumError, TimeoutError, RuntimeError) as error:
        LOG.warning("pet sighting not-label unavailable: %s", logs.reason(error))
        return 503, {"ok": False, "error": "Pet sightings are unavailable. Reload before trying again."}


@functools.lru_cache(maxsize=4)
def _photo_tracks(state: Path, revision: str, second: int) -> dict[str, Sighting]:
    from liberational.people import pet_recognition

    return {s.track_id: s for s in pet_recognition.recorded(state, time.time()) if s.shot}


def photo_sighting(track: str, state: Path, revision: str) -> Sighting | None:
    """One short-lived snapshot shared by pet crop lookups, before taking a decode turn."""
    return _photo_tracks(state, revision, int(time.monotonic() // 2)).get(track)


def photo(track: str, state: Path, *, sighting: Sighting | None = None) -> bytes | None:
    """GET /pets/sightings/photo?track=: a square JPEG of the sighting's largest recorded box, or None when the track
    isn't listed or its frame is no longer recorded. The workers save it as the box comes (people/pet_shots.py), so
    this cuts one only for a track without one, and saves it."""
    from liberational.people import pet_shots
    from liberational.people.face_shots import write

    sighting = sighting if sighting is not None else photo_sighting(track, state, pet_shots.revision(state))
    if sighting is None:
        return None
    jpg = pet_shots.cut(state, sighting)
    if jpg is not None:
        write(state, track, pet_shots.revision(state), jpg, pet_shots.KIND)
    return jpg
