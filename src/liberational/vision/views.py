"""Camera IDs and the owner's names for them (`vision.camera_names`), shared by camera consumers.

A camera is a view, an angle on whatever holds it, not a room (#5): its name says which camera it is ("front
camera"), never where it is. Where a camera's pixel lands comes from its pose (spatial/rig.py, memory/places.py).
Native manifest names are exact ASCII tokens. Name settings also accept legacy numbered aliases unless the literal
token names a configured camera.
"""

import re
import tomllib
from collections.abc import Iterable
from pathlib import Path

from liberational import config as cfg

CAMERA_ID = re.compile(r"[A-Za-z0-9_-]+")
ENTRY = re.compile(r"\s*(.*?)\s*[:=]\s*(.*?)\s*", re.DOTALL)
LEGACY = re.compile(r"(?:cam)?\s*(\d+)", re.IGNORECASE)


def valid_camera_id(value: str) -> bool:
    """Whether value is an exact native camera name, without trimming or normalization."""
    return CAMERA_ID.fullmatch(value) is not None


def manifest_camera_names(path: Path | None = None, *, active_only: bool = False) -> list[str] | None:
    """Exact camera/RTSP names, including disabled history unless active_only; None only for an absent manifest."""
    path = path if path is not None else cfg.ROOT / "config" / "devices.toml"
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except UnicodeError as ex:
        raise ValueError(f"Invalid camera manifest {path}: {ex}") from ex
    try:
        devices = tomllib.loads(text).get("device", [])
        if not isinstance(devices, list):
            raise ValueError("device must be an array of tables")
        out: list[str] = []
        seen: set[str] = set()
        for index, device in enumerate(devices):
            if not isinstance(device, dict) or not isinstance(device.get("kind"), str):
                raise ValueError(f"device {index + 1} needs a kind")
            name = device.get("name")
            if not isinstance(device.get("enabled", True), bool):
                raise ValueError(f"device {index + 1} enabled must be true or false")
            if not isinstance(name, str) or not valid_camera_id(name):
                raise ValueError(f"device {index + 1} needs a name containing only ASCII letters, digits, _ or -")
            if device["kind"] in {"camera", "rtsp"}:
                if name in seen:
                    raise ValueError(f"camera {name!r} is named twice")
                seen.add(name)
                if not active_only or device.get("enabled", True):
                    out.append(name)
        return out
    except (ValueError, UnicodeError) as ex:
        raise ValueError(f"Invalid camera manifest {path}: {ex}") from ex


def camera_ids(*, active_only: bool = False) -> list[str]:
    """Configured IDs, retaining disabled history by default; only Live requests active_only.

    The manifest is the only source: no manifest, like an empty one, means no cameras.
    """
    return manifest_camera_names(active_only=active_only) or []


def configured_camera_mentions(text: str) -> tuple[list[str], str]:
    """Configured native IDs or unambiguous camera names where the question names a camera.

    Explicit, case-exact native IDs win over aliases. Spoken case and separators may vary, but ambiguous
    aliases require an actual ID. Bare ordinary words, numeric times, and temporal dayparts remain text, and so does
    a bare "in the <name>": that is a place or a room, and only "in the <name> camera" names a camera (#34).
    The masked text retains its length for the history parser's positional time handling.
    """
    cameras = camera_ids()
    found: set[str] = set()
    remaining = text

    def mask(start: int, end: int) -> None:
        nonlocal remaining
        remaining = remaining[:start] + " " * (end - start) + remaining[end:]

    # Exact explicit IDs beat aliases for the same mention; a longer name still owns its full phrase.
    matches: list[tuple[int, int, str, str | None]] = []
    for camera in sorted(cameras, key=len, reverse=True):
        token = re.escape(camera)
        contexts = [rf"(?i:cam(?:era)?\s+(?:the\s+)?){token}"]
        if re.fullmatch(r"cam(?:era)?(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)", camera, re.I):
            contexts.append(token)
        for context in contexts:
            pattern = rf"(?<![A-Za-z0-9_-])(?:{context})(?![A-Za-z0-9_-])"
            matches.extend((match.start(), match.end(), camera, camera) for match in re.finditer(pattern, text))

    aliases: dict[str, set[str]] = {}
    for camera, label in [(camera, camera) for camera in cameras] + list(names().items()):
        if camera in cameras:
            normalized = re.sub(r"[\s_-]+", " ", label).strip().casefold()
            if normalized:
                aliases.setdefault(normalized, set()).add(camera)

    for alias in aliases:
        token = r"[\s_-]+".join(re.escape(part) for part in alias.split())
        contexts = [rf"cam(?:era)?\s+(?:the\s+)?{token}"]
        if not alias.isdecimal():
            contexts.extend(
                [
                    rf"(?:the\s+)?{token}\s+cam(?:era)?",
                    rf"(?:on|from)\s+(?:the\s+)?{token}(?:\s+cam(?:era)?)?",
                    rf"{token}(?=\s+(?:see|saw|show|showed|record|recorded|capture|captured)\b)",
                    # "in the front camera" names the camera; a bare "in the kitchen" is a place or a room, never a
                    # camera, since a camera is a view on what holds it, not a room (#34).
                    rf"in\s+(?:the\s+)?{token}\s+cam(?:era)?",
                ]
            )
        pattern = rf"(?<![A-Za-z0-9_-])(?i:{'|'.join(contexts)})(?![A-Za-z0-9_-])"
        matches.extend((match.start(), match.end(), alias, None) for match in re.finditer(pattern, text))
    # A longer camera name owns an overlapping mention, avoiding prefix-camera selections.
    for start, end, alias, exact in sorted(matches, key=lambda item: (-(item[1] - item[0]), item[3] is None, item[0])):
        if remaining[start:end] != text[start:end]:
            continue
        candidates = {exact} if exact is not None else aliases[alias]
        if len(candidates) != 1:
            raise ValueError("Ambiguous camera name; specify an exact camera ID with 'camera <ID>'.")
        found.update(candidates)
        mask(start, end)
    return sorted(found), remaining


def parse(entries: Iterable[object], camera_ids: Iterable[str] = ()) -> dict[str, str]:
    """Parse camera name entries, retaining configured exact IDs before interpreting legacy numbered aliases."""
    known = set(camera_ids)
    out: dict[str, str] = {}
    for entry in entries:
        m = ENTRY.fullmatch(str(entry))
        if m is None or not m[2]:
            raise ValueError(f"{entry!r} is not a camera and its name, like cam0: front camera")
        cam = m[1]
        legacy = LEGACY.fullmatch(cam)
        if cam not in known and legacy is not None:
            cam = f"cam{int(legacy[1])}"
        if not valid_camera_id(cam):
            raise ValueError(f"{entry!r} is not a camera and its name, like cam0: front camera")
        if cam in out:
            raise ValueError(f"{cam} is named twice")
        out[cam] = " ".join(m[2].split())
    return out


def names() -> dict[str, str]:
    """Every named camera's name; a malformed setting names no camera."""
    known = manifest_camera_names() or []
    try:
        return parse(cfg.get("vision.camera_names") or [], known)
    except ValueError:
        return {}


def name(cam: str) -> str:
    """The camera's name, or its exact ID when it has none."""
    return names().get(cam, cam)
