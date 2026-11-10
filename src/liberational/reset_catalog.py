"""The catalog reset as one command: `pack-reset-catalog`.

  pack-reset-catalog --keep-taught                # print the plan, change nothing
  pack-reset-catalog --keep-taught --yes          # a catalog-only reset: the taught stores stay
  pack-reset-catalog --fresh --yes                # nothing crosses: the taught stores are deleted too
  pack-reset-catalog --keep-taught --keep-places --yes  # the places' maps, walks, and splats stay too

In order:

1. Check the plan can run: each camera named with `--camera` is in `config/devices.toml`, and the broker's service is
   installed (if it isn't, the reset refuses before stopping anything and says to run `pack-services install`).
2. Stop every installed service outside the `core` group, then the core services too, and check that nothing is
   running and the broker's socket is closed. A broker that crash-loops (on an old catalog format, say) can bind its
   socket again between the stop and the check, so the reset stops it again until it stays down, and refuses only if
   it's still up after `STOP_WAIT_S`.
3. Delete the catalog, the recorded video and audio it indexes (`tier` and `tier2`), the worker state derived from
   it (see `targets`), and the segment store.
4. Turn on H.264 (`encoding = { enabled = true }` under its `video`) for every camera in `config/devices.toml`, or
   only those named with `--camera`, before the broker first starts on the fresh catalog. Capture encodes only a
   camera's MJPEG, so a camera set to capture anything else (a Mac's USB cameras deliver UYVY) is left as it is, and
   the reset says why. A file with no camera turns nothing on.
5. Start the broker and the other core services that were running, wait for the broker's socket, then start the
   other services that were running. The broker always starts, even if it was stopped or crash-looping when the
   reset began: the reset leaves a fresh catalog it can open, and nothing else runs without it.

From step 3 on, a failure leaves everything stopped and says how to finish by hand: starting a broker on half a
reset is worse than a stopped house.

`--keep-places` keeps `state/places`: each place's revisions, maps, plans, walk clouds, and splats, which take long to
make again (a splat trains for a long time). They're files of their own and read nothing from the catalog. What a
kept place still names from the old catalog is its revisions' `sources`, the ids of the visits they were made from:
visits are rows in the segment store, which the reset deletes, and nothing reads those ids back. So the maps, plans,
fixed cameras' poses, and splats work as before, while visits and the rig's positions start fresh, as does each
camera's history behind a splat (the video it was trained on is gone).

`--keep-taught` leaves the taught state alone: it's in `knowledge.sqlite3` and the files beside it, which no step
reads or writes (a reset after a catalog format change or a model cutover). `--fresh` also deletes every taught store
(`taught_targets`): names, voices, faces, pets, zones, Objects, and device links, the voice clones, the owner's voice
reference, and People photos, the deleted-people folder, the ingest state, and any export bundles. It truncates the
logs too, since they hold names and transcripts. Only settings survive it: `pack.toml` (including `[owner]`, the
owner's name), `config/devices.toml`, the health reports, and model weights.
"""

import argparse
import shutil
import socket as py_socket
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import tomlkit

from liberational import config as cfg
from liberational.service_cascade import BROKER

MODES = ("keep-taught", "fresh")
RUNNING = frozenset(("active", "activating", "reloading", "running"))
SOCKET_WAIT_S = 60.0
STOP_WAIT_S = 30.0  # how long a stopped broker gets to stay down before the reset refuses
STOP_RETRY_S = 1.0


class Refused(RuntimeError):
    """The reset stopped before deleting anything; the services that were stopped have been started again."""


class Stopped(RuntimeError):
    """The reset stopped partway through deleting or after; everything is left stopped."""


class Host(Protocol):
    """The service manager, as the reset drives it."""

    def installed(self) -> list[str]: ...

    def core(self) -> set[str]: ...

    def declared(self) -> set[str]: ...

    def states(self) -> dict[str, str]: ...

    def stop(self, names: Sequence[str]) -> None: ...

    def start(self, names: Sequence[str]) -> None: ...


class Services:
    """This machine's services (liberational.services)."""

    def installed(self) -> list[str]:
        from liberational import services

        declared = list(services.declared())
        return sorted(services.installed(), key=lambda n: declared.index(n) if n in declared else len(declared))

    def core(self) -> set[str]:
        from liberational import services

        return {name for name, svc in services.declared().items() if svc.get("group") == "core"}

    def declared(self) -> set[str]:
        from liberational import services

        return set(services.declared())

    def states(self) -> dict[str, str]:
        from liberational import services

        return services.states()

    def stop(self, names: Sequence[str]) -> None:
        from liberational import services

        services.stop(names)

    def start(self, names: Sequence[str]) -> None:
        from liberational import services

        services.start(names)


@dataclass(frozen=True)
class Plan:
    state: Path  # paths.state
    faces: Path  # paths.faces, which fresh empties
    devices: Path  # config/devices.toml
    cameras: tuple[str, ...] | None = None  # the cameras switched to H.264; None for every camera in the file
    socket: Path | None = None  # the broker's socket, if not state/continuum/broker.sock
    mode: str = "keep-taught"  # "keep-taught" or "fresh"
    voices: Path | None = None  # paths.voices, which fresh empties
    logs: Path | None = None  # paths.logs, whose files fresh truncates (they hold names and transcripts)
    keep_places: bool = False  # keep state/places (see the module docstring)

    @property
    def broker_socket(self) -> Path:
        return self.socket or self.state / "continuum" / "broker.sock"


def targets(plan: Plan) -> list[Path]:
    """What the reset deletes, all under the state directory. Everything else there is kept."""
    continuum = plan.state / "continuum"
    paths = [
        *sorted(continuum.glob("catalog.redb*")),  # the catalog, and any scrub copy beside it
        continuum / "tier",  # the recorded video and audio, which only the catalog indexes
        continuum / "tier2",
        continuum / "catalog-age.json",
        continuum / "vector",  # the vector worker's index, rebuilt from what it follows
        continuum / "decoded",  # where the shared decode kept its frame ring before it moved to paths.decoded
        plan.state / "segments",  # the segment store
        *([] if plan.keep_places else [plan.state / "places"]),  # place revisions, maps, plans, and splats
        # the frames kept for learning surroundings (spatial/surroundings_frames.py), which the places' maps are made of
        *([] if plan.keep_places else [plan.state / "surroundings"]),
        plan.state / "identity.speech.cursor",  # identity's place in the speech stream
        plan.state / "see-zones.json",  # the zones worker's checkpoint of its input positions
        plan.state / "cache",  # caches of what the segment store held, such as the device tallies
        continuum / "movement-worker.lock",  # left by deleted workers
        continuum / "zone-worker.lock",
    ]
    return [p for p in paths if p.exists() or p.is_symlink()]


def taught_targets(plan: Plan) -> list[Path]:
    """What `--fresh` deletes beside `targets`: every store of what someone taught or was recognized by."""
    state, data = plan.state, plan.state.parent
    paths = [
        *sorted(state.glob("knowledge.sqlite3*")),  # people, names, identity, pets, zones, Objects, device links
        state / "knowledge_blobs",  # their attachments: reference media, Object crops
        state / "voiceprints.json",  # named voices
        state / "voice_tags.jsonl",  # voice naming requests
        state / "voice_tags.offset",
        state / "clones",  # voice clones of people
        state / "shots",  # People's photos
        state / "inbox_skips.json",  # People's inbox put-asides: unknown face and voice ids, which restart at 1
        *sorted(state.glob("transcript_revisions*.sqlite3*")),  # the nightly pass's text of the deleted speech
        *sorted(state.glob("ingest.sqlite3*")),  # ingested items, pointing at payloads the reset deletes
        *sorted(plan.faces.glob("*")),  # the face gallery (db.npz), each person's crops, and unknown clusters
        *(sorted(plan.voices.glob("*")) if plan.voices is not None else []),  # the owner's voice reference
        data / "premove",  # people and clones deleted from the dashboard, moved aside
        *sorted(data.glob("taught*.zip")),  # earlier export bundles
    ]
    return [p for p in paths if p.exists() or p.is_symlink()]


def logs(plan: Plan) -> list[Path]:
    """The log files `--fresh` truncates, once the services are stopped."""
    if plan.mode != "fresh" or plan.logs is None:
        return []
    return sorted(p for p in plan.logs.rglob("*") if p.is_file() and not p.is_symlink())


def deleted(plan: Plan) -> list[Path]:
    return targets(plan) + (taught_targets(plan) if plan.mode == "fresh" else [])


def size(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if not item.is_dir() or item.is_symlink():
                total += item.lstat().st_size
        except OSError:
            continue
    return total


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _camera(document: tomlkit.TOMLDocument, camera: str) -> dict | None:
    devices = document.get("device")
    if not isinstance(devices, list):
        return None
    return next((d for d in devices if isinstance(d, dict) and d.get("name") == camera), None)


def no_encoding(document: tomlkit.TOMLDocument, device: dict) -> str | None:
    """Why capture would refuse `video.encoding` for this device, or None if it can encode it. Capture encodes only
    a camera's MJPEG, and reads the format as it merges it: the camera's own `video.format`, else the file's
    top-level `[video]` one, else MJPEG."""
    kind = device.get("kind")
    if kind != "camera":
        return f"it's a {kind} device, and capture encodes only a camera's MJPEG"
    own = device.get("video")
    defaults = document.get("video")
    format = next(
        (v["format"] for v in (own, defaults) if isinstance(v, dict) and isinstance(v.get("format"), str)), "MJPEG"
    )
    if format != "MJPEG":
        return f"it captures {format}, and capture encodes only MJPEG"
    return None


def cameras(plan: Plan) -> list[str]:
    """The cameras the reset turns H.264 on for: those named, else every camera device in the file."""
    if plan.cameras is not None:
        return list(plan.cameras)
    devices = tomlkit.parse(plan.devices.read_text()).get("device")
    if not isinstance(devices, list):
        return []
    return [str(d["name"]) for d in devices if isinstance(d, dict) and d.get("kind") == "camera" and d.get("name")]


def missing(plan: Plan) -> list[str]:
    """The cameras named with --camera that aren't devices in the file."""
    document = tomlkit.parse(plan.devices.read_text())
    return [c for c in plan.cameras or () if _camera(document, c) is None]


def encoding_skipped(devices: Path, camera: str) -> str | None:
    """Why the reset leaves the camera's encoding alone, or None if it turns it on."""
    document = tomlkit.parse(devices.read_text())
    device = _camera(document, camera)
    if device is None:
        raise LookupError(f"{camera} is not a [[device]] in {devices}")
    return no_encoding(document, device)


def encode_camera(devices: Path, camera: str) -> bool:
    """Set `encoding = { enabled = true }` under the camera's `video`, keeping the file's comments and the rest of
    its video settings. Returns whether the file changed: it doesn't for a camera capture can't encode
    (`no_encoding`)."""
    document = tomlkit.parse(devices.read_text())
    device = _camera(document, camera)
    if device is None:
        raise LookupError(f"{camera} is not a [[device]] in {devices}")
    if no_encoding(document, device) is not None:
        return False
    video = device.get("video")
    if video is None:
        video = tomlkit.table()
        device["video"] = video
    encoding = video.get("encoding")
    if isinstance(encoding, dict) and encoding.get("enabled") is True:
        return False
    if isinstance(encoding, dict):
        encoding["enabled"] = True
    else:
        inline = tomlkit.inline_table()
        inline["enabled"] = True
        video["encoding"] = inline
    temporary = devices.with_name(devices.name + ".reset-tmp")
    temporary.write_text(tomlkit.dumps(document))
    temporary.replace(devices)
    return True


def listening(socket: Path) -> bool:
    with py_socket.socket(py_socket.AF_UNIX) as probe:
        try:
            probe.connect(str(socket))
        except OSError:
            return False
        return True


def describe(plan: Plan, host: Host) -> list[str]:
    """The plan, as the dry run prints it."""
    installed = host.installed()
    core = host.core()
    lines = [
        f"State: {plan.state}",
        f"Mode: {plan.mode}",
        f"Stop first: {', '.join(n for n in installed if n not in core) or 'nothing'}",
        f"Stop then: {', '.join(n for n in installed if n in core) or 'nothing'}",
        "Delete:",
    ]
    total = 0
    for path in deleted(plan):
        n = size(path)
        total += n
        lines.append(f"  {path} ({human(n)})")
    lines.append(f"  total {human(total)}")
    truncated = logs(plan)
    if truncated:
        lines.append(f"Truncate {len(truncated)} log files in {plan.logs} ({human(sum(size(p) for p in truncated))})")
    if plan.keep_places:
        lines.append(f"Keep: {plan.state / 'places'} (visits and positions start fresh)")
    for camera in missing(plan):
        lines.append(f"Refuses: {camera} is not a [[device]] in {plan.devices}")
    for camera in [c for c in cameras(plan) if c not in missing(plan)]:
        skipped = encoding_skipped(plan.devices, camera)
        if skipped is None:
            lines.append(f"Turn on H.264 for {camera} in {plan.devices}")
        else:
            lines.append(f"Leave {camera}'s encoding alone: {skipped}")
    if not cameras(plan):
        lines.append(f"Turn on H.264 for no camera: {plan.devices} has none")
    lines.append("Start the broker and the services that were running")
    if BROKER not in installed:
        lines.append(f"Refuses: {BROKER} isn't installed (run `pack-services install` first)")
    if plan.mode == "fresh":
        lines.append("Not touched: settings (pack.toml, including [owner], and config/devices.toml)")
    retired = [n for n in installed if n not in host.declared()]
    if retired:
        lines.append(f"Stopped and not started again (gone from services.toml): {', '.join(retired)}")
    return lines


def run(
    plan: Plan,
    host: Host,
    *,
    say: Callable[[str], None] = print,
    socket_wait_s: float = SOCKET_WAIT_S,
    stop_wait_s: float = STOP_WAIT_S,
) -> None:
    """Do the reset. Raises Refused before anything is deleted, or Stopped after."""
    if plan.mode not in MODES:
        raise Refused(f"the mode is one of {', '.join(MODES)}")
    if gone := missing(plan):
        raise Refused(f"{', '.join(gone)} {'is' if len(gone) == 1 else 'are'} not a [[device]] in {plan.devices}")

    installed = host.installed()
    if BROKER not in installed:
        raise Refused(
            f"{BROKER} isn't installed, so the reset couldn't start the house again; run `pack-services install`, "
            "then the reset. Nothing was stopped or deleted"
        )
    core = host.core()
    states = host.states()
    running = [n for n in installed if states.get(n) in RUNNING]
    retired = [n for n in installed if n not in host.declared()]  # installed, but gone from services.toml
    others = [n for n in installed if n not in core]

    say(f"Stopping {len(others)} services outside core")
    host.stop(others)
    still = [n for n, s in host.states().items() if n in others and s in RUNNING]
    if still:
        host.start([n for n in others if n in running])
        raise Refused(f"these didn't stop: {', '.join(still)}")

    say("Stopping the core services")
    host.stop([n for n in installed if n in core])
    deadline = time.monotonic() + stop_wait_s
    while (still := [n for n, s in host.states().items() if s in RUNNING]) or listening(plan.broker_socket):
        if time.monotonic() > deadline:
            host.start([n for n in installed if n in running])
            raise Refused(f"still running: {', '.join(still) or 'the broker socket'}; nothing was deleted")
        # A crash-looping broker binds its socket again between restarts, and launchd's bootout returns before the
        # job is gone: stop what's still up again rather than refusing on the first look.
        time.sleep(STOP_RETRY_S)
        host.stop(list(dict.fromkeys([*still, BROKER])))

    finish = (
        "Everything is stopped. Finish by hand: delete what's left of the plan's list, turn on encoding for the "
        "cameras that capture MJPEG, then start the services."
    )
    try:
        for path in deleted(plan):
            say(f"Deleting {path}")
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        for path in logs(plan):
            path.write_bytes(b"")
        if plan.mode == "fresh" and plan.logs is not None:
            say(f"Truncated the logs in {plan.logs}")
        for camera in cameras(plan):
            skipped = encoding_skipped(plan.devices, camera)
            if skipped is not None:
                say(f"Left {camera}'s encoding alone: {skipped}")
            elif encode_camera(plan.devices, camera):
                say(f"Turned on H.264 for {camera}")
    except Exception as error:
        raise Stopped(f"{error}. {finish}") from error

    say("Starting the core services")
    host.start([n for n in installed if n in core and (n in running or n == BROKER)])
    deadline = time.monotonic() + socket_wait_s
    while not listening(plan.broker_socket):
        if time.monotonic() > deadline:
            raise Stopped(
                f"the broker didn't come up at {plan.broker_socket}; the other services weren't started. "
                "Check the broker's log, then start them."
            )
        time.sleep(0.2)
    say("Starting the other services")
    host.start([n for n in installed if n not in core and n in running and n not in retired])
    if retired:
        say(
            f"Not started, since they're gone from services.toml: {', '.join(retired)}. Uninstall them with "
            f"`pack-services uninstall {' '.join(retired)}`."
        )
    say("Reset done")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--keep-taught", action="store_true", help="the taught stores stay")
    which.add_argument("--fresh", action="store_true", help="delete every taught store too")
    parser.add_argument(
        "--camera",
        action="append",
        metavar="NAME",
        help="a camera to switch to H.264 (again for more); without it, every camera in config/devices.toml",
    )
    parser.add_argument(
        "--keep-places",
        action="store_true",
        help="keep state/places (maps, plans, walks, and splats); their visits and the positions start fresh",
    )
    parser.add_argument("--yes", action="store_true", help="do it; without this, only print the plan")
    args = parser.parse_args(argv)
    plan = Plan(
        state=cfg.path("paths.state"),
        faces=cfg.path("paths.faces"),
        devices=cfg.devices_file(),
        cameras=tuple(args.camera) if args.camera else None,
        keep_places=args.keep_places,
        mode="fresh" if args.fresh else "keep-taught",
        voices=cfg.path("paths.voices"),
        logs=cfg.path("paths.logs"),
    )
    host = Services()
    if not args.yes:
        print("\n".join(describe(plan, host)))
        print("\nNothing was changed. Add --yes to do it.")
        return
    try:
        run(plan, host)
    except (Refused, Stopped) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
