"""The dashboard's settings API: read and save pack.toml, including first-run setup.

GET  /config   every editable setting (config.schema(), with its label, help, and section from labels.toml, and
               "restarts": what saving it restarts), "control" (may this viewer save), "needs_setup" (no pack.toml
               yet), and "restarts" (everything a save could restart). Secret values are never sent.
POST /config   {"values": {"owner.name": "Alice", ...}} -> write pack.toml (config.save()), reply, then restart
               only the installed services that read a changed setting (readers.py), and re-exec the
               dashboard if it reads one. "restarted" names the services and "restarting" says whether the
               dashboard comes back; a save of only live settings (labels.toml) restarts nothing.
               This machine only, same-origin JSON only, like POST /services. A new owner.name (with
               owner.speaker_label empty) renames the owner in People too, as a rename there would; when that
               is refused or fails, pack.toml is put back and the reply is the error, on field owner.name.
GET  /config/voice_ref         {path, exists, seconds, recorded, prompt}: the owner's voice reference (recorded: when,
                               Unix seconds) and the passage to read;
                               ?audio=1 serves the clip itself.
GET  /config/voices            {clone, kokoro}: whether the assistant speaks in the owner's cloned voice here, and
                               the Kokoro voices tts.voice can name for a machine where it can't.
POST /config/voice_ref         a WAV body (8 to 60 s) -> saved as owner.voice_ref with the passage as its .txt
                               transcript, and the owner's voice enrolled from it. This machine only, same origin.
"""

import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import login_item, service_cascade, services
from liberational.dashboard import routes
from liberational.dashboard.context import LOG

if TYPE_CHECKING:
    from liberational.dashboard.server import H


def shown(path: pathlib.Path) -> str:
    """A path as Settings shows it: relative to the owner's root in a checkout, absolute otherwise."""
    root = cfg.OWNER_ROOT
    return str(path.relative_to(root)) if root is not None and path.is_relative_to(root) else str(path)


def view(control: bool) -> dict[str, Any]:
    path = cfg.local_config()
    return {
        "ok": True,
        "control": control,
        "needs_setup": not path.exists(),
        "path": shown(path),
        "sections": with_restarts(cfg.schema(), everything := restarts()),
        "restarts": everything,
    }


def reference_problem(path: pathlib.Path) -> str | None:
    """What's wrong with a voice reference: "missing", "unreadable" (it isn't audio soundfile can open, which the voice
    service can't clone from either), or None when it can be used."""
    import soundfile as sf

    if not path.is_file():
        return "missing"
    try:
        sf.info(str(path))
    except Exception:
        return "unreadable"
    return None


def with_restarts(sections: list[dict[str, Any]], candidates: list[str]) -> list[dict[str, Any]]:
    """Each row gains "restarts": which of the candidates (restarts()) saving it would restart, so the save bar can
    name just those. [] for a live setting. Rows say nothing when the candidates can't be listed or the services'
    code can't be scanned, and the save bar then doesn't say what restarts."""
    from liberational import readers

    if not candidates:
        return sections
    try:
        each = readers.affected_each([row["key"] for s in sections for row in s["settings"]])
        hits = {key: set(names) for key, names in each.items()}
    except Exception:  # a module that doesn't parse, mid-edit: a save then restarts everything (plan())
        return sections
    for section in sections:
        for row in section["settings"]:
            row["restarts"] = [n for n in candidates if n in hits[row["key"]]]
    return sections


def plan(changed: Sequence[str]) -> tuple[list[str], bool]:
    """(services to install again, whether the dashboard restarts) after a save of `changed`: the installed services
    that read one of them, plus any the new settings start (asr.engine = whisper starts whisper). ([], False) for
    a save of live settings only."""
    from liberational import readers

    try:
        hit = set(readers.affected(changed))
    except Exception:  # the services' code can't be scanned: restart everything
        hit = {*services.declared(), "dashboard"}
    try:
        installed = set(services.installed())
        names = [n for n in restart_names() if n in hit or n not in installed]
    except SystemExit:  # the new pack.toml names an unknown service: the dashboard still comes back to say so
        names = []
    return names, "dashboard" in hit


def can_clone() -> bool:
    """Whether the assistant speaks in the owner's cloned voice on this machine: a voice reference to clone (without
    one, speech/tts.py speaks Kokoro), and voice_clone.can_clone, checked without loading torch into the dashboard:
    MLX on Apple Silicon, or the voice-clone extra with its model and an NVIDIA GPU the service may use (nvidia-smi on
    the PATH stands in for CUDA, and CUDA_VISIBLE_DEVICES must not hide every GPU). A torch build without CUDA still
    passes this; the voice service then logs "no CUDA GPU" and speaks Kokoro."""
    import importlib.util
    import os
    import shutil

    if reference_problem(cfg.path("owner.voice_ref")):
        return False
    if cfg.has_mlx():
        return True
    from liberational.speech import qwen3_cuda

    hidden = os.environ.get("CUDA_VISIBLE_DEVICES", None) in ("", "-1")
    return bool(
        importlib.util.find_spec("faster_qwen3_tts")
        and (qwen3_cuda.model_dir() / "config.json").exists()
        and shutil.which("nvidia-smi")
        and not hidden
    )


def kokoro_voices() -> list[str]:
    """The voices in Kokoro's voices file (an .npz of one array per voice), sorted; [] when it isn't downloaded."""
    import numpy as np

    try:
        with np.load(cfg.model_path("tts_kokoro_voices")) as voices:
            return sorted(voices.files)
    except Exception:  # missing, a platform that skips it (Apple Silicon), or unreadable
        return []


def restarts() -> list[str]:
    """What a save restarts, for Settings' save bar: the services restart_names gives, then the dashboard.
    [] when the services can't be listed (an unknown name in pack.toml): a save still works, it just can't say."""
    try:
        return [*restart_names(), "dashboard"]
    except SystemExit, Exception:
        return []


def _host(handler: H) -> str:
    """The hostname the browser used, without the port ("[::1]" stays bracketed)."""
    host = handler.headers.get("Host") or "localhost"
    return host[: host.index("]") + 1] if host.startswith("[") else host.rsplit(":", 1)[0]


def port_arg(argv: Sequence[str] | None = None) -> int | None:
    """The first command-line argument (sys.argv by default) that is a TCP port, 1 to 65535, or None."""
    args = sys.argv[1:] if argv is None else argv
    return next((n for a in args if (n := cfg.parse_int(a)) is not None and 0 < n < 65536), None)


def port_after_restart() -> int:
    """A port given on the command line (pack-dashboard 8801) wins over ports.dashboard, as in server.main()."""
    return port_arg() or cfg.get("ports.dashboard")


STOPPED = {"inactive", "not loaded"}  # an installed service someone stopped with Stop now


def restart_names() -> list[str]:
    """Services to (re)install after a save: the installed ones still selected, plus any the new settings require
    (asr.engine = whisper starts the whisper service). Never the dashboard: restarting its unit could start a second
    copy when this one was started by hand."""
    installed, chosen = set(services.installed()), services.selected()
    return [n for n in chosen if n != "dashboard" and (n in installed or n in services.required())]


def run_here_changes(before: set[str]) -> tuple[list[str], list[str]]:
    """What a save changed about which services run here, from the ones selected before it: (to install: newly
    selected and not installed yet; to uninstall: installed and no longer selected). Never the dashboard."""
    have, chosen = set(services.installed()), set(services.selected())
    added = sorted(n for n in chosen - before if n not in have and n != "dashboard")
    dropped = sorted(n for n in (before - chosen) & have if n != "dashboard")
    return added, dropped


def _pack_services(verb: str, names: list[str]) -> tuple[int, str]:
    """(return code, output) of `pack-services <verb> <names>`."""
    cascade = verb == "restart" or (
        verb == "install"
        and (not names or bool({"--cascade", "continuum-broker", "continuum-capture"}.intersection(names)))
    )
    timeout = service_cascade.subprocess_timeout() if cascade else 300
    r = subprocess.run(
        [sys.executable, "-m", "liberational.services", verb, *names], capture_output=True, text=True, timeout=timeout
    )
    return r.returncode, (r.stdout + r.stderr).strip()


def _outcome(name: str, code: int, out: str) -> str:
    """done, or failed with pack-services' line about this service (its unit's, never a longer name's) or its exit."""
    mine = (f"pack-{name}.", f"pack-{name}:", f"pack.{name}:")
    bad = next((line for line in out.splitlines() if "FAILED" in line and line.startswith(mine)), None)
    if bad:
        return f"failed: {bad}"
    return "done" if code == 0 else f"failed: {out.splitlines()[-1] if out else f'exit {code}'}"


def here_changed(before: set[str] | None) -> bool:
    """Whether a save changed which services run here (Also run, Don't run, or a setting that needs one)."""
    if before is None:
        return False
    cfg.load.cache_clear()
    try:
        return set(services.selected()) != before
    except SystemExit:
        return False


def apply_run_here(before: set[str] | None) -> list[str]:
    """Carry out a saved Also run or Don't run: uninstall what may no longer run here, logging one line each,
    and say which newly allowed services to install with the restart. None before: nothing is known to have changed."""
    if before is None:
        return []
    added, dropped = run_here_changes(before)
    if dropped:
        code, out = _pack_services("uninstall", dropped)
        for n in dropped:
            LOG.info(
                f"services: off {n} (no longer runs here after a Settings save; uninstalled): {_outcome(n, code, out)}"
            )
    return added


def restart(
    names: Sequence[str] = (),
    dashboard: bool = True,
    before: set[str] | None = None,
) -> threading.Thread:
    """After the reply has gone out: carry out a changed Also run or Don't run (before: the services selected before
    the save), install (and so restart) the named services on the new settings, leaving one stopped with Stop
    now stopped until the computer restarts, then re-exec the dashboard if `dashboard`. Returns the thread doing it."""

    def go() -> None:
        time.sleep(0.5)  # let the browser get the reply before this process goes away
        cfg.load.cache_clear()
        try:
            added = apply_run_here(before)
            state = services.states() if names else {}
            todo = list(dict.fromkeys([*(n for n in names if state.get(n) not in STOPPED), *added]))
            if todo:
                code, out = _pack_services("install", todo)
                for n in added:
                    done = _outcome(n, code, out)
                    LOG.info(f"services: on {n} (runs here after a Settings save; installed and started): {done}")
                LOG.info("settings: restarted %s %s", ", ".join(todo), out[-300:])
        except SystemExit as ex:  # the new pack.toml names an unknown service
            LOG.info("settings: not restarting services: %s", ex)
        except Exception as ex:  # a timeout, say: the dashboard must still come back on the new settings
            LOG.info(f"settings: services not restarted: {ex!r}")
        if not dashboard:
            return
        LOG.info("settings: saved, restarting the dashboard")
        sys.stdout.flush()
        os.execv(sys.executable, [sys.executable, *sys.argv])

    t = threading.Thread(target=go, daemon=True, name="settings-restart")
    t.start()
    return t


REF_MIN_S, REF_MAX_S, REF_MAX_BYTES = 8, 60, 16 << 20


def voice_ref(handler: H) -> None:
    """GET and POST /config/voice_ref (see the module docstring)."""
    import io

    import soundfile as sf

    from liberational.speech.voice_clone import PROMPT, ref_text_for

    path = cfg.path("owner.voice_ref")
    if handler.command in ("GET", "HEAD"):
        if "audio=1" in handler.path:
            if not path.is_file():
                return handler._json({"ok": False, "error": "no voice reference recorded yet"}, 404)
            return handler._media(path.read_bytes(), "audio/wav")
        info = {"ok": True, "path": shown(path)}
        info["problem"] = reference_problem(path)  # missing or unreadable: the voice speaks Kokoro meanwhile
        try:
            info.update(
                exists=True,
                seconds=round(sf.info(str(path)).duration, 1),
                recorded=round(path.stat().st_mtime, 3),
                prompt=ref_text_for(path),
            )
        except Exception:  # missing or unreadable: offer the standard passage
            info.update(exists=False, seconds=0, prompt=PROMPT)
        return handler._json(info)
    origin = handler.headers.get("Origin")
    if not handler._can_control():
        return handler._json({"ok": False, "error": "the voice reference can only be recorded on this machine"}, 403)
    if handler.headers.get_content_type() != "audio/wav" or (
        origin
        and origin
        not in (f"http://{handler.headers.get('Host') or ''}", f"https://{handler.headers.get('Host') or ''}")
    ):
        return handler._json({"ok": False, "error": "same-origin audio/wav only"}, 403)
    try:
        n = int(handler.headers.get("Content-Length") or 0)
        if not 0 < n <= REF_MAX_BYTES:
            raise ValueError
        raw = handler.rfile.read(n)
        dur = sf.info(io.BytesIO(raw)).duration
    except Exception:
        return handler._json({"ok": False, "error": "send a WAV recording"}, 400)
    if not REF_MIN_S <= dur <= REF_MAX_S:
        return handler._json({"ok": False, "error": f"record {REF_MIN_S} to {REF_MAX_S} seconds (got {dur:.1f})"}, 400)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)
    path.with_suffix(".txt").write_text(PROMPT)  # the voice clone needs the clip's transcript
    LOG.info(f"settings: recorded a {dur:.1f} s voice reference at {path}")
    enrolled, note = False, "Set the owner's name to have Liberational learn this voice as the owner."
    owner = cfg.owner_label()
    if owner:  # the owner's name, as the owner check reads it
        name = cfg.owner_name() if cfg.owner_name().lower() == owner else owner
        state = cfg.path("paths.state")
        cmd = [sys.executable, "-m", "liberational.people.voice_edits", "--state-dir", str(state), "enroll"]
        r = subprocess.run([*cmd, name, str(path)], cwd=str(cfg.workdir()), capture_output=True, text=True, timeout=120)
        enrolled = r.returncode == 0
        note = (
            f"Learned as {name}'s voice."
            if enrolled
            else "Saved, but learning the voice failed: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1]
        )
    return handler._json({"ok": True, "seconds": round(dur, 1), "enrolled": enrolled, "note": note})


def owner_moved(name: str, pinned: str) -> bool:
    """Whether a save moved the owner check off the owner's voice: owner.name was name and is now something else, with
    owner.speaker_label (pinned before the save) empty both times, so the voice is matched by name."""
    return bool(name and cfg.owner_name() and not pinned and not cfg.get("owner.speaker_label").strip()) and (
        name.lower() != cfg.owner_label()
    )


def _put_back(path: pathlib.Path, text: str | None) -> None:
    """Restore pack.toml as it was before a save (gone, if there was none)."""
    if text is None:
        path.unlink(missing_ok=True)
    else:
        tmp = path.with_name(path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # may hold secrets, as config.save writes it
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    cfg.load.cache_clear()


def handle(
    handler: H,
    carry_owner: Callable[[str, str], tuple[int, dict[str, Any]]] | None = None,
) -> bool:
    """Serve /config and /config/voice_ref; False for any other path. `carry_owner(old, new)` renames the owner in
    People when a save changes owner.name (server.rename_owner)."""
    route = handler.path.split("?")[0]
    if route == "/config/voice_ref" and handler.command in ("GET", "HEAD", "POST"):
        voice_ref(handler)
        return True
    if route == "/config/voices" and handler.command in ("GET", "HEAD"):
        handler._json({"ok": True, "clone": can_clone(), "kokoro": kokoro_voices()})
        return True
    if route != "/config":
        return False
    if handler.command in ("GET", "HEAD"):
        handler._json(view(handler._can_control()))
        return True
    if handler.command != "POST":
        return False
    origin = handler.headers.get("Origin")
    if not handler._can_control():
        handler._json({"ok": False, "error": "settings can only be changed from this machine"}, 403)
        return True
    if handler.headers.get_content_type() != "application/json" or (
        origin
        and origin
        not in (f"http://{handler.headers.get('Host') or ''}", f"https://{handler.headers.get('Host') or ''}")
    ):
        handler._json({"ok": False, "error": "same-origin JSON only"}, 403)
        return True
    try:
        n = int(handler.headers.get("Content-Length") or 0)
        if not 0 < n <= 65536:
            raise ValueError
        values = json.loads(handler.rfile.read(n)).get("values")
        if not isinstance(values, dict):
            raise ValueError
    except ValueError, AttributeError:
        handler._json({"ok": False, "error": 'send {"values": {"section.key": value}}'}, 400)
        return True
    owner, pinned = cfg.owner_name(), cfg.get("owner.speaker_label").strip()
    path = cfg.local_config()
    try:
        before = path.read_text() if path.exists() else None
        cfg.load.cache_clear()  # what pack.toml says now, hand edits since the dashboard started included
        try:
            selected_before: set[str] | None = set(services.selected())
        except SystemExit:  # an unknown service name in pack.toml: a save may be what fixes it
            selected_before = None
        written, overridden = cfg.save(values)
        if login_item.KEY in values:  # the desktop's autostart entry follows the setting at once (#157)
            try:
                LOG.info(f"settings: {login_item.sync()}")
            except OSError as ex:  # the setting is saved; the application applies it again when it next starts
                LOG.warning(f"settings: start at login not applied: {ex}")
    except cfg.SettingError as ex:
        handler._json({"ok": False, "error": str(ex), "field": ex.key}, 400)
        return True
    except OSError as ex:
        handler._json({"ok": False, "error": f"couldn't write {cfg.local_config()}: {ex.strerror}"}, 500)
        return True
    if carry_owner and owner_moved(owner, pinned):
        try:
            code, body = carry_owner(owner, cfg.owner_name())
        except Exception as ex:  # the rename's own late steps log rather than raise, so nothing has moved yet
            code, body = 500, {"error": f"renaming the owner in People failed: {ex}"}
        if code != 200:
            _put_back(path, before)
            LOG.info(f"settings: owner.name not changed: {body.get('error')}")
            handler._json({"ok": False, "error": body.get("error"), "field": "owner.name"}, code)
            return True
        LOG.info(f"settings: renamed the owner {owner!r} -> {cfg.owner_name()!r} in People")
    LOG.info(f"settings: wrote {', '.join(written) or 'no changes'} to {cfg.local_config()}")
    # Bound before the reply goes out: a test's stand-in for restart() must still be the one called even if the test
    # stops patching the instant it reads the reply (that race once reinstalled real services from a worktree).
    go = restart
    names, dashboard = plan(list(values))
    handler._json(
        {
            "ok": True,
            "written": written,
            "overridden": overridden,
            "restarted": names,
            "restarting": dashboard,
            "url": f"http://{_host(handler)}:{port_after_restart()}/",
        }
    )
    if names or dashboard or here_changed(selected_before):
        go(names, dashboard, selected_before)
    else:
        LOG.info("settings: saved live settings, nothing to restart")
    return True


# ---- routes (dashboard/routes.py) ----


def _carry_owner(old: str, new: str) -> tuple[int, dict[str, Any]]:
    """Rename the owner in People when a save changes owner.name (imported when it's needed: People is built on a
    great deal, and Settings is not)."""
    from liberational.dashboard import people_writes

    return people_writes.rename_owner(old, new)


@routes.get("/config", "/config/voice_ref", "/config/voices")
@routes.post_raw("/config", "/config/voice_ref")
def _route(h: H) -> None:
    handle(h, carry_owner=_carry_owner)
