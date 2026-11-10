"""Start at login (#157, part of #151): off by default, and one setting, `app.start_at_login`, turns it on.

Nothing of the system starts at login unless the owner asks: it runs while the application (pack-app) is open. With
the setting on, the desktop opens the application at login:

- **Linux**: an XDG autostart entry, `$XDG_CONFIG_HOME/autostart/liberational.desktop` (`~/.config/autostart/`),
  which GNOME, KDE, and the other desktops that follow the specification run once the session is up. Turning the
  setting off deletes the file. Nothing is registered with systemd.
- **macOS**: not yet. A login item belongs to an application bundle, and there is none until #164: with the bundle,
  `sync()` calls `SMAppService.mainApp.register()` and `unregister()` (ServiceManagement, macOS 13 and later), which
  lists the application under System Settings, General, Login Items. Until then the setting can be saved but does
  nothing on a Mac, and `sync()` says so.

`sync()` makes the entry match the setting, and is safe to call any number of times. It is called when the
application starts, when its window's checkbox changes, and when Settings saves the key (dashboard/settings.py).
The entry on disk is what counts as "on" for the window (`enabled()`), so it shows a change made anywhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from liberational import config as cfg
from liberational import logs

LOG = logs.get("app")

KEY = "app.start_at_login"
NAME = "liberational.desktop"


def supported() -> bool:
    """Whether this platform has a way to start the application at login yet."""
    return sys.platform.startswith("linux")


def entry_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / "autostart" / NAME


def command() -> list[str]:
    """The application as this install runs it: the `pack-app` beside this interpreter, or the module where there is
    no such script. `PACK_ROOT` and `PACK_CONFIG`, when this process was given them, are given to the entry too, so
    the session opens the same install."""
    script = Path(sys.executable).parent / "pack-app"
    run = [str(script)] if script.exists() else [sys.executable, "-m", "liberational.app_window"]
    kept = [f"{name}={os.environ[name]}" for name in ("PACK_ROOT", "PACK_CONFIG") if os.environ.get(name)]
    return ["env", *kept, *run] if kept else run


def _quoted(arg: str) -> str:
    """One argument of a desktop entry's Exec line: quoted when it has a reserved character, with the four the
    specification names escaped, and a literal percent doubled."""
    arg = arg.replace("%", "%%")
    if not arg or any(c in arg for c in " \t\n\"'\\><~|&;$*?#()`"):
        for c in ("\\", '"', "`", "$"):
            arg = arg.replace(c, "\\" + c)
        return f'"{arg}"'
    return arg


def entry() -> str:
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Liberational\n"
        "Comment=Start recording and open the Liberational window at login\n"
        f"Exec={' '.join(_quoted(a) for a in command())}\n"
        "Terminal=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    )


def enabled() -> bool:
    """Whether the application starts at login now: the entry is there."""
    return supported() and entry_path().is_file()


def sync(want: bool | None = None) -> str:
    """Make the entry match the setting (or `want`), and say what was done, in a sentence for the log."""
    on = bool(cfg.get("app.start_at_login")) if want is None else want
    if not supported():
        return (
            "Start at login isn't available on this platform yet (it needs the application bundle, #164)"
            if on
            else "Start at login is off"
        )
    path = entry_path()
    if not on:
        if path.is_file():
            path.unlink()
            return f"Start at login turned off: removed {path}"
        return "Start at login is off"
    text = entry()
    if path.is_file() and path.read_text() == text:
        return "Start at login is on"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return f"Start at login turned on: wrote {path}"
