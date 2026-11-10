"""The application's window (#156, part of #151): while the system runs, something on screen says so.

  pack-app    # open the application: start the services (supervisor.py) and show its window

The window says whether the system is running and recording (and when cameras or microphones are paused for
privacy), opens the dashboard, has a Start at login checkbox (login_item.py, off by default), and has Quit. Quit, and closing the window, ask the supervisor to drain, then stop,
every service; the window stays, saying it is stopping, until they are gone. So nothing records with no window: the
window is there for as long as any service runs, and if the supervisor stops by itself the window says so.

Toolkit: Tk, through `tkinter`. It comes with the Python the project already installs (Tk 9 in uv's builds, on
Linux and macOS alike), so it adds no package, no system library, and no compiler step. The alternatives each cost
something here:

- A tray icon with pystray needs PyGObject for GNOME's AppIndicator (a C build against system GTK headers, which the
  interpreter uv installs can't import from the distribution's packages), or its X11 backend, which draws nothing
  on GNOME under Wayland. Ubuntu's default desktop shows tray icons only through that AppIndicator path.
- A Qt or GTK shell is tens to hundreds of megabytes of wheels, or the PyGObject build again.
- A native shell per platform is two programs to build and sign.

A window is also the plainer answer to "is it recording?" than an icon that may be hidden. The Mac application
bundle (#164) can put a menu-bar item in front of the same `Presence` later: everything the window shows comes from
`summary()`, which knows nothing of Tk.

The supervisor runs in a thread of this process, so its children are this process's children (on a Mac, that is what
lets the bundle's camera and microphone permission reach them). Tk has the main thread, as macOS requires.
"""

from __future__ import annotations

import signal
import threading
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from liberational import config as cfg
from liberational import login_item, logs, supervisor
from liberational.service_cascade import CAPTURE

LOG = logs.get("app")

REFRESH_MS = 1000
TITLE = "Liberational"

# Where the application is: starting its services, running, stopping them, or ended.
STARTING, RUNNING, STOPPING, STOPPED = "starting", "running", "stopping", "stopped"


@dataclass(frozen=True)
class Summary:
    headline: str  # the one line that answers "is it recording?"
    detail: str  # how many services are up, and what is paused or wrong
    recording: bool  # capture is running and something it reads isn't paused


def paused_devices() -> dict[str, tuple[list[str], int]]:
    """{"camera" | "microphone": (the paused ones' names, how many there are)}, from the capture manifest. Empty when
    the manifest can't be read: the window then says nothing about pauses rather than guess."""
    try:
        from liberational.dashboard import privacy

        rows = privacy.devices()
    except Exception as error:
        LOG.debug(f"paused devices unknown: {logs.reason(error)}")
        return {}
    out: dict[str, tuple[list[str], int]] = {}
    for kind in ("camera", "microphone"):
        of_kind = [row for row in rows if row["kind"] == kind]
        if of_kind:
            out[kind] = ([row["name"] for row in of_kind if row["paused"]], len(of_kind))
    return out


def _paused_text(paused: Mapping[str, tuple[Sequence[str], int]]) -> tuple[str, bool]:
    """(what is paused, in words; whether every device there is, is paused)."""
    parts, everything = [], bool(paused)
    for kind, plural in (("camera", "cameras"), ("microphone", "microphones")):
        names, total = paused.get(kind, ((), 0))
        if total and len(names) == total:
            parts.append(f"{plural} paused" if total > 1 else f"{kind} paused")
        elif names:
            parts.append(f"{len(names)} of {total} {plural} paused")
            everything = False
        elif total:
            everything = False
    return ", ".join(parts), everything and bool(parts)


def summary(
    phase: str, status: Mapping[str, Any] | None, paused: Mapping[str, tuple[Sequence[str], int]] | None = None
) -> Summary:
    """What the window says, from the supervisor's status (Supervisor.status()) and the paused devices."""
    services = dict((status or {}).get("services") or {})
    daemons = {name: s for name, s in services.items() if not s.get("scheduled")}
    up = sum(1 for s in daemons.values() if s.get("state") == "running")
    count = f"{up} of {len(daemons)} services running"
    if phase == STOPPED:
        return Summary("Stopped. Nothing is recording.", "The services have stopped. Close this window.", False)
    if phase == STOPPING:
        return Summary("Stopping...", f"Saving what was recorded, then stopping. {count}.", False)
    capture = services.get(CAPTURE, {}).get("state") == "running"
    waiting = sorted(name for name, s in daemons.items() if s.get("state") == "activating")
    trouble = f" Restarting: {', '.join(waiting)}." if waiting else ""
    if phase == STARTING:
        return Summary("Starting...", f"{count}.{trouble}", False)
    text, everything = _paused_text(paused or {})
    if not capture:
        why = "Capture is off." if CAPTURE not in services else "Capture isn't running."
        return Summary("Running, not recording", f"{why} {count}.{trouble}", False)
    if everything:
        return Summary("Running, paused for privacy", f"Nothing is recording: {text}. {count}.{trouble}", False)
    extra = f" ({text})" if text else ""
    return Summary("Running and recording", f"{count}{extra}.{trouble}", True)


def save_start_at_login(on: bool) -> str:
    """Save the setting (pack.toml's app.start_at_login) and make the desktop's entry match it."""
    cfg.save({login_item.KEY: on})
    return login_item.sync(on)


@dataclass
class Presence:
    """The application behind the window: the supervisor's App, the thread that runs it, and what phase it is in."""

    app: supervisor.App
    missed: Sequence[str] = ()
    paused: Callable[[], Mapping[str, tuple[Sequence[str], int]]] = paused_devices
    open_url: Callable[[str], object] = webbrowser.open
    starts_at_login: Callable[[], bool] = login_item.enabled
    set_start_at_login: Callable[[bool], str] = save_start_at_login
    thread: threading.Thread | None = None
    started: bool = False
    asked: bool = False  # someone asked to quit (the button, closing the window, or a signal)

    def start(self) -> None:
        """Run the supervisor in its own thread: start everything, keep it up, and drain and stop on quit."""

        def run() -> None:
            try:
                self.app.run(self.missed)
            except Exception:
                LOG.exception("the supervisor stopped")

        self.thread = threading.Thread(target=run, name="supervisor", daemon=False)
        self.thread.start()
        self.started = True

    @property
    def phase(self) -> str:
        if self.thread is None or not self.started:
            return STARTING
        if not self.thread.is_alive():
            return STOPPED
        if self.app.quitting.is_set():
            return STOPPING
        services = self.app.supervisor.status()["services"]
        return RUNNING if any(s["pid"] is not None for s in services.values()) else STARTING

    def summary(self) -> Summary:
        phase = self.phase
        return summary(phase, self.app.supervisor.status(), self.paused() if phase == RUNNING else None)

    def quit(self) -> None:
        """Ask the supervisor to drain, then stop, everything. The window closes once it has (`finished`)."""
        self.asked = True
        self.app.quitting.set()

    @property
    def finished(self) -> bool:
        return self.thread is not None and not self.thread.is_alive()

    def open_dashboard(self) -> None:
        self.open_url(f"http://localhost:{cfg.get('ports.dashboard')}/")


class Window:
    """The Tk window over a Presence. Closing it is Quit: the window never goes while a service runs."""

    def __init__(self, presence: Presence, tk: Any = None) -> None:
        if tk is None:
            import tkinter as tk
        self.presence, self.tk = presence, tk
        root = self.root = tk.Tk()
        root.title(TITLE)
        root.resizable(False, False)
        self.dot = tk.Label(root, text="●", font=("TkDefaultFont", 22))
        self.dot.grid(row=0, column=0, padx=(16, 8), pady=(16, 4))
        self.headline = tk.Label(root, text="", font=("TkDefaultFont", 14, "bold"), anchor="w")
        self.headline.grid(row=0, column=1, columnspan=2, sticky="w", padx=(0, 16), pady=(16, 4))
        self.detail = tk.Label(root, text="", anchor="w", justify="left", wraplength=360)
        self.detail.grid(row=1, column=1, columnspan=2, sticky="w", padx=(0, 16), pady=(0, 12))
        self.dashboard = tk.Button(root, text="Open the dashboard", command=presence.open_dashboard)
        self.dashboard.grid(row=2, column=1, sticky="w", pady=(0, 16))
        self.quit = tk.Button(root, text="Quit and stop recording", command=self.ask_quit)
        self.quit.grid(row=2, column=2, sticky="e", padx=(8, 16), pady=(0, 16))
        # Start at login (#157): off unless the owner ticks it. Shown as the entry on disk is, so a change made in
        # Settings shows here too.
        self.at_login = tk.BooleanVar(root, value=presence.starts_at_login())
        self.login = tk.Checkbutton(
            root,
            text="Start at login" if login_item.supported() else "Start at login (not on this platform yet)",
            variable=self.at_login,
            command=self.toggle_login,
            state="normal" if login_item.supported() else "disabled",
        )
        self.login.grid(row=3, column=1, columnspan=2, sticky="w", pady=(0, 12))
        root.protocol("WM_DELETE_WINDOW", self.ask_quit)
        self.refresh()

    def ask_quit(self) -> None:
        """Quit, from the button or from closing the window. Once the services have stopped, closing just closes."""
        if self.presence.finished:
            self.root.destroy()
            return
        self.presence.quit()
        self.show()

    def toggle_login(self) -> None:
        """The checkbox changed: save the setting and add or remove the desktop's entry. A failure leaves the box as
        the entry on disk is."""
        try:
            LOG.info(self.presence.set_start_at_login(bool(self.at_login.get())))
        except Exception as error:
            LOG.error(f"start at login not changed: {logs.reason(error)}")
        self.at_login.set(self.presence.starts_at_login())

    def show(self) -> None:
        self.at_login.set(self.presence.starts_at_login())
        now = self.presence.summary()
        self.headline.config(text=now.headline)
        self.detail.config(text=now.detail)
        self.dot.config(fg="#c62828" if now.recording else "#9e9e9e")  # red while recording, grey while not
        busy = self.presence.phase in (STOPPING, STOPPED)
        self.quit.config(state="disabled" if busy else "normal")
        self.dashboard.config(state="disabled" if busy else "normal")
        self.root.title(f"{TITLE}: {now.headline}")

    def refresh(self) -> None:
        """Show the present state, and again every REFRESH_MS. After a Quit, close once everything has stopped; a
        supervisor that stopped by itself leaves the window up, saying so."""
        if self.presence.finished and self.presence.asked:
            self.root.destroy()
            return
        self.show()
        self.root.after(REFRESH_MS, self.refresh)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    """`pack-app`: the application. Starts the services and shows the window until they have stopped."""
    logs.setup()
    logs.adopt("supervisor")
    with supervisor.application() as (app, missed):
        presence = Presence(app, missed)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: presence.quit())
        try:
            window = Window(presence)
        except Exception as error:  # no display: say so before any service starts, and start none
            raise SystemExit(
                f"can't open a window ({logs.reason(error)}). The application needs a desktop session; "
                "`pack-supervisor run` runs the services without one."
            ) from error
        LOG.info(login_item.sync())  # the desktop's entry follows the setting, whoever changed it last
        presence.start()
        try:
            window.run()
        finally:
            # The window is gone (closed after a stop, or the toolkit failed): never leave services behind it.
            presence.quit()
            if presence.thread is not None:
                presence.thread.join()


if __name__ == "__main__":
    main()
