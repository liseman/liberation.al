"""The application's window (#156, part of #151): while the system runs, something on screen says so.

  pack-app    # open the application: start the services (supervisor.py) and show its window

The window says whether the system is running and recording (and when cameras or microphones are paused for
privacy, or declared but sending nothing), opens the dashboard, has a Start at login checkbox (login_item.py, off by default), and has Quit. Quit, and closing the window, ask the supervisor to drain, then stop,
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

import io
import signal
import subprocess
import sys
import threading
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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


def _listed(items: Sequence[str]) -> str:
    """a; a and b; a, b, and c."""
    if len(items) < 3:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


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
    if not capture and (status or {}).get("restarting"):
        # A restart someone asked for (Settings, a deploy): capture being down is planned, not a fault.
        return Summary(
            f"Restarting ({up} of {len(daemons)} up)",
            "Restarting the services, as asked. Recording resumes once capture is back.",
            False,
        )
    if not capture:
        why = "Capture is off." if CAPTURE not in services else "Capture isn't running."
        return Summary("Running, not recording", f"{why} {count}.{trouble}", False)
    if everything:
        return Summary("Running, paused for privacy", f"Nothing is recording: {text}. {count}.{trouble}", False)
    extra = f" ({text})" if text else ""
    resting = {name for names, _ in (paused or {}).values() for name in names}
    missing = [d for d in (status or {}).get("unregistered") or [] if d["name"] not in resting]
    if missing:
        # A device the manifest declares that capture never registered (#183): capture runs on without it, so
        # only this says the microphone, or a camera, isn't being recorded.
        named = _listed([f"{d['kind']} {d['name']}" for d in missing])
        total = sum(n for _, n in (paused or {}).values())
        some = not total or len(missing) + len(resting) < total
        headline = "Recording, but not from every device" if some else "Running, not recording"
        return Summary(headline, f"Nothing is coming from the {named}. {count}{extra}.{trouble}", some)
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


# ---- how the window looks
#
# The window should look like any other window on the desktop: the desktop's own interface font, its light or dark
# scheme, its scaling, and themed (ttk) widgets. Tk does that by itself on macOS (aqua draws native widgets in the
# system font and follows the appearance). On Linux the Tk that uv's Python carries is built without Xft: it can only
# use the X server's core fonts (`tk::pkgconfig get fontsystem` is "x11"), which have no anti-aliasing and none of the
# desktop's fonts, so a heading asked for in TkDefaultFont came out in the bitmap "fixed" face. There the window's
# words are drawn by FreeType through Pillow (already a dependency) into images: the heading in Figtree, the
# dashboard's own typeface, from the built dashboard's assets, and everything else in the desktop's interface font as
# fontconfig finds it. A Tk built with Xft draws its own text in the desktop's font, as on macOS.

BODY_PT = 11.0  # the body text when the desktop names no size; kept within BODY_RANGE
BODY_RANGE = (10.0, 12.0)
HEADING_STEP = 5.0  # the heading is this many points larger than the body
LINE_SPACING = 1.25  # line height over the font's ascent plus descent
WRAP_PT = 300  # the detail wraps at this width
SANS = ("Inter", "Cantarell", "Ubuntu Sans", "Noto Sans", "DejaVu Sans", "Liberation Sans")
FIGTREE = "Figtree-Variable*.woff2"  # in the built dashboard's assets (frontend/src/shared/fonts)
CHECKBOX_PT = 11  # the tick box's side
SEMIBOLD, REGULAR = 600, 400
WEIGHT_AXIS = (b"Weight", "Weight")


@dataclass(frozen=True)
class Desktop:
    """What the desktop says about how windows should look."""

    font: tuple[str, float] | None = None  # the interface font: family and size in points ("Ubuntu Sans", 11)
    dark: bool = False  # the desktop prefers a dark scheme
    dpi: float | None = None  # the desktop's text DPI (Xft.dpi); None leaves Tk's own scaling


@dataclass(frozen=True)
class Palette:
    background: str
    text: str
    secondary: str  # the detail line, and disabled text
    button: str
    button_active: str
    border: str
    accent: str  # the ticked checkbox
    field: str  # the unticked checkbox
    recording: str  # the dot while recording
    idle: str  # the dot while not


# GNOME's (libadwaita's) window colours, flattened onto the window background. Both pass WCAG AA for every text pair.
LIGHT = Palette(
    "#fafafa", "#1e1e1e", "#5e5c64", "#e6e6e6", "#d6d6d6", "#c6c6c6", "#3584e4", "#ffffff", "#c01c28", "#9a9996"
)
DARK = Palette(
    "#242424", "#ffffff", "#c0bfbc", "#383838", "#454545", "#1a1a1a", "#3584e4", "#2e2e2e", "#f66151", "#77767b"
)


def _ask(argv: Sequence[str]) -> str:
    """A desktop command's output, or "" when it is missing, fails, or hangs."""
    try:
        done = subprocess.run(list(argv), capture_output=True, text=True, timeout=2, check=False)
    except OSError, subprocess.SubprocessError:
        return ""
    return done.stdout if done.returncode == 0 else ""


def parse_font_name(value: str) -> tuple[str, float] | None:
    """GNOME's font-name setting ("'Ubuntu Sans 11'", Pango's "family [style] size") as (family, size), or None."""
    words = value.strip().strip("'\"").split()
    if len(words) < 2:
        return None
    try:
        size = float(words[-1])
    except ValueError:
        return None
    styles = {"bold", "italic", "oblique", "light", "medium", "semi-bold", "semibold", "condensed", "regular"}
    family = list(words[:-1])
    while len(family) > 1 and family[-1].lower() in styles:
        family.pop()
    return (" ".join(family), size) if size > 0 else None


def parse_xft_dpi(resources: str) -> float | None:
    """Xft.dpi from `xrdb -query` (GNOME sets it to 96 times the scale, text scaling included), or None."""
    for line in resources.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "Xft.dpi":
            try:
                dpi = float(value.strip())
            except ValueError:
                return None
            return dpi if dpi > 0 else None
    return None


def read_desktop(ask: Callable[[Sequence[str]], str] = _ask, platform: str = sys.platform) -> Desktop:
    """The desktop's font, scheme, and DPI, from GNOME's settings and the X resources. On macOS Tk follows the system
    itself, so only the appearance is read (for the dot's colours)."""
    if platform == "darwin":
        return Desktop(dark=ask(["defaults", "read", "-g", "AppleInterfaceStyle"]).strip() == "Dark")
    interface = ["gsettings", "get", "org.gnome.desktop.interface"]
    return Desktop(
        font=parse_font_name(ask([*interface, "font-name"])),
        dark="dark" in ask([*interface, "color-scheme"]),
        dpi=parse_xft_dpi(ask(["xrdb", "-query"])),
    )


def body_points(desktop: Desktop) -> float:
    low, high = BODY_RANGE
    return min(max(desktop.font[1] if desktop.font else BODY_PT, low), high)


def font_file(pattern: str, ask: Callable[[Sequence[str]], str] = _ask) -> tuple[str, int] | None:
    """The file (and face index) fontconfig picks for a pattern ("Ubuntu Sans:weight=semibold"), or None. A
    monospace face (what fontconfig falls back to on some systems) is refused: the plain sans is asked for instead."""
    out = ask(["fc-match", "-f", "%{file}\t%{index}\t%{spacing}", pattern]).strip()
    path, _, rest = out.partition("\t")
    index, _, spacing = rest.partition("\t")
    if not path or spacing.strip() == "100":
        return None if pattern.startswith("sans-serif") else font_file("sans-serif" + _style(pattern), ask)
    try:
        return path, int(index or 0)
    except ValueError:
        return path, 0


def _style(pattern: str) -> str:
    return pattern[pattern.index(":") :] if ":" in pattern else ""


def figtree_file() -> Path | None:
    """The dashboard's typeface, as the built dashboard ships it, or None (a checkout with no frontend build)."""
    assets = Path(__file__).parent / "dashboard" / "static" / "app" / "assets"
    found = sorted(assets.glob(FIGTREE)) if assets.is_dir() else []
    return found[0] if found else None


class Typesetter:
    """Text drawn by FreeType (Pillow) into PNG images Tk can show, for a Tk that can't draw the desktop's fonts.
    Sizes are in points, at the window's scaling (pixels per point)."""

    def __init__(self, scaling: float, desktop: Desktop, ask: Callable[[Sequence[str]], str] = _ask) -> None:
        from PIL import ImageFont

        self.scaling = scaling
        self.fonts: dict[str, Any] = {}
        family = desktop.font[0] if desktop.font else "sans-serif"
        body = font_file(family, ask)
        if body is None:
            raise RuntimeError("fontconfig found no font")
        semibold = font_file(f"{family}:weight=semibold", ask) or body
        figtree = figtree_file()
        body_pt = body_points(desktop)
        roles: dict[str, tuple[tuple[str | Path, int], float, int]] = {
            "heading": ((figtree, 0) if figtree else semibold, body_pt + HEADING_STEP, SEMIBOLD),
            "body": (body, body_pt, REGULAR),
            "button": (body, body_pt, REGULAR),
        }
        for role, ((path, index), points, weight) in roles.items():
            font = ImageFont.truetype(str(path), max(1, round(points * scaling)), index=index)
            try:  # a variable font (Figtree is one, and its default is Light): set the weight on its axis
                axes = font.get_variation_axes()
                font.set_variation_by_axes(
                    [float(weight if axis["name"] in WEIGHT_AXIS else axis["default"] or 0) for axis in axes]
                )
            except OSError:
                pass  # a static face: fontconfig already chose the weight
            self.fonts[role] = font

    def lines(self, text: str, role: str, wrap: int | None) -> list[str]:
        """The text broken at spaces to fit `wrap` pixels (and at its own newlines)."""
        font, out = self.fonts[role], []
        for paragraph in text.split("\n"):
            line = ""
            for word in paragraph.split(" "):
                trial = f"{line} {word}" if line else word
                if wrap is not None and line and font.getlength(trial) > wrap:
                    out.append(line)
                    line = word
                else:
                    line = trial
            out.append(line)
        return out

    def render(self, text: str, role: str, colour: str, wrap: int | None = None) -> bytes:
        """The text as a PNG with a transparent background: Tk blends it onto whatever the widget's background is
        (a button's changes with hover)."""
        from PIL import Image, ImageDraw

        font = self.fonts[role]
        ascent, descent = font.getmetrics()
        height = round((ascent + descent) * LINE_SPACING)
        lines = self.lines(text, role, wrap)
        width = max([round(font.getlength(line)) for line in lines] + [1]) + 2
        image = Image.new("RGBA", (width, height * len(lines)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        lead = (height - ascent - descent) // 2
        for n, line in enumerate(lines):
            draw.text((0, n * height + lead), line, font=font, fill=colour)
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()

    def checkbox(self, ticked: bool, fill: str, edge: str, gap: float) -> bytes:
        """A rounded tick box (clam's own draws an X), with `gap` points of space to its right, as a PNG."""
        from PIL import Image, ImageDraw

        side, scale = round(CHECKBOX_PT * self.scaling), 4  # drawn 4x larger and scaled down: smooth edges
        big = side * scale
        image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((0, 0, big - 1, big - 1), radius=big // 5, fill=fill, outline=edge, width=scale)
        if ticked:
            points = [(0.25 * big, 0.52 * big), (0.43 * big, 0.70 * big), (0.76 * big, 0.32 * big)]
            draw.line(points, fill="#ffffff", width=round(0.12 * big), joint="curve")
        image = image.resize((side, side), Image.Resampling.LANCZOS)
        out = Image.new("RGBA", (side + round(gap * self.scaling), side), (0, 0, 0, 0))
        out.paste(image, (0, 0))
        buffer = io.BytesIO()
        out.save(buffer, format="PNG")
        return buffer.getvalue()


class Look:
    """The window's theme, fonts, and scaling, set up on its Tk root."""

    def __init__(self, root: Any, desktop: Desktop, ask: Callable[[Sequence[str]], str] = _ask) -> None:
        import tkinter.font as tkfont
        from tkinter import ttk

        self.root, self.desktop = root, desktop
        self.aqua = root.tk.call("tk", "windowingsystem") == "aqua"
        if desktop.dpi and not self.aqua:
            root.tk.call("tk", "scaling", desktop.dpi / 72.0)  # pixels per point; aqua sets its own
        self.scaling = float(root.tk.call("tk", "scaling"))
        self.palette = DARK if desktop.dark else LIGHT
        self.style = ttk.Style(root)
        self.style.theme_use("aqua" if self.aqua else "clam")
        self.fontsystem = self._fontsystem()
        self.typesetter: Typesetter | None = None
        if not self.aqua and self.fontsystem != "xft":
            try:
                self.typesetter = Typesetter(self.scaling, desktop, ask)
            except Exception as error:  # no fontconfig or no usable font: Tk's own fonts, plainer but readable
                LOG.info(f"the window's text uses Tk's fonts ({logs.reason(error)})")
        body = body_points(desktop)
        default = tkfont.nametofont("TkDefaultFont", root=root)
        if not self.aqua:
            default.configure(family=self._tk_family(tkfont.families(root)), size=round(body))
        for name in ("TkTextFont", "TkMenuFont", "TkCaptionFont"):
            tkfont.nametofont(name, root=root).configure(**{k: default.cget(k) for k in ("family", "size")})
        self.heading_font = tkfont.Font(
            root=root, family=default.cget("family"), size=round(body + HEADING_STEP), weight="bold"
        )
        if not self.aqua:
            self._theme()
            self.style.configure("Detail.TLabel", foreground=self.palette.secondary)
        self.style.configure("Heading.TLabel", font=self.heading_font)

    def _fontsystem(self) -> str:
        try:
            return str(self.root.tk.call("::tk::pkgconfig", "get", "fontsystem"))
        except Exception:
            return ""

    def _tk_family(self, families: Sequence[str]) -> str:
        """The family Tk itself uses (its widgets' fallback): the desktop's if Tk has it, else a common sans."""
        have = {family.lower(): family for family in families}
        wanted = [self.desktop.font[0]] if self.desktop.font else []
        for family in [*wanted, *SANS, "helvetica"]:
            if family.lower() in have:
                return have[family.lower()]
        return "sans-serif"

    def _theme(self) -> None:
        p, style = self.palette, self.style
        self.root.configure(background=p.background)
        style.configure(
            ".",
            background=p.background,
            foreground=p.text,
            bordercolor=p.border,
            darkcolor=p.button,
            lightcolor=p.button,
            focuscolor=p.accent,
            troughcolor=p.button,
            selectbackground=p.accent,
            fieldbackground=p.field,
            font="TkDefaultFont",
        )
        style.map(".", foreground=[("disabled", p.secondary)])
        style.configure("TButton", background=p.button, padding=("10p", "4p"), relief="flat", anchor="center")
        style.map(
            "TButton",
            background=[("disabled", p.background), ("pressed", p.button_active), ("active", p.button_active)],
            lightcolor=[("pressed", p.button_active), ("active", p.button_active)],
            darkcolor=[("pressed", p.button_active), ("active", p.button_active)],
        )
        style.configure(
            "TCheckbutton",
            indicatorbackground=p.field,
            indicatorforeground="#ffffff",
            indicatormargin=(0, 0, "6p", 0),
            indicatorsize=round(10 * self.scaling),
            upperbordercolor=p.border,
            lowerbordercolor=p.border,
        )
        style.map(
            "TCheckbutton",
            background=[("active", p.background)],
            indicatorbackground=[("selected", p.accent), ("disabled", p.background)],
        )
        if self.typesetter is not None:
            self._tick_box(self.typesetter)

    def _tick_box(self, typesetter: Typesetter) -> None:
        """The checkbox's indicator as images: a rounded box, filled with the accent and ticked when selected."""
        import tkinter as tk

        p, gap = self.palette, 7.0
        images = {
            "off": typesetter.checkbox(False, p.field, p.idle, gap),
            "on": typesetter.checkbox(True, p.accent, p.accent, gap),
            "off-disabled": typesetter.checkbox(False, p.background, p.border, gap),
            "on-disabled": typesetter.checkbox(True, p.idle, p.idle, gap),
        }
        self.tick_images = {name: tk.PhotoImage(master=self.root, data=data) for name, data in images.items()}
        t = self.tick_images
        self.style.element_create(
            "Tick.indicator",
            "image",
            t["off"],
            ("disabled", "selected", t["on-disabled"]),
            ("disabled", t["off-disabled"]),
            ("selected", t["on"]),
        )
        self.style.layout(
            "TCheckbutton",
            [
                (
                    "Checkbutton.padding",
                    {
                        "sticky": "nswe",
                        "children": [
                            ("Tick.indicator", {"side": "left", "sticky": ""}),
                            (
                                "Checkbutton.focus",
                                {
                                    "side": "left",
                                    "sticky": "w",
                                    "children": [("Checkbutton.label", {"sticky": "nswe"})],
                                },
                            ),
                        ],
                    },
                )
            ],
        )

    @property
    def dot_colours(self) -> tuple[str, str]:
        """(recording, idle) for the status dot."""
        return self.palette.recording, self.palette.idle

    @property
    def background(self) -> str:
        return "systemWindowBackgroundColor" if self.aqua else self.palette.background


class Window:
    """The Tk window over a Presence. Closing it is Quit: the window never goes while a service runs."""

    def __init__(self, presence: Presence, desktop: Desktop | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.presence, self.tk = presence, tk
        root = self.root = tk.Tk()
        root.title(TITLE)
        root.resizable(False, False)
        look = self.look = Look(root, read_desktop() if desktop is None else desktop)
        self.images: dict[tuple[str, str, str], Any] = {}  # rendered text, kept while Tk shows it
        frame = ttk.Frame(root, padding=("18p", "16p", "18p", "14p"))
        frame.grid(sticky="nsew")
        size = round(10 * look.scaling)
        self.dot = tk.Canvas(frame, width=size, height=size, highlightthickness=0, background=look.background)
        self.dot.create_oval(1, 1, size - 1, size - 1, width=0, tags="light")
        self.dot.grid(row=0, column=0, sticky="w", padx=(0, "9p"))
        self.headline = ttk.Label(frame, style="Heading.TLabel", anchor="w")
        self.headline.grid(row=0, column=1, columnspan=2, sticky="w")
        self.detail = ttk.Label(frame, style="Detail.TLabel", anchor="w", justify="left", wraplength=f"{WRAP_PT}p")
        self.detail.grid(row=1, column=1, columnspan=2, sticky="w", pady=("4p", "14p"))
        self.dashboard = ttk.Button(frame, text="Open the dashboard", command=presence.open_dashboard)
        self.dashboard.grid(row=2, column=1, sticky="w")
        self.quit = ttk.Button(frame, text="Quit and stop recording", command=self.ask_quit)
        self.quit.grid(row=2, column=2, sticky="e", padx=("10p", 0))
        frame.columnconfigure(2, weight=1)
        # Start at login (#157): off unless the owner ticks it. Shown as the entry on disk is, so a change made in
        # Settings shows here too.
        self.at_login = tk.BooleanVar(root, value=presence.starts_at_login())
        self.login = ttk.Checkbutton(
            frame,
            text="Start at login" if login_item.supported() else "Start at login (not on this platform yet)",
            variable=self.at_login,
            command=self.toggle_login,
            state="normal" if login_item.supported() else "disabled",
        )
        self.login.grid(row=3, column=1, columnspan=2, sticky="w", pady=("12p", 0))
        for button in (self.dashboard, self.quit):
            self.typeset(button, button.cget("text"), "button")
        self.typeset(self.login, self.login.cget("text"), "button")
        root.protocol("WM_DELETE_WINDOW", self.ask_quit)
        self.refresh()

    def typeset(self, widget: Any, text: str, role: str, secondary: bool = False, wrap: int | None = None) -> None:
        """Set a widget's text. With a Typesetter it is shown as an image of the text (the text stays the widget's
        `text`, for anything that reads it), and a disabled widget shows it in the secondary colour."""
        widget.configure(text=text)
        typesetter = self.look.typesetter
        if typesetter is None:
            return
        p = self.look.palette
        images = []
        for state, colour in (("", p.secondary if secondary else p.text), ("disabled", p.secondary)):
            key = (text, role, colour)
            if key not in self.images:
                data = typesetter.render(text, role, colour, wrap)
                self.images[key] = self.tk.PhotoImage(master=self.root, data=data)
            images.append((state, self.images[key]))
        normal, disabled = images[0][1], images[1][1]
        widget.configure(image=(normal, "disabled", disabled), compound="image")

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
        if self.headline.cget("text") != now.headline:
            self.typeset(self.headline, now.headline, "heading")
        if self.detail.cget("text") != now.detail:
            wrap = round(WRAP_PT * self.look.scaling)
            self.typeset(self.detail, now.detail, "body", secondary=True, wrap=wrap)
        recording, idle = self.look.dot_colours
        self.dot.itemconfigure("light", fill=recording if now.recording else idle)
        busy = self.presence.phase in (STOPPING, STOPPED)
        self.quit.configure(state="disabled" if busy else "normal")
        self.dashboard.configure(state="disabled" if busy else "normal")
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
