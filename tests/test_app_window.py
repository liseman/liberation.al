"""The application's window (#156): what it says, that Quit drains and stops, and the real Tk window on a virtual
display."""

import importlib.util
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

from liberational import app_window, supervisor
from liberational.app_window import DOT_RECORDING, DOT_WARNING, RUNNING, STARTING, STOPPED, STOPPING
from liberational.service_cascade import BROKER, CAPTURE


def status(**states):
    """A supervisor status with these services; "job" marks a scheduled one."""
    return {
        "services": {
            name.replace("_", "-"): {"state": "scheduled" if state == "job" else state, "scheduled": state == "job"}
            for name, state in states.items()
        }
    }


DEVICES = {"camera": ([], 2), "microphone": ([], 1)}
UP = status(continuum_capture="running", continuum_broker="running", listen="running", video_window="job")


def test_it_says_running_and_recording_and_counts_the_daemons_not_the_jobs():
    now = app_window.summary(RUNNING, UP, {"camera": ([], 2), "microphone": ([], 1)})
    assert (now.headline, now.recording, now.dot) == ("Running and recording", True, DOT_RECORDING)
    assert now.detail == "Everything declared is recording."


def test_some_devices_paused_is_still_recording_and_says_which():
    now = app_window.summary(RUNNING, UP, {"camera": (["cam1"], 2), "microphone": ([], 1)})
    assert (now.headline, now.recording, now.dot) == ("Running and recording", True, DOT_RECORDING)
    assert "cam1 camera paused for privacy." in now.detail


def test_every_device_paused_is_not_recording():
    now = app_window.summary(RUNNING, UP, {"camera": (["cam0", "cam1"], 2), "microphone": (["desk"], 1)})
    assert (now.headline, now.recording) == ("Running, paused for privacy", False)
    assert "All cameras are paused for privacy." in now.detail
    assert "The microphone is paused for privacy." in now.detail


@pytest.mark.parametrize(
    ("services", "why"),
    [
        pytest.param(
            status(continuum_capture="activating", listen="running"), "Recording isn't running yet.", id="down"
        ),
        pytest.param(status(listen="running"), "Recording is off.", id="not-selected"),
    ],
)
def test_without_capture_it_never_claims_to_record(services, why):
    now = app_window.summary(RUNNING, services, {"camera": ([], 1)})
    assert (now.headline, now.recording) == ("Running, not recording", False)
    assert why in now.detail


def test_a_requested_restart_says_restarting_with_how_many_are_up_not_not_recording():
    services = status(continuum_capture="inactive", continuum_broker="running", listen="inactive", video_window="job")
    now = app_window.summary(RUNNING, {**services, "restarting": True}, DEVICES)
    assert (now.headline, now.recording) == ("Restarting (1 of 3 up)", False)
    assert "not recording" not in now.headline


def test_capture_back_up_during_a_restart_says_recording():
    services = status(continuum_capture="running", continuum_broker="running", listen="inactive")
    now = app_window.summary(RUNNING, {**services, "restarting": True}, DEVICES)
    assert (now.headline, now.recording) == ("Running and recording", True)


def test_a_service_waiting_to_restart_is_not_named_in_the_window(caplog):
    services = status(continuum_capture="running", listen="activating", journal="activating")
    with caplog.at_level("INFO", logger="liberational.app"):
        assert "Restarting:" not in app_window.summary(RUNNING, services).detail
    assert any("services still starting: journal, listen" in msg for msg in caplog.messages)


@pytest.mark.parametrize(
    ("phase", "headline"),
    [(STARTING, "Starting..."), (STOPPING, "Stopping..."), (STOPPED, "Stopped. Nothing is recording.")],
)
def test_outside_running_it_never_claims_to_record(phase, headline):
    now = app_window.summary(phase, UP, {"camera": ([], 1)})
    assert (now.headline, now.recording) == (headline, False)


def test_stopped_says_you_can_close_or_start_again():
    now = app_window.summary(STOPPED, UP, DEVICES)
    assert now.detail == "You can close this window or start the services again."


def test_unknown_pauses_say_nothing_about_them():
    now = app_window.summary(RUNNING, UP, {})
    assert (now.headline, now.detail) == ("Running and recording", "Everything declared is recording.")


def test_a_declared_microphone_that_never_registered_is_named():
    # #183: capture ran on without its microphone, and the window said "Running and recording".
    stuck = {**UP, "unregistered": [{"name": "desk", "kind": "microphone"}]}
    now = app_window.summary(RUNNING, stuck, DEVICES)
    assert (now.headline, now.recording, now.dot) == ("Recording, but not from every device", True, DOT_WARNING)
    assert now.detail == "The desk microphone isn't sending audio."
    assert app_window.summary(RUNNING, {**UP, "unregistered": []}, DEVICES).headline == "Running and recording"


def test_several_missing_devices_are_all_named():
    gone = [{"name": n, "kind": k} for n, k in (("cam0", "camera"), ("cam1", "camera"), ("desk", "microphone"))]
    now = app_window.summary(RUNNING, {**UP, "unregistered": gone[:2]}, DEVICES)
    assert "The cam0 camera isn't sending pictures." in now.detail
    assert "The cam1 camera isn't sending pictures." in now.detail
    assert (now.headline, now.recording, now.dot) == ("Recording, but not from every device", True, DOT_WARNING)
    every = app_window.summary(RUNNING, {**UP, "unregistered": gone}, DEVICES)
    assert (every.headline, every.recording) == ("Running, not recording", False)
    assert "The desk microphone isn't sending audio." in every.detail


def test_a_paused_device_is_paused_not_missing():
    stuck = {**UP, "unregistered": [{"name": "desk", "kind": "microphone"}]}
    now = app_window.summary(RUNNING, stuck, {"camera": ([], 2), "microphone": (["desk"], 1)})
    assert (now.headline, now.detail) == ("Running and recording", "The microphone is paused for privacy.")


@pytest.mark.parametrize(
    ("value", "font"),
    [
        pytest.param("'Ubuntu Sans 11'", ("Ubuntu Sans", 11.0), id="ubuntu"),
        pytest.param("'Cantarell 10.5'", ("Cantarell", 10.5), id="fractional"),
        pytest.param("'Noto Sans Bold Italic 12'\n", ("Noto Sans", 12.0), id="styles-dropped"),
        pytest.param("'Inter'", None, id="no-size"),
        pytest.param("", None, id="no-gsettings"),
        pytest.param("'Sans 0'", None, id="zero"),
    ],
)
def test_gnome_font_names_parse_to_a_family_and_size(value, font):
    assert app_window.parse_font_name(value) == font


@pytest.mark.parametrize(
    ("resources", "dpi"),
    [
        pytest.param("Xft.antialias:\t1\nXft.dpi:\t192\n", 192.0, id="scaled"),
        pytest.param("Xft.antialias:\t1\n", None, id="absent"),
        pytest.param("Xft.dpi:\tlots\n", None, id="garbled"),
        pytest.param("", None, id="no-xrdb"),
    ],
)
def test_the_desktop_dpi_comes_from_the_x_resources(resources, dpi):
    assert app_window.parse_xft_dpi(resources) == dpi


def test_a_gnome_desktop_is_read_from_its_settings():
    answers = {
        "font-name": "'Cantarell 11'\n",
        "color-scheme": "'prefer-dark'\n",
        "-query": "Xft.dpi:\t144\n",
    }
    desktop = app_window.read_desktop(lambda argv: answers[argv[-1]], platform="linux")
    assert desktop == app_window.Desktop(font=("Cantarell", 11.0), dark=True, dpi=144.0)


def test_a_desktop_with_no_settings_tools_gets_the_light_defaults():
    desktop = app_window.read_desktop(lambda argv: "", platform="linux")
    assert desktop == app_window.Desktop(font=None, dark=False, dpi=None)
    assert app_window.body_points(desktop) == 11.0


def test_a_mac_is_asked_only_for_its_appearance():
    asked = []

    def ask(argv):
        asked.append(argv[0])
        return "Dark\n"

    assert app_window.read_desktop(ask, platform="darwin") == app_window.Desktop(dark=True)
    assert asked == ["defaults"]


def test_a_monospace_match_falls_back_to_the_plain_sans():
    matches = {
        "Hack": "/fonts/Hack.ttf\t0\t100",
        "sans-serif": "/fonts/DejaVuSans.ttf\t0\t",
        "Ubuntu Sans:weight=semibold": "/fonts/UbuntuSans.ttf\t393216\t",
    }
    ask = lambda argv: matches[argv[-1]]
    assert app_window.font_file("Hack", ask) == ("/fonts/DejaVuSans.ttf", 0)
    assert app_window.font_file("Ubuntu Sans:weight=semibold", ask) == ("/fonts/UbuntuSans.ttf", 393216)


def test_no_fontconfig_means_no_font_file():
    assert app_window.font_file("Ubuntu Sans", lambda argv: "") is None


class Proc:
    def __init__(self, pid):
        self.pid, self.code = pid, None

    def poll(self):
        return self.code


class Harness:
    """A Presence, with the stand-in workers it started and the addresses it opened."""

    def __init__(self, presence, procs, opened):
        self.presence, self.procs, self.opened = presence, procs, opened

    def __getattr__(self, name):
        return getattr(self.presence, name)


@pytest.fixture
def presence(monkeypatch):
    """A Presence over a real supervisor with stand-in workers, and a list of what it opened."""
    monkeypatch.setattr(supervisor, "TICK_S", 0.01)
    procs, opened = {}, []

    def spawn(name, svc):
        procs[name] = Proc(100 + len(procs))
        return procs[name]

    def kill(pid, sig):
        next(p for p in procs.values() if p.pid == pid).code = -sig

    spec = {BROKER: {"command": "b"}, CAPTURE: {"command": "c"}, "listen": {"command": "l"}}
    app = supervisor.App(supervisor.Supervisor(spec, spawn=spawn, kill=kill))
    made = Harness(
        app_window.Presence(
            app,
            paused=lambda: {"camera": ([], 1)},
            open_url=opened.append,
            starts_at_login=lambda: False,
            set_start_at_login=lambda on: "unused",
        ),
        procs,
        opened,
    )
    yield made
    made.quit()
    if made.thread is not None:
        made.thread.join(30)


def wait_for(check, seconds=30.0):
    deadline = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_quit_drains_and_stops_every_worker_and_only_then_is_it_finished(presence):
    assert presence.phase == STARTING
    presence.start()
    wait_for(lambda: presence.phase == RUNNING)
    assert presence.summary().headline == "Running and recording"
    presence.quit()
    wait_for(lambda: presence.finished)
    assert all(proc.code is not None for proc in presence.procs.values())
    assert (presence.phase, presence.summary().recording) == (STOPPED, False)


def test_a_supervisor_that_ends_by_itself_shows_as_stopped_without_anyone_asking(presence):
    presence.start()
    wait_for(lambda: presence.phase == RUNNING)
    presence.app.quitting.set()  # the supervisor's own end, not the window's Quit
    wait_for(lambda: presence.finished)
    assert (presence.phase, presence.asked) == (STOPPED, False)


def test_the_dashboard_opens_on_its_port(presence):
    presence.open_dashboard()
    assert len(presence.opened) == 1
    assert presence.opened[0].startswith("http://localhost:") and presence.opened[0].endswith("/")


WINDOW = textwrap.dedent(
    """
    import sys, threading, time
    from liberational import app_window, supervisor

    class Proc:
        def __init__(self, pid): self.pid, self.code = pid, None
        def poll(self): return self.code

    procs = {}
    def spawn(name, svc):
        procs[name] = Proc(100 + len(procs)); return procs[name]
    def kill(pid, sig):
        next(p for p in procs.values() if p.pid == pid).code = -sig

    supervisor.TICK_S = 0.01
    app_window.REFRESH_MS = 20
    spec = {"continuum-broker": {"command": "b"}, "continuum-capture": {"command": "c"}}
    login = {"on": False, "asked": []}
    def set_login(on):
        login["asked"].append(on); login["on"] = on; return "set"
    presence = app_window.Presence(
        supervisor.App(supervisor.Supervisor(spec, spawn=spawn, kill=kill)), paused=lambda: {"camera": ([], 1)},
        starts_at_login=lambda: login["on"], set_start_at_login=set_login,
    )
    desktop = app_window.Desktop(font=("DejaVu Sans", 11), dark=sys.argv[2] == "dark", dpi=float(sys.argv[1]))
    window = app_window.Window(presence, desktop)
    look = window.look
    assert look.style.theme_use() == "clam" and abs(look.scaling / (float(sys.argv[1]) / 72) - 1) < 0.02
    if look.typesetter is not None:  # a Tk without Xft: the words are drawn by FreeType, never in a core X font
        drawn = window.root.tk.call("image", "height", window.quit.cget("image")[0])
    else:
        import tkinter.font
        assert look.heading_font.actual("family").lower() not in ("fixed", "courier")
        drawn = tkinter.font.nametofont("TkDefaultFont", root=window.root).metrics("linespace")
    assert window.at_login.get() is False  # off unless the owner ticks it
    window.login.invoke()
    assert login["asked"] == [True] and window.at_login.get() is True
    window.login.invoke()
    assert login["asked"] == [True, False] and window.at_login.get() is False
    login["on"] = True  # turned on somewhere else (Settings): the box follows at the next refresh
    window.show()
    assert window.at_login.get() is True
    presence.start()
    seen = []

    def step(n=0):
        seen.append(window.headline.cget("text"))
        if window.headline.cget("text") == "Running and recording" and "closed" not in seen:
            assert window.root.title() == "Liberational: Running and recording"
            assert window.dot.itemcget("light", "fill") == look.dot_colours[0]
            assert window.root.tk.call("image", "height", window.headline.cget("image")[0]) > drawn
            seen.append("closed")
            window.quit_recording()  # the Quit button (closing the window asks first)
            assert str(window.quit.cget("state")) == "disabled"
        window.root.after(10, step)

    window.root.after(10, step)
    window.run()  # returns once the window has closed itself, after the workers stopped
    presence.thread.join(5)
    assert "Running and recording" in seen and "closed" in seen
    assert all(p.code is not None for p in procs.values()), "a worker outlived the window"
    print(drawn, "ok")
    """
)


@pytest.mark.skipif(
    shutil.which("xvfb-run") is None or importlib.util.find_spec("tkinter") is None,
    reason="needs xvfb-run and tkinter for a display nobody is looking at",
)
@pytest.mark.parametrize(("dpi", "scheme"), [pytest.param(96, "light", id="1x"), pytest.param(192, "dark", id="2x")])
def test_the_real_window_shows_recording_and_closing_it_stops_everything(dpi, scheme):
    """Tk itself, on a virtual display: themed widgets, text sized for the display's scale, it says it is recording,
    and closing it quits: the workers are stopped before the window goes."""
    done = subprocess.run(
        ["xvfb-run", "-a", sys.executable, "-c", WINDOW, str(dpi), scheme],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr[-2000:]
    # One line of 11 pt body text is about 1.2 to 1.5 times its size in pixels (11 pt at dpi / 72 pixels per point).
    line = int(done.stdout.split()[-2])
    assert 1.0 <= line / (11 * dpi / 72) <= 1.8


def test_main_refuses_without_a_display_before_starting_any_service(monkeypatch):
    started = []

    class NoSpawn(supervisor.Supervisor):
        def start_all(self, missed=()):
            started.append("start")

    app = supervisor.App(NoSpawn({}))

    class Ready:
        def __enter__(self):
            return app, []

        def __exit__(self, *exc):
            return False

    def no_display(presence):
        raise RuntimeError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr(supervisor, "application", Ready)
    monkeypatch.setattr(app_window, "Window", no_display)
    monkeypatch.setattr(app_window.logs, "setup", lambda: None)
    monkeypatch.setattr(app_window.logs, "adopt", lambda name: None)
    monkeypatch.setattr(app_window.signal, "signal", lambda *a: None)
    monkeypatch.setattr(app_window.login_item, "sync", lambda: pytest.fail("touched the login entry"))
    with pytest.raises(SystemExit, match="can't open a window"):
        app_window.main()
    assert started == []
