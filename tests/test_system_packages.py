"""System packages (config/system-packages.toml): what's missing, the command that installs it, and asking first.
Nothing here installs anything: which(), the library check, the prompt, and the runner are all stand-ins."""

import tomllib

import pytest

from liberational import config as cfg
from liberational import services, system_packages
from liberational.system_packages import Package

SOX = Package("sox", "voice cloning", command="sox", apt="sox", brew="sox", services=("voice",))
PORTAUDIO = Package("portaudio", "the microphone", library="portaudio", apt="libportaudio2", services=("listen",))
FFMPEG = Package("ffmpeg", "recording", command="ffmpeg", mise=True, services=("vision",))
BLUEUTIL = Package(
    "blueutil", "speakers", command="blueutil", brew="blueutil", services=("audio-route",), optional=True
)
LIBUSB = Package("libusb", "mic controls", library="usb-1.0", apt="libusb-1.0-0", services=("listen",), optional=True)
ALL = [SOX, PORTAUDIO, FFMPEG, BLUEUTIL]


def nothing(*_args):
    return None


def none_load(_name, _platform):
    return False


# ---- the declared list --------------------------------------------------------------------------------------------


def test_the_declared_packages_load_and_name_real_services():
    packages = system_packages.load()
    by_name = {p.name: p for p in packages}
    declared = set(tomllib.loads((cfg.RESOURCES / "services.toml").read_text()))
    assert set(by_name) >= {"sox", "espeak-ng", "portaudio", "ffmpeg", "ffprobe", "pkg-config", "python3"}
    for required in ("pkg-config", "python3", "libavcodec-dev", "libpipewire"):
        assert not by_name[required].optional, required
    for p in packages:
        assert set(p.services) <= declared, p.name


def test_default_manifest_follows_the_current_resources_and_explicit_paths_stay_explicit(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_manifest = first / "system-packages.toml"
    first_manifest.write_text('[alpha]\nwhy = "test input"\ncommand = "alpha"\napt = "alpha"\n')
    (second / "system-packages.toml").write_text('[beta]\nwhy = "other test input"\ncommand = "beta"\napt = "beta"\n')
    monkeypatch.setattr(cfg, "RESOURCES", first)
    assert [p.name for p in system_packages.load()] == ["alpha"]
    monkeypatch.setattr(cfg, "RESOURCES", second)
    assert [p.name for p in system_packages.load()] == ["beta"]
    assert [p.name for p in system_packages.load(first_manifest)] == ["alpha"]


@pytest.mark.parametrize(
    ("table", "message"),
    [
        pytest.param(
            'why = "x"\ncommand = "a"\nlibrary = "b"\napt = "a"', "one of a command", id="command-and-library"
        ),
        pytest.param('why = "x"\ncommand = "a"', "where it comes from", id="no-source"),
        pytest.param('command = "a"\napt = "a"', "why", id="no-why"),
        pytest.param('why = "x"\ncommand = "a"\napt = "a"\nyum = "a"', "unknown fields", id="unknown-field"),
        pytest.param('why = "x"\ncommand = "a"\napt = "a"\nname = "b"', "unknown fields", id="name-inside"),
    ],
)
def test_a_badly_declared_package_is_refused_by_name(tmp_path, table, message):
    path = tmp_path / "packages.toml"
    path.write_text(f"[thing]\n{table}\n")
    with pytest.raises(ValueError, match=rf"thing: .*{message}"):
        system_packages.load(path)


# ---- what's missing -------------------------------------------------------------------------------------------------


def test_linux_needs_what_apt_or_mise_provides_and_finds_it_by_command_or_library():
    have = {"ffmpeg"}
    gone = system_packages.missing(ALL, platform="linux", which=lambda c: c if c in have else None, library=none_load)
    assert [p.name for p in gone] == ["sox", "portaudio"]  # blueutil is macOS-only; ffmpeg is there


def test_macos_needs_what_brew_or_mise_provides():
    gone = system_packages.missing(ALL, platform="darwin", which=nothing, library=none_load)
    assert [p.name for p in gone] == ["sox", "ffmpeg", "blueutil"]  # PortAudio comes with the Python package there


def test_only_the_packages_some_given_service_uses():
    gone = system_packages.missing(ALL, services=["voice"], platform="linux", which=nothing, library=none_load)
    assert [p.name for p in gone] == ["sox"]


def test_a_library_counts_as_found_when_it_loads():
    def loads(name, _platform):
        return name == "portaudio"

    gone = system_packages.missing([PORTAUDIO], platform="linux", which=nothing, library=loads)
    assert gone == []


# ---- the commands ---------------------------------------------------------------------------------------------------


def test_apt_installs_through_sudo_once_for_all_and_mise_installs_its_own():
    assert system_packages.commands([SOX, PORTAUDIO, FFMPEG], platform="linux", installer="apt") == [
        "sudo apt-get update",  # a fresh image has no package lists yet
        "sudo apt-get install -y sox libportaudio2",
        "mise install",
    ]


def test_brew_installs_on_macos():
    assert system_packages.commands([SOX, BLUEUTIL], platform="darwin", installer="brew") == [
        "brew install sox blueutil"
    ]


def test_no_command_without_a_package_manager_it_knows():
    assert system_packages.commands([SOX], platform="linux", installer=None) == []
    assert "your system's package manager" in system_packages.describe([SOX], platform="linux", installer=None)


# ---- installing -----------------------------------------------------------------------------------------------------


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(system_packages, "manager", lambda *_a, **_k: "apt")


def install(packages, code=0, **kw):
    ran = []

    def run(argv):
        ran.append(argv)
        return code

    code = system_packages.install(packages, platform="linux", run=run, **kw)
    return code, ran


def test_asks_before_installing_and_runs_the_exact_command_on_yes(linux, capsys):
    code, ran = install([SOX, FFMPEG], interactive=lambda: True, ask=lambda _q: "y")
    assert (code, ran) == (
        0,
        [["sudo", "apt-get", "update"], ["sudo", "apt-get", "install", "-y", "sox"], ["mise", "install"]],
    )
    assert "sox: voice cloning" in capsys.readouterr().out


def test_installs_nothing_when_the_answer_is_no(linux):
    assert install([SOX], interactive=lambda: True, ask=lambda _q: "") == (0, [])


def test_only_prints_the_command_without_a_terminal_to_ask_on(linux, capsys):
    assert install([SOX], interactive=lambda: False) == (0, [])
    assert "sudo apt-get install -y sox" in capsys.readouterr().out


def test_yes_skips_the_question(linux):
    def never(_q):
        raise AssertionError("asked")

    assert install([SOX], yes=True, interactive=lambda: False, ask=never)[1][-1] == [
        "sudo",
        "apt-get",
        "install",
        "-y",
        "sox",
    ]


def test_a_failed_install_returns_its_exit_code_and_stops(linux):
    assert install([SOX, FFMPEG], yes=True, code=100) == (100, [["sudo", "apt-get", "update"]])


def test_with_no_installer_it_knows_it_says_what_to_install_and_lets_setup_go_on(monkeypatch, capsys):
    monkeypatch.setattr(system_packages, "manager", lambda *_a, **_k: None)
    assert install([SOX], yes=True) == (0, [])
    assert "your system's package manager" in capsys.readouterr().out


def test_an_installer_that_isnt_there_is_reported_not_raised(capsys):
    assert system_packages._run(["liberational-no-such-installer", "install"]) == 127
    assert "can't run liberational-no-such-installer" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("platform", "have", "expected"),
    [
        pytest.param("darwin", {"brew"}, "brew", id="mac-with-homebrew"),
        pytest.param("darwin", set(), None, id="mac-without-homebrew"),
        pytest.param("linux", {"apt-get"}, "apt", id="debian"),
        pytest.param("linux", {"dnf"}, None, id="fedora"),
    ],
)
def test_the_package_manager_is_one_this_knows_and_the_system_has(platform, have, expected):
    assert system_packages.manager(platform, which=lambda c: c if c in have else None) == expected


def test_nothing_missing_runs_nothing():
    assert install([], yes=True) == (0, [])


def test_optional_packages_are_only_installed_when_asked(monkeypatch):
    monkeypatch.setattr(system_packages, "load", lambda: [SOX, LIBUSB])
    monkeypatch.setattr(
        system_packages,
        "missing",
        lambda packages, optional=True, **_k: [p for p in packages if optional or not p.optional],
    )
    got = []
    monkeypatch.setattr(system_packages, "install", lambda packages, yes: got.append([p.name for p in packages]) or 0)
    for argv in (["install"], ["install", "--optional"]):
        with pytest.raises(SystemExit):
            system_packages.main(argv)
    assert got == [["sox"], ["sox", "libusb"]]


def test_a_package_declared_for_nvidia_applies_only_where_there_is_one(monkeypatch):
    clone = Package("sox", "voice cloning", command="sox", apt="sox", when="nvidia")
    monkeypatch.setitem(system_packages.WHEN, "nvidia", lambda: False)
    assert system_packages.missing([clone], platform="linux", which=nothing) == []
    monkeypatch.setitem(system_packages.WHEN, "nvidia", lambda: True)
    assert system_packages.missing([clone], platform="linux", which=nothing) == [clone]


def test_a_library_in_homebrews_folder_counts_on_macos(monkeypatch, tmp_path):
    (tmp_path / "libusb-1.0.dylib").write_bytes(b"")
    monkeypatch.setattr(system_packages, "BREW_LIBS", (str(tmp_path),))
    monkeypatch.setattr(system_packages.ctypes.util, "find_library", nothing)
    assert system_packages._library("usb-1.0", "darwin")
    assert not system_packages._library("usb-1.0", "linux")  # Linux's loader would have found it already


def test_the_dashboard_lists_only_needed_packages_the_given_services_use(monkeypatch):
    monkeypatch.setattr(system_packages, "manager", lambda *_a, **_k: "apt")
    monkeypatch.setattr(system_packages.shutil, "which", nothing)  # nothing installed
    monkeypatch.setattr(system_packages, "_library", none_load)
    report = system_packages.status([SOX, PORTAUDIO, LIBUSB], services=["listen"], platform="linux")
    assert [m["name"] for m in report["missing"]] == ["portaudio"]  # libusb is optional; sox is voice's
    assert report["commands"] == ["sudo apt-get update", "sudo apt-get install -y libportaudio2"]
    assert report["missing"][0]["command"] == "sudo apt-get update && sudo apt-get install -y libportaudio2"


def test_each_missing_package_says_the_command_that_installs_it_alone(monkeypatch):
    monkeypatch.setattr(system_packages, "manager", lambda *_a, **_k: "brew")
    monkeypatch.setattr(system_packages.shutil, "which", nothing)
    monkeypatch.setattr(system_packages, "_library", none_load)
    report = system_packages.status([SOX, FFMPEG], platform="darwin")
    assert {m["name"]: m["command"] for m in report["missing"]} == {"sox": "brew install sox", "ffmpeg": "mise install"}
    assert report["commands"] == ["brew install sox", "mise install"]


# ---- the command line -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gone", "code"),
    [
        pytest.param([SOX], 1, id="a-needed-one"),
        pytest.param([LIBUSB], 0, id="only-optional"),
        pytest.param([], 0, id="none"),
    ],
)
def test_check_exits_1_only_when_a_needed_package_is_missing(monkeypatch, capsys, gone, code):
    monkeypatch.setattr(system_packages, "load", lambda: ALL)
    monkeypatch.setattr(system_packages, "missing", lambda *_a, **_k: gone)
    monkeypatch.setattr(system_packages, "manager", lambda *_a, **_k: "apt")
    with pytest.raises(SystemExit) as stop:
        system_packages.main(["check"])
    assert stop.value.code == code
    out = capsys.readouterr().out
    assert ("Install with:" in out) == bool(gone)


# ---- the services preflight and the dashboard -----------------------------------------------------------------------


def test_installing_services_names_their_missing_packages_and_the_fix(monkeypatch, capsys):
    monkeypatch.setattr(system_packages, "load", lambda: ALL)
    monkeypatch.setattr(system_packages, "manager", lambda *_a, **_k: "apt")
    monkeypatch.setattr(system_packages.shutil, "which", nothing)
    monkeypatch.setattr(system_packages, "_library", none_load)
    monkeypatch.setattr(system_packages, "_platform", lambda: "linux")
    services.preflight(["voice"])
    out = capsys.readouterr().out
    assert "sox: voice cloning" in out and "portaudio" not in out  # only what voice uses
    assert "sudo apt-get install -y sox" in out and "pack-system install" in out


def test_the_dashboard_reports_missing_packages_and_checks_again_only_after_a_while(monkeypatch):
    from liberational.dashboard import machine_stats

    calls = []

    def status(**_kwargs):
        calls.append(1)
        return {"missing": [{"name": "sox"}], "commands": ["sudo apt-get install -y sox"]}

    monkeypatch.setattr(system_packages, "status", status)
    stats = machine_stats.Stats()
    assert stats.system_status()["missing"] == [{"name": "sox"}]
    stats.system_status()
    assert len(calls) == 1
    stats.system_t -= machine_stats.SYSTEM_S + 1
    stats.system_status()
    assert len(calls) == 2


AVCODEC_DEV = Package(
    "libavcodec-dev", "FFmpeg headers", pkgconfig="libavcodec", apt="libavcodec-dev", brew="ffmpeg", optional=True
)


def test_a_header_package_is_found_by_pkg_config_not_by_the_runtime_library():
    def loads(_name, _platform):
        return True  # the runtime library is there, as on a machine without the -dev package

    headers = system_packages.missing([AVCODEC_DEV], platform="linux", library=loads, pkgconfig=lambda _n: False)
    assert headers == [AVCODEC_DEV]
    assert system_packages.missing([AVCODEC_DEV], platform="linux", library=loads, pkgconfig=lambda _n: True) == []


def test_a_package_names_exactly_one_way_to_check_it(tmp_path):
    both = tmp_path / "p.toml"
    both.write_text('[x]\nwhy = "y"\nlibrary = "a"\npkgconfig = "liba"\napt = "a"\n')
    with pytest.raises(ValueError, match="one of a command, a library, or a pkg-config name"):
        system_packages.load(both)
    one = tmp_path / "q.toml"
    one.write_text('[x]\nwhy = "y"\npkgconfig = "liba"\napt = "liba-dev"\n')
    assert system_packages.load(one)[0].pkgconfig == "liba"
