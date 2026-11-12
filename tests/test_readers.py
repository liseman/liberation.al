"""Which services read which settings (readers.py), on a small made-up package and on the real code."""

import ast
import pathlib
import textwrap
import threading

import pytest

from liberational import config as cfg
from liberational import readers, services

CACHED = (
    readers.keys,
    readers._sections,
    readers.helpers,
    readers._scan,
    readers.entry_points,
    readers._readers,
    readers.unit_keys,
    readers._indexed,
    readers._known_starts,
)


def _forget():
    for f in CACHED:
        f.cache_clear()


@pytest.fixture
def pack(tmp_path, monkeypatch):
    """A package of three Python services and two others, with its own defaults.toml and pyproject.toml."""
    src = tmp_path / "src"

    def module(name, code):
        path = src / pathlib.Path(*name.split("."))
        path = path / "__init__.py" if name.count(".") == 0 or name.endswith("pkg") else path.with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(code))

    module("liberational", "")
    module(
        "liberational.config",
        'def owner_name():\n    return get("owner.name")\n\ndef greeting():\n    return owner_name()\n\n'
        'def get(key):\n    "tts.voice"\n',
    )
    module("liberational.services", 'def log_path(name):\n    return cfg.path("logs.dir")\n')
    module("liberational.pkg", 'from liberational import config as cfg\nSPEED = cfg.get("vision.height")\n')
    module("liberational.pkg.greeter", "from liberational import config as cfg\nHI = cfg.greeting()\n")
    module("liberational.pkg.cams", 'from liberational import config as cfg\nRES = cfg.get("vision.width")\n')
    module("liberational.pkg.eyes", "from . import cams\n")  # relative import
    module("liberational.pkg.ears", 'import subprocess\nsubprocess.run(["-m", "liberational.pkg.worker"])\n')
    module("liberational.pkg.worker", 'from liberational import config as cfg\nKEY = f"audio.{1}"\n')
    module("liberational.pkg.namer", "from liberational import config as cfg\nWHO = cfg.owner_name()\n")
    module("liberational.pkg.models", 'from liberational import config as cfg\nP = cfg.load()["models"]["x"]\n')
    (tmp_path / "defaults.toml").write_text(
        '[owner]\nname = ""\n[tts]\nvoice = "a"\n[vision]\nwidth = 1\nheight = 1\n[audio]\nsensitivity = 70\ngain = 1\n'
        "[audio.mic_sensitivity]\n[ports]\nwhisper = 1\n[models.files.w]\npath = 'w.bin'\n[logs]\ndir = 'l'\n"
    )
    (tmp_path / "labels.toml").write_text('["audio.sensitivity"]\nlive = true\n')
    (tmp_path / "pyproject.toml").write_text(
        "[project.scripts]\n"
        + "\n".join(f'p-{n} = "liberational.pkg.{n}:main"' for n in ("eyes", "ears", "namer", "models", "greeter"))
    )
    declared = {
        "eyes": {"command": "p-eyes"},
        "ears": {"command": "p-ears"},
        "namer": {"command": "p-namer"},
        "models": {"command": "p-models"},
        "greeter": {"command": "p-greeter"},
        "server": {"command": "bin/server", "args": ["--port", "{ports.whisper}"], "reads_settings": False},
        "nightly": {"command": "p-eyes", "restart": False, "schedule": "03:00"},
        "script": {"command": "tools/sync.sh"},
    }
    monkeypatch.setattr(readers, "SRC", src)
    monkeypatch.setattr(readers, "PYPROJECT", tmp_path / "pyproject.toml")
    monkeypatch.setattr(cfg, "DEFAULTS", tmp_path / "defaults.toml")
    monkeypatch.setattr(cfg, "LABELS", tmp_path / "labels.toml")
    monkeypatch.setattr(services, "declared", lambda: declared)
    _forget()
    yield
    _forget()


@pytest.mark.parametrize(
    "key,services_",
    [
        pytest.param("vision.width", ["eyes", "script"], id="through-a-relative-import"),
        pytest.param("audio.gain", ["ears", "script"], id="in-a-module-it-starts-and-an-f-string"),
        pytest.param("owner.name", ["namer", "greeter", "script"], id="through-a-config-helper-and-one-it-calls"),
        pytest.param("vision.height", ["eyes", "ears", "namer", "models", "greeter", "script"], id="a-package-init"),
        pytest.param(
            "logs.dir",
            ["eyes", "ears", "namer", "models", "greeter", "server", "nightly", "script"],
            id="in-every-unit",
        ),
        pytest.param("models.files.w.path", ["models", "script"], id="a-section-by-subscript"),
        pytest.param("tts.voice", ["script"], id="named-only-in-config-py"),
        pytest.param("ports.whisper", ["server", "script"], id="in-a-service-args"),
        pytest.param("audio.sensitivity", [], id="live"),
        pytest.param("audio.mic_sensitivity", ["ears", "script"], id="an-empty-table"),
        pytest.param(
            "phone.number", ["eyes", "ears", "namer", "models", "greeter", "server", "nightly", "script"], id="unknown"
        ),
    ],
)
def test_a_change_restarts_the_services_that_read_it(pack, key, services_):
    assert readers.affected([key]) == services_


def test_a_scheduled_job_and_a_command_that_never_reads_settings_count_only_their_args(pack):
    assert (readers.readers()["nightly"], readers.readers()["server"]) == ({"logs.dir"}, {"ports.whisper", "logs.dir"})
    assert readers.readers()["script"] is None  # a script that may read anything


def test_a_request_during_the_startup_scan_waits_for_it_instead_of_scanning_again(pack, monkeypatch):
    fake = services.declared  # the pack fixture's
    calls: list[str] = []
    started, second, done = threading.Event(), threading.Event(), threading.Event()

    def slow_declared():  # the scan's first step, held open until a second caller has had its chance to start one
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            started.set()
            second.wait(timeout=1)
        else:
            second.set()
        return fake()

    monkeypatch.setattr(services, "declared", slow_declared)
    warmup = readers.warm()
    assert started.wait(timeout=5)
    request = threading.Thread(target=lambda: (readers.affected_each(["vision.width"]), done.set()))
    request.start()
    request.join(timeout=10)
    warmup.join(timeout=10)

    assert done.is_set()
    assert calls == ["settings-readers"]


def test_only_live_settings_restart_nothing(pack):
    assert readers.affected(["audio.sensitivity"]) == []
    assert readers.affected(["audio.sensitivity", "vision.width"]) == ["eyes", "script"]


# ---- the real code ----


def test_every_python_service_is_scanned():
    unread = [
        n
        for n, svc in services.declared().items()
        if svc["command"].startswith("pack-") and readers.readers()[n] is None
    ]
    assert unread == []


def test_a_voice_setting_leaves_the_cameras_running():
    assert "vision" not in readers.affected(["tts.voice"])
    assert readers.affected(["audio.sensitivity"]) == []


@pytest.mark.parametrize("key", ["asr.engine", "services.enable", "services.disable"])
def test_choosing_which_services_run_leaves_the_cameras_and_the_journal_running(key):
    # What services.py reads to choose services: the journal's sleep window runs pack-services rather than import it,
    # so every service whose code reaches the journal doesn't count as reading these too.
    assert {"see-actions", "journal"}.isdisjoint(readers.affected([key]))


# Every cfg.get, cfg.path, or cfg.model_path whose key isn't written out where it is called. The scan sees a key only
# where it is written, so each of these was checked by hand: the keys it can read are written in the same module.
COMPUTED_KEYS = {
    "audio/twins.py": 'f"audio.{key}" over "dedup_window_s" and "dedup_similarity", both in the same module',
    "memory/presence_counts.py": '"presence.gap_s" and "presence.absence_s", a loop over literals',
    "ollama_health.py": 'f"models.ollama.{role}"',
    "services.py": "{section.key} templates in services.toml, which readers.py reads itself",
    "spatial/camera_clouds.py": '"spatial.pointmap_runtime" and "spatial.pointmap_weights", both named in this module',
    "unifi.py": 'setting(section, key) over unifi.CONSOLES, which writes "unifi." and "unifi_protect." whole',
    "dashboard/network_cameras.py": 'unifi.setting(section, "mode"), its sections those of unifi.CONSOLES',
    "vision/crossings.py": '"crossings." + key over the settings() literals',
    # models.hf.<key>.path and .repo, where key is one of the module's three model constants (TODAY, GRID, CONCEPT), so
    # every key it reads is models.hf.object_segmentation{,concept,_points} with one of those two suffixes.
    "vision/segmentation.py": 'f"models.hf.{key}.path" and ".repo" over CONCEPT, GRID, and TODAY',
}


@pytest.mark.parametrize("key", ["unifi.password", "unifi.api_key", "unifi_protect.password", "unifi_protect.mode"])
def test_saving_a_console_setting_restarts_what_reads_it(key):
    # configured_session reads its section through computed names: the scan must still see every key in it.
    assert {"dashboard", "network-devices"} <= set(readers.affected([key]))


def test_the_dashboard_reads_what_every_module_with_routes_reads():
    # The server loads its route modules by name (routes.load): the scan must follow those names, or a setting only
    # a route module reads restarts no dashboard.
    root = pathlib.Path(readers.SRC) / "liberational" / "dashboard"
    with_routes = {
        f"liberational.dashboard.{path.stem}" for path in root.glob("*.py") if "@routes." in path.read_text()
    }
    assert "liberational.dashboard.network_cameras" in with_routes
    assert with_routes <= readers._closure("liberational.dashboard.server")


def test_every_setting_read_by_a_computed_key_was_checked_by_hand():
    root = pathlib.Path(readers.SRC) / "liberational"
    found = set()
    for path in root.rglob("*.py"):
        if path == root / "config.py":
            continue
        for n in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("get", "path", "model_path")
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "cfg"
                and n.args
                and not isinstance(n.args[0], ast.Constant)
            ):
                found.add(str(path.relative_to(root)))
    assert found == set(COMPUTED_KEYS), "check what keys the new call can read, then list it in COMPUTED_KEYS"


def test_every_settings_restarts_match_the_plain_scan():
    """affected_each looks reads up by key and its starts; it must name the same services as checking every
    read with _under, for every real setting and a few that aren't."""

    def plain(key):
        flagged = readers.live()
        if readers.is_live(key, flagged):
            return []
        known = readers.keys()  # the settings in defaults.toml (a function, not a dict's keys)
        unknown = not any(readers._under(key, s) or readers._under(s, key) for s in known)
        return [n for n, read in readers.readers().items() if unknown or readers.reads(read, key)]

    every = [*sorted(readers.keys()), "audio.mic_sensitivity.USB mic", "nowhere.at_all"]
    assert readers.affected_each(every) == {key: plain(key) for key in every}
    assert readers.affected(["tts.voice", "backfill.sample_fps"]) == [
        n for n in readers.readers() if n in {*plain("tts.voice"), *plain("backfill.sample_fps")}
    ]
