import json
import pathlib
import re
import tomllib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from liberational import config as cfg

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_defaults_are_generic():
    assert cfg.owner_name() == ""
    assert cfg.owner_label() == ""
    assert cfg.owner_or() == "the owner"
    assert cfg.wake_words() == ["hey bob"]


@pytest.mark.parametrize(
    "var, raw, key, want",
    [
        pytest.param("PACK_PORTS_DASHBOARD", "9000", "ports.dashboard", 9000, id="int"),
        pytest.param("PACK_JOURNAL_LEVELS", "60,120", "journal.levels", [60, 120], id="int-list"),
        pytest.param("PACK_VISION_OBJECTS", "false", "vision.objects", False, id="bool"),
        pytest.param("PACK_AUDIO_EXCLUDE_DEVICES", "foo, bar", "audio.exclude_devices", ["foo", "bar"], id="str-list"),
        pytest.param("PACK_STATE", "/tmp/pack-state", "paths.state", "/tmp/pack-state", id="alias-state"),
        pytest.param("PACK_ASR", "whisper", "asr.engine", "whisper", id="alias-asr"),
        pytest.param("PACK_SERVICES_ENABLE", "1,-2", "services.enable", [1, -2], id="untyped-list-of-numbers"),
        pytest.param("PACK_SERVICES_ENABLE", "--1, ²", "services.enable", ["--1", "²"], id="untyped-list-isdigit-trap"),
    ],
)
def test_env_beats_file_and_coerces_types(monkeypatch, var, raw, key, want):
    monkeypatch.setenv(var, raw)
    assert cfg.get(key) == want
    assert type(cfg.get(key)) is type(want)


@pytest.mark.parametrize(
    "raw, want",
    [
        ("0.0.0.0", "http://127.0.0.1:11434"),
        ("gpu-box:11500", "http://gpu-box:11500"),
        ("https://ollama.example.com/", "https://ollama.example.com"),
        ("http://0.0.0.0:11434", "http://127.0.0.1:11434"),
    ],
)
def test_ollama_host_forms(monkeypatch, raw, want):
    monkeypatch.setenv("OLLAMA_HOST", raw)
    assert cfg.ollama_url() == want


def test_override_and_identity_helpers():
    with cfg.override({"owner": {"name": "Alice"}, "assistant": {"name": "Max"}}):
        assert cfg.owner_label() == "alice"
        assert cfg.owner_or() == "Alice"
        assert cfg.wake_words() == ["hey max"]
    assert cfg.owner_name() == ""


@pytest.mark.parametrize(
    "setting, want",
    [
        pytest.param("some/dir/openclaw.json", REPO / "some" / "dir" / "openclaw.json", id="relative"),
        pytest.param("~/.openclaw/openclaw.json", pathlib.Path.home() / ".openclaw" / "openclaw.json", id="home"),
    ],
)
def test_relative_paths_resolve_under_repo(setting, want, monkeypatch):
    monkeypatch.setattr(cfg, "OWNER_ROOT", REPO)  # as in a checkout with no PACK_ROOT
    with cfg.override({"agent": {"openclaw_config": setting}}):
        assert cfg.path("agent.openclaw_config") == want


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """No checkout and no PACK_ROOT, on Linux, with a made-up home directory."""
    monkeypatch.setattr(cfg, "OWNER_ROOT", None)
    monkeypatch.setattr(cfg.sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("PACK_CONFIG", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.mark.parametrize(
    "named, want",
    [
        pytest.param("data/state", ".local/share/liberational/state", id="data"),
        pytest.param("data/models/x/y.onnx", ".local/share/liberational/models/x/y.onnx", id="data-nested"),
        pytest.param("data", ".local/share/liberational", id="data-itself"),
        pytest.param(".env.phone", ".config/liberational/.env.phone", id="beside-pack-toml"),
        pytest.param("database/x", ".config/liberational/database/x", id="data-prefix-is-a-whole-part"),
    ],
)
def test_installed_owner_files_split_between_the_config_and_data_directories(installed, named, want):
    assert cfg.owner_file(named) == installed / want


def test_installed_config_files_are_in_the_user_config_directory(installed):
    assert cfg.local_config() == installed / ".config/liberational/pack.toml"
    assert cfg.devices_file() == installed / ".config/liberational/devices.toml"
    assert cfg.path("paths.logs") == installed / ".local/share/liberational/logs"


def test_installed_directories_follow_absolute_xdg_variables_only(installed, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(installed / "conf"))
    monkeypatch.setenv("XDG_DATA_HOME", "relative/ignored")
    assert cfg.config_dir() == installed / "conf/liberational"
    assert cfg.data_dir() == installed / ".local/share/liberational"


def test_installed_on_macos_everything_is_in_application_support(installed, monkeypatch):
    monkeypatch.setattr(cfg.sys, "platform", "darwin")
    support = installed / "Library/Application Support/liberational"
    assert (cfg.local_config(), cfg.owner_file("data/state")) == (support / "pack.toml", support / "state")


def test_installed_first_save_makes_the_config_directory(installed):
    cfg.write_local('[owner]\nname = "Alice"\n')
    saved = installed / ".config/liberational/pack.toml"
    assert tomllib.loads(saved.read_text()) == {"owner": {"name": "Alice"}}


def test_installed_helpers_run_in_the_data_directory_which_is_made(installed):
    assert cfg.workdir() == installed / ".local/share/liberational"
    assert cfg.workdir().is_dir()


def test_an_owner_root_lays_files_out_as_a_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "OWNER_ROOT", tmp_path)
    monkeypatch.delenv("PACK_CONFIG")
    assert cfg.local_config() == tmp_path / "pack.toml"
    assert cfg.devices_file() == tmp_path / "config/devices.toml"
    assert cfg.owner_file("data/state") == tmp_path / "data/state"
    assert cfg.owner_file("/abs/x") == pathlib.Path("/abs/x")
    assert cfg.workdir() == tmp_path


@pytest.mark.parametrize(
    "package, pyproject, want",
    [
        pytest.param("repo/src/liberational", "repo/pyproject.toml", "repo", id="checkout"),
        pytest.param("venv/lib/site-packages/liberational", "venv/pyproject.toml", None, id="site-packages"),
        pytest.param("repo/src/liberational", "elsewhere/pyproject.toml", None, id="src-without-pyproject"),
    ],
)
def test_a_checkout_is_a_package_in_src_beside_a_pyproject(tmp_path, monkeypatch, package, pyproject, want):
    (tmp_path / package).mkdir(parents=True)
    (tmp_path / pyproject).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / pyproject).write_text("")
    monkeypatch.setattr(cfg, "PACKAGE", tmp_path / package)
    assert cfg._checkout() == (tmp_path / want if want else None)


def test_no_module_names_an_owner_file_by_a_path_relative_to_the_working_directory():
    # Path("config/devices.toml") is the checkout's file only while the process runs in the checkout: an installed
    # package runs elsewhere and keeps the file elsewhere. config.devices_file() and config.owner_file() know both.
    relative = re.compile(r"""Path\(\s*["'](?:config|data)/""")
    found = [
        f"{source.relative_to(REPO)}:{number}"
        for source in sorted((REPO / "src").rglob("*.py"))
        for number, line in enumerate(source.read_text().splitlines(), 1)
        if relative.search(line)
    ]
    assert found == []


def test_the_resources_the_code_reads_are_package_data():
    names = {"defaults.toml", "labels.toml", "system-packages.toml", "services.toml"}
    assert {f.name for f in cfg.RESOURCES.iterdir() if f.is_file()} == names
    assert cfg.RESOURCES.parent == pathlib.Path(cfg.__file__).resolve().parent


def test_runtime_data_defaults_live_under_data():
    # The repo root holds only code and config. Runtime state stays in data/;
    # see-decode's frame ring is empty by default, meaning /dev/shm (or the temp
    # directory): memory, never disk.
    d = tomllib.loads(cfg.DEFAULTS.read_text())
    paths = {f"paths.{k}": v for k, v in d["paths"].items()}
    paths["owner.voice_ref"] = d["owner"]["voice_ref"]
    paths.update({f"models.files.{k}": e["path"] for k, e in d["models"]["files"].items()})
    assert {k: v for k, v in paths.items() if not v.startswith("data/")} == {
        "paths.decoded": "",
    }


def test_agent_token_falls_back_to_openclaw_config(tmp_path, monkeypatch):
    (tmp_path / "openclaw.json").write_text(json.dumps({"gateway": {"auth": {"password": "pw"}}}))
    with cfg.override({"agent": {"openclaw_config": str(tmp_path / "openclaw.json")}}):
        assert cfg.agent_token() == "pw"
        monkeypatch.setenv("PACK_AGENT_TOKEN", "tok")
        assert cfg.agent_token() == "tok"


def test_agent_is_opt_in():
    assert not cfg.agent_configured()
    assert cfg.openclaw_config() == {}  # an empty setting must not read the repo root as a file
    assert cfg.agent_token() == ""
    with cfg.override({"agent": {"url": "http://127.0.0.1:18789/v1/chat/completions"}}):
        assert cfg.agent_configured()
    with cfg.override({"agent": {"url": "  "}}):
        assert not cfg.agent_configured()


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="nope.missing"):
        cfg.get("nope.missing")


@pytest.mark.parametrize("raw", ["-1", "nan", "inf", "-inf", '"oops"'])
@pytest.mark.parametrize(
    "key",
    [
        "dashboard.person_gap_s",
        "dashboard.object_gap_s",
        "backfill.visit_gap_s",
        "timeline.gap_s",
    ],
)
def test_invalid_duration_in_pack_file_names_setting(tmp_path, monkeypatch, key, raw):
    section, name = key.split(".")
    path = tmp_path / "pack.toml"
    path.write_text(f"[{section}]\n{name} = {raw}\n")
    monkeypatch.setenv("PACK_CONFIG", str(path))
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        cfg.nonnegative_seconds(key)


@pytest.mark.parametrize("value", [0.0, 0.5, 10.0])
def test_duration_preserves_valid_seconds_and_environment_override(monkeypatch, value):
    monkeypatch.setenv("PACK_TIMELINE_GAP_S", str(value))
    assert cfg.nonnegative_seconds("timeline.gap_s") == value


@given(st.integers())
def test_parse_int_reads_every_whole_number_back(n):
    assert cfg.parse_int(str(n)) == n


# str.isdigit accepts these, or int() would take them with a sign or spaces; none is a plain whole number.
@pytest.mark.parametrize("text", ["--1", "²", "-", "", "+1", " 1", "1 ", "1.5", "1e3", "١", None, 3])
def test_parse_int_refuses_anything_but_an_optional_minus_and_ascii_digits(text):
    assert cfg.parse_int(text) is None
