import json
import pathlib
import tomllib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from liberational import config as cfg


def test_defaults_are_generic():
    assert cfg.owner_name() == ""
    assert cfg.owner_label() == ""
    assert cfg.owner_or() == "the owner"
    assert cfg.wake_words() == ["hey bob"]


@pytest.mark.parametrize(
    "var, raw, key, want",
    [
        pytest.param("PACK_PORTS_DASHBOARD", "9000", "ports.dashboard", 9000, id="int"),
        pytest.param("PACK_VISION_CAMS", "0,2", "vision.cams", [0, 2], id="int-list"),
        pytest.param("PACK_VISION_HOLD_FPS", "false", "vision.hold_fps", False, id="bool"),
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
        pytest.param("some/dir/openclaw.json", cfg.ROOT / "some" / "dir" / "openclaw.json", id="relative"),
        pytest.param("~/.openclaw/openclaw.json", pathlib.Path.home() / ".openclaw" / "openclaw.json", id="home"),
    ],
)
def test_relative_paths_resolve_under_repo(setting, want):
    with cfg.override({"agent": {"openclaw_config": setting}}):
        assert cfg.path("agent.openclaw_config") == want


def test_runtime_data_defaults_live_under_data():
    # The repo root holds only code and config. Runtime state stays in data/;
    # see-decode's frame ring is empty by default, meaning /dev/shm (or the temp
    # directory): memory, never disk (#1955).
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
