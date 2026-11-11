"""Editing pack.toml from the dashboard: config.schema() and config.save(), and the /config endpoints."""

import http.client
import json
import os
import queue
import stat
import subprocess
import sys
import tomllib
from typing import Any, cast

import numpy as np
import pytest
import soundfile as sf

from liberational import config as cfg
from liberational import login_item, services
from liberational.dashboard import settings


def call(base, method, path, body=None, raw=None, **headers):
    if body is not None or raw is not None:
        headers.setdefault("Content-Type", "application/json")
    # A guard against a hung server, not a speed check: the first GET /config scans the services' code (readers.py,
    # about 2 s idle), and on a loaded machine at test priority that ran past a 5 s timeout.
    c = http.client.HTTPConnection(base.removeprefix("http://"), timeout=60)
    c.request(method, path, raw if raw is not None else json.dumps(body) if body is not None else None, headers)
    r = c.getresponse()
    out = r.status, json.loads(r.read())
    c.close()
    return out


def rows():
    return {r["key"]: r for s in cfg.schema() for r in s["settings"]}


@pytest.fixture
def pack_toml(tmp_path, monkeypatch):
    path = tmp_path / "pack.toml"
    monkeypatch.setattr(cfg, "local_config", lambda: path)
    return path


# A comment block names the section or key below it; a blank line or a value ends the block, and a "New feature:"
# line marks a switch without becoming part of its description.
DEFAULTS = """\
# Who it belongs to.
[owner]
# Their name,
# in full.
name = ""
port = 8765

# Cut off by the blank line below.

[vision]
fov = 160.0
cams = [0]
# New feature: Flipping
# Mirrors the picture.
hold_fps = false

[models.files.llm]
path = "data/models/llm.gguf"
"""


def test_schema_reads_types_comments_and_features_from_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DEFAULTS", tmp_path / "defaults.toml")
    cfg.DEFAULTS.write_text(DEFAULTS)
    cfg.load.cache_clear()
    sections = cfg.schema()
    assert [(s["name"], s["description"]) for s in sections] == [("owner", "Who it belongs to."), ("vision", "")]
    assert {k: (r["type"], r["description"], r["feature"]) for k, r in rows().items()} == {
        "owner.name": ("str", "Their name, in full.", ""),
        "owner.port": ("int", "", ""),
        "vision.fov": ("float", "", ""),
        "vision.cams": ("list", "", ""),
        "vision.hold_fps": ("bool", "Mirrors the picture.", "Flipping"),
    }


def test_shipped_new_feature_switches_are_described_checkboxes():
    marked = {k: r for k, r in rows().items() if r["feature"]}
    assert marked
    assert {k: r["type"] for k, r in marked.items()} == dict.fromkeys(marked, "bool")  # the panel shows checkboxes
    assert [k for k, r in marked.items() if not r["description"]] == []


def test_secrets_are_masked(pack_toml):
    assert (rows()["agent.token"]["value"], rows()["agent.token"]["set"]) == ("", False)
    with cfg.override({"agent": {"token": "s3cret"}}):
        r = rows()["agent.token"]
    assert (r["value"], r["set"], r["secret"], rows()["dashboard.password"]["secret"]) == ("", True, True, True)
    assert "set" not in rows()["owner.name"]


def test_env_override_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("PACK_OWNER_NAME", "Dana")
    monkeypatch.setenv("PACK_STATE", str(tmp_path))
    r = rows()
    assert (r["owner.name"]["env"], r["owner.name"]["value"]) == ("PACK_OWNER_NAME", "Dana")
    assert r["paths.state"]["env"] == "PACK_STATE"  # an older alias still counts
    assert r["tts.voice"]["env"] is None


def test_save_coerces_and_writes_only_non_defaults(pack_toml):
    written, overridden = cfg.save(
        {
            "ports.dashboard": "9000",
            "vision.objects": "no",
            "journal.levels": "60, 120",
            "presence.gap_s": 45,
            "audio.exclude_devices": ["usb"],
            "assistant.name": "Bob",  # the default, so not written
        }
    )
    data = tomllib.loads(pack_toml.read_text())
    assert data["ports"] == {"dashboard": 9000}
    assert data["vision"] == {"objects": False}
    assert data["journal"] == {"levels": [60, 120]}
    assert data["presence"] == {"gap_s": 45.0}
    assert data["audio"] == {"exclude_devices": ["usb"]}
    assert "assistant" not in data
    assert sorted(written) == [
        "audio.exclude_devices",
        "journal.levels",
        "ports.dashboard",
        "presence.gap_s",
        "vision.objects",
    ]
    assert overridden == []
    assert cfg.get("ports.dashboard") == 9000  # the cache was cleared
    assert stat.S_IMODE(pack_toml.stat().st_mode) == 0o600


def test_save_keeps_comments_and_other_keys_and_removes_defaults(pack_toml):
    pack_toml.write_text('# my notes\n[owner]\nname = "Alice"  # me\n\n[assistant]\nname = "Zed"\n')
    cfg.save({"assistant.name": "Bob", "tts.voice": "am_adam"})
    text = pack_toml.read_text()
    assert "# my notes" in text
    assert "# me" in text
    data = tomllib.loads(text)
    assert data["owner"] == {"name": "Alice"}
    assert "name" not in data.get("assistant", {})  # back to the default
    assert data["tts"] == {"voice": "am_adam"}
    cfg.save({"tts.voice": "af_heart"})  # back to the default: the emptied table goes too
    assert "[tts]" not in pack_toml.read_text()


def test_save_with_an_empty_secret_keeps_the_current_one(pack_toml):
    pack_toml.write_text('[agent]\ntoken = "old"\n')
    written, _ = cfg.save({"agent.token": "", "owner.name": "Alice"})
    assert tomllib.loads(pack_toml.read_text())["agent"]["token"] == "old"
    assert written == ["owner.name"]
    cfg.save({"agent.token": "new"})
    assert tomllib.loads(pack_toml.read_text())["agent"]["token"] == "new"


@pytest.mark.parametrize(
    "values, field",
    [
        pytest.param({"nope.x": 1}, "nope.x", id="unknown-key"),
        pytest.param({"owner": "x"}, "owner", id="section-not-key"),
        pytest.param({"models.ollama.llm": "x"}, "models.ollama.llm", id="model-manifest"),
        pytest.param({"ports.dashboard": "abc"}, "ports.dashboard", id="int-from-text"),
        pytest.param({"ports.dashboard": 8765.5}, "ports.dashboard", id="int-from-fraction"),
        pytest.param({"ports.dashboard": True}, "ports.dashboard", id="int-from-bool"),
        pytest.param({"vision.objects": "maybe"}, "vision.objects", id="bool-from-word"),
        pytest.param({"owner.name": 5}, "owner.name", id="text-from-number"),
        pytest.param({"journal.levels": ["a"]}, "journal.levels", id="int-list-from-text"),
        pytest.param({"journal.levels": {"a": 1}}, "journal.levels", id="list-from-dict"),
        pytest.param({"tts.voice": "am_adam", "presence.gap_s": "wide"}, "presence.gap_s", id="one-bad-of-two"),
    ],
)
def test_save_refuses_unknown_and_mistyped_values_without_writing(pack_toml, values, field):
    pack_toml.write_text('[owner]\nname = "Alice"\n')
    with pytest.raises(cfg.SettingError) as e:
        cfg.save(values)
    assert e.value.key == field
    assert pack_toml.read_text() == '[owner]\nname = "Alice"\n'


def test_save_reports_env_overrides(pack_toml, monkeypatch):
    monkeypatch.setenv("PACK_OWNER_NAME", "Dana")
    assert cfg.save({"owner.name": "Alice", "tts.voice": "am_adam"}) == (["owner.name", "tts.voice"], ["owner.name"])


@pytest.fixture
def restarts(monkeypatch):
    calls = queue.Queue()
    monkeypatch.setattr(
        settings,
        "restart",
        lambda names=(), dashboard=True, before=None: calls.put((list(names), dashboard)),
    )
    return calls


@pytest.fixture
def config_url(serve_dashboard, monkeypatch, pack_toml, restarts):
    monkeypatch.setattr(sys, "argv", ["pack-dashboard"])
    return serve_dashboard()


def test_get_reports_setup_and_control(config_url, pack_toml):
    code, body = call(config_url, "GET", "/config")
    assert (code, body["ok"], body["control"], body["needs_setup"]) == (200, True, True, True)
    assert [{k: v for k, v in r.items() if k != "restarts"} for s in body["sections"] for r in s["settings"]] == [
        r for s in cfg.schema() for r in s["settings"]
    ]
    assert body["path"] == str(pack_toml)
    assert not call(config_url, "GET", "/config", Host="pack.example.ts.net")[1]["control"]
    pack_toml.write_text("")
    assert not call(config_url, "GET", "/config")[1]["needs_setup"]


def test_save_writes_replies_and_restarts(config_url, pack_toml, restarts, monkeypatch):
    code, body = call(config_url, "POST", "/config", {"values": {"owner.name": "Alice", "ports.dashboard": 9100}})
    assert code == 200, body
    assert (body["ok"], body["restarting"], body["overridden"]) == (True, True, [])
    assert sorted(body["written"]) == ["owner.name", "ports.dashboard"]
    assert body["url"] == "http://127.0.0.1:9100/"  # where the dashboard comes back
    assert tomllib.loads(pack_toml.read_text())["owner"]["name"] == "Alice"
    restarts.get(timeout=5)  # the restart starts after the reply has gone out
    monkeypatch.setattr(sys, "argv", ["pack-dashboard", "8831", "--dev"])  # a port argument wins over config
    assert (
        call(config_url, "POST", "/config", {"values": {"tts.voice": "am_adam"}})[1]["url"] == "http://127.0.0.1:8831/"
    )
    restarts.get(timeout=5)


def test_saving_start_at_login_adds_and_removes_the_desktop_entry_and_restarts_nothing(
    config_url, pack_toml, restarts, tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(login_item.sys, "platform", "linux")
    entry = tmp_path / "xdg" / "autostart" / "liberational.desktop"
    code, body = call(config_url, "POST", "/config", {"values": {"app.start_at_login": True}})
    assert (code, body["written"], body["restarted"], body["restarting"]) == (200, ["app.start_at_login"], [], False)
    assert entry.is_file()
    code, body = call(config_url, "POST", "/config", {"values": {"app.start_at_login": False}})
    assert code == 200 and not entry.exists()
    assert restarts.empty()


def test_bad_values_and_bodies_are_400_without_restart(config_url, pack_toml, restarts):
    code, body = call(config_url, "POST", "/config", {"values": {"ports.dashboard": "abc"}})
    assert (code, body["field"]) == (400, "ports.dashboard")
    assert call(config_url, "POST", "/config", {"owner.name": "Alice"})[0] == 400  # not wrapped in "values"
    assert call(config_url, "POST", "/config", raw="not json")[0] == 400
    assert restarts.empty()
    assert not pack_toml.exists()


def test_save_refuses_remote_rebound_cross_origin_and_non_json(config_url, pack_toml, restarts):
    on = {"values": {"owner.name": "Mallory"}}
    assert call(config_url, "POST", "/config", on, **{"X-Forwarded-For": "100.64.0.9"})[0] == 403  # via Tailscale
    assert call(config_url, "POST", "/config", on, Host="evil.example")[0] == 403  # DNS rebinding
    assert call(config_url, "POST", "/config", on, Origin="http://evil.example")[0] == 403
    assert call(config_url, "POST", "/config", on, **{"Content-Type": "text/plain"})[0] == 403
    assert restarts.empty()
    assert not pack_toml.exists()


def test_restart_names_restarts_installed_services_and_starts_newly_required_ones(monkeypatch):
    monkeypatch.setattr(services, "installed", lambda: ["listen", "dashboard", "voice"])
    monkeypatch.setattr(cfg, "platform_key", lambda: "linux-x86_64")
    assert settings.restart_names() == ["listen", "voice"]  # never the dashboard
    with cfg.override({"asr": {"engine": "whisper"}}):
        assert settings.restart_names() == ["listen", "voice", "whisper"]


def test_a_save_installs_what_may_now_run_here_and_uninstalls_what_may_not(monkeypatch, caplog):
    monkeypatch.setattr(services, "installed", lambda: ["listen", "dashboard", "notes"])
    monkeypatch.setattr(services, "selected", lambda: {"listen": {}, "dashboard": {}, "phone": {}})
    before = {"listen", "dashboard", "notes"}  # phone added under Also run, notes under Don't run
    assert settings.run_here_changes(before) == (["phone"], ["notes"])
    assert settings.run_here_changes({"listen", "dashboard", "phone"}) == ([], [])  # nothing changed: nothing to do

    ran = []
    caplog.set_level("INFO")
    monkeypatch.setattr(
        settings, "_pack_services", lambda verb, names: ran.append([verb, *names]) or (0, "removed notes")
    )
    assert settings.apply_run_here(before) == ["phone"]  # installed with the restart that follows
    assert ran == [["uninstall", "notes"]]
    assert caplog.messages == ["services: off notes (no longer runs here after a Settings save; uninstalled): done"]
    assert settings.apply_run_here(None) == []  # unknown before: never guesses


@pytest.mark.parametrize(
    "code, out, want",
    [
        pytest.param(0, "removed notes", "done", id="worked"),
        pytest.param(1, "removed notes\nnot installed: notes", "failed: not installed: notes", id="exit-code"),
        pytest.param(0, "pack-notes: FAILED busy", "failed: pack-notes: FAILED busy", id="its-line"),
        pytest.param(0, "pack-notes-sync.service: FAILED busy", "done", id="a-longer-name-is-not-it"),
    ],
)
def test_a_save_says_whether_each_service_change_worked(code, out, want):
    assert settings._outcome("notes", code, out) == want


def test_the_restart_uninstalls_then_installs_then_restarts_the_dashboard_even_after_a_failure(monkeypatch, caplog):
    steps = []
    monkeypatch.setattr(settings.time, "sleep", lambda s: None)
    monkeypatch.setattr(settings.os, "execv", lambda *a: steps.append("exec"))
    monkeypatch.setattr(settings, "run_here_changes", lambda before: (["phone"], ["notes"]))
    monkeypatch.setattr(services, "states", lambda: {"listen": "active", "voice": "inactive"})
    monkeypatch.setattr(settings, "_pack_services", lambda verb, names: steps.append([verb, *names]) or (0, ""))
    caplog.set_level("INFO")
    settings.restart(["listen", "voice", "phone"], True, {"listen", "notes"}).join(5)
    # notes uninstalled; voice, stopped with Stop now, stays stopped; phone installed once; then the dashboard
    assert steps == [["uninstall", "notes"], ["install", "listen", "phone"], "exec"]
    assert "services: on phone (runs here after a Settings save; installed and started): done" in caplog.messages

    steps.clear()

    def timeout(verb, names):
        raise subprocess.TimeoutExpired(verb, 300)

    monkeypatch.setattr(settings, "_pack_services", timeout)
    settings.restart(["listen"], True, {"listen", "notes"}).join(5)
    assert steps == ["exec"]  # the dashboard still comes back on the new settings
    assert any(line.startswith("settings: services not restarted: TimeoutExpired") for line in caplog.messages)


def test_a_save_passes_on_what_ran_here_before_it(config_url, pack_toml, monkeypatch):
    seen = queue.Queue()
    monkeypatch.setattr(settings, "restart", lambda names=(), dashboard=True, before=None: seen.put(before))
    want = set(services.selected())
    assert "phone-bridge" not in want
    assert call(config_url, "POST", "/config", {"values": {"services.enable": "phone-bridge"}})[0] == 200
    assert seen.get(timeout=5) == want  # what ran here before the save, so the restart can act on the difference


@pytest.mark.parametrize(
    "raw, want",
    [("1, -2", [1, -2]), ("--1, ²", ["--1", "²"]), (["--1"], ["--1"])],
    ids=["numbers", "isdigit-trap-text", "isdigit-trap-json"],
)
def test_an_untyped_list_keeps_numbers_as_numbers_and_the_rest_as_text(raw, want):
    assert cfg.coerce_setting("services.enable", raw, []) == want


@pytest.mark.parametrize(
    "argv, port",
    [
        (["8801"], 8801),
        (["--dev", "8801"], 8801),
        (["-1"], None),
        (["0"], None),
        (["70000"], None),
        (["²"], None),
        ([], None),
    ],
    ids=["port", "after-a-flag", "negative", "zero", "too-big", "superscript", "none"],
)
def test_a_port_on_the_command_line_must_be_a_real_port(argv, port):
    assert settings.port_arg(argv) == port


def test_camera_places_are_saved_one_per_camera_in_one_form(pack_toml):
    cfg.save({"vision.camera_names": "cam0: Living  room, 2=kitchen"})  # as Settings sends its tags
    assert tomllib.loads(pack_toml.read_text())["vision"]["camera_names"] == ["cam0: Living room", "cam2: kitchen"]


@pytest.mark.parametrize(
    "raw,match",
    [
        pytest.param("living room", "is not a camera and its name", id="no-camera"),
        pytest.param("cam0:", "is not a camera and its name", id="no-place"),
        pytest.param("cam0: living room, upstairs", "'upstairs' is not a camera", id="comma-in-a-place"),
        pytest.param("cam0: den, 0: kitchen", "cam0 is named twice", id="twice"),
    ],
)
def test_a_camera_place_that_is_not_a_camera_and_a_name_is_refused(pack_toml, raw, match):
    with pytest.raises(cfg.SettingError, match=match):
        cfg.save({"vision.camera_names": raw})
    assert not pack_toml.exists()


def test_a_camera_goes_by_its_place_name_or_else_its_id():
    from liberational.vision import views

    with cfg.override({"vision": {"camera_names": ["cam1: kitchen"]}}):
        assert (views.name("cam1"), views.name("cam0")) == ("kitchen", "cam0")
    with cfg.override({"vision": {"camera_names": ["kitchen"]}}):  # set by hand, wrongly: no camera gets a name
        assert views.name("cam1") == "cam1"


@pytest.mark.parametrize(
    "hand", [pytest.param("kitchen", id="malformed"), pytest.param("cam0: living room, upstairs", id="comma")]
)
def test_camera_places_set_by_hand_never_block_saving_something_else(pack_toml, hand):
    pack_toml.write_text(f'[vision]\ncamera_names = ["{hand}"]\n')
    cfg.load.cache_clear()
    cfg.save(
        {"vision.shared_decode_gpu": "1", "vision.camera_names": hand}
    )  # the form sends every field, as it shows them
    assert tomllib.loads(pack_toml.read_text())["vision"] == {"camera_names": [hand], "shared_decode_gpu": 1}


def test_a_line_break_in_a_camera_place_is_a_space():
    from liberational.vision import views

    assert views.parse(["cam0: living\nroom"]) == {"cam0": "living room"}


def test_every_plain_label_names_a_real_setting_and_a_real_section():
    keys = {r["key"] for s in cfg.schema() for r in s["settings"]}
    raw = tomllib.loads(cfg.LABELS.read_text())
    assert set(raw) - keys == set()  # a renamed or removed setting leaves no orphan label
    assert {m["section"] for m in raw.values()} <= set(cfg.SECTIONS)
    assert all(m.get("label") and m.get("help") for m in raw.values())


def test_a_setting_carries_its_plain_label_help_and_section(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DEFAULTS", tmp_path / "defaults.toml")
    monkeypatch.setattr(cfg, "LABELS", tmp_path / "labels.toml")
    cfg.DEFAULTS.write_text(
        '[owner]\n# The name.\nname = ""\n# New feature: Shiny\n# It shines.\nshiny = true\nplain = 1\n'
    )
    cfg.LABELS.write_text('["owner.name"]\nsection = "you"\nlabel = "Your name"\nhelp = "What it calls you"\n'
                          '["owner.plain"]\nsection = "nowhere"\nlabel = "Plain"\nhelp = "x"\n')  # fmt: skip
    cfg.load.cache_clear()
    got = {r["key"]: (r["label"], r["help"], r["section"]) for s in cfg.schema() for r in s["settings"]}
    assert got == {
        "owner.name": ("Your name", "What it calls you", "you"),
        "owner.shiny": ("", "", "features"),  # a new feature with no label of its own
        "owner.plain": ("Plain", "x", ""),  # an unknown section: Advanced only
    }


def test_only_a_setting_labels_toml_calls_live_is_live(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DEFAULTS", tmp_path / "defaults.toml")
    monkeypatch.setattr(cfg, "LABELS", tmp_path / "labels.toml")
    cfg.DEFAULTS.write_text("[audio]\nsensitivity = 70\nfast = 1\nother = 2\n")
    cfg.LABELS.write_text('["audio.sensitivity"]\nlabel = "S"\nhelp = "s"\nlive = true\n'
                          '["audio.fast"]\nlabel = "F"\nhelp = "f"\nlive = "yes"\n')  # fmt: skip
    cfg.load.cache_clear()
    got = {r["key"]: r["live"] for s in cfg.schema() for r in s["settings"]}
    assert got == {"audio.sensitivity": True, "audio.fast": False, "audio.other": False}  # only a TOML true


@pytest.mark.parametrize(
    ("raw", "saved"),
    [
        pytest.param("Mic A, Mic B", ["Mic A", "Mic B"], id="text"),
        pytest.param('["USB Audio: - (hw:1,0)", "Mic B"]', ["USB Audio: - (hw:1,0)", "Mic B"], id="json-keeps-commas"),
    ],
)
def test_a_list_typed_as_text_is_split_at_commas_unless_it_is_json(raw, saved):
    assert cfg.coerce_setting("audio.exclude_devices", raw, []) == saved


def test_a_broken_json_list_is_refused():
    with pytest.raises(cfg.SettingError):
        cfg.coerce_setting("audio.exclude_devices", '["unclosed', [])


def test_config_says_what_a_save_restarts(monkeypatch):
    monkeypatch.setattr(settings, "restart_names", lambda: ["listen", "voice"])
    assert settings.view(True)["restarts"] == ["listen", "voice", "dashboard"]
    monkeypatch.setattr(settings, "restart_names", lambda: (_ for _ in ()).throw(SystemExit("unknown service")))
    assert settings.view(True)["restarts"] == []


def test_a_broken_labels_file_costs_the_plain_words_and_not_the_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "LABELS", tmp_path / "labels.toml")
    cfg.LABELS.write_text('["owner.name"\nlabel = ')
    assert cfg.labels() == {}
    assert any(r["key"] == "owner.name" for s in cfg.schema() for r in s["settings"])


class FakeGet:
    """Just enough of the dashboard's handler for a GET."""

    command, path = "GET", "/config/voice_ref"

    def _json(self, obj, code=200):
        self.reply = (code, obj)


def test_the_voice_reference_says_when_it_was_recorded(tmp_path):
    ref = tmp_path / "owner_ref.wav"
    sf.write(ref, np.zeros(24000 * 2, dtype=np.float32), 24000)
    os.utime(ref, (1793000000, 1793000000))
    h = FakeGet()
    with cfg.override({"owner": {"voice_ref": str(ref)}}):
        settings.voice_ref(cast(Any, h))
    assert (h.reply[0], h.reply[1]["seconds"], h.reply[1]["recorded"]) == (200, 2.0, 1793000000.0)


def test_lists_the_kokoro_voices_from_their_file(tmp_path, monkeypatch):
    np.savez(tmp_path / "voices.npz", af_heart=np.zeros(2), bm_george=np.zeros(2))
    monkeypatch.setattr(cfg, "model_path", lambda name: tmp_path / "voices.npz")
    assert settings.kokoro_voices() == ["af_heart", "bm_george"]
    monkeypatch.setattr(cfg, "model_path", lambda name: tmp_path / "missing.bin")
    assert settings.kokoro_voices() == []


def test_a_machine_with_mlx_speaks_in_the_cloned_voice_once_there_is_a_reference(tmp_path, monkeypatch):
    ref = tmp_path / "owner_ref.wav"
    monkeypatch.setattr(cfg, "has_mlx", lambda: True)
    with cfg.override({"owner": {"voice_ref": str(ref)}}):
        assert not settings.can_clone()  # nothing to clone yet: it speaks Kokoro
        ref.write_bytes(b"RIFF")
        assert not settings.can_clone()  # not audio the voice could clone from
        sf.write(str(ref), np.zeros(1600, np.float32), 16000)
        assert settings.can_clone()
        monkeypatch.setattr(cfg, "has_mlx", lambda: False)
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)  # no voice-clone extra
        assert not settings.can_clone()


def test_a_section_lists_its_settings_in_labels_order(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DEFAULTS", tmp_path / "defaults.toml")
    monkeypatch.setattr(cfg, "LABELS", tmp_path / "labels.toml")
    cfg.DEFAULTS.write_text("[owner]\na = 1\nb = 2\nc = 3\n")
    cfg.LABELS.write_text(
        '["owner.c"]\nsection = "you"\nlabel = "C"\nhelp = "c"\n["owner.a"]\nsection = "you"\nlabel = "A"\nhelp = "a"\n'
    )
    cfg.load.cache_clear()
    got = {r["key"]: r["order"] for s in cfg.schema() for r in s["settings"]}
    assert got == {"owner.a": 1, "owner.b": 10_000, "owner.c": 0}  # labels.toml's order; one not there after


@pytest.fixture
def two_services(monkeypatch):
    """see-actions and voice installed, as on a machine running both."""
    monkeypatch.setattr(settings, "restart_names", lambda: ["see-actions", "voice"])
    monkeypatch.setattr(services, "installed", lambda: ["see-actions", "voice"])


def test_saving_restarts_only_what_reads_the_change(config_url, restarts, two_services):
    body = call(config_url, "POST", "/config", {"values": {"tts.voice": "am_adam"}})[1]
    assert (body["restarted"], body["restarting"]) == (["voice"], True)  # the cameras keep running
    assert restarts.get(timeout=5) == (["voice"], True)


def test_saving_only_live_settings_restarts_nothing(config_url, pack_toml, restarts, two_services):
    body = call(config_url, "POST", "/config", {"values": {"audio.sensitivity": 40}})[1]
    assert (body["ok"], body["restarted"], body["restarting"]) == (True, [], False)
    assert tomllib.loads(pack_toml.read_text())["audio"]["sensitivity"] == 40
    with pytest.raises(queue.Empty):
        restarts.get(timeout=1)


def test_a_new_service_the_settings_need_is_started(config_url, restarts, monkeypatch):
    monkeypatch.setattr(settings, "restart_names", lambda: ["see-actions", "whisper"])  # asr.engine = whisper needs it
    monkeypatch.setattr(services, "installed", lambda: ["see-actions"])
    monkeypatch.setattr("liberational.readers.affected", lambda changed: [])  # the cameras don't read asr.engine
    body = call(config_url, "POST", "/config", {"values": {"asr.engine": "whisper"}})[1]
    assert "whisper" in body["restarted"] and "see-actions" not in body["restarted"]


def test_each_setting_says_what_saving_it_restarts(config_url, two_services):
    rows = {r["key"]: r["restarts"] for s in call(config_url, "GET", "/config")[1]["sections"] for r in s["settings"]}
    assert (rows["audio.sensitivity"], rows["tts.voice"]) == ([], ["voice", "dashboard"])


def test_a_save_restarts_everything_when_the_code_cant_be_scanned(config_url, restarts, two_services, monkeypatch):
    def broken(changed):
        raise SyntaxError("a module mid-edit")

    monkeypatch.setattr("liberational.readers.affected", broken)
    monkeypatch.setattr("liberational.readers.affected_each", broken)
    assert all("restarts" not in r for s in call(config_url, "GET", "/config")[1]["sections"] for r in s["settings"])
    body = call(config_url, "POST", "/config", {"values": {"tts.voice": "am_adam"}})[1]
    assert (body["restarted"], body["restarting"]) == (["see-actions", "voice"], True)


def test_a_setting_carries_the_group_labels_toml_gives_it(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DEFAULTS", tmp_path / "defaults.toml")
    monkeypatch.setattr(cfg, "LABELS", tmp_path / "labels.toml")
    cfg.DEFAULTS.write_text("[journal]\nenabled = true\nfacts = true\n")
    cfg.LABELS.write_text('["journal.enabled"]\nlabel = "J"\nhelp = "j"\ngroup = "Memory"\n'
                          '["journal.facts"]\nlabel = "F"\nhelp = "f"\ngroup = 3\n')  # fmt: skip
    cfg.load.cache_clear()
    got = {r["key"]: r["group"] for s in cfg.schema() for r in s["settings"]}
    assert got == {"journal.enabled": "Memory", "journal.facts": ""}  # only a string names a group


def test_settings_still_open_in_a_process_that_predates_a_new_setting(tmp_path, monkeypatch, caplog):
    # A merge added [pose] to defaults.toml while the dashboard ran; its config was read before that.
    monkeypatch.setattr(cfg, "DEFAULTS", tmp_path / "defaults.toml")
    cfg.DEFAULTS.write_text(DEFAULTS)
    cfg.load.cache_clear()
    cfg.load()  # what the running process read at start
    cfg.DEFAULTS.write_text(DEFAULTS + "\n[pose]\n# Pose estimation.\nenabled = false\n")
    with caplog.at_level("WARNING"):
        got = rows()
        rows()
    assert got["pose.enabled"]["value"] is False  # its default, from the file
    assert got["owner.name"]["value"] == rows()["owner.name"]["default"]
    assert sum("pose.enabled" in r.getMessage() for r in caplog.records) == 1  # said once, not on every read


@pytest.mark.parametrize(
    "verb,names",
    [("restart", ["listen"]), ("restart", ["continuum-broker"]), ("install", ["continuum-capture"]), ("install", [])],
)
@pytest.mark.parametrize("elapsed", [309, 630], ids=["cold-start-over-five-minutes", "late-recovery"])
def test_settings_service_child_can_finish_cascade_and_recovery(monkeypatch, verb, names, elapsed):
    calls = []

    def child(command, *, capture_output, text, timeout):
        calls.append((command, timeout))
        if timeout < elapsed:
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(command, 0, "recovery finished", "")

    monkeypatch.setattr(settings.subprocess, "run", child)
    assert settings._pack_services(verb, names) == (0, "recovery finished")
    assert 630 <= calls[0][1] <= 900


def test_settings_non_cascade_removal_keeps_its_short_timeout(monkeypatch):
    timeouts = []

    def child(command, *, capture_output, text, timeout):
        timeouts.append(timeout)
        return subprocess.CompletedProcess(command, 0, "removed", "")

    monkeypatch.setattr(settings.subprocess, "run", child)
    assert settings._pack_services("uninstall", ["listen"]) == (0, "removed")
    assert timeouts == [300]
