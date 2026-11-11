import datetime
import json
import os
import pathlib
import plistlib
import shlex
import subprocess
import sys
from unittest.mock import create_autospec

import psutil
import pytest

from liberational import config as cfg
from liberational import logs
from liberational import services as s


def test_the_gateway_listens_on_its_configured_port():
    with cfg.override({"ports": {"gateway": 9100}}):
        args = s.argv(s.declared()["continuum-gateway"])
    assert args[args.index("--listen") + 1] == "0.0.0.0:9100"


@pytest.mark.parametrize(
    "platform, services, on, off",
    [
        # vlm: Apple Silicon only; Continuum's services on by default; phone-bridge and whisper-names: opt-in
        ("linux-x86_64", {}, {"listen", "continuum-broker", "see-actions"}, {"vlm", "phone-bridge", "whisper-names"}),
        ("darwin-arm64", {}, {"listen", "vlm", "continuum-broker"}, {"phone-bridge", "whisper-names"}),
        ("darwin-arm64", {"enable": ["whisper-names"]}, {"whisper-names", "listen"}, set()),
        # enabling doesn't override the platform
        ("linux-x86_64", {"enable": ["phone-bridge", "vlm"], "disable": ["voice"]}, {"phone-bridge"}, {"vlm", "voice"}),
    ],
)
def test_selection_follows_defaults_platform_and_overrides(monkeypatch, platform, services, on, off):
    monkeypatch.setattr(cfg, "platform_key", lambda: platform)
    with cfg.override({"services": services}):
        names = set(s.selected())
    assert on <= names
    assert not off & names


def test_whisper_engine_requires_the_whisper_service():
    assert "whisper" not in s.selected()
    with cfg.override({"asr": {"engine": "whisper"}}):
        assert s.required() == {"whisper"}
        assert "whisper" in s.selected()
        with cfg.override({"services": {"disable": ["whisper"]}}):
            assert "whisper" not in s.selected()  # an explicit disable still wins


def test_unknown_names_are_rejected():
    with cfg.override({"services": {"enable": ["nope"]}}), pytest.raises(SystemExit, match="nope"):
        s.selected()


DAEMON, SCHEDULED = {"command": "pack-x", "env": {"A": "has space"}}, {"command": "pack-y", "schedule": "07:05"}


def plist(svc):
    return plistlib.loads(s.render_launchd("x", svc)["pack.x.plist"].encode())


def test_launchd_daemon_schedule_and_background():
    daemon, sync = plist(DAEMON), plist(SCHEDULED)
    assert (daemon["Label"], daemon["WorkingDirectory"]) == ("pack.x", str(cfg.workdir()))
    assert daemon["ProgramArguments"] == [str(cfg.program("pack-x"))]
    assert (daemon["RunAtLoad"], daemon["KeepAlive"], daemon["ThrottleInterval"]) == (True, True, 10)
    assert sync["ProgramArguments"] == [str(cfg.program("pack-y"))]
    assert (sync["StartCalendarInterval"], "KeepAlive" in sync) == ({"Hour": 7, "Minute": 5}, False)
    assert plist({**SCHEDULED, "schedule": "*:07"})["StartCalendarInterval"] == {"Minute": 7}
    bg = plist({**DAEMON, "background": True, "throttle": 60})
    assert (bg["ProcessType"], bg["ThrottleInterval"]) == ("Background", 60)


def test_systemd_daemon_and_timer():
    daemon, sync = s.render_systemd("x", DAEMON), s.render_systemd("x", SCHEDULED)
    assert list(daemon) == ["pack-x.service"]
    assert {"Restart=always", "Environment='A=has space'", "WantedBy=default.target"} <= set(
        daemon["pack-x.service"].splitlines()
    )
    assert "OnCalendar=*-*-* 07:05:00" in sync["pack-x.timer"]
    assert "Type=oneshot" in sync["pack-x.service"]
    assert "Restart=" not in sync["pack-x.service"]


def test_every_declared_service_renders_on_both_platforms():
    assert all(s.render_launchd(name, svc) and s.render_systemd(name, svc) for name, svc in s.declared().items())


def test_systemd_start_recreates_missing_log_directory(tmp_path):
    logs = tmp_path / "missing logs ' % $"
    with cfg.override({"paths": {"logs": str(logs)}}):
        unit = s.render_systemd("probe", {"command": "/bin/echo", "args": ["hello world ' % $HOME"]})[
            "pack-probe.service"
        ]
    # Run only the harmless rendered command, never the service manager.
    command = next(line.removeprefix("ExecStart=") for line in unit.splitlines() if line.startswith("ExecStart="))
    command = command.removeprefix(":").replace("%%", "%")
    assert "StandardOutput=append:" not in unit
    subprocess.run(shlex.split(command), check=True)
    subprocess.run(shlex.split(command), check=True)
    assert (logs / "probe.log").read_text() == "hello world ' % $HOME\nhello world ' % $HOME\n"


def test_args_expand_config_and_env(monkeypatch):
    monkeypatch.setenv("PACK_TEST_DIR", "/opt/x")
    monkeypatch.delenv("PACK_TEST_UNSET", raising=False)
    svc = {
        "command": "pack-x",
        "args": ["{ports.whisper}", "{env:PACK_TEST_DIR:~/f}/a", "{env:PACK_TEST_UNSET:~/f}/b"],
    }
    with cfg.override({"ports": {"whisper": 8123}}):
        assert s.argv(svc)[1:] == ["8123", "/opt/x/a", str(pathlib.Path.home() / "f/b")]


@pytest.mark.parametrize(
    "setting, template, want",
    [
        pytest.param({"paths": {"state": "data/state"}}, "{paths.state}", "{root}/data/state", id="relative-path"),
        pytest.param({"paths": {"state": "/srv/state"}}, "{paths.state}", "/srv/state", id="absolute-path"),
        pytest.param({"paths": {"decoded": ""}}, "{paths.decoded}", "", id="empty-path-stays-empty"),
        pytest.param({}, "{file:devices}", "{root}/config/devices.toml", id="device-manifest"),
        pytest.param({"ask": {"gpu": "data/x"}}, "{ask.gpu}", "data/x", id="not-a-path-setting"),
    ],
)
def test_path_arguments_are_whole_paths_whatever_the_working_directory(tmp_path, monkeypatch, setting, template, want):
    monkeypatch.setattr(cfg, "OWNER_ROOT", tmp_path)
    monkeypatch.delenv("PACK_STATE", raising=False)
    with cfg.override(setting):
        assert s.argv({"command": "pack-x", "args": [template]})[1:] == [want.replace("{root}", str(tmp_path))]


def test_our_commands_come_from_the_environment_and_others_from_path(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "BIN", tmp_path)
    monkeypatch.setattr(s.shutil, "which", {"ollama": "/opt/bin/ollama"}.get)
    assert s.argv({"command": "pack-x"}) == [str(tmp_path / "pack-x")]
    assert s.argv({"command": "continuum-broker"}) == [str(tmp_path / "continuum-broker")]
    assert s.argv({"command": "ollama"}) == ["/opt/bin/ollama"]
    assert s.argv({"command": "not-installed"}) == ["not-installed"]


def test_env_fills_config_and_drops_a_variable_whose_setting_is_empty():
    svc = {"command": "tools/x", "env": {"HOST": "127.0.0.1:{ports.whisper}", "GPU": "{ask.gpu}", "VK": ""}}
    with cfg.override({"ports": {"whisper": 8123}, "ask": {"gpu": ""}}):
        env = s.environment(svc)
    assert (env["HOST"], env["VK"], "GPU" in env) == ("127.0.0.1:8123", "", False)
    with cfg.override({"ask": {"gpu": "GPU-synthetic"}}):
        assert s.environment(svc)["GPU"] == "GPU-synthetic"


@pytest.mark.parametrize(("setting", "expected"), [(None, "1"), (2, "2")])
def test_ask_ollama_keeps_as_many_models_loaded_as_ask_max_loaded_models(setting, expected):
    svc = s.declared()["ask-ollama"]
    values = {} if setting is None else {"ask": {"max_loaded_models": setting}}
    with cfg.override(values):
        assert s.environment(svc)["OLLAMA_MAX_LOADED_MODELS"] == expected


@pytest.mark.parametrize("layers, kv", [(0, "f16"), (48, "q8_0")])
def test_ask_ollama_cpu_expert_profile_has_bounded_gpu_cache(layers, kv):
    with cfg.override({"ask": {"cpu_expert_layers": layers, "kv_cache_type": kv}}):
        env = s.environment(s.declared()["ask-ollama"])
    assert env["LLAMA_ARG_N_CPU_MOE"] == str(layers)
    assert env["LLAMA_ARG_MOE_CACHE_MIB"] == "0"
    assert env["OLLAMA_KV_CACHE_TYPE"] == kv
    assert env["OLLAMA_FLASH_ATTENTION"] == "1"
    assert env["OLLAMA_NUM_PARALLEL"] == "1"
    assert env.get("LLAMA_ARG_FIT") == ("off" if layers else None)


def test_path_has_venv_first_and_no_duplicates(monkeypatch):
    venv, shims = str(cfg.BIN), str(pathlib.Path.home() / ".local/share/mise/shims")
    monkeypatch.setattr(s, "DARWIN", False)
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", venv, "/usr/bin"]))
    assert s.environment({})["PATH"].split(os.pathsep) == [venv, shims, "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    monkeypatch.setattr(cfg, "CHECKOUT", None)  # an installed package has no mise
    assert s.environment({})["PATH"].split(os.pathsep) == [venv, "/usr/bin", "/bin", "/usr/sbin", "/sbin"]


def test_a_service_built_from_source_is_only_declared_in_a_checkout(monkeypatch):
    assert s.declared()["whisper"]["source_build"] is True
    monkeypatch.setattr(cfg, "CHECKOUT", None)
    assert "whisper" not in s.declared()
    assert "continuum-broker" in s.declared()
    with cfg.override({"asr": {"engine": "whisper"}}), pytest.raises(SystemExit, match="built from source"):
        s.selected()


def test_whisper_server_is_named_by_its_whole_path(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "OWNER_ROOT", tmp_path)
    assert s.argv(s.declared()["whisper"])[1] == str(tmp_path / "data/whisper.cpp/build/bin/whisper-server")


@pytest.mark.parametrize(
    "darwin,tail",
    [
        pytest.param(True, ["/usr/local/bin", "/usr/sbin", "/sbin"], id="launchd"),
        pytest.param(False, ["/usr/sbin", "/sbin"], id="systemd"),
    ],
)
def test_path_keeps_system_directories_whatever_the_installing_shell_had(monkeypatch, darwin, tail):
    # Bob's Mac: a shell PATH ending at /usr/bin:/bin left sysctl (/usr/sbin) out of every plist.
    monkeypatch.setattr(s, "DARWIN", darwin)
    monkeypatch.setenv("PATH", "/opt/tool/bin:/usr/bin:/bin")
    path = s.environment({})["PATH"].split(os.pathsep)
    assert path[2:] == ["/opt/tool/bin", "/usr/bin", "/bin", *tail]


# ---- stopping and starting in a few systemctl calls, not several per service ----


@pytest.fixture
def systemd(tmp_path, monkeypatch):
    """systemd in a temp unit dir. Returns (sent, fail): the commands sent, and the (verb, unit) pairs that fail. A
    failed unit then reads as inactive."""
    sent, fail = [], set()

    def run(*cmd, check=False):
        sent.append(cmd)
        verb, units = (cmd[2], set(cmd[3:])) if cmd[:2] == ("systemctl", "--user") and len(cmd) > 2 else ("", set())
        failed = {u for v, u in fail if v == verb and u in units}
        if verb == "is-active":
            failed = {u for _, u in fail if u in units}
        return s.subprocess.CompletedProcess(cmd, 1 if failed else 0, "", "boom" if failed else "")

    monkeypatch.setattr(s, "DARWIN", False)
    monkeypatch.setattr(s, "_run", run)
    # Package checks have their own tests; service-manager batching must not depend on the host or manifest cache.
    monkeypatch.setattr(s, "preflight", lambda names: None)
    monkeypatch.setattr(s, "unit_dir", lambda: tmp_path / "units")
    monkeypatch.setattr(s, "log_path", lambda name: tmp_path / "logs" / f"{name}.log")
    monkeypatch.setattr(s, "selected", lambda: {"a": DAEMON, "b": DAEMON, "sync": SCHEDULED})
    (tmp_path / "units").mkdir()
    return sent, fail


def test_turning_services_off_is_one_stop_one_disable_and_one_reload(systemd, tmp_path, capsys):
    sent, _ = systemd
    for unit in ("pack-a.service", "pack-b.service", "pack-sync.service", "pack-sync.timer"):
        (tmp_path / "units" / unit).write_text("")
    s.uninstall([], anywhere=True)  # every installed one
    units = ("pack-a.service", "pack-b.service", "pack-sync.timer", "pack-sync.service")
    assert sent == [
        ("systemctl", "--user", "stop", *units),
        ("systemctl", "--user", "disable", "--no-reload", *units),
        ("systemctl", "--user", "daemon-reload"),
    ]
    assert list((tmp_path / "units").iterdir()) == []
    assert capsys.readouterr().out.splitlines() == ["removed a", "removed b", "removed sync"]


def test_turning_one_service_off_leaves_the_others(systemd, tmp_path):
    sent, _ = systemd
    for unit in ("pack-a.service", "pack-b.service"):
        (tmp_path / "units" / unit).write_text("")
    s.uninstall(["b"], anywhere=True)
    assert sent[0] == ("systemctl", "--user", "stop", "pack-b.service")
    assert [p.name for p in (tmp_path / "units").iterdir()] == ["pack-a.service"]


def test_turning_services_on_reloads_once_and_starts_them_together(systemd, capsys):
    sent, _ = systemd
    s.install(["a", "b", "sync"], anywhere=True)
    assert sent[:4] == [
        ("systemctl", "--user", "daemon-reload"),
        ("systemctl", "--user", "enable", "--no-reload", "pack-a.service", "pack-b.service", "pack-sync.timer"),
        ("systemctl", "--user", "start", "pack-sync.timer"),
        ("systemctl", "--user", "restart", "pack-a.service", "pack-b.service"),
    ]
    assert capsys.readouterr().out.splitlines()[:3] == [
        "pack-a.service: running",
        "pack-b.service: running",
        "pack-sync.timer: running",
    ]


@pytest.mark.parametrize("verb", ["enable", "restart"])
def test_a_service_that_fails_is_named_and_the_others_still_run(systemd, capsys, verb):
    _, fail = systemd
    fail.add((verb, "pack-b.service"))
    s.install(["a", "b"], anywhere=True)
    assert capsys.readouterr().out.splitlines()[:2] == ["pack-a.service: running", "pack-b.service: FAILED boom"]


def test_a_failed_restart_never_restarts_the_others_twice(systemd):
    """Retrying a failed restart batch one by one must not restart the healthy ones again (a second
    model load for vision or listen)."""
    sent, fail = systemd
    fail.add(("restart", "pack-b.service"))
    s.install(["a", "b"], anywhere=True)
    assert sum(1 for c in sent if c[2:3] == ("restart",) and "pack-a.service" in c) == 1


def test_a_unit_gives_systemd_20_seconds_to_stop_it_unless_set():
    assert "TimeoutStopSec=20" in s.render_systemd("x", DAEMON)["pack-x.service"].splitlines()
    assert "TimeoutStopSec=5" in s.render_systemd("x", {**DAEMON, "stop_timeout": 5})["pack-x.service"].splitlines()


def test_launchd_waits_as_long_as_systemd_before_killing_a_stopped_agent():
    assert plist(DAEMON)["ExitTimeOut"] == 20
    assert plist({**DAEMON, "stop_timeout": 25})["ExitTimeOut"] == 25


def test_services_run_without_progress_bars():
    env = s.environment({})
    assert (env["HF_HUB_DISABLE_PROGRESS_BARS"], env["TQDM_DISABLE"]) == ("1", "1")


def test_a_services_unit_carries_its_own_rust_log_level():
    svc = {"command": "continuum-gateway"}
    with cfg.override({"logging": {"level": "info", "per_service": ["continuum-gateway=debug"]}}):
        gateway = s.render_systemd("continuum-gateway", svc)
        broker = s.render_systemd("continuum-broker", svc)
    assert "RUST_LOG=debug" in "".join(gateway.values())
    assert "RUST_LOG=info" in "".join(broker.values())


# ---- crash loops: restart counts and each service's last error ----


def show(*blocks: str):
    """A fake systemctl that answers `show` with these blocks, one per unit, as systemd prints them."""

    def run(*cmd, check=False):
        out = "\n\n".join(blocks) + "\n" if "show" in cmd else ""
        return s.subprocess.CompletedProcess(cmd, 0, out, "")

    return run


def test_restarts_and_uptime_come_from_one_systemctl_call_in_any_property_order(systemd, tmp_path, monkeypatch):
    sent, _ = systemd
    for unit in ("pack-a.service", "pack-b.service"):
        (tmp_path / "units" / unit).write_text("")
    fake = show(
        "NRestarts=7\nActiveEnterTimestampMonotonic=100000000\nId=pack-a.service",  # older systemd: Id last
        "Id=pack-b.service\nNRestarts=0\nActiveEnterTimestampMonotonic=0",  # never active: no uptime
    )
    monkeypatch.setattr(s, "_run", lambda *cmd, check=False: sent.append(cmd) or fake(*cmd))
    monkeypatch.setattr(s.time, "monotonic", lambda: 160.0)
    assert s.runs() == {
        "a": {"restarts": 7, "up_s": 60.0, "run_id": "systemd:7:100000000:"},
        "b": {"restarts": 0, "up_s": None, "run_id": "systemd:0:0:"},
    }
    assert len([c for c in sent if "show" in c]) == 1


def test_restarts_say_nothing_when_systemctl_fails(systemd, tmp_path, monkeypatch):
    (tmp_path / "units" / "pack-a.service").write_text("")
    fail = lambda *cmd, check=False: s.subprocess.CompletedProcess(cmd, 1, "", "Failed to connect to bus")
    monkeypatch.setattr(s, "_run", fail)
    assert s.runs() == {}


def test_scheduled_services_and_ones_never_restarted_have_no_count(systemd, tmp_path, monkeypatch):
    for unit in ("pack-a.service", "pack-sync.service", "pack-once.service"):
        (tmp_path / "units" / unit).write_text("")
    monkeypatch.setattr(s, "declared", lambda: {"a": DAEMON, "sync": SCHEDULED, "once": {**DAEMON, "restart": False}})
    blocks = [f"Id=pack-{n}.service\nNRestarts=4\nActiveEnterTimestampMonotonic=0" for n in ("a", "sync", "once")]
    monkeypatch.setattr(s, "_run", show(*blocks))
    assert set(s.runs()) == {"a"}  # a job's runs, or a run nobody restarts, are not a crash loop


def test_restarts_on_a_mac_are_launchd_s_runs_after_the_first(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "DARWIN", True)
    monkeypatch.setattr(s, "unit_dir", lambda: tmp_path)
    (tmp_path / "pack.listen.plist").write_text("")
    printed = "\tstate = running\n\tpid = 4242\n\truns = 4\n\tlast exit code = 1\n"
    monkeypatch.setattr(s, "_run", lambda *cmd, check=False: s.subprocess.CompletedProcess(cmd, 0, printed, ""))

    class Proc:
        def __init__(self, pid):
            assert pid == 4242

        def create_time(self):
            return 1000.0

    monkeypatch.setattr(psutil, "Process", Proc)
    monkeypatch.setattr(s.time, "time", lambda: 1090.0)
    assert s.runs() == {"listen": {"restarts": 3, "up_s": 90.0, "run_id": "launchd:4:4242:1000.0"}}


def write_log(tmp_path, name: str, text: str, written: datetime.datetime | None = None) -> None:
    """A service's log, last written at `written` (its mtime, which dates its lines) when given."""
    log = tmp_path / "logs" / f"{name}.log"
    log.parent.mkdir(exist_ok=True)
    log.write_text(text)
    if written:
        os.utime(log, (written.timestamp(), written.timestamp()))


def ts(day: int, clock: str) -> float:
    """Epoch seconds for a local time on a day in October 2026."""
    return datetime.datetime.fromisoformat(f"2026-10-{day:02d}T{clock}").timestamp()


def found(name: str, since: float | None = None) -> dict:
    """last_error's answer, which the test expects there to be."""
    error = s.last_error(name, since=since)
    assert error is not None
    return error


def test_last_error_is_the_newest_error_line_of_the_log(systemd, tmp_path):
    write_log(
        tmp_path,
        "listen",
        "09:00:00 [listen] INFO started\n"
        "09:01:00 [listen] ERROR the microphone vanished\n"
        "09:02:00 [listen] ERROR the model would not load\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "OSError: no such file\n"  # log.exception's traceback: its ERROR line already says what failed
        "09:03:00 [listen] WARN retrying\n",
        written=datetime.datetime(2026, 10, 30, 9, 3),
    )
    assert s.last_error("listen") == {"at": "09:02:00", "ts": ts(30, "09:02:00"), "text": "the model would not load"}


def test_an_unchanged_log_is_not_opened_again_and_a_written_one_is(systemd, tmp_path, monkeypatch):
    written = datetime.datetime(2026, 10, 30, 9, 3)
    write_log(tmp_path, "listen", "09:01:00 [listen] ERROR the microphone vanished\n", written=written)
    opened = []
    real_open = pathlib.Path.open

    def counting_open(self, *args, **kwargs):
        opened.append(self.name)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", counting_open)
    assert found("listen")["text"] == "the microphone vanished"
    assert found("listen")["text"] == "the microphone vanished"
    assert s.last_error("listen", since=ts(30, "09:02:00")) is None  # the kept error, judged against each `since`
    assert opened.count("listen.log") == 1
    with real_open(tmp_path / "logs" / "listen.log", "a") as f:
        f.write("09:02:00 [listen] ERROR the model would not load\n")
    os.utime(tmp_path / "logs" / "listen.log", (written.timestamp(), written.timestamp()))  # same time, longer
    assert found("listen")["text"] == "the model would not load"
    assert opened.count("listen.log") == 2


def test_last_error_is_a_crash_s_exception_when_python_died_with_no_error_line(systemd, tmp_path):
    write_log(
        tmp_path,
        "voice",
        "09:00:00 [voice] INFO started\n"
        "Traceback (most recent call last):\n"
        '  File "voice.py", line 9, in say\n'
        "    raise RuntimeError(why)\n"
        "RuntimeError: the voice clone has no reference\n",
        written=datetime.datetime(2026, 10, 30, 9, 0, 1),
    )
    assert s.last_error("voice") == {
        "at": "09:00:00",
        "ts": ts(30, "09:00:00"),
        "text": "RuntimeError: the voice clone has no reference",
    }


def test_last_error_reads_only_the_end_of_a_long_log(systemd, tmp_path):
    write_log(
        tmp_path, "see-faces", "09:00:00 [see-faces] ERROR long ago\n" + "09:00:01 [see-faces] INFO fine é\n" * 20_000
    )
    assert s.last_error("see-faces") is None  # the error is far before the tail read, which may start mid-character
    assert s.last_error("nothing-logged") is None


def test_an_error_from_before_the_service_s_current_start_is_left_out(systemd, tmp_path):
    # A running service kept showing a KeyError from a run that a fixed setting had since replaced.
    write_log(
        tmp_path,
        "listen",
        "09:01:00 [listen] ERROR unknown setting 'audio.mic_channel'\n09:05:00 [listen] INFO started\n",
        written=datetime.datetime(2026, 10, 30, 9, 6),
    )
    assert s.last_error("listen", since=ts(30, "09:04:59.5")) is None
    assert found("listen", since=ts(30, "09:01:00.5"))["text"] == "unknown setting 'audio.mic_channel'"


def test_a_crash_with_no_time_before_it_counts_as_before_the_current_start(systemd, tmp_path):
    write_log(
        tmp_path,
        "see-faces",
        'Traceback (most recent call last):\n  File "faces.py", line 69, in app\n'
        "ModuleNotFoundError: No module named 'insightface'\n09:45:29 [see-faces] INFO watching cam0.video\n",
        written=datetime.datetime(2026, 10, 30, 9, 46),
    )
    assert s.last_error("see-faces") == {
        "at": "",
        "ts": None,
        "text": "ModuleNotFoundError: No module named 'insightface'",
    }
    assert s.last_error("see-faces", since=ts(30, "09:45:00")) is None


def test_log_lines_from_before_midnight_are_dated_the_day_before(systemd, tmp_path):
    # Dated by its clock alone, 23:58 would be today's, after the 00:02 start, and shown.
    write_log(
        tmp_path,
        "listen",
        "23:58:00 [listen] ERROR yesterday's\n00:02:00 [listen] INFO started\n",
        written=datetime.datetime(2026, 10, 30, 0, 3),
    )
    assert found("listen")["ts"] == ts(29, "23:58:00")
    assert s.last_error("listen", since=ts(30, "00:01:59")) is None


def test_lines_a_little_out_of_order_stay_on_the_same_day(systemd, tmp_path):
    write_log(
        tmp_path,
        "listen",
        "10:00:05 [listen] ERROR a thread's line\n10:00:04 [listen] INFO another's\n",
        written=datetime.datetime(2026, 10, 30, 10, 0, 4),
    )
    assert found("listen")["ts"] == ts(30, "10:00:05")


@pytest.mark.parametrize(
    "state, run, since",
    [
        ("active", {"restarts": 0, "up_s": 50.0}, 950.0),  # running: errors since its start
        ("active", {"restarts": 2, "up_s": 900.0}, 400.0),  # restarted long ago and fine since
        ("active", {"restarts": 3, "up_s": 50.0}, None),  # just restarted by the manager: its crash is the news
        ("failed", {"restarts": 0, "up_s": 50.0}, None),  # down: its newest error says why
        ("active", None, 400.0),  # a healthy service with no reported uptime
    ],
)
def test_errors_count_from_a_running_service_s_start(monkeypatch, state, run, since):
    monkeypatch.setattr(s.time, "time", lambda: 1000.0)
    assert s._run_began(state, run) == since


# ---- stop now versus turn off: a stop leaves the unit installed, so it comes back on a restart ----


def test_stop_stops_the_units_but_leaves_them_installed_and_enabled(systemd, tmp_path, capsys):
    sent, _ = systemd
    for unit in ("pack-a.service", "pack-sync.service", "pack-sync.timer"):
        (tmp_path / "units" / unit).write_text("")
    s.stop(["a", "sync"], anywhere=True)
    assert ("systemctl", "--user", "stop", "pack-a.service", "pack-sync.timer", "pack-sync.service") in sent
    assert not [c for c in sent if "disable" in c or "daemon-reload" in c]
    assert sorted(p.name for p in (tmp_path / "units").iterdir()) == [
        "pack-a.service",
        "pack-sync.service",
        "pack-sync.timer",
    ]
    assert capsys.readouterr().out.splitlines() == ["pack-a.service: stopped", "pack-sync.timer: stopped"]


def test_start_starts_installed_units_and_names_one_that_fails(systemd, tmp_path, capsys):
    sent, fail = systemd
    for unit in ("pack-a.service", "pack-b.service"):
        (tmp_path / "units" / unit).write_text("")
    fail.add(("start", "pack-b.service"))
    s.start(["a", "b"], anywhere=True)
    assert ("systemctl", "--user", "start", "pack-a.service", "pack-b.service") in sent
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "pack-a.service: running"
    assert out[1].startswith("pack-b.service: FAILED")


def test_stop_and_start_refuse_a_service_that_isnt_installed(systemd, tmp_path):
    with pytest.raises(SystemExit, match="not installed: b"):
        s.stop(["b"], anywhere=True)
    with pytest.raises(SystemExit, match="not installed: b"):
        s.start(["b"], anywhere=True)


def test_stop_with_the_application_closed_and_no_units_says_to_open_it(systemd, monkeypatch):
    monkeypatch.setattr(s, "_supervised", lambda: None)
    with pytest.raises(SystemExit, match="pack-app") as raised:
        s.stop(["b"], anywhere=True)
    message = str(raised.value)
    assert "pack-services install" not in message
    assert "pack-supervisor run" in message
    with pytest.raises(SystemExit, match="pack-app") as started:
        s.start(["b"], anywhere=True)
    assert "pack-services install" not in str(started.value)


def test_stop_of_a_missing_unit_on_an_older_install_still_names_install(systemd, tmp_path, monkeypatch):
    monkeypatch.setattr(s, "_supervised", lambda: None)
    (tmp_path / "units" / "pack-a.service").write_text("")
    with pytest.raises(SystemExit, match="pack-services install"):
        s.stop(["b"], anywhere=True)


def test_list_says_the_star_is_the_selection(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pack-services", "list"])
    monkeypatch.setattr(s, "declared", lambda: {"journal": {"description": "Writes the journal."}})
    monkeypatch.setattr(s, "selected", lambda: {"journal": {}})
    s.main()
    out = capsys.readouterr().out
    assert "* = selected on this machine" in out
    assert "pack-services install" not in out


def test_the_service_manifest_and_the_vlm_probe_do_not_start_a_vision_unit():
    root = pathlib.Path(__file__).parents[1]
    opening = (root / "services/services.toml").read_text().split("\n# Per service:", 1)[0]
    assert "launchd" not in opening and "systemd" not in opening
    assert "pack-app" in opening and "pack-supervisor run" in opening
    probe = (root / "tools/vlm_probe/vlm_probe_quiet.sh").read_text()
    assert "pack.vision" not in probe and "launchctl" not in probe
    assert "pack-services stop vlm" in probe and "pack-services start vlm" in probe


def test_stop_on_a_mac_unloads_the_job_and_keeps_its_plist(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(s, "DARWIN", True)
    monkeypatch.setattr(s, "unit_dir", lambda: tmp_path)
    (tmp_path / "pack.listen.plist").write_text("")
    sent = []
    monkeypatch.setattr(
        s, "_run", lambda *cmd, check=False: sent.append(cmd) or s.subprocess.CompletedProcess(cmd, 0, "", "")
    )
    s.stop(["listen"], anywhere=True)
    assert sent == [("launchctl", "bootout", f"gui/{os.getuid()}/pack.listen")]
    assert (tmp_path / "pack.listen.plist").exists()  # loaded again at the next login
    monkeypatch.setattr(s.time, "sleep", lambda x: None)
    loaded = {"yes": False}

    def run(*cmd, check=False):
        sent.append(cmd)
        code = 0 if cmd[1] != "print" or loaded["yes"] else 113  # print: whether launchd has the job loaded
        return s.subprocess.CompletedProcess(cmd, code, "", "")

    monkeypatch.setattr(s, "_run", run)
    s.start(["listen"], anywhere=True)  # unloaded by the stop: loaded again
    assert sent[-1] == ("launchctl", "bootstrap", f"gui/{os.getuid()}", str(tmp_path / "pack.listen.plist"))
    loaded["yes"] = True
    s.start(["listen"], anywhere=True)  # still loaded (between KeepAlive restarts): a bootstrap would fail
    assert sent[-1] == ("launchctl", "kickstart", f"gui/{os.getuid()}/pack.listen")


def test_an_uninstall_that_fails_says_so(systemd, tmp_path, capsys):
    _, fail = systemd
    for unit in ("pack-a.service", "pack-b.service"):
        (tmp_path / "units" / unit).write_text("")
    fail.add(("stop", "pack-b.service"))
    s.uninstall(["a", "b"], anywhere=True)
    out = capsys.readouterr().out.splitlines()
    assert "removed a" in out
    assert any(line.startswith("pack-b: FAILED") for line in out)


def test_ollama_runs_under_the_log_relay_outermost_tagged_with_its_name():
    svc = s.declared()["ollama"]
    command = s.argv(svc, startup=True, name="ollama")
    relay = [sys.executable, "-m", "liberational.log_relay", "--tag", "ollama", "--"]
    assert command[: len(relay)] == relay
    assert "liberational.gpu_ready" in command[len(relay) :], "the GPU wait runs inside the relay"
    assert "log_relay" not in " ".join(s.argv(svc)), "only the service's own start is relayed"


def test_the_continuum_broker_gets_its_state_dir_and_capture_manifest():
    svc = s.declared()["continuum-broker"]
    assert s.argv(svc) == [
        str(cfg.program("continuum-broker")),
        "--state-dir",
        str(cfg.path("paths.state")),
        "--manifest",
        str(cfg.devices_file()),
    ]


def test_the_continuum_broker_restarts_only_for_the_state_dir_and_its_unit():
    from liberational import readers

    assert readers.readers()["continuum-broker"] == {"paths.state"} | readers.unit_keys()


def test_enable_disable_and_reset_edit_pack_toml_and_nothing_else(tmp_path, monkeypatch):
    # Over SSH, a switch turns services on and off through these, so pack.toml's comments and other keys survive.
    local = tmp_path / "pack.toml"
    local.write_text('# mine\n[paths]\nstate = "s"\n')
    monkeypatch.setenv("PACK_CONFIG", str(local))
    cfg.load.cache_clear()
    try:
        assert s.choose(["continuum-broker", "listen"], enable=True) == (["continuum-broker", "listen"], [])
        assert s.choose(["whisper-names"], enable=False) == (["continuum-broker", "listen"], ["whisper-names"])
        # Disabling one that's enabled moves it; reset takes names out of both lists.
        assert s.choose(["listen"], enable=False) == (["continuum-broker"], ["whisper-names", "listen"])
        assert s.choose(["continuum-broker", "whisper-names", "listen"], enable=None) == ([], [])
        text = local.read_text()
        assert "# mine" in text
        assert 'state = "s"' in text
        with pytest.raises(SystemExit, match="not a declared service"):
            s.choose(["nope"], enable=True)
    finally:
        cfg.load.cache_clear()


@pytest.mark.parametrize(
    ("state", "run", "since"),
    [
        ("active", {"restarts": 0, "up_s": 3600.0}, 400.0),
        ("running", None, 400.0),
        ("active", {"restarts": 1, "up_s": 50.0}, 950.0),
        ("active", {"restarts": 4, "up_s": 50.0}, None),
        ("failed", {"restarts": 4, "up_s": 3600.0}, None),
    ],
)
def test_healthy_service_errors_expire_but_active_failures_keep_their_reason(monkeypatch, state, run, since):
    monkeypatch.setattr(s.time, "time", lambda: 1000.0)
    assert s._run_began(state, run) == since


def test_healthy_service_notice_expires_from_error_time_without_a_restart(systemd, tmp_path, monkeypatch):
    write_log(
        tmp_path,
        "listen",
        "09:01:00 [listen] ERROR temporary failure\n09:02:00 [listen] INFO recovered\n",
        written=datetime.datetime(2026, 10, 30, 9, 2),
    )
    monkeypatch.setattr(s.time, "time", lambda: ts(30, "09:20:00"))
    run = {"restarts": 0, "up_s": 3600.0}
    assert s.last_error("listen", since=s._run_began("active", run)) is None
    assert found("listen", since=s._run_began("failed", run))["text"] == "temporary failure"


@pytest.fixture(params=[datetime.datetime(2024, 2, 3, 9, 2, 1), datetime.datetime(2032, 11, 29, 9, 2, 1)])
def occurrence_clock(request, monkeypatch) -> datetime.datetime:
    now = request.param
    timestamp = now.timestamp()
    monkeypatch.setattr(s.time, "time", lambda: timestamp)
    return now


def test_same_error_in_same_second_has_a_new_occurrence_but_unrelated_writes_do_not(
    systemd, tmp_path, occurrence_clock
):
    text = "09:02:00 [listen] ERROR temporary failure\n"
    write_log(tmp_path, "listen", text, written=occurrence_clock.replace(second=0))
    first = s.last_error("listen", include_id=True)
    assert first is not None and first["id"]
    path = tmp_path / "logs" / "listen.log"
    with path.open("a") as f:
        f.write("09:02:01 [listen] INFO retrying\n")
    os.utime(path, (occurrence_clock.timestamp(), occurrence_clock.timestamp()))
    second = s.last_error("listen", include_id=True)
    assert second is not None and second["id"] == first["id"]
    with path.open("a") as f:
        f.write(text)
    os.utime(path, (occurrence_clock.timestamp(), occurrence_clock.timestamp()))
    third = s.last_error("listen", include_id=True)
    assert third is not None and third["text"] == first["text"] and third["ts"] == first["ts"]
    assert third["id"] != first["id"]


def test_log_occurrence_survives_tail_window_changes_but_not_rotation(systemd, tmp_path, occurrence_clock):
    text = "09:02:00 [listen] ERROR temporary failure\n"
    write_log(tmp_path, "listen", "x" * 100_000 + "\n" + text, written=occurrence_clock.replace(second=0))
    first = s.last_error("listen", include_id=True)
    path = tmp_path / "logs" / "listen.log"
    with path.open("a") as f:
        f.write("09:02:01 [listen] INFO unchanged é\n")
    os.utime(path, (occurrence_clock.timestamp(), occurrence_clock.timestamp()))
    after = s.last_error("listen", include_id=True)
    assert first is not None and after is not None and after["id"] == first["id"]
    path.rename(path.with_suffix(".old"))
    write_log(tmp_path, "listen", text, written=occurrence_clock.replace(second=0))
    rotated = s.last_error("listen", include_id=True)
    assert rotated is not None and rotated["id"] != first["id"]


def test_mandatory_broker_clients_wait_before_exec_on_both_platforms():
    service = {"command": "pack-fixture", "args": ["--flag", "two words"], "wait_for_broker": True}
    plain = s.argv(service)
    launched = plist(service)["ProgramArguments"]
    assert launched[:3] == [sys.executable, "-m", "liberational.continuum_ready"]
    assert launched[launched.index("--") + 1 :] == plain
    unit = s.render_systemd("fixture", service)["pack-fixture.service"]
    start = next(line.removeprefix("ExecStart=:") for line in unit.splitlines() if line.startswith("ExecStart=:"))
    args = shlex.split(start)
    assert args[6:] == launched


def test_gpu_services_wait_for_the_driver_before_exec():
    service = {"command": "ollama", "args": ["serve"], "wait_for_gpu": True}
    plain = s.argv(service)
    launched = plist(service)["ProgramArguments"]
    assert launched[:4] == [sys.executable, "-m", "liberational.gpu_ready", "--"]
    assert launched[4:] == plain
    assert "liberational.gpu_ready" not in plist({"command": "ollama", "args": ["serve"]})["ProgramArguments"]


def test_see_actions_waits_for_the_driver_at_boot():
    # Its models need the GPU and it has no useful CPU fallback, so a boot whose driver comes up late waits
    assert s.declared()["see-actions"].get("wait_for_gpu")


def test_the_shared_ollama_is_a_pack_service_pinned_by_config():
    svc = s.declared()["ollama"]
    assert svc.get("wait_for_gpu") and svc.get("default")
    with cfg.override({"ollama": {"gpu": "GPU-aaaa", "models": "/srv/models"}}):
        env = s.environment(svc)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-aaaa" and env["OLLAMA_MODELS"] == "/srv/models"
    with cfg.override({"ollama": {"gpu": "", "models": ""}}):
        env = s.environment(svc)
    assert "CUDA_VISIBLE_DEVICES" not in env and "OLLAMA_MODELS" not in env


def test_the_shared_ollama_takes_its_server_tuning_from_config():
    svc = s.declared()["ollama"]
    env = s.environment(svc)
    # Defaults: Ollama's own keep-alive (left unset), flash attention on, the stock cache, Ollama's own model count.
    assert "OLLAMA_KEEP_ALIVE" not in env
    assert (env["OLLAMA_FLASH_ATTENTION"], env["OLLAMA_KV_CACHE_TYPE"], env["OLLAMA_MAX_LOADED_MODELS"]) == (
        "True",
        "f16",
        "0",
    )
    tuned = {"keep_alive": "-1", "flash_attention": False, "kv_cache_type": "q8_0", "max_loaded_models": 2}
    with cfg.override({"ollama": tuned}):
        env = s.environment(svc)
    assert (
        env["OLLAMA_KEEP_ALIVE"],
        env["OLLAMA_FLASH_ATTENTION"],
        env["OLLAMA_KV_CACHE_TYPE"],
        env["OLLAMA_MAX_LOADED_MODELS"],
    ) == ("-1", "False", "q8_0", "2")


def test_offline_dashboard_and_gateway_do_not_wait_for_the_broker():
    for name in ("dashboard", "continuum-gateway", "continuum-broker"):
        svc = s.declared()[name]
        assert "liberational.continuum_ready" not in plist(svc)["ProgramArguments"]
    for name in ("listen", "continuum-capture", "continuum-tier-worker", "continuum-vector-worker", "see-actions"):
        assert "liberational.continuum_ready" in plist(s.declared()[name])["ProgramArguments"]


def test_tier_settings_reach_both_service_managers(monkeypatch):
    execute = create_autospec(os.execve)
    monkeypatch.setattr(s.os, "execve", execute)
    monkeypatch.setattr(s.os, "chdir", create_autospec(os.chdir))
    with cfg.override({"paths": {"state": "/srv/pack/state-a"}}):
        service = s.declared()["continuum-tier-worker"]
        launch = plist(service)["ProgramArguments"]
        unit = s.render_systemd("continuum-tier-worker", service)["pack-continuum-tier-worker.service"]
        s.exec_service("continuum-tier-worker")
    command = next(line.removeprefix("ExecStart=:") for line in unit.splitlines() if line.startswith("ExecStart=:"))
    expected = ["--state-dir", "/srv/pack/state-a", "--manifest", str(cfg.devices_file())]
    launcher = [sys.executable, "-m", "liberational.services", "exec", "continuum-tier-worker"]
    assert launch[-len(launcher) :] == launcher
    assert shlex.split(command)[-len(launcher) :] == launcher
    assert execute.call_args.args[1][-len(expected) :] == expected


def test_installed_tier_launcher_reads_settings_changes_on_each_start_without_rerendering(tmp_path, monkeypatch):
    local = tmp_path / "pack.toml"
    local.write_text('[paths]\nstate = "/srv/pack/state-a"\n')
    monkeypatch.setenv("PACK_CONFIG", str(local))
    cfg.load.cache_clear()
    execute = create_autospec(os.execve)
    directory = create_autospec(os.chdir)
    monkeypatch.setattr(s.os, "execve", execute)
    monkeypatch.setattr(s.os, "chdir", directory)
    try:
        # This is the command saved in the installed unit. Restarts run these same arguments, even after
        # Settings writes pack.toml. Prime the old config cache to exercise the launcher's fresh read too.
        installed = plist(s.declared()["continuum-tier-worker"])["ProgramArguments"]
        assert cfg.get("paths.state") == "/srv/pack/state-a"
        launcher = installed[installed.index("--") + 1 :]
        assert launcher[1:4] == ["-m", "liberational.services", "exec"]
        monkeypatch.setattr(s.sys, "argv", ["services", *launcher[3:]])
        local.write_text('[paths]\nstate = "/srv/pack/state-b"\n')
        s.main()
        first = execute.call_args.args[1]
        assert first[0] == str(cfg.program("continuum-tier-worker"))
        assert first[first.index("--state-dir") + 1] == "/srv/pack/state-b"
        assert "liberational.services" not in first, "the startup launcher must exec the worker without recursion"
        local.write_text('[paths]\nstate = "/srv/pack/state-c"\n')
        s.main()
        second = execute.call_args.args[1]
        assert second[second.index("--state-dir") + 1] == "/srv/pack/state-c"
        directory.assert_called_with(cfg.workdir())
    finally:
        cfg.load.cache_clear()


def test_the_tier_launcher_fills_the_av1_archive_settings_in_at_each_start(monkeypatch):
    """The archive's settings go by environment (an older worker ignores them), filled in at start, so a
    Settings change reaches the worker on its next restart without reinstalling the unit."""
    execute = create_autospec(os.execve)
    monkeypatch.setattr(s.os, "execve", execute)
    monkeypatch.setattr(s.os, "chdir", create_autospec(os.chdir))
    with cfg.override({"video_archive": {"av1": True, "after_hours": 6}, "video_window": {"days": 14}}):
        s.exec_service("continuum-tier-worker")
    env = execute.call_args.args[2]
    assert (env["CONTINUUM_AV1_ARCHIVE"], env["CONTINUUM_AV1_AFTER_HOURS"]) == ("True", "6")
    assert (env["CONTINUUM_AV1_CRF"], env["CONTINUUM_VIDEO_WINDOW_DAYS"]) == ("52", "14")
    assert env["SVT_LOG"] == "1", "SVT-AV1's per-chunk banner stays out of the log"
    with cfg.override({"video_archive": {"av1": False}}):
        s.exec_service("continuum-tier-worker")
    assert execute.call_args.args[2]["CONTINUUM_AV1_ARCHIVE"] == "False"


@pytest.mark.parametrize("name", ["unknown-service", "continuum-tier-worker"])
def test_fresh_service_launcher_rejects_unknown_or_disabled_services(name, monkeypatch):
    execute = create_autospec(os.execve)
    monkeypatch.setattr(s.os, "execve", execute)
    with (
        cfg.override({"services": {"disable": ["continuum-tier-worker"]}}),
        pytest.raises(SystemExit, match="cannot launch unselected or unknown service"),
    ):
        s.exec_service(name)
    execute.assert_not_called()


def test_fresh_service_launcher_reports_exec_failure(monkeypatch):
    execute = create_autospec(os.execve, side_effect=FileNotFoundError("worker executable is missing"))
    monkeypatch.setattr(s.os, "execve", execute)
    monkeypatch.setattr(s.os, "chdir", create_autospec(os.chdir))
    with pytest.raises(SystemExit, match="cannot start continuum-tier-worker: worker executable is missing"):
        s.exec_service("continuum-tier-worker")


def test_last_error_reads_dated_stamps_as_written_and_old_ones_back_from_them(systemd, tmp_path):
    # An earlier time-only log continued in dated stamps; mtime is far later, so only the dated line
    # can place the old ones: the one before 23:59 is the day before the dated 00:01 line.
    write_log(
        tmp_path,
        "listen",
        "23:58:00 [listen] ERROR the microphone vanished\n"
        "2026-10-21T00:01:00-07:00 [listen] INFO started\n"
        "2026-10-21T00:02:00-07:00 [listen] WARN retrying\n",
        written=datetime.datetime(2026, 10, 30, 9, 3),
    )
    error = found("listen")
    assert error["at"] == "23:58:00"
    assert error["ts"] == datetime.datetime.fromisoformat("2026-10-20T23:58:00-07:00").timestamp()


def test_last_error_of_a_dated_line_keeps_its_time_and_says_the_clock_time(systemd, tmp_path):
    write_log(
        tmp_path,
        "listen",
        "2026-10-21T09:00:00-07:00 [listen] INFO started\n"
        "2026-10-21T09:01:00-07:00 [listen] ERROR the model would not load\n"
        "2026-10-30T09:00:00-07:00 capture worker kitchen: reconnecting\n",  # stamped free text, as capture writes
        written=datetime.datetime(2026, 10, 30, 9, 3),
    )
    error = found("listen")
    assert (error["at"], error["text"]) == ("09:01:00", "the model would not load")
    assert error["ts"] == datetime.datetime.fromisoformat("2026-10-21T09:01:00-07:00").timestamp()


STARTED = "39f53479d3a045ac8e11786248231fbf"
STOPPED = "9d1aaa27d60140bd96365438aad20286"
SUCCEEDED = "7ad2d189f7e94e70a38c781354912448"
RESTART = "5eb03494b6584870a536b337290809b3"
CRASHED = "d9b373ed55a64feb8242e02dbe79a49c"
EXITED = "98e322203f7a4ed290d09fe03c09fe15"


def lifecycle(at, kind, **extra):
    return {
        "USER_UNIT": "pack-listen.service",
        "__REALTIME_TIMESTAMP": str(int(at * 1e6)),
        "MESSAGE_ID": kind,
        "_BOOT_ID": "boot-a",
        "USER_INVOCATION_ID": "run-a",
        **extra,
    }


def test_service_history_keeps_runs_before_the_current_lifetime_and_pairs_exit_reason():
    view = s._journal_view(
        [
            lifecycle(90, STARTED, USER_INVOCATION_ID="run-b"),
            lifecycle(60, CRASHED, UNIT_RESULT="exit-code"),
            lifecycle(59, EXITED, EXIT_CODE="exited", EXIT_STATUS="7"),
            lifecycle(65, RESTART),
            lifecycle(20, STARTED),
        ],
        ["listen"],
        0,
        100,
        "boot-a",
    )
    row = view["services"][0]
    assert row["spans"] == [
        {"start": 0, "end": 20, "state": "unknown"},
        {"start": 20, "end": 60, "state": "up"},
        {"start": 60, "end": 90, "state": "stopped"},
        {"start": 90, "end": 100, "state": "up"},
    ]
    assert row["events"][1] == {"at": 60, "kind": "crashed", "reason": "exit-code: exited, status 7"}
    assert [event["kind"] for event in row["events"]] == ["started", "crashed", "restart", "started"]


def test_service_history_empty_window_is_unknown_and_ignores_unrelated_or_outside_events():
    view = s._journal_view(
        [
            lifecycle(-1, "irrelevant"),
            lifecycle(101, STOPPED),
            lifecycle(50, STARTED, USER_UNIT="other.service"),
            lifecycle(50, "application-message", MESSAGE="never returned"),
        ],
        ["listen"],
        0,
        100,
        "boot-a",
    )
    assert view["services"] == [
        {"name": "listen", "spans": [{"start": 0, "end": 100, "state": "unknown"}], "events": []}
    ]
    assert "never returned" not in str(view)


def test_service_history_does_not_bridge_a_boot_gap_or_extend_a_previous_boot_to_now():
    view = s._journal_view(
        [
            lifecycle(10, STARTED),
            lifecycle(20, STOPPED),
            lifecycle(50, STARTED, _BOOT_ID="boot-b", USER_INVOCATION_ID="run-b"),
        ],
        ["listen"],
        0,
        100,
        "boot-c",
    )
    assert view["services"][0]["spans"] == [
        {"start": 0, "end": 10, "state": "unknown"},
        {"start": 10, "end": 20, "state": "up"},
        {"start": 20, "end": 100, "state": "unknown"},
    ]


def test_failed_unit_and_start_job_count_once_and_never_use_restart_counters():
    view = s._journal_view(
        [
            lifecycle(20, CRASHED, UNIT_RESULT="exit-code", NRestarts="9"),
            lifecycle(21, "be02cf6855d2428ba40df7e9d022f03d", JOB_RESULT="failed", NRestarts="0"),
            lifecycle(50, STARTED, USER_INVOCATION_ID="run-b", NRestarts="0"),
        ],
        ["listen"],
        0,
        100,
        "boot-a",
    )
    assert [event["kind"] for event in view["services"][0]["events"]] == ["crashed", "started"]


def test_history_reader_limits_lifecycle_fields_time_rows_and_uses_the_declared_units(monkeypatch):
    run = create_autospec(subprocess.run, return_value=subprocess.CompletedProcess([], 0, b"", b""))
    monkeypatch.setattr(s.subprocess, "run", run)
    monkeypatch.setattr(s.shutil, "which", lambda name: "/usr/bin/journalctl")
    s._read_journal(["listen"], 10, 100)
    args = run.call_args_list[0].args[0]
    assert args[:3] == ["/usr/bin/journalctl", "--user", "--no-pager"]
    assert args[args.index("-u") + 1] == "pack-listen.service"
    assert "--since=@10.000000" in args and "--until=@100.000000" in args
    assert "-n" in args and int(args[args.index("-n") + 1]) <= 5000
    fields = next(arg.split("=", 1)[1].split(",") for arg in args if arg.startswith("--output-fields="))
    assert "_COMM=systemd" in args
    assert "MESSAGE" not in fields and "COMMAND" not in fields
    assert {"USER_UNIT", "__REALTIME_TIMESTAMP", "EXIT_CODE", "EXIT_STATUS"} <= set(fields)
    assert run.call_args.kwargs["timeout"] == 5


def test_service_history_is_cached_and_expired_reads_refresh_without_resetting_the_data(tmp_path, monkeypatch):
    times = iter([10.0, 11.0, 50.0])
    monkeypatch.setattr(s.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(s.time, "time", lambda: 100000.0)
    monkeypatch.setattr(s, "DARWIN", False)
    monkeypatch.setattr(s, "_history_cache", {})
    monkeypatch.setattr(s, "declared", lambda: {"listen": {}, "nightly": {"schedule": "01:00"}})
    read = create_autospec(
        s._read_journal, side_effect=lambda names, start, end: s._journal_view([], names, start, end, "boot-a")
    )
    monkeypatch.setattr(s, "_read_journal", read)
    first = s.history(tmp_path, 24)
    assert s.history(tmp_path, 24) is first
    assert s.history(tmp_path, 24) == first
    assert read.call_count == 2
    assert read.call_args.args[0] == ("listen",)


@pytest.mark.parametrize("hours", [0, 1, 169, -24])
def test_service_history_refuses_unbounded_or_unsupported_windows(tmp_path, hours):
    with pytest.raises(ValueError, match="24 hours or 7 days"):
        s.history(tmp_path, hours)


def test_prior_lifecycle_anchor_proves_a_long_running_service_without_inventing_restart_events():
    view = s._journal_view([lifecycle(-50, STARTED)], ["listen"], 0, 100, "boot-a")
    assert view["services"][0] == {"name": "listen", "spans": [{"start": 0, "end": 100, "state": "up"}], "events": []}


def test_exit_reason_without_an_invocation_never_leaks_into_a_later_crash():
    view = s._journal_view(
        [
            lifecycle(5, EXITED, USER_INVOCATION_ID="", EXIT_CODE="exited", EXIT_STATUS="0"),
            lifecycle(50, CRASHED, USER_INVOCATION_ID="", UNIT_RESULT="timeout"),
        ],
        ["listen"],
        0,
        100,
        "boot-a",
    )
    assert view["services"][0]["events"] == [{"at": 50, "kind": "crashed", "reason": "timeout"}]


def test_truncated_window_discards_prior_anchor_and_keeps_omitted_time_unknown(monkeypatch):
    raw = b"\n".join(
        json.dumps(row).encode() for row in [lifecycle(1, STARTED), lifecycle(2, STOPPED), lifecycle(3, STARTED)]
    )
    run = create_autospec(subprocess.run, return_value=subprocess.CompletedProcess([], 0, raw, b""))
    monkeypatch.setattr(s.subprocess, "run", run)
    monkeypatch.setattr(s.shutil, "which", lambda name: "/usr/bin/journalctl")
    monkeypatch.setattr(s, "HISTORY_RECORDS", 2)
    view = s._read_journal(["listen"], 0, 100)
    assert view["truncated"]
    assert view["services"][0]["spans"][0] == {"start": 0, "end": 2, "state": "unknown"}
    assert len(view["services"][0]["events"]) == 2
    assert run.call_count == 1  # no anchor may bridge missing in-window transitions


def test_history_reader_refuses_an_oversized_metadata_reply(monkeypatch):
    run = create_autospec(subprocess.run, return_value=subprocess.CompletedProcess([], 0, b"oversized", b""))
    monkeypatch.setattr(s.subprocess, "run", run)
    monkeypatch.setattr(s.shutil, "which", lambda name: "/usr/bin/journalctl")
    monkeypatch.setattr(s, "HISTORY_BYTES", 4)
    with pytest.raises(ValueError, match="bounded reply"):
        s._read_journal(["listen"], 0, 100)


def test_reader_uses_a_bounded_prior_transition_without_turning_it_into_a_restart(monkeypatch):
    anchor = json.dumps(lifecycle(-50, STARTED, _BOOT_ID="boota")).encode()
    run = create_autospec(
        subprocess.run,
        side_effect=[subprocess.CompletedProcess([], 0, b"", b""), subprocess.CompletedProcess([], 0, anchor, b"")],
    )
    monkeypatch.setattr(s.subprocess, "run", run)
    monkeypatch.setattr(s.shutil, "which", lambda name: "/usr/bin/journalctl")
    monkeypatch.setattr(pathlib.Path, "read_text", create_autospec(pathlib.Path.read_text, return_value="boot-a"))
    view = s._read_journal(["listen"], 0, 100)
    assert view["services"][0] == {"name": "listen", "spans": [{"start": 0, "end": 100, "state": "up"}], "events": []}
    prior = run.call_args_list[1].args[0]
    assert not any(arg.startswith("--since=") for arg in prior)
    assert "--until=@-0.000001" in prior and "-n" in prior


@pytest.mark.parametrize(
    "start,stopped_start", [(0, 5), (5.5, 5.5)], ids=["clean-exit-in-window", "clean-exit-prior-anchor"]
)
def test_successful_daemon_exit_closes_running_time_without_a_crash_or_restart(start, stopped_start):
    view = s._journal_view(
        [lifecycle(1, STARTED), lifecycle(5, SUCCEEDED, UNIT_RESULT="success")], ["listen"], start, 6, "boot-a"
    )
    row = view["services"][0]
    assert row["spans"][-1] == {"start": stopped_start, "end": 6, "state": "stopped"}
    assert not any(event["kind"] in {"crashed", "restart"} for event in row["events"])
    assert all(span["end"] <= 5 for span in row["spans"] if span["state"] == "up")


@pytest.mark.parametrize(
    "first_start,first_invocation,next_invocation",
    [(10, "run-a", "run-b"), (-20, "run-a", "run-b"), (10, "", "")],
    ids=["in-window-start", "prior-window-anchor", "missing-invocation-ids"],
)
def test_new_invocation_without_a_recorded_prior_end_leaves_the_intervening_time_unknown(
    first_start, first_invocation, next_invocation
):
    view = s._journal_view(
        [
            lifecycle(first_start, STARTED, USER_INVOCATION_ID=first_invocation),
            lifecycle(90, STARTED, USER_INVOCATION_ID=next_invocation),
        ],
        ["listen"],
        0,
        100,
        "boot-a",
    )
    assert view["services"][0]["spans"] == [
        {"start": 0, "end": 90, "state": "unknown"},
        {"start": 90, "end": 100, "state": "up"},
    ]


def test_duplicate_start_for_the_same_invocation_preserves_continuous_observed_uptime():
    view = s._journal_view([lifecycle(10, STARTED), lifecycle(90, STARTED)], ["listen"], 0, 100, "boot-a")
    row = view["services"][0]
    assert row["spans"] == [
        {"start": 0, "end": 10, "state": "unknown"},
        {"start": 10, "end": 100, "state": "up"},
    ]
    assert not any(event["kind"] in {"crashed", "restart"} for event in row["events"])


@pytest.mark.parametrize("result", ["dependency", "start-limit-hit"])
def test_failed_start_job_alone_is_a_marker_not_a_crash_or_proof_of_stopped_state(result):
    view = s._journal_view(
        [lifecycle(5, "be02cf6855d2428ba40df7e9d022f03d", JOB_RESULT=result)], ["listen"], 0, 6, "boot-a"
    )
    assert view["services"][0] == {
        "name": "listen",
        "spans": [{"start": 0, "end": 6, "state": "unknown"}],
        "events": [{"at": 5, "kind": "start-failed", "reason": result}],
    }


@pytest.mark.parametrize(
    "start,expected",
    [
        (0, [{"start": 0, "end": 1, "state": "unknown"}, {"start": 1, "end": 6, "state": "up"}]),
        (5.5, [{"start": 5.5, "end": 6, "state": "up"}]),
    ],
    ids=["in-window-marker", "prior-window-marker"],
)
def test_failed_start_job_neither_closes_a_running_unit_nor_replaces_its_anchor(start, expected):
    view = s._journal_view(
        [
            lifecycle(1, STARTED),
            lifecycle(5, "be02cf6855d2428ba40df7e9d022f03d", JOB_RESULT="dependency", USER_INVOCATION_ID="run-b"),
        ],
        ["listen"],
        start,
        6,
        "boot-a",
    )
    assert view["services"][0]["spans"] == expected


@pytest.mark.parametrize("job_at", [4, 6], ids=["job-before-unit-failure", "job-after-unit-failure"])
def test_real_unit_failure_supersedes_same_invocation_start_job_failure_in_either_order(job_at):
    view = s._journal_view(
        [
            lifecycle(1, STARTED),
            lifecycle(job_at, "be02cf6855d2428ba40df7e9d022f03d", JOB_RESULT="failed"),
            lifecycle(5, CRASHED, UNIT_RESULT="exit-code"),
        ],
        ["listen"],
        0,
        8,
        "boot-a",
    )
    assert view["services"][0]["events"] == [
        {"at": 1, "kind": "started", "reason": None},
        {"at": 5, "kind": "crashed", "reason": "exit-code"},
    ]
    assert view["services"][0]["spans"][-1] == {"start": 5, "end": 8, "state": "stopped"}


@pytest.mark.parametrize(
    "first_start,terminal", [(10, CRASHED), (-20, STOPPED)], ids=["crash-new-owner", "stopped-new-owner-after-anchor"]
)
def test_stateful_transition_owned_by_a_different_invocation_leaves_unobserved_running_gap_unknown(
    first_start, terminal
):
    view = s._journal_view(
        [
            lifecycle(first_start, STARTED),
            lifecycle(90, terminal, USER_INVOCATION_ID="run-b"),
        ],
        ["listen"],
        0,
        100,
        "boot-a",
    )
    assert view["services"][0]["spans"] == [
        {"start": 0, "end": 90, "state": "unknown"},
        {"start": 90, "end": 100, "state": "stopped"},
    ]


def test_last_error_still_reads_right_after_its_log_is_trimmed(systemd, tmp_path):
    write_log(
        tmp_path,
        "listen",
        "09:00:00 [listen] INFO started\n09:01:00 [listen] ERROR the microphone vanished\n",
        written=datetime.datetime(2026, 10, 30, 9, 2),
    )
    log = tmp_path / "logs" / "listen.log"
    assert logs.rotate(log, cap=50) > 0
    with log.open("a") as f:
        f.write("09:02:00 [listen] INFO retrying\n")
    os.utime(log, (ts(30, "09:02:00"), ts(30, "09:02:00")))
    assert found("listen") == {"at": "09:01:00", "ts": ts(30, "09:01:00"), "text": "the microphone vanished"}


def test_install_removes_the_logs_of_services_no_longer_declared(systemd, tmp_path):
    folder = tmp_path / "logs"
    folder.mkdir()
    for name in ("listen.log", "listen.log.1", "cascade.log", "frontend.log", "see-faces.log", "see-faces.log.1"):
        (folder / name).write_text("x\n")
    (folder / "profiles").mkdir()
    s.install(["a"], anywhere=True)
    assert sorted(p.name for p in folder.iterdir()) == [
        "cascade.log",
        "frontend.log",
        "listen.log",
        "listen.log.1",
        "profiles",
    ]


# ---- a supervised service's environment: the session's variables, and nothing else of the launcher's (#183) ----

# The application opened from a terminal inside an AppImage-based program.
LAUNCHER = {
    "PATH": "/tmp/.mount_editorAbC/usr/bin:/opt/editor/bin:/opt/tool/bin:/usr/bin",
    "HOME": "/somewhere/else",
    "LD_LIBRARY_PATH": "/tmp/.mount_editorAbC/usr/lib",
    "LD_PRELOAD": "/tmp/.mount_editorAbC/usr/lib/hook.so",
    "APPDIR": "/opt/editor",
    "APPIMAGE": "/opt/Editor.AppImage",
    "ARGV0": "editor",
    "OWD": "/home",
    "FLATPAK_ID": "org.example.Editor",
    "SNAP_NAME": "editor",
    "EDITOR_SESSION": "7",
    "PYTHONPATH": "/opt/editor/python",
    "XDG_RUNTIME_DIR": "/run/user/4242",
    "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/4242/bus",
    "WAYLAND_DISPLAY": "wayland-0",
    "DISPLAY": ":0",
    "XDG_SESSION_TYPE": "wayland",
    "LOGNAME": "alice",
    "USER": "alice",
    "LANG": "en_US.UTF-8",
    "LC_TIME": "en_GB.UTF-8",
}


def test_a_supervised_service_gets_the_session_and_nothing_else_from_the_launcher(monkeypatch):
    monkeypatch.setattr(s, "DARWIN", False)
    env = s.environment({}, "listen", launcher=LAUNCHER)
    session = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "WAYLAND_DISPLAY", "DISPLAY", "XDG_SESSION_TYPE")
    session += ("USER", "LOGNAME", "LANG", "LC_TIME")
    assert {k: env[k] for k in session} == {k: LAUNCHER[k] for k in session}
    own = {"PATH", "HOME", "RUST_LOG", "HF_HUB_DISABLE_PROGRESS_BARS", "TQDM_DISABLE", "TRANSFORMERS_VERBOSITY"}
    assert set(env) == set(session) | own | {"OPENCV_FFMPEG_LOGLEVEL", "OPENCV_LOG_LEVEL"}
    assert env["HOME"] == str(pathlib.Path.home())
    # the launcher's mount and its own tree are out of PATH; a tool directory of the owner's stays
    assert env["PATH"].split(os.pathsep)[-5:] == ["/opt/tool/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    assert "editor" not in env["PATH"]


def test_a_service_keeps_a_loader_variable_its_declaration_sets_on_purpose():
    env = s.environment({"env": {"LD_LIBRARY_PATH": "/opt/vendor/lib"}}, launcher=LAUNCHER)
    assert env["LD_LIBRARY_PATH"] == "/opt/vendor/lib"


def test_what_was_dropped_names_the_launchers_variables_and_its_path_entries():
    assert s.dropped(LAUNCHER) == [
        "APPDIR",
        "APPIMAGE",
        "ARGV0",
        "FLATPAK_ID",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "OWD",
        "SNAP_NAME",
        "PATH entry /tmp/.mount_editorAbC/usr/bin",
        "PATH entry /opt/editor/bin",
    ]
    assert s.dropped({"PATH": "/usr/bin", "XDG_RUNTIME_DIR": "/run/user/4242"}) == []


def test_a_unit_file_carries_no_session_variables(monkeypatch):
    # The desktop's service manager supplies them to a unit; written into the file they would go stale.
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/4242")
    monkeypatch.setenv("DISPLAY", ":0")
    assert not {"XDG_RUNTIME_DIR", "DISPLAY"} & set(s.environment({}))
