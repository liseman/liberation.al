"""Moving an install off the per-service units (#165): the plan, the dry run, and carrying it out, with a stand-in
service manager and a temporary unit directory."""

import subprocess

import pytest

from liberational import unit_migration
from liberational.service_cascade import BROKER, CAPTURE, TIER

NAMES = ["listen", "journal", "dashboard", "continuum-gateway", BROKER, CAPTURE, TIER, "video-window"]


class Manager:
    """Records every command; loginctl says whether linger is on; a command with a word in `fail` fails."""

    def __init__(self, linger="yes", fail=()):
        self.commands, self.linger, self.fail = [], linger, fail

    def __call__(self, command):
        self.commands.append(list(command))
        if command[:2] == ["loginctl", "show-user"]:
            return subprocess.CompletedProcess(command, 0, f"Linger={self.linger}\n", "")
        bad = any(word in command for word in self.fail)
        return subprocess.CompletedProcess(command, 1 if bad else 0, "", "unit is masked" if bad else "")

    @property
    def changes(self):
        return [c for c in self.commands if c[:2] != ["loginctl", "show-user"]]


@pytest.fixture
def units(tmp_path, monkeypatch):
    """A Linux user's unit directory with the pack's units, a job's timer, an override directory, and someone
    else's unit."""
    directory = tmp_path / "systemd/user"
    directory.mkdir(parents=True)
    for name in NAMES:
        (directory / f"pack-{name}.service").write_text("[Service]\n")
    (directory / "pack-video-window.timer").write_text("[Timer]\n")
    (directory / "pack-listen.service.d").mkdir()
    (directory / "pack-listen.service.d/override.conf").write_text("[Service]\n")
    (directory / "other.service").write_text("[Service]\n")
    monkeypatch.setattr(unit_migration, "systemd_dir", lambda: directory)
    monkeypatch.setattr(unit_migration.sys, "platform", "linux")
    return directory


@pytest.fixture
def manager(monkeypatch):
    made = Manager()
    monkeypatch.setattr(unit_migration, "_run", made)

    def no_broker(*args):
        raise RuntimeError("no broker in a test")

    monkeypatch.setattr(unit_migration.service_cascade, "SDKReadiness", no_broker)  # main() never waits on a drain
    return made


def test_the_plan_names_every_unit_of_ours_and_nobody_elses(units, manager):
    plan = unit_migration.find()
    assert sorted(plan.units) == sorted(NAMES)
    assert [p.name for p in plan.units["video-window"]] == ["pack-video-window.service", "pack-video-window.timer"]
    assert [p.name for p in plan.dropins] == ["pack-listen.service.d"]
    assert plan.linger is True


def test_a_dry_run_says_each_command_and_file_and_changes_nothing(units, manager, capsys):
    before = sorted(p.name for p in units.iterdir())
    unit_migration.main(dry_run=True, disable_linger=False)
    said = capsys.readouterr().out
    assert said.startswith("Dry run: nothing is changed.\n8 services run as 9 unit files here.")
    assert f"stop capture: systemctl --user stop pack-{CAPTURE}.service" in said
    assert f"delete {units / 'pack-video-window.timer'}" in said
    assert f"delete {units / 'pack-listen.service.d'}" in said
    assert "Linger is on for this user" in said and "disable-linger" in said
    assert manager.changes == []  # only the one question to loginctl
    assert sorted(p.name for p in units.iterdir()) == before


def test_units_stop_in_the_quit_order_with_the_drain_between_capture_and_the_tier_worker(units, manager):
    said = []

    def drain(deadline):
        manager.commands.append(["drain"])
        return 0

    assert unit_migration.migrate(unit_migration.find(), say=said.append, drain=drain) is True
    stops = [c[3:] if c[:3] == ["systemctl", "--user", "stop"] else c for c in manager.changes]
    assert stops[:6] == [
        ["pack-journal.service", "pack-listen.service", "pack-video-window.service", "pack-video-window.timer"],
        [f"pack-{CAPTURE}.service"],
        ["drain"],
        [f"pack-{TIER}.service"],
        [f"pack-{BROKER}.service"],
        ["pack-continuum-gateway.service", "pack-dashboard.service"],
    ]
    assert manager.changes[-2][:3] == ["systemctl", "--user", "disable"]
    assert manager.changes[-1] == ["systemctl", "--user", "daemon-reload"]
    assert "Tier 0: flushed" in said


def test_afterwards_our_units_are_gone_and_others_are_untouched_and_a_second_run_does_nothing(units, manager, capsys):
    unit_migration.main(dry_run=False, disable_linger=False)
    assert sorted(p.name for p in units.iterdir()) == ["other.service"]
    assert not any(c[0] == "loginctl" and "disable-linger" in c for c in manager.commands)  # linger is left alone
    manager.commands.clear()
    capsys.readouterr()
    unit_migration.main(dry_run=False, disable_linger=False)
    assert manager.changes == []
    assert "nothing to migrate" in capsys.readouterr().out


def test_linger_is_turned_off_only_when_asked(units, manager):
    unit_migration.main(dry_run=False, disable_linger=True)
    assert manager.changes[-1][:2] == ["loginctl", "disable-linger"]


def test_a_step_that_fails_is_said_and_the_rest_still_run(units, monkeypatch, capsys):
    failing = Manager(fail=(f"pack-{CAPTURE}.service",))
    monkeypatch.setattr(unit_migration, "_run", failing)

    def no_broker(*args):
        raise RuntimeError("no broker in a test")

    monkeypatch.setattr(unit_migration.service_cascade, "SDKReadiness", no_broker)
    with pytest.raises(SystemExit, match="some steps failed"):
        unit_migration.main(dry_run=False, disable_linger=False)
    said = capsys.readouterr().out
    assert f"systemctl --user stop pack-{CAPTURE}.service: FAILED unit is masked" in said
    assert ["systemctl", "--user", "stop", f"pack-{BROKER}.service"] in failing.changes  # it went on
    assert not (units / f"pack-{BROKER}.service").exists()


def test_a_drain_that_cannot_be_checked_never_stops_the_migration(units, manager):
    def unknown(deadline):
        raise RuntimeError("broker socket refused")

    said = []
    assert unit_migration.migrate(unit_migration.find(), say=said.append, drain=unknown) is True
    assert any("couldn't check" in line for line in said)


def test_on_a_mac_each_agent_is_booted_out_and_its_plist_removed(tmp_path, monkeypatch, manager):
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    for name in ("listen", CAPTURE, BROKER):
        (agents / f"pack.{name}.plist").write_text("<plist/>")
    (agents / "com.example.other.plist").write_text("<plist/>")
    monkeypatch.setattr(unit_migration, "launchd_dir", lambda: agents)
    monkeypatch.setattr(unit_migration.sys, "platform", "darwin")
    plan = unit_migration.find()
    assert (sorted(plan.units), plan.linger) == (sorted(["listen", CAPTURE, BROKER]), None)
    assert unit_migration.migrate(plan, say=lambda line: None) is True
    assert [c[2].rsplit("/", 1)[1] for c in manager.changes] == ["pack.listen", f"pack.{CAPTURE}", f"pack.{BROKER}"]
    assert all(c[:2] == ["launchctl", "bootout"] for c in manager.changes)
    assert sorted(p.name for p in agents.iterdir()) == ["com.example.other.plist"]
