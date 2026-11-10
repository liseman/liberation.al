import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "profile_services", Path(__file__).resolve().parents[1] / "tools" / "profile_services.py"
)
assert _spec is not None and _spec.loader is not None
P = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = P
_spec.loader.exec_module(P)


def test_stat_comm_parentheses_do_not_shift_cpu_or_identity():
    fields = ["S", "7"] + ["0"] * 9 + ["120", "30"] + ["999", "999", "0", "0", "1", "0", "42"]
    row = P.parse_stat(9, "9 (worker (odd) name) " + " ".join(fields))
    assert (row.parent, row.user, row.system, row.start) == (7, 120, 30, 42)


def test_nearest_named_service_owns_descendants_without_double_counting():
    rows = {
        1: P.Process(1, 0, 1, 0, 0, "capture"),
        2: P.Process(2, 1, 2, 0, 0),
        3: P.Process(3, 2, 3, 0, 0),
        4: P.Process(4, 2, 4, 0, 0, "dashboard"),
        5: P.Process(5, 4, 5, 0, 0),
        6: P.Process(6, 6, 6, 0, 0),
    }
    assert P.assign_owners(rows) == {1: "capture", 2: "capture", 3: "capture", 4: "dashboard", 5: "dashboard"}


def test_survey_counts_actual_intervals_and_marks_exit_birth_and_pid_reuse():
    snapshots = iter(
        [
            {1: P.Process(1, 0, 1, 100, 50, "one"), 2: P.Process(2, 1, 2, 50, 10)},
            {1: P.Process(1, 0, 1, 150, 60, "one"), 2: P.Process(2, 1, 9, 900, 900), 3: P.Process(3, 1, 3, 20, 10)},
            {1: P.Process(1, 0, 1, 175, 65, "one"), 3: P.Process(3, 1, 3, 45, 15)},
        ]
    )
    now = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    report = P.survey(1.5, lambda: next(snapshots), ticks=100, clock=lambda: now[0], sleep=sleep)
    assert waits == [1.0, 0.5]
    (group,) = report["services"]
    assert group["user_seconds"] == 1.0
    assert group["system_seconds"] == pytest.approx(0.2)
    assert group["cores"] == pytest.approx(0.8)
    by_identity = {(r["pid"], r["start_ticks"]): r for r in group["processes"]}
    assert by_identity[(1, 1)]["partial"] is False
    assert by_identity[(2, 2)]["partial"] is True
    assert by_identity[(2, 9)]["partial"] is True
    assert by_identity[(3, 3)]["partial"] is True
    assert by_identity[(3, 3)]["observed_seconds"] == 0.5


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), 3601])
def test_invalid_duration_never_reads_processes(seconds):
    def read():
        pytest.fail("invalid survey read processes")

    with pytest.raises(ValueError, match="seconds"):
        P.survey(seconds, read, ticks=100)


def test_private_report_repairs_mode_and_refuses_symlink(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("old")
    target.chmod(0o644)
    P.write_report(target, {"services": []})
    assert target.stat().st_mode & 0o777 == 0o600
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(OSError):
        P.write_report(link, {})
    assert target.read_text().startswith("{")


def test_fake_proc_matches_named_and_exact_path_commands_without_saving_args(tmp_path):
    proc = tmp_path / "10"
    proc.mkdir()
    (proc / "cmdline").write_bytes(b"/usr/bin/python\0/venv/bin/pack-listen\0private-argument\0")
    fields = ["S", "1"] + ["0"] * 9 + ["3", "4"] + ["0"] * 6 + ["42"]
    (proc / "stat").write_text("10 (python) " + " ".join(fields))
    rows = P.snapshot(tmp_path, ({"pack-listen": "listen"}, {}))
    assert rows[10].service == "listen"
    assert "private-argument" not in repr(rows)


def test_orphan_retains_service_ownership_after_root_exits():
    samples = iter(
        [
            {1: P.Process(1, 0, 1, 0, 0, "capture"), 2: P.Process(2, 1, 2, 0, 0)},
            {2: P.Process(2, 0, 2, 10, 0)},
            {2: P.Process(2, 0, 2, 30, 0)},
        ]
    )
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    report = P.survey(2, lambda: next(samples), ticks=100, clock=lambda: now[0], sleep=sleep)
    (group,) = report["services"]
    assert group["service"] == "capture"
    assert group["user_seconds"] == pytest.approx(0.3)
    assert group["cores"] == pytest.approx(0.15)


def profile(start=10, final=True, capture="one", elapsed=2):
    return {
        "version": 1,
        "identity": {
            "service": "listen",
            "pid": 7,
            "process_start_ticks": 8,
            "capture_id": capture,
            "started_at_ns": start,
        },
        "final": final,
        "tick_hz": 100,
        "elapsed_s": elapsed,
        "process_ticks": 200,
        "observed_thread_ticks": 190,
        "excluded_sampler_ticks": 10,
        "python_attributed_ticks": 150,
        "native_unattributed_ticks": 30,
        "process_minus_observed_thread_ticks": 10,
        "functions": {
            "outer": {"self_ticks_estimate": 10, "inclusive_ticks_estimate": 150},
            "inner": {"self_ticks_estimate": 140, "inclusive_ticks_estimate": 140},
        },
    }


def test_profile_summary_selects_latest_final_capture_and_ranks_self_ticks(tmp_path):
    import json

    (tmp_path / "old.json").write_text(json.dumps(profile(start=5)))
    (tmp_path / "latest.json").write_text(json.dumps(profile(start=10, capture="two")))
    (tmp_path / "running.json").write_text(json.dumps(profile(start=20, final=False)))
    rows = P.read_profiles(tmp_path)
    (row,) = rows
    assert row["identity"]["capture_id"] == "two"
    assert row["cores"] == 1.0
    assert [r["location"] for r in row["top_functions"]] == ["inner", "outer"]
    assert row["top_functions"][0]["self_cores_estimate"] == 0.7
    assert row["native_functions"] == "unavailable"
    assert row["accounting_ticks"]["process_minus_observed_thread_ticks"] == 10
    assert row["stack_filter"] == "all-endpoint-states"


def test_running_only_report_keeps_idle_endpoint_cpu_unassigned_and_labels_method(tmp_path):
    import json

    row = profile()
    row.update(
        stack_filter="running-only",
        attribution_method="thread-cpu-delta-running-state-v2",
        nonrunning_unattributed_ticks=120,
        python_attributed_ticks=30,
        functions={"busy": {"self_ticks_estimate": 30, "inclusive_ticks_estimate": 30}},
    )
    (tmp_path / "final.json").write_text(json.dumps(row))
    (result,) = P.read_profiles(tmp_path)
    assert result["accounting_ticks"]["nonrunning_unattributed_ticks"] == 120
    assert result["top_functions"][0]["self_cores_estimate"] == 0.15
    assert result["cores"] == 1.0
    assert result["stack_filter"] == "running-only"
    assert result["attribution_method"] == "thread-cpu-delta-running-state-v2"
    assert "stack filter running-only" in P.markdown({"profiles": [result]})


def test_profile_only_report_does_not_survey_and_preserves_independent_windows(tmp_path, capsys):
    import json

    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "final.json").write_text(json.dumps(profile()))
    saved = tmp_path / "survey.json"
    saved.write_text(
        json.dumps({"schema_version": 1, "duration_seconds": 60, "services": [{"service": "listen", "cores": 0.2}]})
    )
    output = tmp_path / "combined.json"
    P.main(["--profiles", str(profiles), "--survey", str(saved), "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["duration_seconds"] == 60
    assert result["services"][0]["cores"] == 0.2
    assert result["profiles"][0]["cores"] == 1.0
    text = capsys.readouterr().out
    assert "| listen | 0.200 | 1.000 |" in text
    assert "must not be added" in text
    assert "Native functions unavailable" in text
    assert output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "change,message",
    [
        ({"version": 2}, "version"),
        ({"process_ticks": -1}, "process_ticks"),
        ({"elapsed_s": 0}, "elapsed_s"),
        ({"functions": None}, "function"),
    ],
)
def test_bad_profile_schema_is_actionable(tmp_path, change, message):
    import json

    row = profile()
    row.update(change)
    (tmp_path / "bad.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match=message):
        P.read_profiles(tmp_path)


def test_running_only_profiles_do_not_claim_finished_results(tmp_path):
    import json

    (tmp_path / "running.json").write_text(json.dumps(profile(final=False)))
    with pytest.raises(ValueError, match="No final"):
        P.read_profiles(tmp_path)


def test_profiles_alone_is_offline_and_top_functions_are_bounded(tmp_path, capsys):
    import json

    row = profile()
    row["functions"] = {f"function{i}": {"self_ticks_estimate": i, "inclusive_ticks_estimate": i} for i in range(12)}
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "final.json").write_text(json.dumps(row))
    output = tmp_path / "report.json"
    P.main(["--profiles", str(profiles), "--output", str(output)])
    saved = json.loads(output.read_text())
    assert saved["services"] == []
    assert len(saved["profiles"][0]["top_functions"]) == 10
    assert saved["profiles"][0]["top_functions"][0]["location"] == "function11"
    assert "native work may be assigned" in capsys.readouterr().out


def test_profile_report_retains_each_pid_and_reused_pid_identity(tmp_path):
    import json

    old = profile(start=5, capture="older")
    latest = profile(start=10, capture="server")
    child = profile(start=20, capture="review")
    child["identity"]["pid"] = 9
    reused = profile(start=30, capture="reused")
    reused["identity"]["process_start_ticks"] = 88
    (tmp_path / "old.json").write_text(json.dumps(old))
    (tmp_path / "server.json").write_text(json.dumps(latest))
    (tmp_path / "child.json").write_text(json.dumps(child))
    (tmp_path / "reused.json").write_text(json.dumps(reused))
    rows = P.read_profiles(tmp_path)
    assert {
        (r["identity"]["pid"], r["identity"]["process_start_ticks"], r["identity"]["capture_id"]) for r in rows
    } == {(7, 8, "server"), (9, 8, "review"), (7, 88, "reused")}
    text = P.markdown({"profiles": rows, "services": [{"service": "listen", "cores": 0.5}]})
    assert "| listen | 0.500 | per-PID rates:" in text
    assert "server" in text and "review" in text and "reused" in text
    assert text.count("| Python location (estimate)") == 3
    assert "3.000" not in text


def test_wait_sites_report_wakeup_rates_apart_from_cpu_and_old_captures_say_unavailable(tmp_path):
    import json

    row = profile()
    row.update(
        wait_method="voluntary-switch-delta-at-nonrunning-endpoint-v1",
        voluntary_switches=500,
        wait_site_wakeups=400,
        running_endpoint_wakeups=60,
        native_wakeups=40,
        wait_sites=[
            {
                "stack": ["main", "loop", "queue.get at line 9"],
                "wakeups": 80,
                "nonrunning_ticks": 4,
                "samples": 9,
                "threads": 1,
            },
            {"stack": ["main", "poll at line 3"], "wakeups": 320, "nonrunning_ticks": 10, "samples": 40, "threads": 2},
        ],
        threads=[
            {"native_tid": 5, "cpu_ticks": 20, "voluntary_switches": 320},
            {"native_tid": 6, "cpu_ticks": 2, "voluntary_switches": 0},
        ],
    )
    (tmp_path / "waits.json").write_text(json.dumps(row))
    (result,) = P.read_profiles(tmp_path)
    waits = result["waits"]
    assert [site["stack"][-1] for site in waits["top_sites"]] == ["poll at line 3", "queue.get at line 9"]
    assert waits["top_sites"][0]["wakeups_per_s"] == 160
    assert waits["top_sites"][0]["nonrunning_cores"] == 0.05
    assert waits["top_threads"] == [{"native_tid": 5, "wakeups_per_s": 160, "cores": 0.1}]
    assert waits["wakeups_per_s"]["voluntary_switches"] == 250
    assert [f["location"] for f in result["top_functions"]] == ["inner", "outer"]
    text = P.markdown({"profiles": [result]})
    assert "Wait sites between wakeups (not CPU attribution" in text
    assert "| poll at line 3 < main | 160.0 | 0.0500 | 2 |" in text
    assert "| 5 | 160.0 | 0.1000 |" in text

    (tmp_path / "waits.json").write_text(json.dumps(profile()))
    (old,) = P.read_profiles(tmp_path)
    assert old["waits"] == "unavailable"
    assert "Wait sites between wakeups: unavailable" in P.markdown({"profiles": [old]})
