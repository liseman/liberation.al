"""Finite self-profiling uses exact thread CPU, never wall-time sample counts or frame contents."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from liberational import service_profile as P


@pytest.mark.parametrize("state", ["R", "S", "D"])
def test_proc_stat_parses_parentheses_spaces_and_distinct_start_identity(state):
    suffix = [state, *(["0"] * 20)]
    suffix[11], suffix[12], suffix[19] = "17", "6", "123456"
    assert P._stat("31 (worker (odd) name )) " + " ".join(suffix)) == P._Counter(123456, 23, state)
    with pytest.raises(ValueError, match="incomplete"):
        P._stat("31 (worker) R 0")


def test_python_frames_are_read_next_to_their_own_counters_before_later_native_tasks(monkeypatch):
    trace = []
    frame_maps = []
    private = "PRIVATE-LOCAL-DO-NOT-SERIALIZE"

    def first_python_frame(argument):
        assert argument
        return sys._getframe()

    def second_python_frame(argument):
        assert argument
        return sys._getframe()

    first, second = first_python_frame(private), second_python_frame(private)

    class ProcPath:
        def __init__(self, text):
            self.text = text

        @property
        def name(self):
            return self.text.rsplit("/", 1)[-1]

        def __truediv__(self, name):
            return ProcPath(self.text + "/" + name)

        def iterdir(self):
            return iter(ProcPath(f"/proc/self/task/{tid}") for tid in (11, 900, 901, 12, 902))

        def read_text(self):
            assert all(not frames for frames in frame_maps)
            trace.append(self.text)
            tid = int(self.text.split("/")[-2]) if "/task/" in self.text else 0
            if self.text.endswith("/status"):
                return f"Name:\tfake\nvoluntary_ctxt_switches:\t{tid + 1000}\nnonvoluntary_ctxt_switches:\t3\n"
            fields = ["S" if tid == 12 else "R", *(["0"] * 20)]
            fields[11], fields[12], fields[19] = str(tid + 1), "2", str(tid + 100)
            return f"{tid} (fake task) " + " ".join(fields)

    def current_frames():
        trace.append("frames")
        frames = {1111: first, 2222: second}
        frame_maps.append(frames)
        return frames

    monkeypatch.setattr(P, "Path", ProcPath)
    monkeypatch.setattr(
        P.threading,
        "enumerate",
        lambda: [SimpleNamespace(native_id=11, ident=1111), SimpleNamespace(native_id=12, ident=2222)],
    )
    monkeypatch.setattr(P.sys, "_current_frames", current_frames)
    reader = object.__new__(P._LinuxCounters)
    snapshot, stacks, lines = reader.sample(3)
    assert trace == [
        "/proc/self/stat",
        "/proc/self/task/11/status",
        "/proc/self/task/11/stat",
        "frames",
        "/proc/self/task/900/status",
        "/proc/self/task/900/stat",
        "/proc/self/task/901/status",
        "/proc/self/task/901/stat",
        "/proc/self/task/12/status",
        "/proc/self/task/12/stat",
        "frames",
        "/proc/self/task/902/status",
        "/proc/self/task/902/stat",
    ]
    assert set(snapshot.threads) == {11, 12, 900, 901, 902}
    assert snapshot.threads[11] == P._Counter(111, 14, "R", 1011)
    assert snapshot.threads[12] == P._Counter(112, 15, "S", 1012)
    assert set(stacks) == set(lines) == {11, 12}
    assert lines[11] == first.f_lineno
    assert stacks[11][-1].endswith(
        ":test_python_frames_are_read_next_to_their_own_counters_before_later_native_tasks.<locals>.first_python_frame"
    )
    assert stacks[12][-1].endswith(
        ":test_python_frames_are_read_next_to_their_own_counters_before_later_native_tasks.<locals>.second_python_frame"
    )
    assert private not in str(stacks)
    assert frame_maps == [{}, {}]
    trace.clear()
    frame_maps.clear()
    assert reader.snapshot() == snapshot
    assert [path for path in trace if path.endswith("/stat")] == [
        "/proc/self/stat",
        "/proc/self/task/11/stat",
        "/proc/self/task/900/stat",
        "/proc/self/task/901/stat",
        "/proc/self/task/12/stat",
        "/proc/self/task/902/stat",
    ]
    assert frame_maps == []


def test_cpu_deltas_weight_running_functions_and_leave_waiting_and_sampler_out():
    initial = P._Snapshot(
        P._Counter(1, 100), {1: P._Counter(10, 40), 2: P._Counter(20, 50), 3: P._Counter(30, 3), 99: P._Counter(99, 7)}
    )
    accumulator = P._Accumulator(initial, 99, 10, 10)
    accumulator.add(
        P._Snapshot(
            P._Counter(1, 130),
            {1: P._Counter(10, 60), 2: P._Counter(20, 50), 3: P._Counter(30, 7), 99: P._Counter(99, 10)},
        ),
        {1: ("module", "busy"), 2: ("module", "waiting"), 99: ("sampler",)},
    )
    report = accumulator.report()
    assert report["process_ticks"] == 30
    assert report["observed_thread_ticks"] == 27
    assert report["excluded_sampler_ticks"] == 3
    assert report["python_attributed_ticks"] == 20
    assert report["native_unattributed_ticks"] == 4
    assert report["process_minus_observed_thread_ticks"] == 3
    assert report["functions"] == {
        "module": {"inclusive_ticks_estimate": 20, "self_ticks_estimate": 0},
        "busy": {"inclusive_ticks_estimate": 20, "self_ticks_estimate": 20},
    }
    assert accumulator.stacks == {("[thread 1 start 10]", "module", "busy"): 20}
    assert next(row for row in report["threads"] if row["native_tid"] == 3)["native_or_unmapped_ticks"] == 4


@pytest.mark.parametrize("state", ["S", "D", "T", "I"])
def test_nonrunning_python_endpoints_do_not_receive_or_redistribute_cpu_deltas(state):
    def counter(start, ticks, state="R"):
        return P._Counter(start=start, ticks=ticks, state=state)

    accumulator = P._Accumulator(
        P._Snapshot(P._Counter(1, 100), {tid: counter(tid, 0) for tid in (1, 2, 3, 4, 99)}), 99, 10, 10
    )
    accumulator.add(
        P._Snapshot(
            P._Counter(1, 133),
            {
                1: counter(1, 10, "R"),
                2: counter(2, 10, state),
                3: counter(3, 4, "R"),
                4: counter(4, 4, "S"),
                99: counter(99, 2, "R"),
            },
        ),
        {1: ("root", "running"), 2: ("root", "epoll_wait"), 99: ("sampler",)},
    )
    report = accumulator.report()
    assert report["functions"] == {
        "root": {"inclusive_ticks_estimate": 10, "self_ticks_estimate": 0},
        "running": {"inclusive_ticks_estimate": 10, "self_ticks_estimate": 10},
    }
    assert accumulator.stacks == {("[thread 1 start 1]", "root", "running"): 10}
    assert report["python_attributed_ticks"] == report["nonrunning_unattributed_ticks"] == 10
    assert report["native_unattributed_ticks"] == 8
    assert report["excluded_sampler_ticks"] == 2
    assert report["process_ticks"] == 33 and report["observed_thread_ticks"] == 30
    assert report["process_minus_observed_thread_ticks"] == 3
    threads = {row["native_tid"]: row for row in report["threads"]}
    assert threads[2]["cpu_ticks"] == threads[2]["nonrunning_ticks"] == 10
    assert threads[2]["native_or_unmapped_ticks"] == 0
    assert threads[3]["native_or_unmapped_ticks"] == threads[4]["native_or_unmapped_ticks"] == 4
    assert threads[3]["nonrunning_ticks"] == threads[4]["nonrunning_ticks"] == 0
    accumulator.add(
        P._Snapshot(
            P._Counter(1, 138),
            {1: counter(1, 10), 2: counter(2, 15, "R"), 3: counter(3, 4), 4: counter(4, 4, "S"), 99: counter(99, 2)},
        ),
        {2: ("root", "resumed")},
    )
    resumed = accumulator.report()
    assert resumed["functions"]["resumed"]["self_ticks_estimate"] == 5
    assert resumed["python_attributed_ticks"] == 15
    assert resumed["nonrunning_unattributed_ticks"] == 10
    assert resumed["process_ticks"] == 38 and resumed["observed_thread_ticks"] == 35


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Name:\tx\nvoluntary_ctxt_switches:\t42\nnonvoluntary_ctxt_switches:\t7\n", 42),
        ("Name:\tx\nnonvoluntary_ctxt_switches:\t7\n", P._UNREAD),
        ("voluntary_ctxt_switches:\tlots\n", P._UNREAD),
    ],
)
def test_status_voluntary_switches_parse_or_stay_unread(text, expected):
    assert P._voluntary(text) == expected


def test_wakeups_are_tabled_at_wait_sites_without_feeding_cpu_functions():
    def counter(ticks, state, voluntary):
        return P._Counter(start=1, ticks=ticks, state=state, voluntary=voluntary)

    accumulator = P._Accumulator(
        P._Snapshot(
            P._Counter(1, 0),
            {
                1: counter(0, "S", 100),
                2: counter(0, "S", 100),
                3: counter(0, "R", 100),
                4: counter(0, "S", 100),
                5: counter(0, "S", P._UNREAD),
                99: counter(0, "S", 100),
            },
        ),
        99,
        10,
        10,
    )
    for ticks, poller, other in ((1, 105, 104), (3, 111, 106)):
        accumulator.add(
            P._Snapshot(
                P._Counter(1, ticks + 9),
                {
                    1: counter(ticks, "S", poller),
                    2: counter(0, "S", other),
                    3: counter(5, "R", 103),
                    4: counter(0, "S", 102),
                    5: counter(0, "S", 500),
                    99: counter(0, "S", 140),
                },
            ),
            {1: ("root", "poll"), 2: ("root", "poll"), 3: ("root", "busy"), 5: ("root", "unreadable")},
            {1: 12, 2: 12, 3: 30},
        )
    report = accumulator.report()
    assert report["wait_sites"] == [
        {"stack": ["root", "poll at line 12"], "wakeups": 17, "nonrunning_ticks": 3, "samples": 4, "threads": 2}
    ]
    assert report["wait_site_wakeups"] == 17
    assert report["running_endpoint_wakeups"] == 3
    assert report["native_wakeups"] == 2
    assert report["voluntary_switches"] == 22
    assert report["excluded_sampler_switches"] == 40
    assert report["nonrunning_unattributed_ticks"] == 3
    assert report["functions"] == {
        "root": {"inclusive_ticks_estimate": 5, "self_ticks_estimate": 0},
        "busy": {"inclusive_ticks_estimate": 5, "self_ticks_estimate": 5},
    }
    threads = {row["native_tid"]: row["voluntary_switches"] for row in report["threads"]}
    assert threads == {1: 11, 2: 6, 3: 3, 4: 2, 5: 0}


def test_missing_exited_new_and_reused_thread_counters_do_not_invent_cpu():
    accumulator = P._Accumulator(
        P._Snapshot(P._Counter(1, 50), {1: P._Counter(10, 5), 2: P._Counter(20, 7), 3: P._Counter(30, 8)}), 99, 10, 10
    )
    accumulator.add(
        P._Snapshot(P._Counter(1, 70), {1: P._Counter(11, 100), 4: P._Counter(40, 80)}, (2,)),
        {1: ("reused",), 4: ("new",)},
    )
    report = accumulator.report()
    assert report["process_ticks"] == 20
    assert report["observed_thread_ticks"] == 0
    assert report["process_minus_observed_thread_ticks"] == 20
    assert report["reused_or_reset_threads"] == report["new_thread_baselines"] == 1
    assert report["unreadable_thread_samples"] == report["exited_threads"] == 1
    assert report["functions"] == {}


def test_bounded_stack_and_function_tables_keep_overflow_cpu_visible():
    accumulator = P._Accumulator(P._Snapshot(P._Counter(1, 0), {1: P._Counter(1, 0)}), 99, 1, 1)
    accumulator.add(P._Snapshot(P._Counter(1, 10), {1: P._Counter(1, 10)}), {1: ("root", "first")})
    accumulator.add(P._Snapshot(P._Counter(1, 20), {1: P._Counter(1, 20)}), {1: ("root", "second")})
    report = accumulator.report()
    assert len(accumulator.stacks) == len(report["functions"]) == 1
    assert report["stack_overflow_ticks"] == 10
    assert (
        sum(f["self_ticks_estimate"] for f in report["functions"].values()) + report["function_overflow_self_ticks"]
        == 20
    )
    assert sum(accumulator.stacks.values()) + report["stack_overflow_ticks"] == report["python_attributed_ticks"] == 20


def test_identical_python_stacks_preserve_thread_weights_and_reused_native_tid_identity():
    accumulator = P._Accumulator(
        P._Snapshot(P._Counter(1, 0), {12: P._Counter(45, 0), 13: P._Counter(46, 0)}), 99, 10, 10
    )
    stack = ("module", "busy")
    accumulator.add(
        P._Snapshot(P._Counter(1, 10), {12: P._Counter(45, 7), 13: P._Counter(46, 3)}), {12: stack, 13: stack}
    )
    accumulator.add(P._Snapshot(P._Counter(1, 12), {12: P._Counter(99, 0), 13: P._Counter(46, 3)}), {12: stack})
    accumulator.add(P._Snapshot(P._Counter(1, 17), {12: P._Counter(99, 5), 13: P._Counter(46, 3)}), {12: stack})
    assert accumulator.stacks == {
        ("[thread 12 start 45]", "module", "busy"): 7,
        ("[thread 13 start 46]", "module", "busy"): 3,
        ("[thread 12 start 99]", "module", "busy"): 5,
    }
    report = accumulator.report()
    assert report["functions"] == {
        "module": {"inclusive_ticks_estimate": 15, "self_ticks_estimate": 0},
        "busy": {"inclusive_ticks_estimate": 15, "self_ticks_estimate": 15},
    }
    assert report["python_attributed_ticks"] == report["observed_thread_ticks"] == 15
    assert report["process_minus_observed_thread_ticks"] == 2
    assert report["reused_or_reset_threads"] == 1


def test_thread_root_labels_consume_the_existing_stack_byte_budget(monkeypatch):
    monkeypatch.setattr(P, "_STACK_BYTES", 8)  # root + busy fit; the required native thread identity does not.
    accumulator = P._Accumulator(P._Snapshot(P._Counter(1, 0), {12: P._Counter(45, 0)}), 99, 10, 10)
    accumulator.add(P._Snapshot(P._Counter(1, 8), {12: P._Counter(45, 8)}), {12: ("root", "busy")})
    assert accumulator.stacks == {}
    assert accumulator.stack_bytes == 0
    report = accumulator.report()
    assert report["stack_overflow_ticks"] == report["python_attributed_ticks"] == 8
    assert report["functions"]["busy"]["self_ticks_estimate"] == 8


def test_identical_callstacks_on_distinct_threads_consume_distinct_stack_slots():
    accumulator = P._Accumulator(
        P._Snapshot(P._Counter(1, 0), {12: P._Counter(45, 0), 13: P._Counter(46, 0)}), 99, 10, 1
    )
    accumulator.add(
        P._Snapshot(P._Counter(1, 10), {12: P._Counter(45, 7), 13: P._Counter(46, 3)}),
        {12: ("busy",), 13: ("busy",)},
    )
    assert accumulator.stacks == {("[thread 12 start 45]", "busy"): 7}
    report = accumulator.report()
    assert report["stack_overflow_ticks"] == 3
    assert report["functions"]["busy"]["self_ticks_estimate"] == report["python_attributed_ticks"] == 10


def test_stack_capture_is_depth_bounded_and_does_not_serialize_locals():
    secret = "PRIVATE-ARGUMENT-AND-LOCAL-CONTENT"
    stack = P._stack(sys._getframe(), 1)
    assert len(stack) <= 2  # one code location and the explicit truncation marker
    assert secret not in " ".join(stack)
    assert stack[0] == "[stack truncated]"
    assert stack[-1].endswith(":test_stack_capture_is_depth_bounded_and_does_not_serialize_locals")


def test_process_counter_replacement_is_an_error_rather_than_fabricated_cpu():
    accumulator = P._Accumulator(P._Snapshot(P._Counter(1, 20), {}), 99, 10, 10)
    with pytest.raises(ValueError, match="identity changed or reset"):
        accumulator.add(P._Snapshot(P._Counter(2, 1000), {}), {})


def test_private_atomic_artifacts_rotate_without_removing_other_services(tmp_path, monkeypatch):
    profiler = P.Profiler("worker", tmp_path, max_artifacts=2)
    accumulator = P._Accumulator(P._Snapshot(P._Counter(1, 0), {1: P._Counter(1, 0)}), 99, 10, 10)
    accumulator.add(P._Snapshot(P._Counter(1, 7), {1: P._Counter(1, 7)}), {1: ("root", "busy")})
    clock = iter(range(10, 20))
    monkeypatch.setattr(P.time, "time_ns", lambda: next(clock))
    unrelated = tmp_path / "other.json"
    unrelated.write_text("unchanged")
    identity = {"capture_id": "test", "pid": 123, "service": "worker", "process_start_ticks": 1}
    profiler._write(accumulator, identity, 100, 1.0, False)
    profiler._write(accumulator, identity, 100, 2.0, False)
    profiler._write(accumulator, identity, 100, 3.0, True)
    artifacts = sorted(tmp_path.glob("profile-*.json"))
    assert len(artifacts) == len(list(tmp_path.glob("*.collapsed"))) == 2
    assert json.loads(artifacts[-1].read_text())["final"] is True
    assert artifacts[-1].with_suffix(".collapsed").read_text() == "[thread 1 start 1];root;busy 7\n"
    assert json.loads(artifacts[-1].read_text())["stack_scope"] == "native-thread"
    assert {p.stat().st_mode & 0o777 for p in [*artifacts, *tmp_path.glob("*.collapsed")]} == {0o600}
    assert not list(tmp_path.glob(".profile-*"))
    assert unrelated.read_text() == "unchanged"


def test_nonrunning_cpu_is_written_as_unattributed_with_explicit_filter_version(tmp_path):
    profiler = P.Profiler("worker", tmp_path)
    accumulator = P._Accumulator(P._Snapshot(P._Counter(1, 0), {1: P._Counter(1, 0)}), 99, 10, 10)
    accumulator.add(P._Snapshot(P._Counter(1, 8), {1: P._Counter(1, 8, "S")}), {1: ("waiting",)})
    profiler._write(accumulator, {"capture_id": "test"}, 100, 1, True)
    (artifact,) = tmp_path.glob("*.json")
    report = json.loads(artifact.read_text())
    assert report["stack_filter"] == "running-only"
    assert report["attribution_method"] == "thread-cpu-delta-running-state-v3"
    assert report["stack_pairing"] == "per-python-thread-adjacent"
    assert report["nonrunning_unattributed_ticks"] == report["observed_thread_ticks"] == 8
    assert report["functions"] == {}
    assert artifact.with_suffix(".collapsed").read_text() == "[nonrunning_unattributed_ticks] 8\n"


def test_failed_json_publication_removes_its_uncommitted_collapsed_companion(tmp_path, monkeypatch):
    profiler = P.Profiler("worker", tmp_path)
    accumulator = P._Accumulator(P._Snapshot(P._Counter(1, 0), {}), 99, 10, 10)
    atomic = P._atomic

    def publish(path, data):
        if path.suffix == ".json":
            raise OSError("write refused")
        atomic(path, data)

    monkeypatch.setattr(P, "_atomic", publish)
    with pytest.raises(OSError, match="write refused"):
        profiler._write(accumulator, {"capture_id": "test"}, 100, 1, True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "kwargs",
    [{"interval_s": 0}, {"duration_s": float("inf")}, {"delay_s": -1}, {"max_depth": True}, {"max_artifacts": 0}],
)
def test_invalid_capture_settings_are_rejected_before_any_thread_or_file(tmp_path, kwargs):
    with pytest.raises(ValueError):
        P.Profiler("worker", tmp_path / "untouched", **kwargs)
    assert not (tmp_path / "untouched").exists()


def test_unsupported_platform_is_actionable_only_on_explicit_start(tmp_path, monkeypatch):
    monkeypatch.setattr(P.sys, "platform", "unsupported")
    profiler = P.Profiler("worker", tmp_path / "untouched")
    with pytest.raises(RuntimeError, match="requires Linux procfs"):
        profiler.start()
    assert not (tmp_path / "untouched").exists()
    assert not profiler.running


def test_stop_interrupts_warmup_without_collecting_or_writing(tmp_path, monkeypatch):
    class Counters:
        def snapshot(self):
            pytest.fail("warmup must not read CPU counters")

    monkeypatch.setattr(P, "_LinuxCounters", Counters)
    profiler = P.Profiler("worker", tmp_path, delay_s=60).start()
    profiler.stop()
    assert profiler.wait(1)
    assert not profiler.running
    assert list(tmp_path.iterdir()) == []


def test_finite_capture_duration_begins_after_warmup(tmp_path, monkeypatch):
    reads = []

    class Counters:
        tick_hz = 100

        def snapshot(self):
            reads.append(time.monotonic())
            return P._Snapshot(P._Counter(1, 0), {})

        def sample(self, depth):
            return self.snapshot(), {}, {}

    monkeypatch.setattr(P, "_LinuxCounters", Counters)
    started = time.monotonic()
    profiler = P.Profiler("worker", tmp_path, interval_s=0.01, delay_s=0.03, duration_s=0.03).start()
    assert profiler.wait(2)
    (artifact,) = tmp_path.glob("*.json")
    report = json.loads(artifact.read_text())
    assert reads[0] - started >= 0.03
    assert reads[-1] - reads[0] >= 0.03
    assert report["elapsed_s"] >= 0.03
    assert report["delay_s"] == 0.03
    assert report["final"] is True


def test_capture_failure_stops_the_sampler_without_stopping_the_service(tmp_path, monkeypatch):
    class Counters:
        def snapshot(self):
            raise OSError("counter unavailable")

    monkeypatch.setattr(P, "_LinuxCounters", Counters)
    profiler = P.Profiler("worker", tmp_path).start()
    assert profiler.wait(1)
    assert profiler.error == "OSError: counter unavailable"
    assert not profiler.running


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux procfs CPU accounting")
def test_finite_child_capture_attributes_cpu_to_busy_work_instead_of_waiting(tmp_path):
    source = """
import pathlib, threading, time
from liberational.service_profile import Profiler
done = threading.Event()
def waiting_worker():
    done.wait()
def busy_work():
    until = time.perf_counter() + .25
    total = 0
    while time.perf_counter() < until:
        total += 1
    return total
waiter = threading.Thread(target=waiting_worker)
waiter.start()
profiler = Profiler('child-test', pathlib.Path(__import__('sys').argv[1]), interval_s=.01, duration_s=.4).start()
busy_work()
assert profiler.wait(5), 'capture did not stop automatically'
assert profiler.error is None, profiler.error
done.set()
waiter.join()
"""
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    subprocess.run(
        [sys.executable, "-c", source, str(tmp_path)], env=environment, check=True, timeout=10, capture_output=True
    )
    (artifact,) = tmp_path.glob("*.json")
    report = json.loads(artifact.read_text())
    busy = sum(row["self_ticks_estimate"] for name, row in report["functions"].items() if name.endswith(":busy_work"))
    waiting = sum(
        row["self_ticks_estimate"] for name, row in report["functions"].items() if name.endswith(":waiting_worker")
    )
    assert busy > waiting
    assert busy > 0
    assert report["process_ticks"] >= busy
    assert report["final"] is True
    assert report["identity"]["pid"] != os.getpid()
    assert report["excluded_sampler_ticks"] >= 0
    assert report["elapsed_s"] >= 0.4


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux procfs CPU accounting")
def test_ten_millisecond_poller_shows_at_its_sleep_line_near_one_hundred_wakeups_per_second(tmp_path):
    source = """
import pathlib, sys, threading, time
from liberational.service_profile import Profiler
done = threading.Event()
def poll_loop():
    while not done.is_set():
        time.sleep(.01)
poller = threading.Thread(target=poll_loop)
poller.start()
profiler = Profiler('poll-test', pathlib.Path(sys.argv[1]), interval_s=.05, duration_s=1).start()
assert profiler.wait(10), 'capture did not stop automatically'
assert profiler.error is None, profiler.error
done.set()
poller.join()
"""
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    subprocess.run(
        [sys.executable, "-c", source, str(tmp_path)], env=environment, check=True, timeout=20, capture_output=True
    )
    (artifact,) = tmp_path.glob("*.json")
    report = json.loads(artifact.read_text())
    top = report["wait_sites"][0]
    assert top["stack"][-1].endswith(":poll_loop at line 7")
    assert 60 <= top["wakeups"] / report["elapsed_s"] <= 130
    assert not any(name.endswith(":poll_loop") for name in report["functions"])
