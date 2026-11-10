"""This machine's per-service usage over time (dashboard/machine_usage.py, #2245)."""

import subprocess

import pytest

from liberational.dashboard import machine_usage as mu

GIB = 1024**3
MIB = 1024**2


def snapshot(*procs, cpu=50.0, mem_used=10 * GIB, cores=4):
    return {"cpu": cpu, "cpu_count": cores, "mem_total": 32 * GIB, "mem_used": mem_used, "procs": list(procs)}


def proc(pid, role, cpu=0.0, rss=0):
    return {"pid": pid, "role": role, "cpu": cpu, "rss": rss}


def test_an_ollama_runner_counts_toward_the_server_that_started_it():
    procs = [proc(1, "ask-ollama"), proc(2, "ollama"), proc(11, mu.RUNNER), proc(12, mu.RUNNER), proc(13, mu.RUNNER)]
    parents = {11: 1, 12: 2, 13: 999}
    assert mu.groups(procs, parents.get) == {1: "ask-ollama", 2: "ollama", 11: "ask-ollama", 12: "ollama", 13: "ollama"}


def test_a_sample_is_each_service_s_share_of_the_machine_and_the_rest_is_everything_else():
    taken = mu.sample(
        snapshot(
            proc(1, "listen", cpu=100.0, rss=2 * GIB), proc(2, "listen", cpu=20.0, rss=GIB), proc(3, "vision", cpu=40.0)
        ),
        {1: "listen", 2: "listen", 3: "vision"},
        None,
        ts=100.0,
    )
    # 120% of one core on a four-core machine is 30% of it; the machine's 50% less 30% and 10% is 10%.
    assert taken == {
        "ts": 100.0,
        "cpu": {"listen": 30.0, "vision": 10.0, mu.OTHER: 10.0},
        "mem": {"listen": 3 * GIB, "vision": 0, mu.OTHER: 7 * GIB},
        "gpu": [],
    }


def test_the_rest_of_the_machine_never_goes_below_zero():
    taken = mu.sample(snapshot(proc(1, "listen", cpu=400.0, rss=20 * GIB), cpu=50.0), {1: "listen"}, None, ts=0.0)
    assert (taken["cpu"][mu.OTHER], taken["mem"][mu.OTHER]) == (0.0, 0)


def test_gpu_memory_is_grouped_by_service_per_device_and_unlisted_processes_are_everything_else():
    gpus = [
        {
            "index": 0,
            "uuid": "a",
            "name": "G0",
            "total": 24 * GIB,
            "used": 10 * GIB,
            "util": 40,
            "apps": {1: 6 * GIB, 11: GIB},
        },
        {
            "index": 1,
            "uuid": "b",
            "name": "G1",
            "total": 24 * GIB,
            "used": 3 * GIB,
            "util": None,
            "apps": {77: 2 * GIB},
        },
    ]
    taken = mu.sample(snapshot(), {1: "see-actions", 11: "ask-ollama"}, gpus, ts=0.0)
    assert taken["gpu"] == [
        {
            "index": 0,
            "used": 10 * GIB,
            "util": 40,
            "apps": {"see-actions": 6 * GIB, "ask-ollama": GIB, mu.OTHER: 3 * GIB},
        },
        {"index": 1, "used": 3 * GIB, "util": None, "apps": {mu.OTHER: 3 * GIB}},
    ]


def history(gpus=None):
    return mu.History(gpu_reader=lambda: gpus, parent=lambda pid: None, step_s=5.0, window_s=20.0)


def test_history_keeps_one_sample_a_step_for_its_window_and_serves_what_is_newer():
    h = history()
    for t in (0.0, 2.0, 5.0, 9.0, 10.0, 30.0):
        h.record(snapshot(proc(1, "listen", cpu=4.0)), now=t)
    assert [s["ts"] for s in h.view()["samples"]] == [10.0, 30.0], "2 and 9 too soon; 0 and 5 out of the window"
    assert [s["ts"] for s in h.view(since=10.0)["samples"]] == [30.0]


def test_history_says_what_each_resource_stacks_up_to():
    h = history([{"index": 0, "uuid": "a", "name": "RTX", "total": 24 * GIB, "used": 0, "util": 0, "apps": {}}])
    h.record(snapshot(), now=1.0)
    view = h.view()
    assert (view["cpu_count"], view["mem_total"], view["gpus"]) == (
        4,
        32 * GIB,
        [{"index": 0, "name": "RTX", "total": 24 * GIB}],
    )


def test_without_psutil_or_a_gpu_history_has_nothing_to_draw_with():
    h = history(None)
    h.record({"error": "psutil not installed", "procs": []}, now=1.0)
    assert h.view()["samples"] == []
    h.record(snapshot(), now=2.0)
    assert (h.view()["gpus"], h.view()["samples"][0]["gpu"]) == ([], [])


def test_nvidia_smi_is_read_per_device(monkeypatch):
    replies = {
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu": "0, GPU-a, RTX 3090, 1000, 24576, 37\n"
        "1, GPU-b, RTX 3090, 500, 24576, [N/A]\n",
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory": "GPU-a, 10, 600\nGPU-b, 10, 100\nGPU-b, 11, 300\n",
    }

    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=replies[cmd[1]], stderr="")

    monkeypatch.setattr(mu.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(mu.subprocess, "run", run)
    assert mu._smi() == [
        {
            "index": 0,
            "uuid": "GPU-a",
            "name": "RTX 3090",
            "total": 24576 * MIB,
            "used": 1000 * MIB,
            "util": 37,
            "apps": {10: 600 * MIB},
        },
        {
            "index": 1,
            "uuid": "GPU-b",
            "name": "RTX 3090",
            "total": 24576 * MIB,
            "used": 500 * MIB,
            "util": None,
            "apps": {10: 100 * MIB, 11: 300 * MIB},
        },
    ]


@pytest.mark.parametrize(
    "failure",
    [subprocess.TimeoutExpired("nvidia-smi", 1.0), subprocess.CalledProcessError(9, "nvidia-smi"), FileNotFoundError()],
)
def test_no_working_nvidia_smi_means_no_gpus(monkeypatch, failure):
    def run(cmd, **kwargs):
        raise failure

    monkeypatch.setattr(mu.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(mu.subprocess, "run", run)
    assert mu._smi() is None


def test_a_machine_without_nvidia_smi_has_no_gpus(monkeypatch):
    monkeypatch.setattr(mu.shutil, "which", lambda name: None)
    assert mu._smi() is None


def test_the_sampler_samples_with_no_poll_survives_a_failed_sample_and_stops():
    h = mu.History(gpu_reader=lambda: None, parent=lambda pid: None, step_s=0.0001, window_s=1.0)
    calls = 0
    enough = mu.threading.Event()

    def snap():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("psutil hiccup")
        if calls >= 4:
            enough.set()
        return snapshot(proc(1, "listen", cpu=4.0))

    sampler = mu.Sampler(h, snap, step_s=0.001)
    sampler.start()
    sampler.start()  # a second start doesn't make a second thread
    first = sampler.thread
    assert enough.wait(5), "it kept sampling after the failed one"
    sampler.stop()
    assert first is not None
    first.join(5)
    assert not first.is_alive()
    assert sampler.thread is first
    assert len(h.view()["samples"]) >= 2


def test_the_track_route_is_gone(serve, monkeypatch):
    import http.client
    import urllib.parse

    from liberational.dashboard import server

    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    host = urllib.parse.urlsplit(serve(server.H)).netloc
    c = http.client.HTTPConnection(host, timeout=5)
    c.request("POST", "/stats/history/track", "{}", {"Content-Type": "application/json"})
    status = c.getresponse().status
    c.close()
    assert status == 404


def test_nvml_reads_what_nvidia_smi_does():
    from liberational import nvml

    found = nvml.gpus()
    if found is None:
        pytest.skip("needs an NVIDIA GPU and its driver's NVML")
    smi = mu._smi()
    assert smi is not None
    assert [(g["index"], g["uuid"], g["name"], g["total"]) for g in found] == [
        (g["index"], g["uuid"], g["name"], g["total"]) for g in smi
    ]
    for ours, theirs in zip(found, smi, strict=True):
        assert abs(ours["used"] - theirs["used"]) <= 64 * MIB  # read a moment apart
        assert set(ours["apps"]) == set(theirs["apps"])


def test_without_nvml_the_gpus_come_from_nvidia_smi(monkeypatch):
    from liberational import nvml

    monkeypatch.setattr(nvml, "gpus", lambda: None)
    monkeypatch.setattr(mu, "_smi", lambda: [{"index": 0, "uuid": "GPU-x", "name": "smi", "total": 1, "used": 0}])
    assert mu.nvidia() == [{"index": 0, "uuid": "GPU-x", "name": "smi", "total": 1, "used": 0}]
