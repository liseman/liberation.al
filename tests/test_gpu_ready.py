"""The GPU wait: fake nvidia-smi probes and a fake clock, never the real driver or a real exec."""

from liberational import gpu_ready as ready


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def test_no_nvidia_tooling_starts_at_once():
    def probe(smi: str) -> list[str]:
        raise AssertionError("nothing to probe without nvidia-smi")

    assert ready.wait_for_gpu(which=lambda name: None, probe=probe)


def test_waits_until_the_driver_lists_a_gpu():
    clock = Clock()
    answers = iter([[], [], ["GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-x)"]])
    up = ready.wait_for_gpu(
        which=lambda name: "/usr/bin/nvidia-smi", probe=lambda smi: next(answers), sleep=clock.sleep, clock=clock
    )
    assert up
    assert clock.now == 2 * ready.POLL_S


def test_gives_up_after_the_wait_and_starts_anyway():
    clock = Clock()
    up = ready.wait_for_gpu(
        which=lambda name: "/usr/bin/nvidia-smi", probe=lambda smi: [], sleep=clock.sleep, clock=clock, wait_s=20
    )
    assert not up
    assert 20 <= clock.now < 20 + ready.POLL_S


def test_a_failing_nvidia_smi_reads_as_no_gpu(tmp_path):
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 'NVIDIA-SMI has failed' >&2\nexit 9\n")
    smi.chmod(0o755)
    assert ready.gpus(str(smi)) == []
    smi.write_text("#!/bin/sh\necho 'GPU 0: Fake (UUID: GPU-a)'\necho 'GPU 1: Fake (UUID: GPU-b)'\n")
    assert len(ready.gpus(str(smi))) == 2


def test_main_execs_the_command_after_the_wait(monkeypatch):
    calls = []
    monkeypatch.setattr(ready, "wait_for_gpu", lambda: calls.append("wait") or True)
    monkeypatch.setattr(ready.os, "execv", lambda path, argv: calls.append(argv))
    ready.main(["--", "/usr/bin/ollama", "serve"])
    assert calls == ["wait", ["/usr/bin/ollama", "serve"]]
