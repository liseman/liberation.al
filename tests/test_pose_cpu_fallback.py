"""The pose model falls back to the CPU when its accelerated session fails at run time."""

import numpy as np
import pytest

from liberational.vision import pose_model
from liberational.vision.onnx_runtime import DetectorUnavailable


class Fails:
    def run(self, names, feed):
        raise RuntimeError("CoreML node failed")


class Works:
    def __init__(self):
        self.calls = 0

    def run(self, names, feed):
        self.calls += 1
        n = len(feed["x"])
        return [np.ones((n, 17, 384), np.float32), np.ones((n, 17, 512), np.float32)]


def test_a_failing_accelerated_session_switches_to_the_cpu_once():
    cpu = Works()
    made = []

    def make():
        made.append(1)
        return cpu

    m = pose_model.PoseModel(Fails(), "x", "rtmpose", "v", "CoreMLExecutionProvider", cpu=make)
    blob = np.zeros((3, 3, 256, 192), np.float32)
    x, _ = m._run(blob)
    assert x.shape == (3, 17, 384) and m.provider == "CPUExecutionProvider" and m.cpu is None
    m._run(blob)
    assert made == [1] and cpu.calls == 2


def test_with_no_cpu_fallback_the_failure_is_reported():
    m = pose_model.PoseModel(Fails(), "x", "rtmpose", "v", "CPUExecutionProvider")
    with pytest.raises(DetectorUnavailable):
        m._run(np.zeros((1, 3, 256, 192), np.float32))
