"""On a Mac, /devices matches manifest cameras by AVFoundation unique ID."""

import subprocess
from types import SimpleNamespace

from liberational.dashboard import devices

FFMPEG = """[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] Arducam 1080P Low Light
[AVFoundation indev @ 0x1] [1] Arducam 1080P Low Light
[AVFoundation indev @ 0x1] AVFoundation audio devices:
"""
PROFILER = '{"SPCameraDataType": [{"_name": "Arducam 1080P Low Light", "spcamera_unique-id": "0x114"}, {"_name": "Arducam 1080P Low Light", "spcamera_unique-id": "0x113"}]}'


def test_mac_cameras_carry_their_unique_ids(monkeypatch):
    def run(cmd, **kw):
        if cmd[0] == "system_profiler":
            return SimpleNamespace(stdout=PROFILER, stderr="")
        return SimpleNamespace(stdout="", stderr=FFMPEG)

    monkeypatch.setattr(devices.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(subprocess, "run", run)
    assert devices._mac_cameras() == [(0, "Arducam 1080P Low Light", "0x114"), (1, "Arducam 1080P Low Light", "0x113")]
