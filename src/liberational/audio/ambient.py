"""Microphone selection rules: which input devices count as microphones to record (the dashboard's device page,
dashboard/devices.py).

Capture, ASR, and the transcript moved to Continuum: continuum-capture records the microphones, the listen worker
(listen/worker.py, with the VAD and speech gate in listen/gate.py) transcribes them onto the broker's speech log,
and voices are voiceprints matched through the vector worker (listen/voices.py).
"""

import json
import re
import subprocess
import sys
import threading
from collections.abc import Collection, Iterable
from typing import Any

from liberational import config as cfg
from liberational import logs
from liberational.people import registry

PA_LOCK = threading.RLock()


LOG = logs.get("ambient")


def log(*a: object, level: int = logs.INFO) -> None:
    LOG.log(level, " ".join(str(x) for x in a))


EXCLUDE_SUBSTR = [x.strip().lower() for x in cfg.get("audio.exclude_devices") if x.strip()]


def _excluded(name: str) -> bool:
    return any(x in name.lower() for x in EXCLUDE_SUBSTR)


_ALSA_HW = re.compile(r"\(hw:\d+,\d+\)$")


def select_input_devices(
    devs: Iterable[tuple[int, str, int, float, str]],
    platform: str | None = None,
    include: Collection[str] | None = None,
) -> list[tuple[int, str, int, float]]:
    """Pick the devices to capture from PortAudio's list.

    devs: [(index, name, max_input_channels, default_samplerate, hostapi_name)]
    include: device names to keep after the rules below (None or empty: every real input)
    -> [(index, name, channels, samplerate)], at most one entry per name.

    macOS and Windows: every input device that isn't excluded (one path per device there already).
    Linux: PortAudio lists each mic several times over (raw ALSA hw:, ALSA plugins like default, pulse,
    pipewire, sysdefault, and the rate converters, plus the PulseAudio host API), and also lists every
    output's .monitor source. Opening the same hardware through several of those at once is what made
    PipeWire time out and wedge the daemon. So: when the PulseAudio host API has inputs (PipeWire or
    PulseAudio is running), take only its real sources (no monitors, no "Default Source" alias); otherwise
    take only raw ALSA hw: devices.
    """
    platform = platform or sys.platform
    devs = [d for d in devs if d[2] > 0 and not _excluded(d[1])]
    if platform.startswith("linux"):
        pulse = [d for d in devs if d[4] == "PulseAudio"]
        if pulse:
            devs = [d for d in pulse if not d[1].endswith(".monitor") and d[1] != "Default Source"]
        else:
            devs = [d for d in devs if d[4] == "ALSA" and _ALSA_HW.search(d[1])]
    if include:
        devs = [d for d in devs if d[1] in include]
    seen, out = {}, []
    # Identical names are CoreAudio aggregate oddities. Two names with one slug would share a folder under state/audio
    # and their lines' keys, so only the first is opened.
    for i, name, ch, sr, _ in devs:
        slug = registry.slugify(name)
        if slug in seen:
            if seen[slug] != name:
                log(f"{name!r} not opened: its slug {slug!r} is {seen[slug]!r}'s", level=logs.WARN)
            continue
        seen[slug] = name
        out.append((i, name, ch, sr))
    return out


def scan_raw_devices() -> list[tuple[Any, ...]]:
    """PortAudio's whole device list from a child process, as select_input_devices() takes it. A child process
    because PortAudio caches its list in-process, and so callers (the dashboard) never initialize PortAudio."""
    code = (
        "import sounddevice as sd, json; h = [a['name'] for a in sd.query_hostapis()]; print(json.dumps("
        "[(i, d['name'], d['max_input_channels'], d['default_samplerate'], h[d['hostapi']])"
        " for i, d in enumerate(sd.query_devices())]))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20).stdout.strip()
    return [tuple(x) for x in json.loads(out or "[]")]
