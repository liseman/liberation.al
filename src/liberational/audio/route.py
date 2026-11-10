"""Prefer the connected Bluetooth headset (audio.bluetooth_output), otherwise built-in speakers. Never modifies input.
macOS only: requires brew install switchaudio-osx blueutil. Polls every three seconds.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from typing import Any

from liberational import config as cfg
from liberational import logs

LOG = logs.get("audio-route")


def address() -> str:
    """Configured headset address in blueutil form (aa-bb-cc-dd-ee-ff), or ""."""
    return cfg.get("audio.bluetooth_output").strip().lower().replace(":", "-")


def preferred_uid() -> str | None:
    return address().upper() + ":output" if address() else None


def select_output(
    devices: Iterable[Mapping[str, Any]],
    connected: bool,
    current_uid: str | None = None,
    preferred: str | None = None,
) -> Mapping[str, Any] | None:
    preferred = preferred or preferred_uid()
    by_uid = {d["uid"]: d for d in devices}
    if connected and preferred in by_uid:
        return by_uid[preferred]
    if "BuiltInSpeakerDevice" in by_uid:
        return by_uid["BuiltInSpeakerDevice"]
    # Avoid automatically choosing HDMI/USB with no attached speaker.
    if current_uid != preferred:
        return by_uid.get(current_uid)
    return None


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True, timeout=5).strip()


def tools() -> tuple[str, str | None]:
    """(SwitchAudioSource, blueutil or None); exits with a clear message when they're unavailable."""
    if sys.platform != "darwin":
        sys.exit("audio_route.py is macOS only (it uses SwitchAudioSource and blueutil).")
    switch, blueutil = shutil.which("SwitchAudioSource"), shutil.which("blueutil")
    if not switch or (address() and not blueutil):
        sys.exit("audio_route.py needs SwitchAudioSource and blueutil: brew install switchaudio-osx blueutil")
    return switch, blueutil


def route_once(switch: str, blueutil: str | None) -> Mapping[str, Any]:
    connected = False
    if address():
        assert blueutil is not None, "tools() requires blueutil when a headset is configured"
        connected = run(blueutil, "--is-connected", address()) == "1"
    devices = [json.loads(line) for line in run(switch, "-a", "-t", "output", "-f", "json").splitlines()]
    current = json.loads(run(switch, "-c", "-t", "output", "-f", "json"))
    target = select_output(devices, connected, current["uid"])
    if target is None:
        raise RuntimeError("No safe output available")
    changed = []
    for kind in ("output", "system"):
        selected = json.loads(run(switch, "-c", "-t", kind, "-f", "json"))
        if selected["uid"] != target["uid"]:
            run(switch, "-t", kind, "-u", target["uid"])
            verified = json.loads(run(switch, "-c", "-t", kind, "-f", "json"))
            if verified["uid"] != target["uid"]:
                raise RuntimeError(f"{kind} route verification failed")
            changed.append(kind)
    if changed:
        LOG.info(json.dumps({"selected": target["name"], "connected": connected, "changed": changed}))
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    switch, blueutil = tools()
    if args.once:
        print(json.dumps(route_once(switch, blueutil)))
        return
    last_error = None
    while True:
        try:
            route_once(switch, blueutil)
            last_error = None
        except Exception as exc:
            # A failed Bluetooth query is not treated as a disconnect.
            if str(exc) != last_error:
                LOG.warning(f"routing: {exc}")
                last_error = str(exc)
        time.sleep(3)


if __name__ == "__main__":
    main()
