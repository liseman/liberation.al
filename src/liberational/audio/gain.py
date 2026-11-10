"""Auto gain (#410, #798): steer a microphone's level so speech peaks sit near audio.auto_gain_target_dbfs.

Joe: "We need an option to auto adjust mic gain", and on a mic whose input volume can be set over USB (his Elgato
Wave:3) it should move that, not scale samples. So each mic gets a gain stage: the mic's own capture volume when the
OS can set it (HardwareGain: on Linux the ALSA mixer control of the card PipeWire opened, through amixer), else a
software gain applied to the samples before the VAD, the level log, and the speech gate (SoftwareGain).

On Continuum the listen worker runs it (listen/worker.py): a MicGain per microphone in config/devices.toml hears each
utterance the VAD hands over and moves the stage between them. Which mics, from [audio.mic_auto_gain] in pack.toml
as it is now (followed within about a second, no restart):
- An entry is keyed by the mic's device name, the one Settings shows and writes (on Linux its exact PipeWire node
  name, as v1 keyed it), or by its name in config/devices.toml ("wave3"). The device name wins when both are set.
  The listen worker finds a mic's device name the way continuum-capture finds the device (device_name: the
  manifest entry's `match` against PipeWire's sources, or on macOS its `name` against the input devices).
- A mic with no entry has auto gain when it has a hardware control (Joe, 2026-10-30: capture sets the gain, as v1
  did with it turned on); software gain is off until turned on. `false` turns it off.

AutoGain decides, from the utterances the VAD hands over:
- It measures each utterance's speech peak: the 95th percentile of its 20 ms RMS, in dBFS. A sample peak would
  follow one plosive; the median would follow the quiet parts.
- It moves the gain only while nobody is talking: QUIET_S after the last utterance ended, and MIN_INTERVAL_S after
  its last change, so a voice never changes level in the middle of a sentence.
- One step goes toward the median peak of the last few utterances: at most UP_DB louder or DOWN_DB quieter, nothing
  within DEADBAND_DB of the target, and DOWN_DB quieter at once after an utterance that clipped (CLIP_SAMPLES samples
  at CLIP_DBFS, and its speech peak above the target, so a click in quiet speech doesn't count); after that step it
  doesn't rise again for CLIP_HOLD_S. Slow on purpose: a minute of speech moves a quiet mic about 5 dB.
- The gain stays within the hardware control's range (the Wave:3's is 0 to 40 dB), or SOFTWARE_DB for software
  gain.
Turning auto gain off leaves a hardware control where it is (it is the mic's level now) and puts software gain back
to 0 dB.
"""

import dataclasses
import functools
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from liberational import config as cfg
from liberational.audio import sensitivity

log = logging.getLogger(__name__)

TABLE = "mic_auto_gain"  # [audio.mic_auto_gain]: device name (or devices.toml name) -> true or false
FRAME_S = 0.02  # the RMS window a speech peak is measured over
PEAK_PERCENTILE = 95
MIN_UTTERANCE_S = 0.5  # shorter ones (a cough, a click) don't count
HISTORY = 5  # utterances the median is taken over
QUIET_S = 3.0  # after an utterance ends, before the gain may move
MIN_INTERVAL_S = 20.0  # between two moves
REREAD_S = 10.0  # how often a hardware gain is read back (it is an amixer call), to notice changes from outside
DEADBAND_DB = 3.0
UP_DB = 1.5
DOWN_DB = 3.0
CLIP_DBFS = -1.0  # a sample this loud is clipping
CLIP_SAMPLES = 4  # an utterance clipped when at least this many samples did: one click or plosive isn't a clip
CLIP_HOLD_S = 120.0  # after a step down for clipping, how long before the gain may rise again
SOFTWARE_DB = (-12.0, 30.0)


def setting(keys: Sequence[str]) -> bool | None:
    """The mic's own [audio.mic_auto_gain] entry, from pack.toml as it is now: the first of `keys` (its device name,
    then its devices.toml name) that has one, or None."""
    table = sensitivity._local_audio().get(TABLE)
    if not isinstance(table, dict):
        return None
    for key in keys:
        value = table.get(key)
        if isinstance(value, bool):
            return value
    return None


def enabled(keys: Sequence[str], hardware: bool) -> bool:
    """Whether auto gain is on for a mic: its own entry, else on when it has a hardware control."""
    own = setting(keys)
    return hardware if own is None else own


@functools.cache
def _shipped_target() -> object:
    return tomllib.loads(cfg.DEFAULTS.read_text())["audio"]["auto_gain_target_dbfs"]


def target_dbfs() -> float:
    """audio.auto_gain_target_dbfs now: pack.toml as it is now, else defaults.toml."""
    local = sensitivity._local_audio()
    value = local.get("auto_gain_target_dbfs")
    if value is None:
        value = _shipped_target()
    if isinstance(value, bool) or not isinstance(value, int | float) or not -60 <= value <= -3:
        raise ValueError("audio.auto_gain_target_dbfs must be a number from -60 to -3")
    return float(value)


def save(device: str, on: bool) -> None:
    """Turn auto gain on or off for one mic in pack.toml, keeping the rest of the file."""
    import tomlkit

    with cfg.LOCAL_WRITE:
        path = cfg.local_config()
        doc = tomlkit.parse(path.read_text()) if path.exists() else tomlkit.document()
        audio = doc.setdefault("audio", tomlkit.table())
        table = audio.setdefault(TABLE, tomlkit.table())
        table[device] = on  # kept when off too: a mic with a hardware control is on without an entry
        cfg.write_local(tomlkit.dumps(doc))


def speech_peak(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """(speech peak in dBFS, sample peak in dBFS) of one utterance."""
    n = max(1, int(sr * FRAME_S))
    frames = len(audio) // n
    if frames == 0:
        return -120.0, -120.0
    rms = np.sqrt(np.mean(np.square(audio[: frames * n].reshape(frames, n), dtype=np.float64), axis=1))
    level = float(np.percentile(rms, PEAK_PERCENTILE))
    top = float(np.max(np.abs(audio)))
    return 20 * math.log10(max(level, 1e-6)), 20 * math.log10(max(top, 1e-6))


class AutoGain:
    """Decides when and how far to move one mic's gain (pure: no clock or device of its own)."""

    def __init__(self, lo: float, hi: float, step_db: float = 0.0) -> None:
        self.lo, self.hi = lo, hi
        self.step_db = step_db  # the stage's own step: a move is at least one, or a coarse control never moves
        self.peaks: deque[float] = deque(maxlen=HISTORY)
        self.clip_peak: float | None = None  # the loudest speech peak of an utterance that clipped since the last move
        self.last_speech = -math.inf
        self.last_move = -math.inf
        self.last_clip_step = -math.inf

    def heard(
        self, peak_dbfs: float, top_dbfs: float, duration_s: float, ended: float, *, clipped_samples: int = 0
    ) -> None:
        """An utterance ended at `ended` (monotonic seconds); `clipped_samples` of its samples reached CLIP_DBFS."""
        self.last_speech = max(self.last_speech, ended)
        if top_dbfs >= CLIP_DBFS and clipped_samples >= CLIP_SAMPLES:
            self.clip_peak = max(peak_dbfs, self.clip_peak if self.clip_peak is not None else -math.inf)
        if duration_s >= MIN_UTTERANCE_S:
            self.peaks.append(peak_dbfs)

    def step(self, now: float, in_speech: bool, gain_db: float, target: float) -> float | None:
        """The gain to set now, or None to leave it."""
        if in_speech or now - self.last_speech < QUIET_S or now - self.last_move < MIN_INTERVAL_S:
            return None
        # A clip steps down only when that utterance was loud too: on devbox's Wave 3 a few clipped transients in
        # speech at -42 dBFS stepped the gain down, the quiet speech raised it again, and it hunted (2026-10-30).
        clip = self.clip_peak is not None and self.clip_peak > target
        self.clip_peak = None
        if clip:
            change = -DOWN_DB
        elif self.peaks:
            error = target - float(np.median(self.peaks))
            if abs(error) <= DEADBAND_DB:
                return None
            change = min(UP_DB, error) if error > 0 else max(-DOWN_DB, error)
            if change > 0 and now - self.last_clip_step < CLIP_HOLD_S:
                return None  # it clipped lately at a higher gain: don't climb straight back
        else:
            return None
        if 0 < abs(change) < self.step_db:
            change = math.copysign(self.step_db, change)
        new = min(self.hi, max(self.lo, gain_db + change))
        if abs(new - gain_db) < 0.25:
            return None  # at a bound already
        self.last_move = now
        if clip:
            self.last_clip_step = now
        return new

    def moved(self, by_db: float) -> None:
        """The gain changed by by_db (as the mixer now reads, which may differ from the step asked for, or come from
        outside): the speech heard so far would sound that much louder now, so one quiet stretch isn't counted
        twice."""
        if by_db:
            self.peaks = deque((p + by_db for p in self.peaks), maxlen=HISTORY)


class Stage(Protocol):
    kind: str  # "hardware" or "software"
    lo: float
    hi: float
    step_db: float  # the smallest change it makes (0: any)

    def get_db(self) -> float: ...
    def set_db(self, db: float) -> None: ...


@dataclasses.dataclass
class SoftwareGain:
    """A gain applied to the samples: the stand-in for a mic the OS can't set."""

    db: float = 0.0
    kind: str = "software"
    lo: float = SOFTWARE_DB[0]
    hi: float = SOFTWARE_DB[1]
    step_db: float = 0.0

    def get_db(self) -> float:
        return self.db

    def set_db(self, db: float) -> None:
        self.db = min(self.hi, max(self.lo, db))

    @property
    def factor(self) -> float:
        return 10 ** (self.db / 20)


Run = Callable[[list[str]], str]


def _run(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=5, check=True).stdout


_DB = re.compile(r"\[(-?\d+(?:\.\d+)?)dB\]")
_MINMAX = re.compile(r"dBminmax-min=(-?\d+(?:\.\d+)?)dB,max=(-?\d+(?:\.\d+)?)dB")
_SCALE = re.compile(r"dBscale-min=(-?\d+(?:\.\d+)?)dB,step=(\d+(?:\.\d+)?)dB")
_STEPS = re.compile(r"type=INTEGER,.*?min=(-?\d+),max=(-?\d+)")


@dataclasses.dataclass
class HardwareGain:
    """A mic's own capture volume: an ALSA simple mixer control, read and set in dB through amixer. Only its capture
    side: a headset's "Mic" can be the sidetone (playback) as well."""

    card: int
    control: str  # "Mic", or "Capture,1" past the first of a name
    lo: float
    hi: float
    run: Run = _run
    kind: str = "hardware"
    step_db: float = 0.0

    def get_db(self) -> float:
        out = self.run(["amixer", "-c", str(self.card), "sget", self.control])
        for line in out.split("Limits:", 1)[-1].splitlines():
            if "Capture" in line and (found := _DB.search(line.split("Capture", 1)[1])):
                return float(found.group(1))
        raise ValueError(f"amixer shows no capture dB for {self.control!r} on card {self.card}")

    def set_db(self, db: float) -> None:
        db = min(self.hi, max(self.lo, db))
        # "--" so a negative value isn't read as an option; "capture" so a sidetone is left alone
        self.run(["amixer", "-q", "-c", str(self.card), "--", "sset", self.control, "capture", f"{db:.2f}dB"])


PREFERRED = ("Mic", "Capture", "Mic Capture", "Internal Mic", "Microphone")


def _pw_dump(run: Run) -> list[dict[str, Any]]:
    """PipeWire's objects, or [] without pw-dump or when it fails."""
    if not shutil.which("pw-dump") and run is _run:
        return []
    try:
        objs = json.loads(run(["pw-dump"]) or "[]")
    except OSError, subprocess.SubprocessError, ValueError:
        return []
    return [o for o in objs if isinstance(o, dict)] if isinstance(objs, list) else []


def _props(obj: dict[str, Any]) -> dict[str, Any]:
    return (obj.get("info") or {}).get("props") or {}


def _alsa_card(device: str, run: Run) -> int | None:
    """The ALSA card PipeWire opened for this source node (its api.alsa.pcm.card), or None."""
    for o in _pw_dump(run):
        props = _props(o)
        if props.get("media.class") == "Audio/Source" and props.get("node.name") == device:
            card = props.get("api.alsa.pcm.card", props.get("api.alsa.card"))
            try:
                return int(card)
            except TypeError, ValueError:
                return None
    return None


CAPTURE_EXCLUDE = ("arducam", "iphone", "ipad", "continuity")  # continuum-capture's default [audio] exclude


def _usb_id(text: object) -> tuple[int, int] | None:
    """ "0fd9:0078" as (vendor, product), as continuum-capture's parse_usb reads it."""
    vendor, _, product = str(text).partition(":")
    try:
        return int(vendor, 16), int(product, 16)
    except ValueError:
        return None


def _hex(value: object) -> int | None:
    try:
        return int(str(value).removeprefix("0x"), 16)
    except ValueError:
        return None


def pipewire_node(match: Mapping[str, Any], exclude: Sequence[str] = CAPTURE_EXCLUDE, run: Run = _run) -> str | None:
    """The PipeWire node name of the source a config/devices.toml entry's `match` picks, as continuum-capture picks
    it (supervisor.rs `matches` over pipewire.rs `source_info`): every field given must match, `name` a
    case-insensitive piece of its description (else its nick, else its node name), `usb` its vendor and product,
    `path` its node name or serial, and a source with one of the manifest's [audio] exclude pieces in its name is
    skipped. None without PipeWire, or when no source or more than one matches: the wrong mic's gain never moves."""
    objs = _pw_dump(run)
    devices = {o.get("id"): _props(o) for o in objs if o.get("type") == "PipeWire:Interface:Device"}
    found = []
    for o in objs:
        p = _props(o)
        if p.get("media.class") != "Audio/Source" or not p.get("node.name"):
            continue
        node = str(p["node.name"])
        name = str(p.get("node.description") or p.get("node.nick") or node).lower()
        if any(piece.lower() in name for piece in exclude):
            continue
        if "name" in match and str(match["name"]).lower() not in name:
            continue
        if "usb" in match:
            dev = devices.get(p.get("device.id")) or {}
            vendor = _hex(p.get("device.vendor.id", dev.get("device.vendor.id")))
            product = _hex(p.get("device.product.id", dev.get("device.product.id")))
            want = _usb_id(match["usb"])
            if want is None or want != (vendor, product):
                continue
        if "path" in match and str(match["path"]) not in (node, str(p.get("object.serial", ""))):
            continue
        found.append(node)
    return found[0] if len(found) == 1 else None


def hardware(device: str, run: Run = _run) -> HardwareGain | None:
    """The mic's own capture volume, when the OS lets us set it: on Linux, a capture-volume control with a dB range
    on the ALSA card behind its PipeWire node (the Wave:3's "Mic", 0 to 40 dB; devbox's built-in "Capture", -17.25 to
    +30 dB). None elsewhere (macOS, a mic with no such control, no PipeWire), where software gain stands in."""
    if sys.platform != "linux" or (run is _run and not shutil.which("amixer")):
        return None
    card = _alsa_card(device, run)
    if card is None:
        return None
    try:
        controls = run(["amixer", "-c", str(card), "scontents"])
        contents = run(["amixer", "-c", str(card), "contents"])
    except OSError, subprocess.SubprocessError:
        return None
    capture: list[tuple[str, int]] = []
    for block in re.split(r"(?m)^Simple mixer control ", controls):
        m = re.match(r"'([^']+)',(\d+)", block)
        if m and "cvolume" in block:
            capture.append((m.group(1), int(m.group(2))))
    for name, index in sorted(
        capture, key=lambda c: (c[0] not in PREFERRED, PREFERRED.index(c[0]) if c[0] in PREFERRED else 0, c[1])
    ):
        limits = _limits(contents, name, index)
        if limits is not None:
            lo, hi, step = limits
            return HardwareGain(card, name if index == 0 else f"{name},{index}", lo, hi, run=run, step_db=step)
    return None


def _limits(contents: str, name: str, index: int = 0) -> tuple[float, float, float] | None:
    """The dB range amixer contents gives a simple control's capture volume: its element is "<name> Capture Volume"
    or, for a capture-only name like "Capture", "<name> Volume"; the range is a dBminmax, or a dBscale's minimum and
    step over its integer steps (an HDA codec's)."""
    elements = {f"name='{name} Capture Volume'", f"name='{name} Volume'"}
    for block in re.split(r"(?m)^numid=", contents):
        head = block.split("\n", 1)[0]
        if not any(e in head for e in elements) or ("Playback" in head):
            continue
        if (f",index={index}" in head) != (index != 0):
            continue
        steps = _STEPS.search(block)
        count = int(steps.group(2)) - int(steps.group(1)) if steps else 0
        if m := _MINMAX.search(block):
            lo, hi = float(m.group(1)), float(m.group(2))
            step = (hi - lo) / count if count > 0 else 0.0
        elif (m := _SCALE.search(block)) and count > 0:
            lo, step = float(m.group(1)), float(m.group(2))
            hi = lo + step * count
        else:
            continue
        if hi > lo:
            return lo, hi, step
    return None


def stage(device: str, run: Run = _run) -> Stage:
    """The mic's gain stage: its hardware control when there is one, else software gain."""
    try:
        found = hardware(device, run)
    except Exception:  # an odd mixer never stops capture: software gain stands in
        found = None
    return found if found is not None else SoftwareGain()


def _input_names(run: Run = _run) -> list[str]:
    """Every input device's name as PortAudio lists it, from a child process (PortAudio caches its list in-process,
    and the listen worker never opens a device itself)."""
    code = (
        "import json, sounddevice as sd; "
        "print(json.dumps([d['name'] for d in sd.query_devices() if d['max_input_channels'] > 0]))"
    )
    try:
        names = json.loads(run([sys.executable, "-c", code]) or "[]")
    except OSError, subprocess.SubprocessError, ValueError:
        return []
    return [str(n) for n in names] if isinstance(names, list) else []


def device_name(match: Mapping[str, Any], exclude: Sequence[str] = CAPTURE_EXCLUDE, run: Run = _run) -> str | None:
    """The device name Settings shows for a config/devices.toml entry, so [audio.mic_auto_gain] means the same mic
    to both: on Linux its PipeWire node (pipewire_node); elsewhere (macOS) the one input device whose name has the
    entry's `name` piece, as continuum-capture matches a CoreAudio device by its name (`usb` and `path` can't be
    checked there, so an entry without `name` finds none)."""
    if sys.platform.startswith("linux"):
        return pipewire_node(match, exclude, run)
    if "name" not in match:
        return None
    want = str(match["name"]).lower()
    found = {n for n in _input_names(run) if want in n.lower() and not any(p.lower() in n.lower() for p in exclude)}
    return found.pop() if len(found) == 1 else None


TICK_S = 1.0  # how often a MicGain looks at its setting and its gain
FIND_AGAIN_S = 30.0  # how often a hardware control is looked for again: after one failed, or while software stands in


class MicGain:
    """One microphone's auto gain as the listen worker runs it, the way v1's ambient ran it for each device: the
    stage is found the first time it ticks (PipeWire's node for the mic, then its mixer), [audio.mic_auto_gain] is
    followed live, the gain is read back every REREAD_S (a control rounds to its steps, and someone may move it by
    hand), and moved when AutoGain says so, each move logged in one line. A setting or mixer that can't be read
    keeps things as they are, logged once; a hardware control that fails, or software gain standing in, is
    looked for again every FIND_AGAIN_S.

    `name` is the mic's name in config/devices.toml; `device` gives its device name (`device_name`: its PipeWire
    node, or its input device on macOS; None when none is found); `find` gives a device's stage (`stage`, or a
    test's fake)."""

    def __init__(self, name: str, device: Callable[[], str | None], find: Callable[[str], Stage] = stage) -> None:
        self.name, self._device, self._find = name, device, find
        self.device: str | None = None
        self.stage: Stage | None = None
        self.auto: AutoGain | None = None
        self.on = False
        self.db: float | None = None
        self.read_at = self.ticked = self.lost_at = self.found_at = -math.inf
        self.error = ""
        self._changed_db = 0.0  # how far the gain moved since take_change()

    @property
    def keys(self) -> list[str]:
        """Its [audio.mic_auto_gain] keys, the one that wins first."""
        return [k for k in (self.device, self.name) if k]

    @property
    def factor(self) -> float:
        """What the samples are multiplied by now: software gain's factor, else 1."""
        return self.stage.factor if isinstance(self.stage, SoftwareGain) else 1.0

    def take_change(self) -> float:
        """How many dB the gain moved since the last call (a move of ours, one from outside, or software gain put
        back to 0 dB), so whatever learned the mic's levels (the VAD's floor) moves with it."""
        changed, self._changed_db = self._changed_db, 0.0
        return changed

    def heard(self, audio: np.ndarray, sr: int, ended: float) -> None:
        """An utterance the VAD handed over (after software gain), ended at `ended` (monotonic seconds)."""
        if self.auto is not None:
            peak, top = speech_peak(audio, sr)
            clipped = int(np.count_nonzero(np.abs(audio) >= 10 ** (CLIP_DBFS / 20)))
            self.auto.heard(peak, top, len(audio) / sr, ended, clipped_samples=clipped)

    def tick(self, now: float, in_speech: bool) -> None:
        """Follow the setting and move the gain if it's time (at most every TICK_S; returns at once otherwise)."""
        if now - self.ticked < TICK_S:
            return
        self.ticked = now
        try:
            self._tick(now, in_speech)
            self.error = ""
        except Exception as ex:  # a broken setting or mixer never stops listening
            if str(ex) != self.error:
                self.error = str(ex)
                log.warning("%s: auto gain: %s: %s", self.name, type(ex).__name__, ex)
            if isinstance(self.stage, HardwareGain) and isinstance(ex, OSError | subprocess.SubprocessError):
                self.stage, self.auto, self.db, self.lost_at = None, None, None, now

    def _tick(self, now: float, in_speech: bool) -> None:
        if self.stage is None:
            if now - self.lost_at < FIND_AGAIN_S:
                return
            self.found_at = now
            self.device = self._device()
            self._use(self._find(self.device) if self.device else SoftwareGain())
        elif isinstance(self.stage, SoftwareGain) and now - self.found_at >= FIND_AGAIN_S:
            self._find_again(now)
        stage_ = self.stage
        if stage_ is None:
            return
        on = enabled(self.keys, stage_.kind == "hardware")
        if on and self.auto is None:
            self.auto = AutoGain(stage_.lo, stage_.hi, stage_.step_db)
        if (on or self.on) and (self.db is None or now - self.read_at >= REREAD_S):
            self._read(stage_, now)
        if on and self.auto is not None and self.db is not None:
            peaks = list(self.auto.peaks)
            target = target_dbfs()
            new = self.auto.step(now, in_speech, self.db, target)
            if new is not None:
                was = self.db
                stage_.set_db(new)
                self._read(stage_, now)
                heard = f"speech peaks {float(np.median(peaks)):.1f} dBFS" if peaks else "near clipping"
                log.info(
                    "%s: auto gain %+.1f -> %+.1f dB (%s; %s, target %g)",
                    self.name,
                    was,
                    self.db,
                    stage_.kind,
                    heard,
                    target,
                )
        if on != self.on:
            if not on:
                if isinstance(stage_, SoftwareGain):
                    stage_.set_db(0.0)  # software gain is ours: back to as captured
                    self._changed_db -= self.db or 0.0
                    self.db = 0.0
                self.auto = None  # its history was at the old gain
            now_db = "" if self.db is None else f", {self.db:+.1f} dB now"
            log.info("%s: auto gain %s (%s gain%s)", self.name, "on" if on else "off", stage_.kind, now_db)
        self.on = on

    def _use(self, stage_: Stage) -> None:
        self.stage = stage_
        where = f" ({stage_.control} on card {stage_.card})" if isinstance(stage_, HardwareGain) else ""
        log.info(
            "%s: %s gain%s, %g to %g dB; device %s",
            self.name,
            stage_.kind,
            where,
            stage_.lo,
            stage_.hi,
            self.device or "not found",
        )

    def _find_again(self, now: float) -> None:
        """Software gain may only stand in because a lookup failed (PipeWire or the mixer not ready yet, the mic
        unplugged when a failed control was looked for again): look again every FIND_AGAIN_S, and take the mic's
        hardware control once there is one."""
        self.found_at = now
        device = self._device()
        if not device:
            return
        self.device = device
        found = self._find(device)
        if found.kind != "hardware":
            return
        self._changed_db -= self.db or 0.0  # the samples lose software gain's dB
        self.auto, self.db = None, None  # its history was at the old gain
        self._use(found)

    def _read(self, stage_: Stage, now: float) -> None:
        """Read the gain back, and tell AutoGain how far it really moved."""
        db = stage_.get_db()
        if self.db is not None:
            self._changed_db += db - self.db
            if self.auto is not None:
                self.auto.moved(db - self.db)
        self.db, self.read_at = db, now

    def status(self) -> dict[str, Any]:
        """{"name", "auto", "kind", "db"}: what Settings shows under the mic's toggle."""
        return {
            "name": self.name,
            "auto": self.on,
            "kind": self.stage.kind if self.stage is not None else None,
            "db": None if self.db is None else round(self.db, 1),
        }


STATUS = "listen.gain.json"  # under paths.state
FRESH_S = 30.0  # an older status file means the listen worker isn't running


def write_status(path: Path, gains: Iterable[MicGain]) -> None:
    """Each mic's gain status, by device name (else devices.toml name), for Settings: replaced whole."""
    mics = {g.device or g.name: g.status() for g in gains}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"ts": time.time(), "mics": mics}))
    tmp.replace(path)


def read_status(path: Path) -> dict[str, dict[str, Any]]:
    """The listen worker's gain status by device name, or {} when it is missing, unreadable, or stale."""
    try:
        data = json.loads(path.read_text())
        if time.time() - float(data["ts"]) > FRESH_S:
            return {}
        mics = data["mics"]
    except OSError, ValueError, KeyError, TypeError:
        return {}
    return {k: v for k, v in mics.items() if isinstance(v, dict)} if isinstance(mics, dict) else {}
