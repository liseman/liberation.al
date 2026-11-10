"""ReSpeaker XVF3800 USB 4-Mic Array (Seeed, non-XIAO, USB firmware): host-side driver.

Phase 2b of the Liberational Pack: ambient mic array + bone-conduction speaker on
one USB-C cable. The XMOS does AEC/beamforming/DoA/VAD in hardware; this module
reads those results over the USB vendor control endpoint (pyusb) and finds the
UAC2 audio device for capture/playback (sounddevice).

  python xvf3800.py --probe              # enumerate USB + audio, read version/DoA/mute/routing
  python xvf3800.py --doa [--hz 10]      # live DoA/VAD stream (Ctrl-C to stop)
  python xvf3800.py --record 5 out.wav   # capture N s from the array (all channels)
  python xvf3800.py --play file.wav      # play through the array's speaker/AUX output
  python xvf3800.py --tone               # 1 kHz, -20 dBFS, 1 s test tone to the speaker header
  python xvf3800.py --mute / --unmute    # software mute (X0D30), same line as the button
  python xvf3800.py --amp on|off         # amplifier enable (X0D31, active-low)
  python xvf3800.py --get NAME / --set NAME v1 v2 ...   # any command in PARAMETERS
  python xvf3800.py --selftest           # exercise the protocol against a fake device (no HW)

Protocol (from respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY python_control/xvf_host.py):
  write: ctrl_transfer(OUT|VENDOR|DEVICE, bRequest=0, wValue=cmdid,      wIndex=resid, payload)
  read : ctrl_transfer(IN |VENDOR|DEVICE, bRequest=0, wValue=0x80|cmdid, wIndex=resid, len+1)
         response[0] = status (0 ok, 64 = retry), response[1:] = little-endian payload.
See NOTES-xvf3800.md for sources, register IDs and the speaker wiring checklist.
"""

import argparse
import ctypes.util
import math
import os
import pathlib
import struct
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

VID, PID = 0x2886, 0x001A  # Seeed "reSpeaker 3800" / "ReSpeaker XVF3800 4-Mic Array"
AUDIO_NAME_HINTS = ("xvf3800", "respeaker", "reSpeaker 3800")
RATE_DEFAULT = 16000  # default USB firmware is 16 kHz (6ch image = 16k6ch; 48k2ch variant exists)
if sys.platform == "darwin":
    LIBUSB_CANDIDATES = (
        "/opt/homebrew/lib/libusb-1.0.dylib",
        "/opt/homebrew/opt/libusb/lib/libusb-1.0.dylib",
        "/usr/local/lib/libusb-1.0.dylib",
    )
    LIBUSB_HINT = "brew install libusb"
else:
    LIBUSB_CANDIDATES = tuple(
        dict.fromkeys(
            p
            for p in (
                ctypes.util.find_library("usb-1.0"),
                "libusb-1.0.so.0",
                "/usr/lib/x86_64-linux-gnu/libusb-1.0.so.0",
                "/usr/lib/aarch64-linux-gnu/libusb-1.0.so.0",
                "/usr/lib64/libusb-1.0.so.0",
                "/usr/lib/libusb-1.0.so.0",
            )
            if p
        )
    )
    LIBUSB_HINT = "install libusb-1.0, e.g. apt install libusb-1.0-0"

# ---- USB capture channel layout -------------------------------------------------------------
# Default firmware (v2.1.0): L = processed auto-select beam (cat 8 / "user chosen" -> copies cat 6 src 3),
# R = ASR output of the auto-selected beam (per host_control README; wiki says R may be silence on
# some builds; read AUDIO_MGR_OP_R to confirm). 16k6ch image adds CH3..CH6; on v2.1.0 they are
# configurable (AUDIO_MGR_OP_CH3..6), on v2.0.8 they are fixed raw mics 0..3.
CH_PROCESSED = 0  # beamformed + AEC + NS + AGC (feed to ASR / diarization)
CH_ASR = 1  # ASR-tuned auto-select beam (less post-processing, no AGC)
CH_RAW_MIC0, CH_RAW_MIC1, CH_RAW_MIC2, CH_RAW_MIC3 = 2, 3, 4, 5  # only with 6-ch firmware + routing below
# (category, source) pairs for AUDIO_MGR_OP_*
SRC_SILENCE = (0, 0)
SRC_RAW_MIC = lambda i: (1, i)  # raw PDM mics, no gain/delay (0..3)
SRC_MIC_AMP_DELAY = lambda i: (3, i)  # what the SHF algorithm actually sees
SRC_REF_16K = (4, 0)  # far-end reference (our TTS) resampled to 16 kHz
SRC_BEAM_FOCUSED = lambda i: (6, i)  # i in 0,1
SRC_BEAM_SCANNING = (6, 2)
SRC_BEAM_AUTO = (6, 3)  # recommended beamformed output
SRC_ASR_BEAM = lambda i: (7, i)  # AEC residual / ASR beam
SRC_USER = lambda i: (8, i)
ROUTING_2CH_PLUS_RAW = {
    "AUDIO_MGR_OP_L": SRC_BEAM_AUTO,
    "AUDIO_MGR_OP_R": SRC_ASR_BEAM(3),
    "AUDIO_MGR_OP_CH3": SRC_RAW_MIC(0),
    "AUDIO_MGR_OP_CH4": SRC_RAW_MIC(1),
    "AUDIO_MGR_OP_CH5": SRC_RAW_MIC(2),
    "AUDIO_MGR_OP_CH6": SRC_RAW_MIC(3),
}

# ---- Beam indices for AEC_AZIMUTH_VALUES / AEC_SPENERGY_VALUES ---------------------------------
BEAM_FOCUS1, BEAM_FOCUS2, BEAM_FREE, BEAM_AUTO = 0, 1, 2, 3

# ---- Mic geometry (AEC_MIC_ARRAY_GEO, metres; 33 mm square) ----
MIC_GEO = {0: (0.033, -0.033), 1: (0.033, 0.033), 2: (-0.033, 0.033), 3: (-0.033, -0.033)}

# ---- GPIO ----
GPI_PINS = ("X1D09_MUTE_BTN", "X1D13", "X1D34")  # GPI_READ_VALUES order; X1D09 high = released
GPO_PINS = ("X0D11", "X0D30_MUTE", "X0D31_AMP_nEN", "X0D33_LED_PWR", "X0D39")  # GPO_READ_VALUES order
GPO_MUTE, GPO_AMP, GPO_LEDPWR = 1, 2, 3  # pin index for GPO_WRITE_VALUE
LED_OFF, LED_BREATH, LED_RAINBOW, LED_SOLID, LED_DOA, LED_RING = 0, 1, 2, 3, 4, 5

# name: (resid, cmdid, count, access, type, description), a subset of upstream xvf_host.py PARAMETERS
PARAMETERS = {
    "VERSION": (48, 0, 3, "ro", "uint8", "firmware MAJOR MINOR PATCH"),
    "BLD_MSG": (48, 1, 50, "ro", "char", "build config name"),
    "REBOOT": (48, 7, 1, "wo", "uint8", "reboot, params back to saved config"),
    "USB_BIT_DEPTH": (48, 8, 2, "rw", "uint8", "IN,OUT bit depth 16/24/32 (reboots)"),
    "SAVE_CONFIGURATION": (48, 9, 1, "wo", "uint8", "persist current params to flash"),
    "CLEAR_CONFIGURATION": (48, 10, 1, "wo", "uint8", "revert to defaults (then REBOOT)"),
    "AIC3104_HP_LEVEL": (48, 11, 1, "rw", "uint8", "3.5mm headphone level 0..9"),
    "AIC3104_LINEOUT_LEVEL": (48, 12, 1, "rw", "uint8", "line-out level 0..9 (feeds the speaker amp)"),
    "SHF_BYPASS": (33, 70, 1, "rw", "uint8", "bypass whole AEC/beamformer"),
    "AEC_MIC_ARRAY_GEO": (33, 74, 12, "ro", "float", "4x XYZ metres"),
    "AEC_AZIMUTH_VALUES": (33, 75, 4, "ro", "radians", "azimuth: focus1, focus2, free, auto"),
    "AEC_SPENERGY_VALUES": (33, 80, 4, "ro", "float", "speech energy per beam (>0 = speech)"),
    "AEC_FIXEDBEAMSAZIMUTH_VALUES": (33, 81, 2, "rw", "radians", "fixed-mode beam azimuths"),
    "AEC_FIXEDBEAMSONOFF": (33, 37, 1, "rw", "int32", "fixed focused beam mode"),
    "AEC_AECCONVERGED": (33, 3, 1, "ro", "int32", "AEC converged flag"),
    "AEC_RT60": (33, 9, 1, "ro", "float", "RT60 estimate s (neg = invalid)"),
    "AEC_FAR_EXTGAIN": (33, 5, 1, "rw", "float", "dB gain on far-end reference"),
    "AEC_ASROUTONOFF": (33, 35, 1, "rw", "int32", "0 = AEC residual per mic, 1 = ASR beams"),
    "AUDIO_MGR_MIC_GAIN": (35, 0, 1, "rw", "float", "pre-SHF mic gain"),
    "AUDIO_MGR_REF_GAIN": (35, 1, 1, "rw", "float", "pre-SHF reference gain"),
    "AUDIO_MGR_SELECTED_AZIMUTHS": (35, 11, 2, "ro", "radians", "[processed DoA, auto-beam DoA]; NaN = no speech"),
    "AUDIO_MGR_OP_L": (35, 15, 2, "rw", "uint8", "USB ch1 (category, source)"),
    "AUDIO_MGR_OP_R": (35, 19, 2, "rw", "uint8", "USB ch2"),
    "AUDIO_MGR_OP_CH3": (35, 28, 2, "rw", "uint8", "USB ch3 (6ch fw v2.1.0+)"),
    "AUDIO_MGR_OP_CH4": (35, 29, 2, "rw", "uint8", "USB ch4"),
    "AUDIO_MGR_OP_CH5": (35, 30, 2, "rw", "uint8", "USB ch5"),
    "AUDIO_MGR_OP_CH6": (35, 31, 2, "rw", "uint8", "USB ch6"),
    "AUDIO_MGR_SYS_DELAY": (35, 26, 1, "rw", "int32", "ref delay samples (tune for AEC w/ speaker)"),
    "GPO_READ_VALUES": (20, 0, 5, "ro", "uint8", "X0D11 X0D30 X0D31 X0D33 X0D39"),
    "GPO_WRITE_VALUE": (20, 1, 2, "wo", "uint8", "(pin_index, level)"),
    "LED_EFFECT": (20, 12, 1, "rw", "uint8", "0 off 1 breath 2 rainbow 3 solid 4 doa 5 ring"),
    "LED_BRIGHTNESS": (20, 13, 1, "rw", "uint8", "0..255"),
    "LED_COLOR": (20, 16, 1, "rw", "uint32", "0xRRGGBB"),
    "LED_DOA_COLOR": (20, 17, 2, "rw", "uint32", "(base, doa) colours"),
    "DOA_VALUE": (20, 18, 2, "ro", "uint16", "[deg 0..359, speech 0/1]"),
    "GPI_READ_VALUES": (36, 0, 3, "ro", "uint8", "X1D09(mute btn, 1=released) X1D13 X1D34"),
    "GPI_INDEX": (36, 1, 1, "rw", "uint8", "select GPI pin (0 = mute button)"),
    "GPI_EVENT_CONFIG": (36, 2, 1, "rw", "uint8", "0 none 1 falling(default) 2 rising 3 both"),
    "GPI_EVENT_PENDING": (36, 5, 1, "ro", "uint8", "read-and-clear event for selected pin"),
    "GPI_EVENT_PENDING_ALL": (36, 7, 1, "ro", "uint32", "read-and-clear event bitmap"),
    "MUTE_FUNCTION_ENABLE": (36, 8, 1, "rw", "uint8", "built-in mute button action on/off"),
    "PP_AGCONOFF": (17, 10, 1, "rw", "int32", "AGC on processed output"),
    "PP_MIN_NS": (17, 21, 1, "rw", "float", "stationary NS gain floor"),
    "PP_MIN_NN": (17, 22, 1, "rw", "float", "non-stationary NS gain floor"),
}
CONTROL_SUCCESS, SERVICER_COMMAND_RETRY = 0, 64
_FMT = {"uint8": "B", "char": "s", "uint16": "H", "uint32": "I", "int32": "i", "float": "f", "radians": "f"}
_SIZE = {"uint8": 1, "char": 1, "uint16": 2, "uint32": 4, "int32": 4, "float": 4, "radians": 4}


class NotFound(RuntimeError):
    pass


def _encode(ptype: str, count: int, values: str | bytes | Sequence[float]) -> bytes:
    if ptype == "char":
        # Text, or byte values: a char parameter never takes floats.
        b = values.encode() if isinstance(values, str) else bytes(cast(Sequence[int], values))
        return b[:count].ljust(count, b"\0")
    if len(values) != count:
        raise ValueError(f"expected {count} values, got {len(values)}")
    conv = float if ptype in ("float", "radians") else int
    return struct.pack("<" + _FMT[ptype] * count, *[conv(v) for v in values])


def _decode(ptype: str, count: int, payload: bytes) -> str | tuple[Any, ...]:
    if ptype == "char":
        return payload[:count].rstrip(b"\0").decode("utf-8", "ignore")
    return struct.unpack("<" + _FMT[ptype] * count, payload[: count * _SIZE[ptype]])


class XVF3800:
    """Vendor-control-endpoint client. Pass dev=None to try to open the real device."""

    TIMEOUT_MS = 2000
    MAX_RETRY = 100

    def __init__(self, dev: Any = None, vid: int = VID, pid: int = PID) -> None:
        self.dev: Any = dev if dev is not None else self._find(vid, pid)
        try:
            import usb.util as _u

            self._OUT = _u.CTRL_OUT | _u.CTRL_TYPE_VENDOR | _u.CTRL_RECIPIENT_DEVICE
            self._IN = _u.CTRL_IN | _u.CTRL_TYPE_VENDOR | _u.CTRL_RECIPIENT_DEVICE
        except ImportError:  # selftest without pyusb
            self._OUT, self._IN = 0x40, 0xC0
        self.last_rtt_ms: float | None = None

    # -- discovery --------------------------------------------------------------------------
    @staticmethod
    def backend() -> Any:
        """libusb backend; pyusb's default search misses Homebrew's lib dir on macOS, so try known paths too."""
        import usb.backend.libusb1 as lb

        be = lb.get_backend()
        if be is None:
            for p in LIBUSB_CANDIDATES:
                if "/" not in p or pathlib.Path(p).exists():  # bare sonames go to the dynamic loader
                    be = lb.get_backend(find_library=lambda _x, p=p: p)
                    if be:
                        break
        return be

    @classmethod
    def _find(cls, vid: int, pid: int) -> Any:
        try:
            import usb.core
        except ImportError as e:
            raise NotFound("pyusb missing: uv sync --inexact --extra mic-array") from e
        be = cls.backend()
        if be is None:
            raise NotFound(f"libusb not found ({LIBUSB_HINT})")
        dev = usb.core.find(idVendor=vid, idProduct=pid, backend=be)
        if dev is None:
            raise NotFound(f"XVF3800 not attached (no USB device {vid:04x}:{pid:04x})")
        return dev

    @staticmethod
    def list_usb() -> list[tuple[int, int, str]]:
        """All USB devices visible via libusb (for the probe)."""
        import usb.core

        be = XVF3800.backend()
        out: list[tuple[int, int, str]] = []
        for d in usb.core.find(find_all=True, backend=be) or []:
            try:
                name = f"{d.manufacturer or ''} {d.product or ''}".strip()
            except Exception:
                name = "?"
            out.append((d.idVendor, d.idProduct, name))
        return out

    # -- raw protocol -----------------------------------------------------------------------
    def read(self, name: str) -> Any:
        # A str for a char parameter, else a tuple of numbers: which depends on the name.
        resid, cmdid, count, access, ptype, _ = PARAMETERS[name]
        if access == "wo":
            raise ValueError(f"{name} is write-only")
        length = count * _SIZE[ptype] + 1
        t0 = time.perf_counter()
        for attempt in range(self.MAX_RETRY):
            resp = bytes(self.dev.ctrl_transfer(self._IN, 0, 0x80 | cmdid, resid, length, self.TIMEOUT_MS))
            if resp[0] == CONTROL_SUCCESS:
                break
            if resp[0] != SERVICER_COMMAND_RETRY:
                raise OSError(f"{name}: device status {resp[0]}")
            time.sleep(0.01)
        else:
            raise OSError(f"{name}: retry limit")
        self.last_rtt_ms = (time.perf_counter() - t0) * 1e3
        return _decode(ptype, count, resp[1:])

    def write(self, name: str, values: str | bytes | Sequence[float]) -> None:
        resid, cmdid, count, access, ptype, _ = PARAMETERS[name]
        if access == "ro":
            raise ValueError(f"{name} is read-only")
        payload = _encode(ptype, count, values)
        self.dev.ctrl_transfer(self._OUT, 0, cmdid, resid, payload, self.TIMEOUT_MS)

    def close(self) -> None:
        try:
            import usb.util

            usb.util.dispose_resources(self.dev)
        except Exception:
            pass

    # -- high-level ---------------------------------------------------------------------------
    def version(self) -> str:
        return ".".join(map(str, self.read("VERSION")))

    def get_doa(self) -> dict[str, Any]:
        """Direction of arrival of the auto-selected beam.
        Returns dict: deg (0..359, Seeed convention: 0° = USB-connector edge, increasing CCW when
        viewed from the mic/logo side; 90° left, 180° opposite the USB port, 270° right),
        speech (bool), beams_deg [focus1, focus2, free, auto], processed_deg (None if no speech)."""
        deg, speech = self.read("DOA_VALUE")
        az = self.read("AEC_AZIMUTH_VALUES")
        try:
            sel = self.read("AUDIO_MGR_SELECTED_AZIMUTHS")
            proc = None if math.isnan(sel[0]) else math.degrees(sel[0]) % 360
        except Exception:
            proc = None
        return {
            "deg": int(deg),
            "speech": bool(speech),
            "beams_deg": [math.degrees(a) % 360 for a in az],
            "processed_deg": proc,
            "rtt_ms": self.last_rtt_ms,
        }

    def get_vad(self) -> dict[str, Any]:
        """Speech energy per beam (float, 0 = no speech; bigger = louder/closer) + DOA_VALUE speech flag.
        Returns dict: speech (bool, firmware flag), energy [f1, f2, free, auto], energy_auto."""
        e = self.read("AEC_SPENERGY_VALUES")
        _, speech = self.read("DOA_VALUE")
        return {"speech": bool(speech), "energy": list(e), "energy_auto": e[BEAM_AUTO], "rtt_ms": self.last_rtt_ms}

    def get_mute(self) -> dict[str, Any]:
        """mute button (X1D09, 1 = released) and mute line (X0D30, 1 = mics muted + red LED)."""
        gpi = self.read("GPI_READ_VALUES")
        gpo = self.read("GPO_READ_VALUES")
        return {
            "button_pressed": gpi[0] == 0,
            "muted": bool(gpo[GPO_MUTE]),
            "amp_enabled": gpo[GPO_AMP] == 0,
            "led_power": bool(gpo[GPO_LEDPWR]),
            "gpi": list(gpi),
            "gpo": list(gpo),
        }

    def set_mute(self, on: bool) -> None:
        self.write("GPO_WRITE_VALUE", [GPO_MUTE, 1 if on else 0])

    def set_amp(self, on: bool) -> None:
        self.write("GPO_WRITE_VALUE", [GPO_AMP, 0 if on else 1])  # active low

    def set_led(self, effect: int = LED_DOA, color: int | None = None, brightness: int | None = None) -> None:
        self.write("LED_EFFECT", [effect])
        if color is not None:
            self.write("LED_COLOR", [color])
        if brightness is not None:
            self.write("LED_BRIGHTNESS", [brightness])

    def get_routing(self) -> dict[str, tuple[Any, ...] | str]:
        out: dict[str, tuple[Any, ...] | str] = {}
        for k in (
            "AUDIO_MGR_OP_L",
            "AUDIO_MGR_OP_R",
            "AUDIO_MGR_OP_CH3",
            "AUDIO_MGR_OP_CH4",
            "AUDIO_MGR_OP_CH5",
            "AUDIO_MGR_OP_CH6",
        ):
            try:
                out[k] = tuple(self.read(k))
            except Exception as e:  # 2-ch firmware has no CH3..6
                out[k] = f"n/a ({type(e).__name__})"
        return out

    def set_routing(self, table: Mapping[str, tuple[int, int]] = ROUTING_2CH_PLUS_RAW, save: bool = False) -> None:
        for k, v in table.items():
            self.write(k, list(v))
        if save:
            self.write("SAVE_CONFIGURATION", [1])


# ---- UAC2 audio side (sounddevice / CoreAudio) ------------------------------------------------
def find_audio_device() -> tuple[int | None, int | None, dict[str, Any] | None]:
    """(input_index, output_index, info) for the array's CoreAudio device(s), or (None, None, None)."""
    import sounddevice as sd

    inp = outp = info = None
    for i, d in enumerate(sd.query_devices()):
        n = d["name"].lower()
        if any(h.lower() in n for h in AUDIO_NAME_HINTS):
            info = d
            if d["max_input_channels"] > 0 and inp is None:
                inp = i
            if d["max_output_channels"] > 0 and outp is None:
                outp = i
    return inp, outp, info


def record(
    seconds: float, path: str | os.PathLike[str] | None = None, rate: int | None = None, channels: int | None = None
) -> tuple[np.ndarray, int]:
    import sounddevice as sd
    import soundfile as sf

    inp, _, info = find_audio_device()
    if inp is None:
        raise NotFound("XVF3800 audio input not present")
    assert info is not None  # found with the input
    rate = int(rate or info["default_samplerate"] or RATE_DEFAULT)
    channels = channels or info["max_input_channels"]
    a = sd.rec(int(seconds * rate), samplerate=rate, channels=channels, dtype="float32", device=inp)
    sd.wait()
    if path:
        sf.write(path, a, rate, subtype="PCM_24")
    return a, rate


def play(audio: ArrayLike, rate: int, device: int | str | None = None, block: bool = True) -> float:
    """Play mono/stereo float audio out of the array (speaker header + AUX). Resamples if needed."""
    import sounddevice as sd

    if device is None:
        _, device, info = find_audio_device()
        if device is None:
            raise NotFound("XVF3800 audio output not present")
        assert info is not None  # found with the output
        target = int(info["default_samplerate"] or RATE_DEFAULT)
    else:
        target = int(sd.query_devices(device)["default_samplerate"])
    audio = np.asarray(audio, np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if rate != target:
        n = int(round(len(audio) * target / rate))
        x = np.linspace(0, 1, len(audio), endpoint=False)
        xi = np.linspace(0, 1, n, endpoint=False)
        audio = np.stack([np.interp(xi, x, audio[:, c]) for c in range(audio.shape[1])], 1).astype(np.float32)
    t0 = time.perf_counter()
    sd.play(audio, target, device=device)
    if block:
        sd.wait()
    return (time.perf_counter() - t0) * 1e3


def tone(freq: float = 1000.0, secs: float = 1.0, dbfs: float = -20.0, rate: int = RATE_DEFAULT) -> NDArray[np.float32]:
    t = np.arange(int(secs * rate)) / rate
    env = np.minimum(1, np.minimum(t, secs - t) / 0.02)  # 20 ms fades
    return (10 ** (dbfs / 20) * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)


# ---- fake device for --selftest ---------------------------------------------------------------
class _FakeDev:
    """Emulates the firmware's control endpoint well enough to test encoding/decoding/retry."""

    def __init__(self) -> None:
        self.store: dict[tuple[int, int], bytes] = {
            (48, 0): _encode("uint8", 3, [2, 1, 0]),
            (20, 18): _encode("uint16", 2, [135, 1]),
            (33, 75): _encode("radians", 4, [math.radians(52.36), 0, math.pi / 2, math.radians(52.36)]),
            (33, 80): _encode("float", 4, [2080656.0, 0.0, 2083455.0, 2080656.0]),
            (35, 11): _encode("radians", 2, [math.radians(50), math.radians(52.36)]),
            (36, 0): _encode("uint8", 3, [1, 0, 0]),
            (20, 0): _encode("uint8", 5, [0, 0, 1, 1, 0]),
            (35, 15): _encode("uint8", 2, [8, 0]),
            (35, 19): _encode("uint8", 2, [7, 3]),
            (35, 28): _encode("uint8", 2, [1, 0]),
            (35, 29): _encode("uint8", 2, [1, 1]),
            (35, 30): _encode("uint8", 2, [1, 2]),
            (35, 31): _encode("uint8", 2, [1, 3]),
        }
        self.retry_once = {(33, 80)}
        self.writes: list[tuple[int, int, bytes]] = []

    def ctrl_transfer(
        self, bm: int, breq: int, wval: int, widx: int, data_or_len: int | bytes, timeout: int
    ) -> bytes | int:
        assert breq == 0
        if bm & 0x80:  # IN
            assert isinstance(data_or_len, int)  # a read passes the length
            key = (widx, wval & 0x7F)
            if key in self.retry_once:
                self.retry_once.discard(key)
                return bytes([SERVICER_COMMAND_RETRY]) + bytes(data_or_len - 1)
            return bytes([CONTROL_SUCCESS]) + self.store[key].ljust(data_or_len - 1, b"\0")
        assert isinstance(data_or_len, bytes)  # a write passes the payload
        self.writes.append((widx, wval, bytes(data_or_len)))
        if (widx, wval) == (20, 1):  # GPO_WRITE_VALUE -> reflect into GPO_READ_VALUES
            pin, lvl = data_or_len[0], data_or_len[1]
            g = bytearray(self.store[(20, 0)])
            g[pin] = lvl
            self.store[(20, 0)] = bytes(g)
        return len(data_or_len)


def selftest() -> bool:
    x = XVF3800(dev=_FakeDev())
    assert x.version() == "2.1.0"
    d = x.get_doa()
    assert d["deg"] == 135 and d["speech"] and abs(d["beams_deg"][BEAM_FREE] - 90) < 1e-3
    assert abs(d["processed_deg"] - 50) < 1e-3
    v = x.get_vad()
    assert v["speech"] and v["energy_auto"] > 0 and v["energy"][BEAM_FOCUS2] == 0
    m = x.get_mute()
    assert not m["muted"] and not m["button_pressed"] and not m["amp_enabled"]
    x.set_mute(True)
    assert x.get_mute()["muted"]
    x.set_amp(True)
    assert x.get_mute()["amp_enabled"]
    r = x.get_routing()
    assert r["AUDIO_MGR_OP_CH6"] == (1, 3)
    x.set_routing(ROUTING_2CH_PLUS_RAW)
    assert x.dev.writes[-1] == (35, 31, b"\x01\x03") and (35, 15, b"\x06\x03") in x.dev.writes
    x.write("LED_COLOR", [0xFF0000])
    assert x.dev.writes[-1] == (20, 16, b"\x00\x00\xff\x00")
    x.write("AEC_FIXEDBEAMSAZIMUTH_VALUES", [0.0, math.pi])
    assert len(x.dev.writes[-1][2]) == 8
    n, t0 = 2000, time.perf_counter()
    for _ in range(n):
        x.get_doa()
    us = (time.perf_counter() - t0) / n * 1e6
    print(
        f"selftest OK: protocol encode/decode/retry paths pass; get_doa() host overhead {us:.0f} µs "
        f"(3 control reads, excl. USB)"
    )
    return True


# ---- CLI ------------------------------------------------------------------------------------
def probe() -> int:
    print("== XVF3800 probe ==")
    try:
        import usb
        import usb.core

        be = XVF3800.backend()
        print(f"pyusb {usb.__version__}, libusb backend: {'ok' if be else f'MISSING ({LIBUSB_HINT})'}")
        if be:
            devs = XVF3800.list_usb()
            print(f"USB devices via libusb: {len(devs)}")
            for v, p, n in devs:
                print(f"  {v:04x}:{p:04x} {n}")
    except ImportError:
        print("pyusb not installed")
    try:
        import sounddevice as sd

        inp, outp, info = find_audio_device()
        print(
            f"Audio: array input={inp} output={outp}"
            + (
                f" ({info['name']}, in={info['max_input_channels']} "
                f"out={info['max_output_channels']} sr={info['default_samplerate']})"
                if info
                else ": not present"
            )
        )
        print("  other audio devices:", ", ".join(d["name"] for d in sd.query_devices()))
    except Exception as e:
        print(f"sounddevice error: {e}")
    try:
        x = XVF3800()
    except NotFound as e:
        print(f"control endpoint: {e}")
        print("-> nothing to read yet; run --selftest for the no-hardware check.")
        return 1
    try:
        print(f"firmware: {x.version()}  build: {x.read('BLD_MSG')!r}  rtt {x.last_rtt_ms:.1f} ms")
        print("geometry:", [round(g, 3) for g in x.read("AEC_MIC_ARRAY_GEO")])
        print("doa:", x.get_doa())
        print("vad:", x.get_vad())
        print("mute/gpio:", x.get_mute())
        print("routing:", x.get_routing())
        print(
            "levels: hp",
            x.read("AIC3104_HP_LEVEL"),
            "lineout",
            x.read("AIC3104_LINEOUT_LEVEL"),
            "sys_delay",
            x.read("AUDIO_MGR_SYS_DELAY"),
            "aec_converged",
            x.read("AEC_AECCONVERGED"),
        )
        print("usb bit depth (in,out):", x.read("USB_BIT_DEPTH"))
    finally:
        x.close()
    return 0


def doa_loop(hz: float) -> None:
    x = XVF3800()
    try:
        period = 1 / hz
        while True:
            t0 = time.perf_counter()
            d, v = x.get_doa(), x.get_vad()
            print(
                f"{time.strftime('%H:%M:%S')} doa={d['deg']:3d}° speech={int(d['speech'])} "
                f"proc={d['processed_deg'] if d['processed_deg'] is None else round(d['processed_deg'])} "
                f"E_auto={v['energy_auto']:.0f} beams={[round(b) for b in d['beams_deg']]} "
                f"rtt={d['rtt_ms']:.1f}ms",
                flush=True,
            )
            time.sleep(max(0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        x.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--doa", action="store_true")
    ap.add_argument("--hz", type=float, default=10)
    ap.add_argument("--record", nargs=2, metavar=("SECS", "WAV"))
    ap.add_argument("--play", metavar="WAV")
    ap.add_argument("--tone", action="store_true")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--unmute", action="store_true")
    ap.add_argument("--amp", choices=["on", "off"])
    ap.add_argument("--led", type=int, metavar="EFFECT")
    ap.add_argument("--route-default6", action="store_true", help="L/R=auto beam, CH3-6=raw mics (needs 16k6ch fw)")
    ap.add_argument("--save", action="store_true", help="SAVE_CONFIGURATION after writes")
    ap.add_argument("--get", metavar="NAME")
    ap.add_argument("--set", nargs="+", metavar="NAME_OR_VALUE")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, (r, c, n, acc, t, d) in PARAMETERS.items():
            print(f"{k:32s} resid={r:<3d} cmd={c:<3d} n={n:<2d} {acc} {t:8s} {d}")
        return 0
    if a.selftest:
        return 0 if selftest() else 1
    if a.probe:
        return probe()
    try:
        if a.doa:
            doa_loop(a.hz)
            return 0
        if a.record:
            secs, path = float(a.record[0]), a.record[1]
            x, r = record(secs, path)
            print(f"wrote {path}: {x.shape} @ {r} Hz")
            return 0
        if a.play or a.tone:
            if a.play:
                import soundfile as sf

                audio, r = sf.read(a.play, dtype="float32")
            else:
                audio, r = tone(), RATE_DEFAULT
            ms = play(audio, r)
            print(f"played {len(audio) / r:.2f} s (blocking call {ms:.0f} ms)")
            return 0
        if any([a.mute, a.unmute, a.amp, a.led is not None, a.route_default6, a.get, a.set]):
            x = XVF3800()
            try:
                if a.mute:
                    x.set_mute(True)
                if a.unmute:
                    x.set_mute(False)
                if a.amp:
                    x.set_amp(a.amp == "on")
                if a.led is not None:
                    x.set_led(a.led)
                if a.route_default6:
                    x.set_routing(ROUTING_2CH_PLUS_RAW)
                if a.get:
                    print(a.get, x.read(a.get.upper()))
                if a.set:
                    name, vals = a.set[0].upper(), a.set[1:]
                    conv = float if PARAMETERS[name][4] in ("float", "radians") else (lambda s: int(s, 0))
                    x.write(name, [conv(v) for v in vals])
                    print(name, "<-", vals)
                if a.save:
                    x.write("SAVE_CONFIGURATION", [1])
            finally:
                x.close()
            return 0
    except NotFound as e:
        print(f"not available: {e}", file=sys.stderr)
        return 2
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
