#!/usr/bin/env python3
"""Make a synthetic two-speaker conversation wav (Kokoro af_heart + am_adam) for ambient.py --wav tests.
Usage: ambient_testwav.py out.wav [--gap 1.2]
Prints the ground-truth script as JSON lines (speaker, text) to stdout.
"""

import argparse
import json
import sys

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load as load_tts

from liberational import config as cfg

SCRIPT = [
    ("A", "Hey, did you get a chance to look at the collar wiring diagram last night?"),
    ("B", "Yeah, I went through it. I think we should move the throat mic transmitter to the left side of the collar."),
    ("A", "Okay, that works for me. Let's decide that now, left side it is."),
    ("B", "Also, remind me to order a spare battery for the Rode receiver before Friday."),
    ("A", "I'll add that to the list. By the way, Sarah said she can test the bone conduction speaker on Tuesday."),
    ("B", "Perfect. Tell her to bring the four ohm driver, not the eight ohm one."),
    ("A", "Will do. One more thing, the Pi for the camera stitching keeps overheating, we might need a heatsink."),
    ("B", "Let's just order the active cooler, it's cheap. I'll do it tonight."),
]
VOICES = {"A": "af_heart", "B": "am_adam"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--gap", type=float, default=1.2)
    a = ap.parse_args()
    m = load_tts(cfg.hf_repo("tts_kokoro"))
    sr = 24000
    parts = [np.zeros(int(0.8 * sr), np.float32)]
    for who, text in SCRIPT:
        aud = np.concatenate(
            [np.asarray(r.audio, dtype=np.float32) for r in m.generate(text=text, voice=VOICES[who], speed=1.0)]
        )
        parts += [aud, np.zeros(int(a.gap * sr), np.float32)]
        print(json.dumps({"speaker": who, "voice": VOICES[who], "text": text, "dur_s": round(len(aud) / sr, 2)}))
    y = np.concatenate(parts)
    sf.write(a.out, y, sr)
    print(f"wrote {a.out}: {len(y) / sr:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
