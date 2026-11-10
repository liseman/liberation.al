#!/usr/bin/env python3
"""Phase 0: load every model concurrently and hammer them for N minutes. Apple Silicon (MLX) only.
Pass = no errors, latency stable (no thermal throttling drift)."""

import sys
import time
from typing import Any

import numpy as np
import requests

from liberational import config as cfg

MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 10


def ollama(model: str, prompt: str, images: list[str] | None = None) -> tuple[float, str]:
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": -1,
        "options": {"num_predict": 30, "num_ctx": 4096},  # as every service asks: another size reloads the model
    }
    if images:
        body["images"] = images
    t0 = time.perf_counter()
    r = requests.post(f"{cfg.ollama_url()}/api/generate", json=body, timeout=120).json()
    return time.perf_counter() - t0, r.get("response", "")[:60]


def main() -> None:
    print("loading MLX audio models...")
    from liberational.speech.asr import ASR

    asr = ASR()
    from mlx_audio.tts.utils import load as load_tts

    tts = load_tts(cfg.hf_repo("tts_kokoro"))
    # test image: grey 448x448 jpeg
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (448, 448), (120, 120, 120)).save(buf, "JPEG")
    img = base64.b64encode(buf.getvalue()).decode()

    stats: dict[str, list[float]] = {k: [] for k in ["asr", "tts", "llm", "vlm"]}
    end = time.time() + MINUTES * 60
    i = 0
    while time.time() < end:
        i += 1
        t0 = time.perf_counter()
        asr(np.random.randn(16000).astype(np.float32) * 0.01)
        stats["asr"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        list(tts.generate(text="Testing one two three.", voice=cfg.get("tts.voice")))
        stats["tts"].append(time.perf_counter() - t0)
        d, _ = ollama(cfg.get("models.ollama.llm"), f"Say a random word. #{i}")
        stats["llm"].append(d)
        d, _ = ollama(cfg.get("models.ollama.vlm"), "Describe this image in 5 words.", [img])
        stats["vlm"].append(d)
        if i % 5 == 0:
            print(
                f"[{(end - time.time()) / 60:.1f} min left] "
                + " ".join(f"{k}={1000 * np.mean(v[-5:]):.0f}ms" for k, v in stats.items())
            )
    print("\n=== summary (first 5 vs last 5 iterations) ===")
    ok = True
    for k, v in stats.items():
        a, b = np.mean(v[:5]), np.mean(v[-5:])
        drift = b / a
        print(f"{k}: start {1000 * a:.0f} ms  end {1000 * b:.0f} ms  drift x{drift:.2f}")
        if drift > 1.5:
            ok = False
    print("PASS" if ok else "FAIL (latency drifted >1.5x, thermal?)")


if __name__ == "__main__":
    main()
