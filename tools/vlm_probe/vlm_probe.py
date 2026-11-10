#!/usr/bin/env python3
import sys
import time

import cv2
import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

from liberational import config as pack_config
from liberational.vision.mlx_types import chat_prompt, loaded_processor, pil_images

P_ORIG = (
    "You are the eyes of a wearable assistant. In at most 20 words, describe what matters "
    "in this view: people, signs/text, vehicles, hazards, and anything that changed. Plain text, no preamble."
)
P_TERSE = (
    "You are the eyes of a wearable assistant. One terse sentence, max 15 words: what matters in this view "
    "(people, signs/text, vehicles, hazards, changes). Skip what is absent. No preamble."
)
repo = sys.argv[1]
pxs = [int(x) for x in sys.argv[2].split(",")]
t0 = time.perf_counter()
model, loaded = load(repo)
processor = loaded_processor(loaded)
cfg = load_config(repo)
print(f"{repo} load {time.perf_counter() - t0:.1f}s", flush=True)


def preview(c: int) -> cv2.typing.MatLike:
    path = pack_config.path("paths.state") / "preview" / f"cam{c}.jpg"
    im = cv2.imread(str(path))
    assert im is not None, f"no camera preview at {path}"
    return im


imgs = [cv2.cvtColor(preview(c)[:-50], cv2.COLOR_BGR2RGB) for c in (0, 1, 2)]


def run(tag: str, prompt_txt: str, px: int, max_tokens: int, reps: int = 1) -> None:
    prompt = chat_prompt(apply_chat_template, processor, cfg, prompt_txt)
    lat = []
    for _ in range(reps):
        for i, im in enumerate(imgs):
            h, w = im.shape[:2]
            s = px / max(h, w)
            im2 = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else im
            t0 = time.perf_counter()
            r = generate(
                model,
                processor,
                prompt,
                image=pil_images(Image.fromarray(im2)),
                max_tokens=max_tokens,
                temperature=0.0,
                verbose=False,
            )
            dt = (time.perf_counter() - t0) * 1000
            lat.append(dt)
            print(
                f"  {tag} px={px} cam{i} {dt:6.0f} ms ptok={r.prompt_tokens} gtok={r.generation_tokens} ptps={r.prompt_tps:.0f} gtps={r.generation_tps:.0f} | {r.text.strip()!r}",
                flush=True,
            )
    lat = sorted(lat)
    print(f"{tag} px={px}: median {lat[len(lat) // 2]:.0f} ms min {lat[0]:.0f} max {lat[-1]:.0f}", flush=True)


run("warm", P_ORIG, 448, 40)
for px in pxs:
    run("orig", P_ORIG, px, 40)
    run("terse", P_TERSE, px, 28)
print("peak mem GB", mx.get_peak_memory() / 1e9)
