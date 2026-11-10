#!/usr/bin/env python3
import time

import cv2
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

from liberational import config as pack_config
from liberational.vision.mlx_types import chat_prompt, loaded_processor, pil_images

repo = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
model, loaded = load(repo)
processor = loaded_processor(loaded)
cfg = load_config(repo)
BASE = (
    "You are the eyes of a wearable assistant. Describe THIS image in at most 20 words: people, signs/text, "
    "vehicles, hazards, notable objects. Skip what is absent. Plain text, no preamble."
)
V = {
    "E_changed_only_if_salient": lambda prev: (
        BASE[: -len("Plain text, no preamble.")] + "Earlier caption (may be wrong, do not copy it): " + prev + "\n"
        "Start with 'Changed:' ONLY if a person, vehicle, sign or hazard has appeared or disappeared since then; otherwise just describe. Plain text, no preamble."
    ),
    "F_prev_as_anticopy": lambda prev: BASE + " Do not repeat the earlier caption: " + prev,
    "G_two_step": lambda prev: (
        "You are the eyes of a wearable assistant. Earlier caption of this view: " + prev + "\n"
        "Look at THIS image. Answer in at most 20 words. First word: 'Same' if the scene matches that caption, 'Changed' if not. "
        "Then describe: people, signs/text, vehicles, hazards, notable objects. Skip what is absent. No preamble."
    ),
}


def preview(c: int) -> cv2.typing.MatLike:
    path = pack_config.path("paths.state") / "preview" / f"cam{c}.jpg"
    im = cv2.imread(str(path))
    assert im is not None, f"no camera preview at {path}"
    return im


imgs = [cv2.cvtColor(preview(c)[:-50], cv2.COLOR_BGR2RGB) for c in (0, 1, 2)]


def cap(prompt_txt: str, im: cv2.typing.MatLike) -> tuple[float, str]:
    prompt = chat_prompt(apply_chat_template, processor, cfg, prompt_txt)
    t0 = time.perf_counter()
    r = generate(
        model, processor, prompt, image=pil_images(Image.fromarray(im)), max_tokens=32, temperature=0.0, verbose=False
    )
    return 1000 * (time.perf_counter() - t0), r.text.strip()


first = [cap(BASE, im)[1] for im in imgs]
for i, f in enumerate(first):
    print(f"BASE cam{i} | {f!r}")
for name, fn in V.items():
    for i, im in enumerate(imgs):
        ms, txt = cap(fn(first[i]), im)
        print(f"{name:26s} cam{i} steady {ms:5.0f} ms | {txt!r}", flush=True)
        ms, txt = cap(fn(txt), im)
        print(f"{name:26s} cam{i} steady2 {ms:5.0f} ms | {txt!r}", flush=True)
        ms, txt = cap(fn("A man in a red jacket stands by the workbench, holding a drill."), im)
        print(f"{name:26s} cam{i} person-gone {ms:5.0f} ms | {txt!r}", flush=True)
