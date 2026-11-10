"""The VLM behind the caption worker: the backends (Ollama, mlx-vlm in
process, and the mlx-vlm server, vision/vlm_server.py), the caption prompt, and the tile and JPEG helpers they use.

`make_vlm` picks a backend with fallbacks; each backend has `caption(tile, memory)`, and Ollama's has `ground(frame)`
for object boxes (vision/objects.py's prompt and parser).
"""

import base64
import threading
import time
from collections.abc import Sequence

import cv2
import numpy as np
import requests

from liberational import config as cfg
from liberational import logs, model_runtime, ollama_health, voice_turn
from liberational.vision import objects
from liberational.vision.mlx_types import chat_prompt, loaded_processor, pil_images

VLM = cfg.get("models.ollama.vlm")
MLX_VLM = cfg.hf_repo("vlm")
MLX_SERVER_PORT = cfg.get("ports.vlm")  # liberational/vision/vlm_server.py, localhost only
# AVFoundation on macOS, V4L2 on Linux, OpenCV's choice elsewhere.
TILE_PX = 448
PROMPT = (
    "Describe only this single camera image for later factual questions, in at most 85 words. "
    "State visible person count (including zero, or unknown if obscured), positions, posture and observable activity. "
    "Describe notable objects and their positions and interactions. Only quote text when every character "
    "is clearly readable; never guess brands or complete partial letters. Otherwise say text unreadable. "
    "Report light fixtures/screens as on/off only if the fixture itself is clearly visible; otherwise uncertain. "
    "Room brightness alone is not proof. "
    "Explicitly mark blur, occlusion and uncertainty. Do not infer identity, intent, direction of travel, "
    "arrival/departure or elapsed time from one image. Plain factual text, no preamble."
)
# Prompt variants tried against the real previews (logs/vlm-mlx-probe-prompt.log): "Previous observations: ..." as
# a prefix makes the model parrot the earlier caption verbatim for every camera; "start with 'Changed:' if different"
# latches on at temperature 0 and dumps object lists. A forced first word Same/Changed judged against the earlier
# caption was right in both directions (steady scene -> Same; person from the earlier caption gone -> Changed).
PROMPT_WITH_PREV = (
    "You are the eyes of a wearable assistant. Earlier caption of this view: {prev}\n"
    "Look at THIS image. Answer in at most 20 words. First word: 'Same' if the scene matches that caption, "
    "'Changed' if not. Then describe: people, signs/text, vehicles, hazards, notable objects. "
    "Skip what is absent. No preamble."
)
MAX_NEW_TOKENS = 144  # bounded independent observation, not an abbreviated change label
VLM_NUM_CTX = ollama_health.VLM_NUM_CTX  # every call to the Ollama VLM, its warm-up included


def t() -> float:
    return time.perf_counter()


LOG = logs.get("see-vlm")


def log(msg: str, level: int = logs.INFO) -> None:
    LOG.log(level, msg)


def downscale(img: np.ndarray, px: int = TILE_PX) -> np.ndarray:
    h, w = img.shape[:2]
    s = px / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def to_jpeg_b64(img: np.ndarray, q: int = 80) -> str:
    _ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()


# ---------- VLM ----------
def build_prompt(memory: Sequence[str]) -> str:
    """Keep the legacy memory argument; never prime inference with stale observations."""
    return PROMPT  # independent observations: previous descriptions can anchor hallucinated presence


def _clean(txt: str) -> str:
    return " ".join(txt.strip().replace("\n", " ").split()[:110])  # defensive cap beyond the requested 85 words


class OllamaVLM:
    """Caption via Ollama /api/chat (qwen3-vl). Kept as the fallback path."""

    name = "ollama"

    def __init__(self, model: str = VLM, timeout: float = 30) -> None:
        self.model, self.timeout = model, timeout

    def warm(self) -> None:
        voice_turn.wait_quiet()
        with ollama_health.call(self.model):
            r = requests.post(
                f"{cfg.ollama_url()}/api/generate",
                json={"model": self.model, "keep_alive": -1, "options": {"num_ctx": VLM_NUM_CTX}},
                timeout=120,
            )
            r.raise_for_status()

    def caption(self, img: np.ndarray, memory: Sequence[str]) -> str:
        """memory = recent captions (so the model can say what changed)."""
        try:
            voice_turn.wait_quiet()
            with ollama_health.call(self.model):
                r = requests.post(
                    f"{cfg.ollama_url()}/api/chat",
                    timeout=self.timeout,
                    json={
                        "model": self.model,
                        "stream": False,
                        "keep_alive": -1,
                        "options": {
                            "num_predict": MAX_NEW_TOKENS,
                            "temperature": 0.2,
                            "num_ctx": VLM_NUM_CTX,
                        },  # never the 262k default
                        "messages": [{"role": "user", "content": build_prompt(memory), "images": [to_jpeg_b64(img)]}],
                    },
                )
                r.raise_for_status()
            return _clean(r.json()["message"]["content"])
        except ollama_health.Unavailable:
            return ""  # logged once by ollama_health, not on every frame
        except Exception as e:
            log(f"  vlm error: {e}", level=logs.WARN)
            return ""

    def ground(self, img: np.ndarray, prompt: str = objects.PROMPT) -> str:
        """Notable objects with boxes as the model's raw text (objects.parse reads it), or what `prompt` asks for
        in the same form (see/caption_animals.py asks for animals only). Raises on HTTP errors, and
        GroundingTruncated when the reply ran out of tokens."""
        voice_turn.wait_quiet()
        with ollama_health.call(self.model):
            r = requests.post(
                f"{cfg.ollama_url()}/api/chat",
                timeout=self.timeout,
                json={
                    "model": self.model,
                    "stream": False,
                    "keep_alive": -1,
                    "format": objects.SCHEMA,
                    "options": {"num_predict": objects.NUM_PREDICT, "temperature": 0, "num_ctx": VLM_NUM_CTX},
                    "messages": [{"role": "user", "content": prompt, "images": [to_jpeg_b64(img, 85)]}],
                },
            )
            r.raise_for_status()
        reply = r.json()
        if reply.get("done_reason") == "length":
            raise objects.GroundingTruncated(f"the grounding reply hit its {objects.NUM_PREDICT}-token limit")
        return reply["message"]["content"]


class MlxVLM:
    """Caption in-process with mlx-vlm (Qwen3-VL). Image tokens = (H/32)*(W/32) after the processor's
    resize, so a 448-px tile is ~100-200 tokens instead of Ollama's forced >=1024. Thread-affine:
    construct and call from the same (main) thread."""

    name = "mlx"

    def __init__(self, repo: str = MLX_VLM, px: int = TILE_PX, *, background: bool = True) -> None:
        from mlx_vlm import generate, load  # imported lazily: optional dependency
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        self.background = background
        self.repo, self.px, self._generate, self._tmpl = repo, px, generate, apply_chat_template
        self.thread = threading.get_ident()
        t0 = t()
        self.model, processor = load(repo)
        self.processor = loaded_processor(processor)
        self.cfg = load_config(repo)
        model_runtime.loaded("models.hf.vlm", repo, "camera descriptions", "Apple GPU", owner=self)
        log(f"mlx-vlm: loaded {repo} in {t() - t0:.1f}s")

    def _prompt(self, memory: Sequence[str]) -> str:
        return chat_prompt(self._tmpl, self.processor, self.cfg, build_prompt(memory))

    def warm(self) -> None:
        """First generate() pays Metal kernel compilation (~1-2 s); do it before the loop starts."""
        self.caption(np.full((self.px // 2, self.px, 3), 128, np.uint8), [])

    def caption(self, img: np.ndarray, memory: Sequence[str]) -> str:
        from PIL import Image

        assert threading.get_ident() == self.thread, "MlxVLM must be called on the thread that loaded it"
        try:
            if self.background:
                voice_turn.wait_quiet()
            img = downscale(img, self.px)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            r = self._generate(
                self.model,
                self.processor,
                self._prompt(memory),
                image=pil_images(pil),
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                verbose=False,
            )
            return _clean(r.text)
        except Exception as e:
            log(f"  vlm error: {e}", level=logs.WARN)
            return ""


class MlxServerVLM:
    """Caption via liberational/vision/vlm_server.py (mlx-vlm in its own process, launchd pack.vlm). Same prompt/model as
    MlxVLM, ~5 ms HTTP+JPEG overhead, and the camera process stays free of Metal work."""

    name = "mlx-server"

    def __init__(self, port: int = MLX_SERVER_PORT, timeout: float = 30) -> None:
        self.url, self.timeout = f"http://127.0.0.1:{port}", timeout
        h = requests.get(f"{self.url}/health", timeout=5).json()  # raises if the server is down -> fallback
        self.repo = h.get("model", "?")
        log(f"mlx-server: {self.url} serving {self.repo}")

    def warm(self) -> None:
        pass  # the server warms itself

    def caption(self, img: np.ndarray, memory: Sequence[str]) -> str:
        try:
            voice_turn.wait_quiet()
            r = requests.post(
                f"{self.url}/caption",
                timeout=self.timeout,
                json={"image_b64": to_jpeg_b64(img, 90), "memory": memory[-1:]},
            )
            r.raise_for_status()
            return _clean(r.json()["caption"])
        except Exception as e:
            log(f"  vlm error: {e}", level=logs.WARN)
            return ""


type Backend = OllamaVLM | MlxVLM | MlxServerVLM


def make_vlm(backend: str, ollama_model: str = VLM, mlx_repo: str = MLX_VLM, px: int = TILE_PX) -> Backend:
    """backend: 'mlx-server' (pack.vlm over localhost; falls back to in-process mlx, then ollama),
    'mlx' (in-process; falls back to ollama), or 'ollama'."""
    if backend == "mlx-server":
        try:
            return MlxServerVLM()
        except Exception as e:
            log(f"mlx-server unavailable ({type(e).__name__}: {e}); trying in-process mlx", level=logs.WARN)
    if backend in ("mlx-server", "mlx"):
        try:
            return MlxVLM(mlx_repo, px)
        except Exception as e:
            log(f"mlx-vlm unavailable ({type(e).__name__}: {e}); falling back to ollama", level=logs.WARN)
    return OllamaVLM(ollama_model)
