"""Tiny localhost caption server: Qwen3-VL via mlx-vlm, one model, one thread. Apple Silicon (MLX) only.

Why a separate process: running Metal/MLX in the camera process coincided with capture fps decaying to zero
and AVFoundation re-opens hanging, while the out-of-process Ollama path had been stable for hours. This keeps the GPU
work out of the caption worker (and lets other tools, e.g. identity.py, reuse the resident model).

  python vlm_server.py [--port PORT] [--model REPO] [--px 448]   # defaults: ports.vlm, models.hf.vlm

  POST /caption  {"image_b64": <jpeg/png>, "memory": [..recent captions..]}  ->  {"caption": str, "ms": int}
  GET  /health   ->  {"ok": true, "model": ..., "n": captions served}

Single-threaded HTTPServer on purpose: MLX streams are thread-affine and the GPU serialises anyway.
The request/response contract lives in see/vlm.py's MlxServerVLM.
"""

import argparse
import base64
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar

import cv2
import numpy as np

from liberational import config as cfg
from liberational import logs, model_runtime
from liberational.see import vlm as vl

LOG = logs.get("vlm")


def log(msg: str, level: int = logs.INFO) -> None:
    LOG.log(level, msg)


class Handler(BaseHTTPRequestHandler):
    vlm: ClassVar[vl.MlxVLM]  # set by main() before serving
    n = 0

    def log_message(self, format: str, *args: Any) -> None:
        pass  # quiet; we log captions ourselves

    def _json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._json(200, {"ok": True, "model": self.vlm.repo, "n": Handler.n})
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path not in ("/caption", "/inspect"):
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 4 * 1024 * 1024:
                return self._json(400, {"error": "invalid request size"})
            req = json.loads(self.rfile.read(length))
            buf = np.frombuffer(base64.b64decode(req["image_b64"]), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                return self._json(400, {"error": "bad image"})
            t0 = time.perf_counter()
            if self.path == "/inspect":
                from liberational.vision.review_inference import inspect

                observation = inspect(self.vlm, img, req.get("question", "Describe the recorded scene"))
                return self._json(200, {"observation": observation, "ms": round(1000 * (time.perf_counter() - t0))})
            cap = self.vlm.caption(img, req.get("memory") or [])
            ms = round(1000 * (time.perf_counter() - t0))
            Handler.n += 1
            if Handler.n % 50 == 1:
                log(f"caption #{Handler.n} {ms} ms {img.shape[1]}x{img.shape[0]} {cap[:60]!r}")
            self._json(200, {"caption": cap, "ms": ms})
        except Exception as e:
            log(f"{e}", level=logs.ERROR)
            self._json(500, {"error": str(e)})


@model_runtime.reporting("vlm")
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=vl.MLX_SERVER_PORT)
    ap.add_argument("--model", default=vl.MLX_VLM)
    ap.add_argument("--px", type=int, default=vl.TILE_PX)
    a = ap.parse_args()
    if not cfg.has_mlx():
        sys.exit("vlm_server.py needs Apple Silicon with MLX installed; elsewhere the caption worker uses Ollama")
    # The caption client waits before HTTP. Waiting again here would block foreground /inspect on this thread.
    Handler.vlm = vl.MlxVLM(a.model, a.px, background=False)
    t0 = time.perf_counter()
    Handler.vlm.warm()
    log(f"warm {1000 * (time.perf_counter() - t0):.0f} ms; serving http://{a.host}:{a.port}")
    HTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
