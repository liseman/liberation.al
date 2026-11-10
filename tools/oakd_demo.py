#!/usr/bin/env python3
"""DepthAI 3 RGB + metric stereo demo; private captures and optional LAN viewer."""

import argparse
import datetime
import fcntl
import io
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import depthai as dai
import numpy as np
from PIL import Image

PAGE = b"""<!doctype html><meta name="viewport" content="width=device-width"><title>OAK-D live RGB + depth</title><style>body{background:#151920;color:#eee;font:18px system-ui;margin:24px}img{width:48%;max-width:800px}pre{white-space:pre-wrap}</style><h1>OAK-D live RGB + metric depth</h1><p>RGB (left), stereo depth (right). Blue = near, red = far (0-5 m); black = invalid. Views are not pixel-aligned.</p><img id="rgb"><img id="depth"><pre id="status">Starting...</pre><script>async function tick(){let t=Date.now();document.getElementById('rgb').src='/rgb.jpg?t='+t;document.getElementById('depth').src='/depth.jpg?t='+t;try{let s=await(await fetch('/status')).json();document.getElementById('status').textContent=JSON.stringify(s,null,2)}catch(e){}setTimeout(tick,500)}tick()</script>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=20, help="0 means continuous")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--state", type=Path, default=Path.home() / ".local/state/oakd-demo")
    args = ap.parse_args()
    args.state.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = open(args.state / "device.lock", "w")  # noqa: SIM115  (held open for the process lifetime as an flock)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    status: dict[str, Any] = {"state": "starting"}  # declared, so ty does not narrow shared["status"] to str values
    shared: dict[str, Any] = {"status": status, "rgb": None, "depth": None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/":
                data, mime = PAGE, "text/html"
            elif path == "/status":
                data = json.dumps(shared["status"]).encode()
                mime = "application/json"
            elif path in ("/rgb.jpg", "/depth.jpg"):
                data = shared[path[1:-4]]
                mime = "image/jpeg"
                if data is None:
                    self.send_error(503)
                    return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = None
    if args.serve:
        server = ThreadingHTTPServer((args.bind, args.port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    counts = {"rgb": 0, "depth": 0}
    latest: dict[str, dict[str, Any]] = {}
    first: dict[str, float] = {}
    last_seen: dict[str, float] = {}
    valid_frames = 0

    def jpeg(array: np.ndarray) -> bytes:
        out = io.BytesIO()
        Image.fromarray(array).save(out, format="JPEG", quality=85)
        return out.getvalue()

    try:
        with dai.Pipeline() as pipeline:
            device = pipeline.getDefaultDevice()
            info = {
                "device_id": device.getDeviceId(),
                "usb_speed": str(device.getUsbSpeed()),
                "depthai": dai.__version__,
            }
            print(json.dumps(info), flush=True)
            rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = pipeline.create(dai.node.StereoDepth)
            left.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=10).link(stereo.left)
            right.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=10).link(stereo.right)
            stereo.setLeftRightCheck(True)
            stereo.setExtendedDisparity(True)
            stereo.initialConfig.setDepthUnit(dai.LengthUnit.MILLIMETER)
            queues = {
                "rgb": rgb.requestOutput((640, 360), type=dai.ImgFrame.Type.RGB888i, fps=10).createOutputQueue(
                    maxSize=2, blocking=False
                ),
                "depth": stereo.depth.createOutputQueue(maxSize=2, blocking=False),
            }
            pipeline.start()
            start = time.monotonic()
            report_at = start
            while pipeline.isRunning() and not stopped.is_set():
                now = time.monotonic()
                if args.seconds and now - start >= args.seconds:
                    break
                for name, q in queues.items():
                    msg = q.tryGet()
                    if msg is None:
                        continue
                    counts[name] += 1
                    last_seen[name] = now
                    timestamp = msg.getTimestamp().total_seconds()
                    first.setdefault(name, timestamp)
                    latest[name] = {
                        "sequence": msg.getSequenceNum(),
                        "device_timestamp_s": timestamp,
                        "width": msg.getWidth(),
                        "height": msg.getHeight(),
                    }
                    if name == "rgb":
                        arr = np.asarray(msg.getData(), dtype=np.uint8).reshape(msg.getHeight(), msg.getWidth(), 3)
                        shared[name] = jpeg(arr)
                    else:
                        arr = np.asarray(msg.getData()).view(np.uint16).reshape(msg.getHeight(), msg.getWidth())
                        valid = arr[arr > 0]
                        valid_frames += int(valid.size > 0)
                        latest[name].update(
                            valid_pixels=int(valid.size),
                            valid_fraction=float(valid.size / arr.size),
                            units="mm",
                            min_mm=int(valid.min()) if valid.size else None,
                            median_mm=float(np.median(valid)) if valid.size else None,
                            max_mm=int(valid.max()) if valid.size else None,
                        )
                        z = np.clip(arr.astype(float) / 5000, 0, 1)
                        color = np.stack((z * 255, (1 - abs(z * 2 - 1)) * 255, (1 - z) * 255), axis=-1).astype(np.uint8)
                        color[arr == 0] = 0
                        shared[name] = jpeg(color)
                        np.save(args.state / "depth-mm.npy", arr)
                if now - report_at >= 1:
                    shared["status"] = {
                        **info,
                        "state": "streaming",
                        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
                        "elapsed_s": round(now - start, 2),
                        "frames": dict(counts),
                        "valid_depth_frames": valid_frames,
                        "latest": dict(latest),
                        "first_device_timestamp_s": first.copy(),
                    }
                    print(json.dumps(shared["status"]), flush=True)
                    for name in queues:
                        if shared[name]:
                            (args.state / (name + ".jpg")).write_bytes(shared[name])
                    (args.state / "status.json").write_text(json.dumps(shared["status"], indent=2))
                    report_at = now
                if now - start > 12 and any(now - last_seen.get(k, start) > 10 for k in queues):
                    raise RuntimeError("RGB/depth frame watchdog expired")
                time.sleep(0.005)
            if not all(counts.values()) or not valid_frames:
                raise RuntimeError("No valid RGB + nonzero metric depth captured")
            shared["status"].update(state="stopped", frames=counts, valid_depth_frames=valid_frames)
            (args.state / "status.json").write_text(json.dumps(shared["status"], indent=2))
            print("PASS " + json.dumps(shared["status"]), flush=True)
    finally:
        if server:
            server.shutdown()
            server.server_close()
        lock.close()


if __name__ == "__main__":
    main()
