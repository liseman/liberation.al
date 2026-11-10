#!/usr/bin/env python3
"""Probe which capture modes every configured camera delivers at once; writes <logs>/camprobe.txt."""

import json
import sys
import time

import cv2

from liberational import config as cfg

CAP_BACKEND = (
    cv2.CAP_AVFOUNDATION
    if sys.platform == "darwin"
    else cv2.CAP_V4L2
    if sys.platform.startswith("linux")
    else cv2.CAP_ANY
)
res = {}
for W, H in [(1280, 720), (960, 540), (640, 480)]:
    caps = []
    ok = []
    for i in cfg.get("vision.cams"):
        c = cv2.VideoCapture(i, CAP_BACKEND)
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        c.set(cv2.CAP_PROP_FPS, 15)
        time.sleep(1.5)
        got = False
        for _ in range(20):
            r, f = c.read()
            if r:
                got = True
                break
            time.sleep(0.1)
        ok.append(got and f.shape[1])
        caps.append(c)
    res[f"{W}x{H}"] = ok
    for c in caps:
        c.release()
    time.sleep(2)
out = cfg.path("paths.logs") / "camprobe.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res))
print(out, json.dumps(res))
