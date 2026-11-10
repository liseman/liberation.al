"""Object memory by voice, with no model: "remember this as my wallet" saves the view for the Objects page, and "where
did I leave my wallet?" answers from where it was last seen. It asks the dashboard's /objects API; call only
after the existing authorization succeeds."""

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any

from liberational import config as cfg


def request(path: str, data: dict[str, Any] | None = None) -> Any:
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        cfg.local_url("dashboard", "/objects" + path), data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=4) as r:
        return json.load(r)


def parse(text: str) -> tuple[str, str, str | None] | None:
    text = text.strip()
    capture = re.fullmatch(r"(?:please\s+)?remember\s+(?:this|that)(?:\s+(?:as\s+)?(.+?))?[.!?]*", text, re.I)
    if capture:
        name = (capture.group(1) or "").strip(" .!?")
        cam = re.search(r"\s+(?:on|from|in)\s+cam(?:era)?\s*(\d+)$", name, re.I)
        camera = "cam" + cam.group(1) if cam else "cam0"
        if cam:
            name = name[: cam.start()]
        return ("remember", name, camera)
    if re.match(
        r"^(?:please\s+)?(?:where (?:did I (?:leave|put)|(?:is|are|was|were) (?:my|the))|find (?:my|the)|when did you (?:last )?see|last seen)\b",
        text,
        re.I,
    ):
        return ("query", text, None)
    return None


def handle(text: str) -> str | None:
    parsed = parse(text)
    if not parsed:
        return None
    kind, value, camera = parsed
    try:
        if kind == "remember":
            assert camera is not None  # parse() names a camera for every capture
            request("/api/capture", {"camera": camera, "draft": True, "name": value})
            return f"I saved the view from camera {camera.removeprefix('cam')}. Mark the object in Object Memory on the dashboard to finish remembering it."
        data = request("/api/query?q=" + urllib.parse.quote(value))
        if data.get("status") != "last_seen":
            return data["answer"]
        e = data["evidence"]
        name = data["object"]["name"]
        when = time.strftime("%I:%M %p on %b %d", time.localtime(e["captured_at"]))
        verb = "You marked" if e["kind"] == "manual_label" else "I found a possible visual match for"
        return f"{verb} {name} in camera {e['camera'].removeprefix('cam')} at {when}. I cannot confirm it is still there. The image is in Object Memory."
    except Exception:
        return "Object memory could not retrieve or save that evidence. Nothing new is confirmed."
