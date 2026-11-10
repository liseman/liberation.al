"""What the cameras show now: "what do you see", "what's on camera 2". The captions come from the vision log
(`camera.caption` on Continuum); this reads them, it keeps nothing.

Only the present is answered here. A question about the past ("what did cam0 see in the last hour", "what was on
camera at 3pm") is a history question: the voice loop asks the one reader (memory/history_qa.asks_about_the_past, from
memory/question.py's reading) and sends it to Ask, which reads the same captions over the window the reading names.
So this module reads no time out of a question, and a past question is never answered with the current view.
"""

import argparse
import json
import re
import time
from collections.abc import Mapping
from typing import Any

from liberational import vision_log
from liberational.vision import views

RECENT = 600  # "now" is the newest caption of the last ten minutes
GAP = 30  # a newest caption older than this is called stale

# Whether a transcribed line is a camera request at all. The voice loop checks every line it hears with this before
# anything is read by a model, so it stays a word check: one model call for each line heard isn't affordable.
CAMERA_Q = re.compile(r"\b(?:cameras?|cams?\s*(?:[0-9]+|zero|one|two)|what do you see)\b", re.I)


def cameras(request: str) -> list[str]:
    """The cameras a request names (configured IDs and names, and numbered ones), or all of them when it names none.
    Raises ValueError for an ambiguous name."""
    return views.cameras_named(request)[0] or views.camera_ids()


def query(request: str, log: vision_log.VisionLog | None = None, now: float | None = None) -> dict[str, Any]:
    """What each camera the request names shows now, from the vision log (`vision_log.at()` by default): its newest
    caption of the last ten minutes, or when it was last captioned."""
    now = time.time() if now is None else now
    log = log or vision_log.at()
    start = now - RECENT
    result: dict[str, Any] = {"request": request, "start": start, "end": now, "cameras": []}
    for cam in cameras(request):
        rows = [(r["ts"], r["caption"]) for r in log.captions(start, now + 0.001, [cam]) if r.get("caption")]
        last = rows[-1][0] if rows else _last_seen(log, cam, now)
        result["cameras"].append(
            dict(
                cam=cam,
                count=len(rows),
                last=last,
                stale=last is not None and now - last > GAP,
                latest=rows[-1][1] if rows else None,
            )
        )
    result["answer"] = render(result)
    return result


def _last_seen(log: vision_log.VisionLog, cam: str, now: float) -> float | None:
    """When `cam` was last captioned before `now`, looked for over the last day."""
    rows = log.captions(now - 86400, now + 0.001, [cam])
    return rows[-1]["ts"] if rows else None


def stamp(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def render(result: Mapping[str, Any]) -> str:
    lines = []
    for c in result["cameras"]:
        name = c["cam"]
        if not c["count"]:
            lines.append(
                f"{name}: no observations in the last {RECENT // 60} minutes."
                + (f" Last observation {stamp(c['last'])}, stale." if c["last"] else " No saved history.")
            )
            continue
        lines.append(
            f"{name}, observed at {stamp(c['last'])}" + (" (stale)" if c["stale"] else "") + ": " + c["latest"]
        )
    return " ".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    try:
        r = query(a.question)
    except ValueError as e:
        p.error(str(e))
    print(json.dumps(r, indent=2) if a.json else r["answer"])


if __name__ == "__main__":
    main()
