"""What the owner asked the assistant to remember: `memory.jsonl` under `paths.state`, one JSON object a line,
`{"ts", "op": "remember" | "forget", "text"}`, appended by the phone bridge ("remember that ...", "forget ...").

`items` is the one reader, for the phone bridge, the in-person voice path, and the dashboard's person card: they used
to read the log three ways, and a "forget everything" said on the phone still reached the voice agent (#3123).
Forgetting "everything", "all", "all of it", or nothing clears what came before; any other forget drops the notes
that contain its words or that its words contain, ignoring case.
"""

import json
import time
from pathlib import Path

CLEAR_ALL = ("everything", "all", "all of it")


def items(path: Path) -> list[str]:
    """The notes still remembered, oldest first ([] without the file). A line that isn't a JSON object is skipped."""
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    out: list[str] = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not isinstance(e, dict):
            continue
        op, text = e.get("op"), str(e.get("text") or "").strip()
        if op == "remember" and text:
            out.append(text)
        elif op == "forget":
            key = text.lower()
            if not key or key in CLEAR_ALL:
                out.clear()
            else:
                out = [i for i in out if key not in i.lower() and i.lower() not in key]
    return out


def append(path: Path, op: str, text: str) -> None:
    """Add a remember or forget to the log."""
    with path.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "op": op, "text": text}) + "\n")
