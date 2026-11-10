"""Post-call review: for each new call under calls/, write calls/<day>/<dir>/review.md with timing stats and the
transcript, append a one-line entry to state/call_reviews.jsonl, and print a digest. Run by the pack.call-review
automation; Bob reads the digest and decides what to change in liberational/phone/bridge.py."""

import json
import re
from typing import Any

from liberational import config as cfg
from liberational.phone import bridge_log

CALLS, OUT = cfg.path("paths.calls"), cfg.path("paths.state") / "call_reviews.jsonl"


def main() -> None:
    seen: set[str] = set()
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                seen.add(json.loads(ln)["dir"])
            except Exception:
                pass
    log = bridge_log().splitlines()
    digest: list[dict[str, Any]] = []
    for d in sorted(CALLS.glob("*/*"))[-12:]:
        if d.name in seen or not (d / "call.json").exists():
            continue
        meta = json.loads((d / "call.json").read_text())
        sid = meta.get("callSid", "")
        rows = (
            [json.loads(l) for l in (d / "transcript.jsonl").read_text().splitlines() if l.strip()]
            if (d / "transcript.jsonl").exists()
            else []
        )
        lines = [l for l in log if sid and sid in l]
        lat = [float(m.group(1)) for l in lines for m in [re.search(r"first audio ([0-9.]+)s", l)] if m]
        barge = sum(1 for l in lines if "barge-in" in l)
        empty = sum(1 for l in lines if "heard" in l and ": ''" in l)
        peaks = [float(m.group(1)) for l in lines for m in [re.search(r"peak rms while .+? spoke: ([0-9.]+)", l)] if m]
        fallback = sum(1 for l in lines if "answering locally" in l)
        errors = sum(1 for l in lines if "failed" in l or "Traceback" in l)
        bot = [r for r in rows if r.get("speaker") == "bot"]
        caller = [r for r in rows if r.get("speaker") != "bot"]
        long_bot = sum(1 for r in bot if len((r.get("text") or "").split()) > 30)
        rec = dict(
            dir=d.name,
            day=d.parent.name,
            started=meta.get("started"),
            dur_s=round(meta.get("durationSeconds") or 0, 1),
            turns=len(rows),
            caller_turns=len(caller),
            bot_turns=len(bot),
            lat_median_s=(sorted(lat)[len(lat) // 2] if lat else None),
            lat_max_s=(max(lat) if lat else None),
            barge_ins=barge,
            empty_hears=empty,
            peak_rms_max=(max(peaks) if peaks else None),
            agent_fallbacks=fallback,
            errors=errors,
            long_bot_turns=long_bot,
            hung_up_by=("caller" if any("caller asked to hang up" in l for l in lines) else "end"),
        )
        md = [
            "# Call review " + str(meta.get("started")),
            "",
            "- " + ", ".join(f"{k}: {v}" for k, v in rec.items() if k not in ("dir", "day")),
            "",
            "## Transcript",
            "",
        ]
        md += [f"[{r.get('t', 0):7.2f}s] {r.get('speaker_label') or r.get('speaker')}: {r.get('text')}" for r in rows]
        (d / "review.md").write_text("\n".join(md) + "\n")
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        digest.append(rec)
        digest.append(
            {"transcript": [(r.get("speaker_label") or r.get("speaker"), (r.get("text") or "")[:160]) for r in rows]}
        )
    print(json.dumps(digest, indent=1) if digest else "NO_NEW_CALLS")


if __name__ == "__main__":
    main()
