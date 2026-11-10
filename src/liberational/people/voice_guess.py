"""The "Who's this?" rows for unnamed voices ("Speaker N"), in the same shape as the face rows (vision/face_guess.py):
a best guess at whose voice it is, its score and band, and why the guess is as sure as it is (issue #347).

The guess compares the voice's centroid (the average of its stored ReDimNet2 embeddings) with every named person's
voices, and takes each person's closest. The score is the lower of likeness and lead, as for faces. LOW and HIGH
span 0.20 below and above listen's anonymous cluster admission threshold, placing that threshold at 50, with the
same lead over the runner-up. These are ranking heuristics, not probabilities or automatic person assignment
gates. The bands and their wording are the faces' too.

A named voice counts by its centroid and by each of its stored embeddings (its refs), as listen matches a line to
the nearest stored one. A voiceprint of another size, or one that isn't finite, is left out rather than failing
every guess.

The reason line names what holds the guess back: a close runner-up, few lines heard, short clips, or lines that
don't all sound alike: the median match of all its lines to the voice is low, which is what a voice holding several
people looks like, and what "Multiple speakers" is for.
"""

import statistics
from typing import Any
from urllib.parse import quote

import numpy as np
from numpy.typing import NDArray

from liberational.listen import voices
from liberational.vision.face_guess import MAYBE, PROBABLE, band, lead

LOW, HIGH = voices.THRESHOLD - 0.20, voices.THRESHOLD + 0.20  # cluster admission at a likeness of 50
FEW = 5  # heard fewer lines than this: too little to go on
SHORT = 2.0  # median clip under this many seconds: too short to tell
MIXED = voices.THRESHOLD + 0.07  # median line similarity below this suggests a mixed cluster


def likeness(sim: float) -> int:
    return round(100 * min(1.0, max(0.0, (sim - LOW) / (HIGH - LOW))))


def _unit(v: NDArray[np.floating]) -> NDArray[np.float32]:
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _vector(v: Any) -> NDArray[np.float32] | None:
    try:
        a = np.asarray(v, np.float32).reshape(-1)
    except TypeError, ValueError:
        return None
    return a if a.size and np.all(np.isfinite(a)) and np.any(a) else None


def best_guess(centroid: Any, names: list[str], centroids: Any) -> dict[str, Any] | None:
    """The named person whose voice is most like this one: {"name", "score", "band", "close"}, or None with no
    voiceprint to compare (this voice's, or any named one of the same size)."""
    vec = _vector(centroid)
    if vec is None:
        return None
    pairs = [(n, e) for n, c in zip(names, centroids) if (e := _vector(c)) is not None and e.shape == vec.shape]
    if not pairs:
        return None
    sims = _unit(np.stack([e for _, e in pairs])) @ _unit(vec)
    per: dict[str, float] = {}
    for (n, _), s in zip(pairs, sims.tolist()):
        if n not in per or s > per[n]:
            per[n] = s
    ranked = sorted(per.items(), key=lambda kv: -kv[1])
    (name, sim), second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    alike = likeness(sim)
    ahead = lead(sim - second[1]) if second else 100
    pts = min(alike, ahead)
    return {"name": name, "score": pts, "band": band(pts), "close": second[0] if second and ahead < alike else None}


def reason(
    guess: dict[str, Any] | None, count: int, dur: float | None, match: float | None, anyone: bool = True
) -> str:
    """One line on why the guess is as sure as it is: what holds it back, or that nothing does. anyone: whether
    anybody is named at all."""
    if guess is None:
        return "No voiceprint to compare yet" if anyone else "Nobody is named yet, so there's no one to compare with"
    issues = []
    if guess.get("close") and guess["band"] != "probable":
        issues.append(f"sounds like {guess['close']} too")
    if match is not None and match < MIXED:
        issues.append("its lines don't all sound alike")
    if count < FEW:
        issues.append(f"only {count} {'line' if count == 1 else 'lines'} heard")
    if dur is not None and dur < SHORT:
        issues.append("short clips")
    if issues:
        text = " and ".join(issues[:2])
        return text[0].upper() + text[1:]
    if guess["band"] == "probable":
        return f"A clear voice, close to {guess['name']}'s"
    if guess["band"] == "maybe":
        return f"A clear voice, but only partly like {guess['name']}'s"
    return "A clear voice, unlike anyone named"


def rows(
    voices: list[dict[str, Any]],
    named: list[dict[str, Any]],
    clips: list[dict[str, Any]],
    sims: dict[int, list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Rows for the unnamed voices ([{"id", "label", "centroid", "n", "last_seen"}]) against the named ones
    ([{"label", "centroid", "refs"}]), with their clips ([{"id", "ts", "line", "text", "dur", "sim"}], most useful first) and
    every playable line's match to its voice (sims, by voice id), best guess first."""
    names: list[str] = []
    prints: list[Any] = []
    for v in named:
        for c in [v.get("centroid"), *(v.get("refs") or [])]:
            if c is not None:
                names.append(str(v["label"]))
                prints.append(c)
    out: list[dict[str, Any]] = []
    for v in voices:
        mine = [c for c in clips if c["id"] == v["id"]]
        guess = best_guess(v.get("centroid"), names, prints)
        if guess is not None:
            guess = {**guess, "id": guess["name"]}  # the name to send back, as faces' and pets' guesses (#2366)
        count = int(v.get("n") or 0)
        dur = statistics.median(float(c["dur"]) for c in mine) if mine else None
        heard = (sims or {}).get(v["id"]) or []
        match = statistics.median(heard) if heard else None
        out.append(
            {
                "id": int(v["id"]),
                "label": str(v["label"]),
                "count": count,
                "last_seen": v.get("last_seen"),
                # each clip by its line's key (#247): two microphones can start lines in the same millisecond
                "clips": [
                    {k: c[k] for k in ("ts", "line", "text", "dur")} | {"url": f"/clip?line={quote(c['line'])}"}
                    for c in mine
                ],
                "guess": guess if guess and guess["band"] != "none" else None,
                "score": guess["score"] if guess else 0,
                "band": guess["band"] if guess else "none",
                "reason": reason(guess, count, dur, match, anyone=bool(names)),
            }
        )
    rank = {"probable": 0, "maybe": 1, "none": 2}
    out.sort(key=lambda r: (rank[r["band"]], -r["score"] if r["band"] != "none" else 0, -(r["last_seen"] or 0)))
    return out


__all__ = ["MAYBE", "PROBABLE", "best_guess", "reason", "rows"]
