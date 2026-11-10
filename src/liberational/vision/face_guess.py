"""Live's "Who's this?": each unnamed face group with its pictures, how often it was seen, a best guess at who it is,
and why the guess is as sure as it is.

The guess compares the group's average embedding (vision/faces.py keeps one per "Face N") with every photo of a named
person (faces/db.npz), and takes each person's closest photo. The score (0 to 100) is the lower of two parts:

- Likeness: cosine similarity on a straight line from LOW (0) to HIGH (100), so the live match threshold (0.45) sits
  at 50.
- Lead: how far the best person is ahead of the runner-up, from 50 when they tie to 100 at a gap of 2 x CLOSE, so a
  gap of CLOSE gives 75. With nobody else named the lead doesn't limit the score.

A score is how sure the guess is, not a probability that it's right. Bands, for the colour and the wording:

- "probable" (75 and up): clearly past the live threshold, and clearly ahead of anyone else.
- "maybe" (50 to 74): about as alike as one live frame needs to be named, or alike but close to someone else too.
- "none" (under 50), or nobody named yet: no guess is shown.

The reason line names what holds the guess back, from measurements already stored: a close runner-up (the lead is
what set the score), how often the group was seen, the face detector's confidence on its sightings (low for a face
turned away or blurred), how small the face was in the frame, and how dark its photo is. A group's sightings are the
unnamed faces the face worker put in it (its face.match cluster, "Face N"), as the group's filmstrip on People reads
them (people/face_photos.py).
"""

import re
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from liberational import config as cfg
from liberational.vision.sightings import FaceBatch

LOW, HIGH = 0.25, 0.65  # similarity at a likeness of 0 and of 100
CLOSE = 0.04  # a runner-up this near in similarity holds the score at 75
PROBABLE, MAYBE = 75, 50  # band floors, in score points
MATCH = 0.45  # the live match threshold (vision/faces.py THRESH), shown with a face's name
DET_MIN = 0.5  # a sighting the detector is less sure of than this is left out of pictures and quality
SCAN = 20_000  # recent unnamed sightings read for pictures and quality
THUMBS = 6  # pictures per row
SCAN_STEP_S = 86400.0  # the unnamed sightings are read a day at a time, newest first
FEW = 5  # seen fewer times than this: too little to go on
TURNED = 0.7  # median detector confidence under this: side-on, partly hidden, or blurred
SMALL = 0.08  # median face height under this share of the frame: far from the camera
DIM = 70  # mean brightness (0 to 255) of the group's photo under this: dim


def likeness(sim: float) -> int:
    """A similarity as a 0 to 100 likeness."""
    return round(100 * min(1.0, max(0.0, (sim - LOW) / (HIGH - LOW))))


def lead(gap: float) -> int:
    """How far ahead of the runner-up, as 50 (a tie) to 100 (2 x CLOSE or more)."""
    return round(50 + 50 * min(1.0, max(0.0, gap / (2 * CLOSE))))


def band(score: int) -> str:
    return "probable" if score >= PROBABLE else "maybe" if score >= MAYBE else "none"


def _unit(v: NDArray[np.floating]) -> NDArray[np.float32]:
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def best_guess(vec: NDArray[np.floating], names: list[str], embs: NDArray[np.floating]) -> dict[str, Any] | None:
    """The named person most like a group's embedding: {"name", "score", "band", "close"}, or None with nobody named.
    close names the runner-up when the lead is what held the score down."""
    if not len(names):
        return None
    sims = _unit(embs) @ _unit(vec)
    per: dict[str, float] = {}
    for n, s in zip(names, sims.tolist()):
        if str(n) not in per or s > per[str(n)]:
            per[str(n)] = s
    ranked = sorted(per.items(), key=lambda kv: -kv[1])
    (name, sim), second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    alike = likeness(sim)
    ahead = lead(sim - second[1]) if second else 100
    pts = min(alike, ahead)
    return {"name": name, "score": pts, "band": band(pts), "close": second[0] if second and ahead < alike else None}


def brightness(photo: Path) -> float | None:
    """Mean brightness (0 to 255) of a photo, or None when there is none to read."""
    try:
        from PIL import Image, ImageStat

        with Image.open(photo) as im:
            return float(ImageStat.Stat(im.convert("L")).mean[0])
    except Exception:
        return None


def reason(
    guess: dict[str, Any] | None, count: int, det: float | None, height: float | None, light: float | None
) -> str:
    """One line on why the guess is as sure as it is: what holds it back, or that nothing does."""
    if guess is None:
        return "Nobody is named yet, so there's no one to compare with"
    issues = []
    if guess.get("close") and guess["band"] != "probable":
        issues.append(f"looks like {guess['close']} too")
    if count < FEW:
        issues.append(f"only seen {count} times")
    if det is not None and det < TURNED:
        issues.append("side-on or blurred")
    if height is not None and height < SMALL:
        issues.append("far from the camera")
    if light is not None and light < DIM:
        issues.append("dim")
    if issues:
        text = " and ".join(issues[:2])
        return text[0].upper() + text[1:]
    if guess["band"] == "probable":
        return f"A clear face, close to {guess['name']}'s photos"
    if guess["band"] == "maybe":
        return f"A clear face, but only partly like {guess['name']}"
    return "A clear face, unlike anyone named"


def _scan(gids: list[int], state: Path | None, face_batch: FaceBatch | None = None) -> list[dict[str, Any]]:
    """For each group in gids, its newest sightings and their quality, from one read of the recent unnamed
    sightings: {"shots": [{"id", "ts"}], "times", "last", "det", "height"}."""
    from liberational.vision import sightings as S

    by_group: dict[int, list[dict[str, Any]]] = {g: [] for g in gids}
    if gids:
        # Keep the original daily joins and ordering before the SCAN cap. A request-owned sparse snapshot serves
        # these windows in memory; dense histories retain day-at-a-time reads rather than a partial answer.
        now = time.time()
        windows = []
        end = now + 1
        while end > now - S.SHOT_DAYS * 86400:
            start = max(end - SCAN_STEP_S, now - S.SHOT_DAYS * 86400)
            windows.append((start, end))
            end = start
        batch = (
            face_batch.faces(windows, newest_first=True)
            if face_batch is not None
            else S.bounded_faces(windows[-1][0], windows[0][1], state=state, windows=windows, newest_first=True)
        )
        unnamed = [f for f in batch if f["name"] is None and (f["det"] or 0.0) > DET_MIN] if batch is not None else []
        for start, end in windows if batch is None else []:
            if len(unnamed) >= SCAN:
                break
            day = [f for f in S.faces(start, end, state=state) if f["name"] is None and (f["det"] or 0.0) > DET_MIN]
            unnamed += sorted(day, key=lambda f: (f["ts"], f["id"]), reverse=True)
        for f in unnamed[:SCAN]:
            cluster = f["cluster"]  # "Face N": group N
            group = int(cluster[5:]) if isinstance(cluster, str) and re.fullmatch(r"Face \d+", cluster) else None
            if group in by_group:
                by_group[group].append(f)
    out = []
    for g in gids:
        mine = by_group[g]  # newest first
        heights = [float(f["bbox_norm"][3]) - float(f["bbox_norm"][1]) for f in mine if f["bbox_norm"]]
        out.append(
            {
                "shots": [{"id": f["id"], "ts": float(f["ts"])} for f in mine[:THUMBS]],
                "all_shots": [{"id": f["id"], "ts": float(f["ts"])} for f in mine],
                "shot_count": len(mine),  # every recent shot it has, for Who's this? "Show all (N)"
                "times": [float(f["ts"]) for f in mine],  # when each was seen, for who was heard then
                "last": float(mine[0]["ts"]) if mine else None,
                "det": statistics.median(float(f["det"]) for f in mine) if mine else None,
                "height": statistics.median(heights) if heights else None,
            }
        )
    return out


def groups(
    waiting: list[dict[str, int]],
    state: Path | None = None,
    times: dict[int, list[float]] | None = None,
    *,
    face_batch: FaceBatch | None = None,
    all_shots: bool = False,
    nots: Mapping[int, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Live's "Who's this?" rows for the face groups waiting for a name ([{"id", "count"}], as /people lists them),
    best guess first: the probable ones by score, then the maybes, then the rest by when they were last seen. times,
    when given, gets when each group's recent sightings were (people/together.py reads who was heard then). `nots`:
    each group's people (by name) the owner said it isn't (people/face_nots.py), never its guess, and listed
    as the row's `not`. A guess's `id` is the name to send back, as pets' and voices' guesses have."""
    from liberational.vision import faces as fid

    ids, vecs, _, _ = fid._unk_load()
    index = {int(g): i for i, g in enumerate(ids)}
    waiting = [g for g in waiting if int(g["id"]) in index]
    mine = _unit(np.stack([vecs[index[int(g["id"])]] for g in waiting])) if waiting else np.zeros((0, 512), np.float32)
    names, embs = fid.gallery()  # the owner's rows and the learned ones
    photos = cfg.path("paths.faces") / "_unknown"
    out: list[dict[str, Any]] = []
    for g, vec, sc in zip(waiting, mine, _scan([int(g["id"]) for g in waiting], state, face_batch)):
        gid = int(g["id"])
        if times is not None:
            times[gid] = sc["times"]
        denied = (nots or {}).get(gid, set())
        folded = {d.casefold() for d in denied}
        keep = [k for k, n in enumerate(names) if str(n).casefold() not in folded]
        guess = best_guess(vec, [names[k] for k in keep], embs[keep]) if denied else best_guess(vec, names, embs)
        if guess is not None:
            guess = {**guess, "id": guess["name"]}
        out.append(
            {
                "id": gid,
                "label": f"Face {gid}",
                "count": int(g["count"]),
                "last_seen": sc["last"],  # its newest sighting; Live marks a face in view now itself
                "shots": [{**s, "url": f"/faces/shot/{s['id']}.jpg"} for s in sc["shots"]],
                **(
                    {"_all_shots": [{**s, "url": f"/faces/shot/{s['id']}.jpg"} for s in sc["all_shots"]]}
                    if all_shots
                    else {}
                ),
                "shot_count": sc["shot_count"],
                "guess": guess if guess and guess["band"] != "none" else None,
                "score": guess["score"] if guess else 0,
                "band": guess["band"] if guess else "none",
                "reason": reason(guess, int(g["count"]), sc["det"], sc["height"], brightness(photos / f"{gid}.jpg")),
                "not": sorted(denied),
            }
        )
    rank = {"probable": 0, "maybe": 1, "none": 2}
    out.sort(key=lambda r: (rank[r["band"]], -r["score"] if r["band"] != "none" else 0, -(r["last_seen"] or 0)))
    return out
