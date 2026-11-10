#!/usr/bin/env python3
"""Remove the unnamed face groups and learned face rows made from the side or back of a head (#3155), once.

Since #3155, see-actions never starts or joins an unnamed group ("Face N") with a look that doesn't show the face
(`see/head_faces.py:shows_face`), and never learns from one. This clears what was made before:

- A group is removed when it has looks in the window, every one has its `face.signal`, and none shows the face. A
  group the owner touched is kept (`owner_kept`): put aside in People's inbox (`inbox_skips.json`), marked not
  someone (`people/face_nots.py`), a shot taken out (`vision/face_rejects.py`), or a name undone. When those marks
  can't be read, the tool stops and removes nothing.
- A learned row (`faces/learned.npz`) is removed when it is the embedding of a look that doesn't show the face.

Named people, the owner's enrolled rows (`faces/db.npz`), and the recorded `face.match` rows are left as they are.
Both files change under the locks the services write them with (`vision/faces.py` `editing_unknown` and `editing`),
so see-actions and the dashboard can keep running. It prints counts only. A dry run is the default:

    .venv/bin/python tools/drop_side_faces.py           # counts only
    .venv/bin/python tools/drop_side_faces.py --apply   # remove them

Running it again finds nothing more to remove.
"""

import argparse
import time
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import segment_log
from liberational.see.head_faces import shows_face
from liberational.vision import face_rejects
from liberational.vision import faces as F

DAY_S = 86400.0


def group_number(cluster: object) -> int | None:
    """The group number of a `face.match` cluster label ("Face 12"), or None."""
    if not isinstance(cluster, str) or not cluster.startswith("Face "):
        return None
    try:
        return int(cluster[5:])
    except ValueError:
        return None


def facing(signals: Iterable[Mapping[str, Any]]) -> dict[str, bool]:
    """Whether each look shows the face, by its match id (`face.signal`)."""
    return {
        str(s["match_id"]): shows_face(
            float(s.get("frontal") or 0.0), None if s.get("norm") is None else float(s["norm"])
        )
        for s in signals
        if s.get("match_id")
    }


def side_only_groups(matches: Iterable[Mapping[str, Any]], shows: Mapping[str, bool], keep: set[int]) -> set[int]:
    """The groups whose every look has a signal and none shows the face, less `keep`."""
    side: dict[int, bool] = {}
    for m in matches:
        n = group_number(m.get("cluster"))
        if n is None:
            continue
        shown = shows.get(str(m["id"]))
        side[n] = side.get(n, True) and shown is False  # a look with no signal can't be judged: keep the group
    return {n for n, only in side.items() if only and n not in keep}


def side_learned(
    learned: np.ndarray, embeddings: Iterable[Mapping[str, Any]], shows: Mapping[str, bool]
) -> list[np.ndarray]:
    """The learned rows (N x 512) that are the embedding of a look that doesn't show the face."""
    if not len(learned):
        return []
    hit = np.zeros(len(learned), bool)
    for e in embeddings:
        if shows.get(str(e.get("match_id"))) is not False or e.get("vector") is None:
            continue
        v = np.asarray(e["vector"], np.float32)
        hit |= learned @ (v / (np.linalg.norm(v) + 1e-9)) >= 0.9999
    return [learned[k] for k in np.nonzero(hit)[0]]


def touched_by_owner(matches: Iterable[Mapping[str, Any]], rejected: frozenset[str]) -> set[int]:
    """The groups the owner took a shot out of ("Not this person")."""
    out: set[int] = set()
    for m in matches:
        n = group_number(m.get("cluster"))
        if n is not None and str(m["id"]) in rejected:
            out.add(n)
    return out


def owner_kept(matches: Iterable[Mapping[str, Any]]) -> set[int]:
    """The groups the owner touched, which stay: the ones the store holds for them (`faces._owner_held`: put aside,
    marked not someone, a shot taken out, or a name undone), and the ones a rejected shot in the window belongs to
    (shots taken out before the store held such groups). Raises when People's marks can't be read, so an unreadable
    mark never counts as none."""
    return F._owner_held() | touched_by_owner(matches, face_rejects.rejected())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=30.0, help="how far back to read looks (default 30)")
    ap.add_argument("--apply", action="store_true", help="remove them (default: a dry run that only counts)")
    args = ap.parse_args()

    state = cfg.path("paths.state")
    log = segment_log.at(state)
    now = time.time()
    start = now - args.days * DAY_S
    matches = log.range(segment_log.FACE_MATCHES, start, now + 1)
    signals = log.range(segment_log.FACE_SIGNALS, start, now + 1)
    shows = facing(signals)
    keep = owner_kept(matches)
    groups = side_only_groups(matches, shows, keep)
    stored = {int(i) for i in F._unk_load()[0]}
    drop = groups & stored

    learned = F.load_learned()
    rows = np.asarray(learned["embs"], np.float32).reshape(-1, 512)
    views: list[np.ndarray] = []
    if len(rows):
        since = max(start, float(np.min(learned["ts"])) - DAY_S)  # a row is learned when its track ends
        views = side_learned(rows, log.range(segment_log.FACE_EMBEDDINGS, since, now + 1), shows)

    print(f"looks read: {len(matches)} matches, {len(signals)} signals, over {args.days:g} days")
    print(f"unnamed groups stored: {len(stored)}; side or back only: {len(groups)} ({len(drop)} still stored)")
    print(f"groups kept because the owner touched them: {len(keep)}")
    print(f"learned rows: {len(rows)}; from side or back looks: {len(views)}")
    if not args.apply:
        print("dry run: nothing removed (--apply to remove)")
        return
    gone = F.drop_groups(drop)
    unlearned = F.unlearn_views(views)
    print(f"removed {gone} groups and {unlearned} learned rows")
    print(f"unnamed groups stored now: {len(F._unk_load()[0])}; learned rows now: {len(F.load_learned()['names'])}")


if __name__ == "__main__":
    main()
