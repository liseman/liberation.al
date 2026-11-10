"""What the owner hid on Who's this?: pictures ignored one by one, and faces blocked.

Both are owner decisions, so they're kept in the knowledge store (outside the broker's reset) and travel with
`pack-knowledge export`:

- `face.ignore`: one record a picture (a `face.match` sighting, by its id), with the face group it was in. An ignored
  picture is never shown again; new pictures of the same face still are.
- `face.block`: one record a "Block", with the unit mean embedding of the pictures blocked, their ids, and their
  group. An unnamed face group whose face is at least BLOCK_SIM like a blocked one is not listed for a name
  (`blocked_groups`), whichever pictures it holds and however many more come.

The newest record of a key wins, and taking one back writes an inactive one, as people/face_nots.py does. The pictures
themselves leave their group through vision/face_rejects.py, which is what the readers apply; these records are what
the owner sees listed and can take back.

BLOCK_SIM is vision/faces.py's FOLD_SIM (0.70), the similarity at which two unnamed groups are folded as one person.
It was measured on the centroids of 247 named tracks: no two different people's were that alike except through one
track that was likely misnamed, while 0.45 (THRESH and UNKNOWN_JOIN, the bars for naming one look and for joining a
group) would have joined 180 different-person pairs. A block compares the same two things, a group's face with the
mean of the blocked pictures, so it hides the same face seen again (a poster, a screen, a neighbor) and leaves a
look-alike waiting.
"""

import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from liberational import knowledge
from liberational.people import registry

IGNORES = "face.ignore"
BLOCKS = "face.block"
BLOCK_SIM = 0.70  # vision/faces.py FOLD_SIM: two unnamed groups this alike are one person
MAX_SHOTS = 5000  # pictures in one Ignore or Block
LISTED = 200  # the most ignored pictures and the most blocks one listing gives, newest first
BLOCK_SHOWN = 6  # a block's pictures the listing shows


def _store(state: Path) -> knowledge.Store:
    return knowledge.at(registry.knowledge_path(state))


def _latest(state: Path, collection: str) -> dict[str, dict[str, Any]]:
    """The collection's newest owner record of each key, when it is active: {key: its value, with `at`}. None,
    without creating the store, when there's no store."""
    if not registry.knowledge_path(state).exists():
        return {}
    newest: dict[str, dict[str, Any]] = {}
    for r in sorted((r.record for r in _store(state).query(collection)), key=lambda r: r.start_ns):
        if r.key is not None and r.value.get("source") == "owner":
            newest[r.key] = {**r.value, "at": r.start_ns / 1e9}
    return {key: row for key, row in newest.items() if row.get("active") is True}


def ignored(state: Path) -> dict[str, dict[str, Any]]:
    """Every ignored picture: {shot id: {shot, group, at}}."""
    return _latest(state, IGNORES)


def blocks(state: Path) -> dict[str, dict[str, Any]]:
    """Every block: {its key: {vector, shots, group, at}}."""
    return {k: r for k, r in _latest(state, BLOCKS).items() if isinstance(r.get("vector"), list)}


def _shots(shots: object) -> list[str]:
    if not isinstance(shots, list) or not shots or len(shots) > MAX_SHOTS:
        raise ValueError(f"send 1 to {MAX_SHOTS} pictures")
    if not all(isinstance(s, str) and s for s in shots):
        raise ValueError("a picture is named by its sighting id")
    return list(dict.fromkeys(shots))


def ignore(state: Path, group: int, shots: object, now: float | None = None) -> list[str]:
    """Ignore these pictures of face group `group`. Returns their keys (the shot ids), which `restore` takes."""
    ids = _shots(shots)
    now = time.time() if now is None else now
    records = []
    for k, shot in enumerate(ids):
        at = int((now + k * 1e-6) * 1e9)  # newest wins
        body = {"shot": shot, "group": int(group), "active": True, "source": "owner"}
        records.append(knowledge.Record(str(uuid.uuid7()), IGNORES, shot, at, at, at, body))
    _store(state).put(records)
    return ids


def block(state: Path, group: int, shots: object, members: np.ndarray, now: float | None = None) -> str:
    """Block the face these pictures of face group `group` show: `members` are their unit embeddings (N x 512), and
    the block keeps their unit mean. Returns the block's key, which `restore` takes. Raises ValueError with no
    embedding to block by."""
    ids = _shots(shots)
    kept = np.asarray(members, np.float32).reshape(-1, 512)
    if not len(kept):
        raise ValueError("these pictures have no face on record to block by")
    mean = kept.mean(axis=0)
    unit = mean / max(float(np.linalg.norm(mean)), 1e-9)
    at = int((time.time() if now is None else now) * 1e9)
    key = str(uuid.uuid7())
    body = {
        "vector": [float(v) for v in unit],
        "shots": ids,
        "group": int(group),
        "active": True,
        "source": "owner",
    }
    _store(state).put([knowledge.Record(str(uuid.uuid7()), BLOCKS, key, at, at, at, body)])
    return key


def restore(state: Path, kind: object, keys: object, now: float | None = None) -> list[dict[str, Any]]:
    """Take back ignored pictures (`kind` "ignored", by shot id) or blocks ("blocked", by key). Returns what was
    taken back, as [{group, shots}], so the caller can put the pictures back in their groups. A key that isn't
    hidden is left out."""
    if kind not in ("ignored", "blocked"):
        raise ValueError('kind is "ignored" or "blocked"')
    if not isinstance(keys, list) or not keys or len(keys) > MAX_SHOTS or not all(isinstance(k, str) for k in keys):
        raise ValueError(f"send 1 to {MAX_SHOTS} keys")
    collection, have = (IGNORES, ignored(state)) if kind == "ignored" else (BLOCKS, blocks(state))
    now = time.time() if now is None else now
    records, back = [], []
    for k, key in enumerate(dict.fromkeys(keys)):
        row = have.get(key)
        if row is None:
            continue
        at = int((now + k * 1e-6) * 1e9)
        body = {"group": row.get("group"), "active": False, "source": "owner"}
        records.append(knowledge.Record(str(uuid.uuid7()), collection, key, at, at, at, body))
        shots = [str(row["shot"])] if kind == "ignored" else [str(s) for s in row.get("shots") or []]
        back.append({"group": row.get("group"), "shots": shots})
    if records:
        _store(state).put(records)
    return back


def blocked_groups(state: Path, ids: Sequence[int], faces: np.ndarray) -> set[int]:
    """The groups among `ids` whose face (`faces`, one row each) is at least BLOCK_SIM like a blocked one, in one
    product: every group against every block."""
    walls = [r["vector"] for r in blocks(state).values() if len(r["vector"]) == np.asarray(faces).shape[-1]]
    if not walls or not len(ids):
        return set()
    mine = np.asarray(faces, np.float32)
    mine = mine / (np.linalg.norm(mine, axis=1, keepdims=True) + 1e-9)
    nearest = (mine @ np.asarray(walls, np.float32).T).max(axis=1)
    return {int(g) for g, sim in zip(ids, nearest, strict=True) if sim >= BLOCK_SIM}


def listing(state: Path) -> dict[str, Any]:
    """The ignored pictures and the blocks, newest first, at most LISTED of each, with how many there are in all."""
    shots = sorted(ignored(state).values(), key=lambda r: -r["at"])
    walls = sorted(blocks(state).items(), key=lambda kv: -kv[1]["at"])
    return {
        "ignored": [{"id": str(r["shot"]), "group": r.get("group"), "at": r["at"]} for r in shots[:LISTED]],
        "ignored_count": len(shots),
        "blocked": [
            {
                "id": key,
                "group": r.get("group"),
                "at": r["at"],
                "shots": [str(s) for s in (r.get("shots") or [])[:BLOCK_SHOWN]],
                "count": len(r.get("shots") or []),
            }
            for key, r in walls[:LISTED]
        ],
        "blocked_count": len(walls),
        "block_similarity": BLOCK_SIM,
    }
