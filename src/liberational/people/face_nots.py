"""The owner's "this face group is not <person>", as pets' "not this pet".

Each is a `face.not` record in the knowledge store (outside the broker's reset), keyed by its face group and
person, the newest of a pair winning; taking it back writes an inactive one (Undo). It's kept by person id, so it
follows a rename or a merge. A face group marked not Alice never has Alice as its guess on Who's this? and is never
named Alice automatically (vision/faces.py autoname), and the group's face becomes one of Alice's negatives in the face
matcher (`faces.add_not`), so its next look isn't named Alice either. A single shot that isn't someone is the card's own correction
(POST /people/correct "remove"), which /faces/not reuses for its `shots`.
"""

import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from liberational import knowledge
from liberational.domain.kernel import PersonId
from liberational.people import registry

NOTS = "face.not"
MAX_GROUPS = 500  # face groups in one call


def _store(state: Path) -> knowledge.Store:
    return knowledge.at(registry.knowledge_path(state))


def latest(rows: Iterable[Mapping[str, Any]]) -> dict[int, set[str]]:
    """Each face group's person ids the owner said it isn't, from `face.not` rows: the newest row of a pair wins."""
    newest: dict[tuple[int, str], Mapping[str, Any]] = {}
    for r in sorted(rows, key=lambda r: float(r["ts"])):
        if r.get("source") == "owner" and type(r.get("group")) is int and isinstance(r.get("person"), str):
            newest[(r["group"], r["person"])] = r
    out: dict[int, set[str]] = {}
    for (gid, person), r in newest.items():
        if r.get("active") is True:
            out.setdefault(gid, set()).add(person)
    return out


def read(state: Path) -> dict[int, set[str]]:
    """Every face group's person ids the owner said it isn't. None, without creating the store, when there's none."""
    if not registry.knowledge_path(state).exists():
        return {}
    return latest({**r.record.value, "ts": r.record.start_ns / 1e9} for r in _store(state).query(NOTS))


def names(state: Path) -> dict[int, set[str]]:
    """`read`, as each person's name now (a person since forgotten is left out)."""
    found = read(state)
    if not found:
        return {}
    reg = registry.at(state)
    out: dict[int, set[str]] = {}
    for gid, people in found.items():
        for pid in people:
            person = reg.person(PersonId(pid))
            if person is not None:
                out.setdefault(gid, set()).add(person.display_name)
    return out


def mark(state: Path, name: object, groups: object, active: bool = True, now: float | None = None) -> int:
    """Say these face groups are not the person called `name`, or take it back (`active` False). Returns how many."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name is the person the faces aren't")
    if (
        not isinstance(groups, list)
        or not groups
        or len(groups) > MAX_GROUPS
        or any(type(g) is not int or g <= 0 for g in groups)
    ):
        raise ValueError(f"groups is 1 to {MAX_GROUPS} face group numbers")
    reg = registry.at(state)
    pid = reg.person_id(name)
    person = reg.person(pid) if pid is not None else None
    if pid is None or person is None:
        raise LookupError(f"Nobody called {name.strip()} yet.")
    now = time.time() if now is None else now
    records = []
    for k, gid in enumerate(groups):
        at = int((now + k * 1e-6) * 1e9)  # newest wins
        body = {"group": gid, "person": str(pid), "active": active, "source": "owner"}
        records.append(knowledge.Record(str(uuid.uuid7()), NOTS, f"{gid}:{pid}", at, at, at, body, (str(pid),)))
    _store(state).put(records)
    from liberational.vision import faces

    for gid in groups:  # and the matcher's negative for them: the group's face, so its next look isn't them
        if not active:
            faces.remove_not(person.display_name, f"group:{gid}")
        elif (vec := faces.group_vector(gid)) is not None:
            faces.add_not(person.display_name, vec, f"group:{gid}")
    return len(groups)
