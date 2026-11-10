"""Who a face record is about, as Continuum subjects (the rule on person subjects).

A named face's subject is the person's `PersonId` from the people registry: names are only attributes, so a rename
never changes it. A face nobody is named for carries its face group's id (`registry.subject_uuid` of
`registry.face_group(n)` for "Face N"), never a person; naming it later is a `person_of` decision, not a rewrite.
Forgetting a person purges every id that resolves to them (`Registry.forget_subjects`).
"""

import re
from typing import Protocol

from liberational.people import registry

FACE_GROUP = re.compile(r"Face (\d+)")


class People(Protocol):
    """Names to people: the `PersonId` of the person called `name` now, or None for nobody."""

    def person(self, name: str, /) -> str | None: ...


class NoPeople:
    """Knows nobody: names give no subject (tests, or a worker run without the registry)."""

    def person(self, name: str, /) -> str | None:
        return None


class RegistryPeople:
    """The people registry under the state directory."""

    def __init__(self, reg: registry.Registry) -> None:
        self.reg = reg

    def person(self, name: str, /) -> str | None:
        pid = self.reg.person_id(name)
        return str(pid) if pid is not None else None


def group(cluster: object) -> str | None:
    """The subject of an unnamed face's group ("Face 3"), or None for anything else."""
    m = FACE_GROUP.fullmatch(cluster) if isinstance(cluster, str) else None
    return registry.subject_uuid(registry.face_group(int(m.group(1)))) if m else None
