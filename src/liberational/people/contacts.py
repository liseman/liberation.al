"""Phone numbers and the names added by hand, read from the person registry.

A person's phone number, and whether the owner added them by hand (People: Add), are fields of their `Person` in the
registry (`people/registry.py`). A rename or a merge changes that one record, so the number goes with the person.

Two files are read models of these fields. Every change here writes the registry and the file (a dual write, as
people/registry.py does for labels), so older readers and tools still find them:

    contacts.json       {name: number}, chmod 600
    people_added.json   [name]

Nothing reads a file as the source of a number the registry has. A reader falls back to a file only for a name the
registry has no number (or no added mark) for: a fresh install before `pack-registry backfill` copied the files in, a
name someone wrote into contacts.json by hand, or a registry write that failed. When both have a number, the
registry's wins, so after the backfill a number is changed on the People tab or with `pack-registry phone`, not by
editing the file.
"""

import dataclasses
import pathlib
import re

from liberational import config as cfg
from liberational import json_state
from liberational.domain.identity import Person
from liberational.people import registry

CONTACTS, ADDED = "contacts.json", "people_added.json"
PHONE = re.compile(r"\+[1-9]\d{7,14}")  # a number as the People tab takes it, such as +14155550123


def key(name: str) -> str:
    """The key a name is matched by: as `registry.norm`, and "Alice (phone)" is Alice, as the People tab reads it."""
    return registry.norm(re.sub(r"\s*\(phone\)$", "", str(name or "")))


@dataclasses.dataclass(frozen=True, slots=True)
class Book:
    """The phone numbers and added names of the people in the registry at `state`, with their two read models."""

    state: pathlib.Path
    contacts_file: pathlib.Path
    added_file: pathlib.Path

    @classmethod
    def at(cls, state: str | pathlib.Path | None = None) -> Book:
        """The book for `state` (`paths.state` by default), with its files in that folder."""
        s = pathlib.Path(state) if state is not None else cfg.path("paths.state")
        return cls(s, s / CONTACTS, s / ADDED)

    # ---- reading ----
    def _people(self) -> list[Person]:
        """Everyone in the registry, or nobody before the event log exists (reading never creates it) or when it
        can't be read."""
        people: list[Person] = []
        if registry.log_exists(self.state):
            with registry.safely("reading people"):
                people = registry.at(self.state).people()
        return people

    def file_phones(self) -> dict[str, str]:
        """contacts.json as {name: number}, notes ("_" keys) left out. A name with an empty number is kept, as the
        People tab has always listed it."""
        d = json_state.read(self.contacts_file, {})
        if not isinstance(d, dict):
            return {}
        return {k: v.strip() if isinstance(v, str) else "" for k, v in d.items() if not k.startswith("_")}

    def file_added(self) -> list[str]:
        d = json_state.read(self.added_file, [])
        return [n for n in d if isinstance(n, str)] if isinstance(d, list) else []

    def phones(self) -> dict[str, str]:
        """{name: number} for everyone with a phone number: the registry's, then contacts.json's for a name the
        registry has no number for."""
        out = {p.display_name: p.phone for p in self._people() if p.phone}
        have = {key(n) for n in out}
        for name, number in self.file_phones().items():
            if key(name) not in have:
                out[name] = number
                have.add(key(name))
        return out

    def number(self, name: str) -> str | None:
        """The phone number of the person called `name`, or None."""
        return next((v for k, v in self.phones().items() if key(k) == key(name) and v), None)

    def added_names(self) -> list[str]:
        """The people the owner added by hand: the registry's, then people_added.json's for a name the registry has
        not marked."""
        names = [p.display_name for p in self._people() if p.added]
        have = {key(n) for n in names}
        for name in self.file_added():
            if key(name) not in have:
                names.append(name)
                have.add(key(name))
        return names

    # ---- writing: the registry, then the read models ----
    def add(self, name: str, number: str = "", source: str = "dashboard", **attributes: str) -> None:
        """Add someone to the People list, with a phone number if one is given (a new one replaces theirs).
        `attributes` are a role and an organization typed with the name."""
        # The registry first, as set_phone does: one it couldn't record raises, and the file isn't written.
        with registry.recorded("person added"):
            reg = registry.at(self.state)
            reg.describe(name, source, attributes)
            reg.contact(name, source, phone=number or None, added=True)
        added = self.file_added()
        if name not in added:
            added.append(name)
        json_state.write(self.added_file, added, durable=True)
        if number:
            self._write_phone(name, number)

    def set_phone(self, name: str, number: str, source: str) -> None:
        """Give the person called `name` this number, or take theirs away with "". The registry write is the one that
        counts, so a failure there is raised; the file is written only after it."""
        registry.at(self.state).contact(name, source, phone=number)
        self._write_phone(name, number or None)

    def _write_phone(self, name: str, number: str | None) -> None:
        c = json_state.read(self.contacts_file, {})
        c = c if isinstance(c, dict) else {}
        for k in [k for k in c if not k.startswith("_") and key(k) == key(name)]:
            c.pop(k)
        if number:
            c[name] = number
        json_state.write(self.contacts_file, c, private=True, durable=True, indent=1)

    def follow(self, old: str, new: str) -> None:
        """Bring the read models in line with a rename or a merge of `old` into `new`, which the registry records
        (dashboard/server.py, rename_person). On a merge the survivor keeps their own number, as in the registry, and
        the file takes the registry's number for them whenever it has one."""
        if self.contacts_file.exists():
            c = json_state.read(self.contacts_file, {})
            if isinstance(c, dict):
                for k in [k for k in c if not k.startswith("_") and key(k) == key(old)]:
                    num = c.pop(k)
                    if not any(key(x) == key(new) for x in c if not x.startswith("_")):
                        c[new] = num
                known = next((p.phone for p in self._people() if key(p.display_name) == key(new) and p.phone), None)
                if known:
                    for k in [k for k in c if not k.startswith("_") and key(k) == key(new)]:
                        c.pop(k)
                    c[new] = known
                json_state.write(self.contacts_file, c, private=True, durable=True, indent=1)
        if self.added_file.exists():
            added = [new if key(n) == key(old) else n for n in self.file_added()]
            json_state.write(self.added_file, list(dict.fromkeys(added)), durable=True)

    def forget(self, name: str) -> None:
        """Take the person called `name` out of the read models. The registry forgets them in
        `Registry.forget`, which purges their number and added mark with the rest of their events."""
        if self.contacts_file.exists():
            c = json_state.read(self.contacts_file, {})
            if isinstance(c, dict):
                for k in [k for k in c if not k.startswith("_") and key(k) == key(name)]:
                    c.pop(k)
                json_state.write(self.contacts_file, c, private=True, durable=True, indent=1)
        if self.added_file.exists():
            json_state.write(self.added_file, [n for n in self.file_added() if key(n) != key(name)], durable=True)
