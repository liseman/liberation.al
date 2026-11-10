"""Owner-named pets as owner records in the knowledge store (`pet.profile`, outside the broker's reset),
separate from human identity and visual recognition.

One active pet of a species permits an explicitly labeled household assumption for one observed animal. Several
named pets, or several simultaneous animals, remain anonymous. Revision order is the order written. Malformed
profiles and exhausted history bounds stay actionable; they never become an empty household.
"""

import re
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from liberational import config, knowledge

PET_PROFILES = "pet.profile"
SPECIES = ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")  # COCO's animals
HOUSEHOLD = ("dog", "cat", "bird")  # the species a pet is offered as, most likely first
TRACK = re.compile(r"[\w.-]{1,64}:[a-z]{1,20}:\d{1,12}")  # a recorded animal track's id: "<run>:<species>:<n>"
MAX_HIDDEN = 500  # sightings hidden from one pet's card


def offered(listing: Iterable[Mapping[str, Any]]) -> list[str]:
    """The species the dashboard offers: the household ones, then any other a named pet already is."""
    return [*HOUSEHOLD, *sorted({str(p["species"]) for p in listing} - set(HOUSEHOLD))]


MAX_REVISIONS = 4096
MAX_PAGES = 64


def _identity(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("pet id must be a UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise ValueError("pet id must be a UUID") from error
    if str(parsed) != value or parsed.int == 0:
        raise ValueError("pet id must be a canonical, nonempty UUID")
    return value


def _name(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 80:
        raise ValueError("pet name must contain 1 to 80 characters")
    if any(ord(c) < 32 for c in value):
        raise ValueError("pet name must be one line")
    return " ".join(value.split())


SEXES = ("female", "male")


def _sex(value: object) -> str | None:
    """A pet's sex, optional: "female", "male", or None when the owner hasn't said (Ask's "she" and "he")."""
    if value in (None, ""):
        return None
    if value not in SEXES:
        raise ValueError("pet sex is female, male, or left unset")
    return str(value)


def _track(value: object) -> str:
    if not isinstance(value, str) or not TRACK.fullmatch(value):
        raise ValueError("track is a recorded animal track's id")
    return value


def _photo(value: object) -> str | None:
    """A stored profile's chosen photo: the track of the sighting its avatar shows, or None for the newest."""
    return None if value is None else _track(value)


def _hidden(value: object) -> list[str]:
    """A stored profile's sightings hidden from its card."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_HIDDEN:
        raise ValueError("hidden is a bounded list of tracks")
    return [_track(v) for v in value]


def _species(value: object) -> str:
    if not isinstance(value, str) or value not in SPECIES:
        raise ValueError("choose a supported animal species")
    return value


def read_profiles(store: knowledge.Store) -> list[dict[str, Any]]:
    """Current active profiles in the order written, with each revision's record ID as `evidence`."""
    rows = store.query(PET_PROFILES, limit=MAX_REVISIONS + 1)
    if len(rows) > MAX_REVISIONS:
        raise RuntimeError("pet profile history exceeds the safe read bound")
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        body = row.record.value
        try:
            if type(body.get("active")) is not bool or body.get("source") != "owner":
                raise ValueError("invalid profile revision")
            ident = _identity(body.get("pet_id"))
            latest[ident] = {
                "id": ident,
                "name": _name(body.get("name")),
                "species": _species(body.get("species")),
                "sex": _sex(body.get("sex")),
                "photo": _photo(body.get("photo")),
                "hidden": _hidden(body.get("hidden")),
                "active": body["active"],
                "evidence": row.record.id,
            }
        except (ValueError, TypeError) as error:
            raise RuntimeError("stored pet profile is malformed") from error
    return sorted(
        [{k: v for k, v in p.items() if k != "active"} for p in latest.values() if p["active"]],
        key=lambda p: (p["name"].casefold(), p["id"]),
    )


def resolve_species(profiles: list[dict[str, Any]], species: str, *, observed_count: int = 1) -> dict[str, Any] | None:
    """A household assumption only for one observed animal and exactly one named pet of its species."""
    if type(observed_count) is not int or observed_count != 1:
        return None
    matches = [p for p in profiles if p.get("species") == species]
    if len(matches) != 1:
        return None
    return {**matches[0], "assumption": "single_named_pet_of_species"}


class PetRegistry:
    """Synchronous dashboard facade over the knowledge store's pet profiles."""

    def __init__(self, knowledge_db: Path) -> None:
        self.path = knowledge_db
        self.lock = threading.RLock()

    @property
    def store(self) -> knowledge.Store:
        """The store, opened (and created) on first use."""
        return knowledge.at(self.path)

    def listing(self) -> list[dict[str, Any]]:
        """The current named pets; raises rather than return a partial roster."""
        return read_profiles(self.store) if self.path.exists() else []

    def resolve(self, species: str, *, observed_count: int = 1) -> dict[str, Any] | None:
        """Resolve one animal through the explicit single-pet assumption, never through appearance."""
        return resolve_species(self.listing(), species, observed_count=observed_count)

    def save(self, name: object, species: object, pet_id: object = None, sex: object = None) -> dict[str, Any]:
        """Add a pet, or revise an existing pet without changing its stable subject ID."""
        name, species, chosen = _name(name), _species(species), _sex(sex)
        with self.lock:
            current = self.listing()
            prior = None
            if pet_id is not None:
                ident = _identity(pet_id)
                prior = next((p for p in current if p["id"] == ident), None)
                if prior is None:
                    raise LookupError("pet not found")
            else:
                ident = str(uuid.uuid7())
            # The card's choices (its photo, the sightings hidden from it) are the owner's too: an edit keeps them.
            photo, hidden = (prior["photo"], prior["hidden"]) if prior is not None else (None, [])
            evidence = self._write(ident, name, species, True, prior, chosen, photo, hidden)
            return {
                "id": ident,
                "name": name,
                "species": species,
                "sex": chosen,
                "photo": photo,
                "hidden": hidden,
                "evidence": evidence,
            }

    def set_photo(self, pet_id: object, track: object) -> dict[str, Any]:
        """Choose the sighting (its track) the pet's avatar shows, or None for its newest again. A hidden
        sighting chosen comes back to the card."""
        photo = None if track is None else _track(track)
        with self.lock:
            prior = self._current(pet_id)
            hidden = [t for t in prior["hidden"] if t != photo]
            return self._revise(prior, photo, hidden)

    def hide(self, pet_id: object, track: object, hidden: bool = True) -> dict[str, Any]:
        """Hide a sighting from the pet's card, or show it again. Hiding its chosen photo unchooses it."""
        tid = _track(track)
        with self.lock:
            prior = self._current(pet_id)
            kept = [t for t in prior["hidden"] if t != tid]
            if hidden:
                if len(kept) >= MAX_HIDDEN:
                    raise ValueError(f"at most {MAX_HIDDEN} sightings can be hidden from one card")
                kept.append(tid)
            photo = None if hidden and prior["photo"] == tid else prior["photo"]
            return self._revise(prior, photo, kept)

    def _current(self, pet_id: object) -> dict[str, Any]:
        ident = _identity(pet_id)
        prior = next((p for p in self.listing() if p["id"] == ident), None)
        if prior is None:
            raise LookupError("pet not found")
        return prior

    def _revise(self, prior: dict[str, Any], photo: str | None, hidden: list[str]) -> dict[str, Any]:
        evidence = self._write(
            prior["id"], prior["name"], prior["species"], True, prior, prior.get("sex"), photo, hidden
        )
        return {**prior, "photo": photo, "hidden": hidden, "evidence": evidence}

    def remove(self, pet_id: object) -> None:
        """Retire an owner binding; recorded detections, tracks, and media are preserved."""
        ident = _identity(pet_id)
        with self.lock:
            prior = next((p for p in self.listing() if p["id"] == ident), None)
            if prior is None:
                raise LookupError("pet not found")
            self._write(
                ident, prior["name"], prior["species"], False, prior, prior.get("sex"), prior["photo"], prior["hidden"]
            )

    def _write(
        self,
        ident: str,
        name: str,
        species: str,
        active: bool,
        prior: dict[str, Any] | None,
        sex: str | None = None,
        photo: str | None = None,
        hidden: list[str] | None = None,
    ) -> str:
        revision = str(uuid.uuid7())
        now = time.time_ns()
        record: dict[str, Any] = {
            "pet_id": ident,
            "name": name,
            "species": species,
            "active": active,
            "source": "owner",
        }
        if sex is not None:
            record["sex"] = sex
        if photo is not None:
            record["photo"] = photo
        if hidden:
            record["hidden"] = list(hidden)
        self.store.put([knowledge.Record(revision, PET_PROFILES, ident, now, now, now, record, (ident,))])
        return revision


_REGISTRIES: dict[Path, PetRegistry] = {}
_LOCK = threading.Lock()


def at(state: str | Path | None = None) -> PetRegistry:
    """The pet registry for the configured state directory, or an explicitly supplied one (tests)."""
    from liberational.people import registry

    own = state is None or Path(state).resolve() == config.path("paths.state").resolve()
    path = registry.knowledge_path(None if own else state)
    with _LOCK:
        if path not in _REGISTRIES:
            _REGISTRIES[path] = PetRegistry(path)
        return _REGISTRIES[path]
