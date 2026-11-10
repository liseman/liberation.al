"""Telling pets of one species apart by appearance, from the animal embeddings in the segment store: see-actions'
`animal.embedding` rows (see/action_animals.py) and see-captions' `animal.grounding_embedding` rows
(see/caption_animals.py), `segment_log.ANIMAL_EMBEDDING_STREAMS`.

A **sighting** is one camera-local animal track: the mean of its embeddings (configured DINO, unit length), its species, camera,
and first and last time. Sightings of a species are **grouped** greedily in time order: one joins the group whose
centroid it is most similar to, at `GROUP_SIM` or more, else starts its own. Groups are only a way to name many
sightings at once; they never say who an animal is.

The owner **labels** sightings, a whole group at a time, on People's Pets section: as one of the named pets
(people/pets.py), as "another animal" (a visiting cat, say), or clears a label. Each label is a `pet.sighting` record in the
knowledge store (outside the broker's reset), keyed by the track it was judged from. The newest label of a
track wins. Labeling a sighting as a pet or "another animal" also writes its mean embedding there as a `pet.reference`
record (`reference`), so the label keeps its example when a reset empties the segment store. Labels made before that
was written get theirs from their saved photos (tools/restore_pet_references.py).

Unlabeled embeddings remain labelable for `pets.label_days`. Active owner labels, including "another animal", retain
their whole sighting as a reference without an age cutoff. Clearing the newest label makes that sighting ordinary
unlabeled detail again. Listing, labeling, and photo selection use the same filter over every embedding the segment
store still keeps.

**Recognition** names an unlabeled sighting by its nearest labeled sighting's class (a pet, or "another animal"):
the best class must reach `MATCH_SIM` and beat the next class by `MARGIN`, else it stays unknown. It must also beat
every labeled sighting of the other species by `MARGIN`: the detector sometimes boxes the dog as a cat, and
such a box is more like the dog's labels than a cat's, so it stays unknown rather than take a cat's name. It is
**off** for a species until the labels pass a **held-out check**: each labeled sighting is recognized from the others, leaving out
every labeled sighting within `HOLD_OUT_S` of it (the same visit, re-tracked, would make the check too easy). It is on
when there are at least `MIN_LABELED` labeled sightings of at least two classes (two pets, or a pet and another
animal: with one class nothing could be got wrong), at most 1 in 10 decided ones are wrong, and at least half are
decided. Until then a sighting's best match is shown as a suggestion only, and nothing answers for a pet by
appearance. The single-named-pet assumption is separate and unchanged.

A species with **exactly one** named pet (Bess, the only dog) can't pass that check: there is no second class to get
wrong. Its labeled sightings are a **gallery** instead: an unlabeled sighting is that pet when its nearest
labeled sighting is at least the gallery's own threshold, the `GALLERY_PERCENTILE`th percentile of how alike each
labeled sighting is to its nearest labeled one from another visit (held out by `HOLD_OUT_S`), and never below
`MIN_GALLERY_SIM`. It is on with `MIN_LABELED` labeled sightings, at least half of them with another visit to compare
with. A sighting labeled "another animal" still has to be beaten by `MARGIN`.

The owner can also say a sighting is **not** a pet: By pet's unchecked tiles, or a card's Not <name>…. Each is
a `pet.not` record keyed by its track and pet, the newest of a pair winning. Recognition then never names or suggests
that pet for that sighting (`recognize`'s `exclude`), and saying it of a sighting labeled as that pet clears the label.

The thresholds were first guesses, then set from the owner's labels. The held-out check, not the
thresholds, is what decides whether recognition is used.

**Reading the embeddings**: every recorded animal embedding is kept in this process an hour at a time
(`embedding_rows`). Each read stats the store's files (milliseconds), and reads again only the hours whose files
changed, which is a new hour or a row written late into an old one, and starts over when the deletions log moves.
Re-reading everything took about half a second per label, under the evidence lock, so quick labels queued behind
each other. The kept rows cost about 7 MB of vectors (4,590 embeddings of 384 float32s).

**One embedding space**: every reader (listing, labeling, photos, Review's boxes) takes only rows in the
current space (`embedding_model`: the configured model and its crop), as references are. A row of another space (an
earlier square crop) doesn't compare with the rest, so it's no sighting, and a label can only keep an example of
the current space.

**Review's pet boxes** name a track by appearance through `resolve`, which never scans history. Its reference
gallery is built from the owner's labels alone: each label records its sighting's camera, so the labeled tracks'
embeddings are taken from the kept rows by track and camera, plus the imported `pet.reference` records. A label
without a camera and start can't be found that way and isn't a reference. The gallery and its checks are cached per
state directory, keyed by the knowledge store's label and reference generation (which counts deletions) and the
household's pets, and rebuilt at least every `GALLERY_TTL_S`, since segment-store retention can drop embeddings without
touching the knowledge store. Each request then reads only the requested camera and time window's embeddings of the
requested tracks, and a track gets a name only when the same check and threshold as the Pets page recognize it as a
named pet. Anything less stays unknown.
"""

import calendar
import functools
import math
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config, knowledge, locking, segment_log
from liberational.people.pets import offered

# Measured on the Linux machine against labeled photos cut tight to their box: 863 of two
# cats and 773 of a dog. A labeled cat's nearest same-pet sighting from another visit was a median 0.74 alike. At
# MATCH_SIM 0.70 (with MARGIN 0.05) the cats' held-out check had 574 right, 3 wrong, and 286 unknown; square crops at
# 0.75 had 513, 5, and 403 of 921. At GROUP_SIM 0.75, 1,636 sightings made 557 groups, 1 sighting in a group led by
# the other cat (square crops at 0.80: 810 groups, 2).
# MATCH_SIM is 0.05 under what those photos supported: references are embedded from the 192 px saved photos,
# live sightings from full frames, and one track's photo was a median 0.85 alike to its own live sightings (cats; the
# dog 0.88), so live sightings score lower against the references than the references do against each other. In live
# sightings, 0.70 named 17 of 36 cat sightings and held back 8 clear ones at 0.59 to 0.69; at 0.65 the
# cats' held-out check had 700 right, 5 wrong, and 157 unknown of 862.
GROUP_SIM = 0.75  # a sighting joins a group whose centroid is at least this similar
MATCH_SIM = 0.65  # recognition's nearest labeled sighting must be at least this similar
MARGIN = 0.05  # and beat the nearest of any other class by this much
HOLD_OUT_S = 600.0  # the held-out check leaves out labeled sightings this close in time to the one it checks
MIN_LABELED = 6  # labeled sightings of a species before recognition can be on
MAX_WRONG = 0.1  # the most of its decided held-out answers that may be wrong
MIN_DECIDED = 0.5  # the least share of held-out answers that must be decided
GALLERY_PERCENTILE = 10  # a one-pet species' threshold: this percentile of its labeled sightings' held-out similarity
MIN_GALLERY_SIM = 0.7  # and never below this, since with one pet nothing measures how alike other animals look
OTHER = "other"  # the label for an animal that is none of the named pets
IGNORED = "ignored"  # put aside by the owner: off the list, and never an example for recognition
ANIMAL = "animal"  # a sighting the detector gave a species no pet here is (a "bear" that's the dog): any household one
SEEN_MAX = 12  # a named pet's newest sightings the dashboard shows
EVIDENCE_LOCK = Path("continuum/pet-evidence")
LABELS = "pet.sighting"  # owner labels, in the knowledge store
NOTS = "pet.not"  # the owner's "this sighting is not that pet", in the knowledge store
REFERENCES = "pet.reference"  # labeled tracks' embeddings from before the reset
GALLERY_TTL_S = 300.0  # the longest a cached reference gallery is used before it's read again


@dataclass
class Sighting:
    """One animal track's embeddings."""

    track_id: str
    species: str
    cam: str
    start: float
    end: float
    vectors: list[np.ndarray] = field(default_factory=list)
    shot: dict[str, Any] = field(default_factory=dict)  # the largest box seen: {ts, box}, for its photo

    @functools.cached_property
    def mean(self) -> np.ndarray:
        """The unit mean of its vectors, worked out once: /pets/sightings compares it about 70k times, and
        vectors are only added while sightings() builds it, before anything reads this."""
        v = np.mean(self.vectors, axis=0)
        return v / max(float(np.linalg.norm(v)), 1e-9)


def sightings(rows: Iterable[Mapping[str, Any]]) -> list[Sighting]:
    """`animal.embedding` records as sightings, in order of first seen. A malformed row is skipped."""
    out: dict[str, Sighting] = {}
    for r in rows:
        vector, tid, species = r.get("vector"), r.get("track_id"), r.get("species")
        if not isinstance(tid, str) or not isinstance(species, str) or not isinstance(vector, list | np.ndarray):
            continue
        if not len(vector):
            continue
        v = np.asarray(vector, dtype=np.float32)
        if not np.isfinite(v).all() or not np.linalg.norm(v) or (tid in out and v.shape != out[tid].vectors[0].shape):
            continue
        ts = float(r["ts_start"])
        s = out.setdefault(tid, Sighting(tid, species, str(r.get("cam") or ""), ts, ts))
        s.start, s.end = min(s.start, ts), max(s.end, ts)
        s.vectors.append(v / np.linalg.norm(v))
        box = r.get("box")
        if isinstance(box, list) and len(box) >= 4:
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > s.shot.get("area", -1.0):
                s.shot = {"ts": ts, "box": box[:4], "area": area}
    return sorted(out.values(), key=lambda s: (s.start, s.track_id))


def groups(found: Sequence[Sighting]) -> list[list[Sighting]]:
    """Each species' sightings grouped greedily in time order (see the module's docs), largest group first.

    Each sighting joins the group of its species whose centroid it's most like (the latest of equals), if that's at
    least GROUP_SIM, else starts one. It's compared with all of them in one product with their unit centroids, kept up
    to date as groups grow: one at a time, 1,000 groups made it about 480,000 norms and dots, 5 s of a request."""
    created: list[list[Sighting]] = []  # every group, in the order it was made
    by_species: dict[str, _Centroids] = {}
    for s in found:
        cs = by_species.setdefault(s.species, _Centroids(len(s.mean)))
        k = cs.nearest(s.mean, GROUP_SIM)
        if k is None:
            cs.add(len(created), s.mean)
            created.append([s])
        else:
            created[cs.group[k]].append(s)
            cs.grow(k, s.mean)
    return sorted(created, key=lambda m: (-len(m), m[0].start))


class _Centroids:
    """One species' group centroids (sums of unit means) and their unit vectors, as rows, for `groups`."""

    def __init__(self, dims: int) -> None:
        self.sums = np.zeros((16, dims), dtype=np.float32)
        self.unit = np.zeros((16, dims), dtype=np.float32)
        self.group: list[int] = []  # each row's group, as an index into `created`

    def nearest(self, mean: np.ndarray, at_least: float) -> int | None:
        """The row most like `mean`, the latest of equals, when that's at least `at_least`."""
        n = len(self.group)
        if not n:
            return None
        sims = self.unit[:n] @ mean
        best = float(sims.max())
        if best < at_least:
            return None
        return int(np.flatnonzero(sims == best)[-1])

    def add(self, group: int, mean: np.ndarray) -> None:
        n = len(self.group)
        if n == len(self.sums):
            self.sums = np.concatenate([self.sums, np.zeros_like(self.sums)])
            self.unit = np.concatenate([self.unit, np.zeros_like(self.unit)])
        self.sums[n] = mean
        self.unit[n] = mean / max(float(np.linalg.norm(mean)), 1e-9)
        self.group.append(group)

    def grow(self, row: int, mean: np.ndarray) -> None:
        self.sums[row] += mean
        self.unit[row] = self.sums[row] / max(float(np.linalg.norm(self.sums[row])), 1e-9)


def recognize(
    query: Sighting,
    labeled: Sequence[tuple[Sighting, str]] | References,
    min_sim: float = MATCH_SIM,
    exclude: Iterable[str] = (),
) -> tuple[str | None, float | None, str | None]:
    """The class of the nearest labeled sighting of the query's species: (label, similarity, None) when it is clear
    (at least `min_sim`, and beating any other class, and every labeled sighting of another species, by `MARGIN`),
    else (None, best similarity, the nearest label as a suggestion). (None, None, None) with nothing to compare. The
    classes in `exclude` (the pets the owner said it isn't) are left out, so the next class is the answer.
    Asking about many sightings, pass `References` built once instead of the labeled list."""
    refs = labeled if isinstance(labeled, References) else References(labeled)
    found = refs.of(query.species)
    if found is None:
        return None, None, None
    tracks, labels, means = found
    sims = means @ query.mean
    others = tracks != query.track_id
    excluded = set(exclude)
    best: dict[str, float] = {}
    for label in dict.fromkeys(labels.tolist()):
        if label not in excluded and (keep := others & (labels == label)).any():
            best[label] = float(sims[keep].max())
    label, sim, suggestion = _decide(best, min_sim)
    rivals = refs.besides(query.species)
    if label is not None and sim is not None and rivals is not None:
        rival_labels, rival_means = rivals
        keep = ~np.isin(rival_labels, list(excluded))
        if keep.any() and sim - float((rival_means[keep] @ query.mean).max()) < MARGIN:
            return None, sim, label  # more like another species' labeled animal: the detector's species is doubtful
    return label, sim, suggestion


class References:
    """Labeled sightings by species, with their unit means as one matrix, for recognizing many sightings against them
    (Pets' list asks about every unlabeled one; restacking the means for each was most of what remained of its time)."""

    def __init__(self, labeled: Sequence[tuple[Sighting, str]]) -> None:
        self.labeled = labeled
        self._of: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
        self._besides: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def of(self, species: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """(track ids, labels, means) of the species' labeled sightings, or None when it has none."""
        if species not in self._of:
            mine = [(s, label) for s, label in self.labeled if s.species == species]
            self._of[species] = (
                (
                    np.array([s.track_id for s, _ in mine]),
                    np.array([label for _, label in mine]),
                    _means([s for s, _ in mine]),
                )
                if mine
                else None
            )
        return self._of[species]

    def besides(self, species: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(labels, means) of every other species' labeled sightings, or None when there are none."""
        if species not in self._besides:
            rest = [(s, label) for s, label in self.labeled if s.species != species]
            self._besides[species] = (
                (np.array([label for _, label in rest]), _means([s for s, _ in rest])) if rest else None
            )
        return self._besides[species]


def _decide(best: Mapping[str, float], min_sim: float) -> tuple[str | None, float | None, str | None]:
    """recognize's answer from each class's best similarity."""
    if not best:
        return None, None, None
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    label, sim = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else -1.0
    if sim >= min_sim and sim - runner >= MARGIN:
        return label, sim, None
    return None, sim, label


def _means(found: Sequence[Sighting]) -> np.ndarray:
    return np.stack([s.mean for s in found])


_BLOCK = 1024  # rows of the similarity matrix worked out at a time, so a long reference list stays small in memory


def _held_out_rows(mine: Sequence[Sighting]) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    """Each sighting's index, its similarity to every one of `mine`, and which of them are from another visit (more
    than HOLD_OUT_S apart, which also leaves out itself): the pairs the held-out checks compare, as one matrix product
    instead of a dot product a pair in Python (/pets/sightings took 1.8 s of CPU for 935 labeled sightings)."""
    means = _means(mine)
    starts = np.array([s.start for s in mine])
    for first in range(0, len(mine), _BLOCK):
        sims = means[first : first + _BLOCK] @ means.T
        far = np.abs(starts[first : first + _BLOCK, None] - starts[None, :]) > HOLD_OUT_S
        for k in range(sims.shape[0]):
            yield first + k, sims[k], far[k]


def held_out(labeled: Sequence[tuple[Sighting, str]], species: str) -> dict[str, Any]:
    """The held-out check for a species (see the module's docs): its counts, and whether recognition is on."""
    mine = [(s, label) for s, label in labeled if s.species == species]
    right = wrong = unknown = 0
    if mine:
        labels = np.array([label for _, label in mine])
        classes_of = {label: labels == label for label in dict.fromkeys(labels.tolist())}
        for i, sims, far in _held_out_rows([s for s, _ in mine]):
            best = {label: float(sims[keep].max()) for label, of in classes_of.items() if (keep := far & of).any()}
            guess, _, _ = _decide(best, MATCH_SIM)
            if guess is None:
                unknown += 1
            elif guess == mine[i][1]:
                right += 1
            else:
                wrong += 1
    classes = len({label for _, label in mine})
    decided = right + wrong
    reason = None
    if len(mine) < MIN_LABELED:
        reason = f"needs {MIN_LABELED - len(mine)} more labeled sightings"
    elif classes < 2:
        reason = "needs a labeled sighting of another animal of this species, so a mistake could show"
    elif decided < MIN_DECIDED * len(mine):
        reason = "too many held-out sightings stayed unknown"
    elif wrong > MAX_WRONG * decided:
        reason = "too many held-out sightings were recognized wrongly"
    return {
        "species": species,
        "labeled": len(mine),
        "right": right,
        "wrong": wrong,
        "unknown": unknown,
        "on": reason is None,
        "reason": reason,
    }


def gallery(labeled: Sequence[tuple[Sighting, str]], species: str, pet: str) -> dict[str, Any]:
    """The check for a species with one named pet (see the module's docs): its counts, its threshold, and whether
    recognition is on. "right" counts labeled sightings whose nearest labeled one from another visit reaches the
    threshold, "unknown" the rest."""
    mine = [s for s, label in labeled if s.species == species and label == pet]
    nearest = [float(sims[far].max()) for _, sims, far in _held_out_rows(mine) if far.any()] if mine else []
    threshold = max(MIN_GALLERY_SIM, float(np.percentile(nearest, GALLERY_PERCENTILE))) if nearest else None
    reason = None
    if len(mine) < MIN_LABELED:
        reason = f"needs {MIN_LABELED - len(mine)} more labeled sightings"
    elif len(nearest) < MIN_DECIDED * len(mine):
        reason = "needs labeled sightings from more separate visits"
    right = sum(1 for v in nearest if threshold is not None and v >= threshold)
    return {
        "species": species,
        "labeled": len(mine),
        "right": right,
        "wrong": 0,
        "unknown": len(mine) - right,
        "on": reason is None,
        "reason": reason,
        "threshold": threshold,  # exact: recognition compares against it
    }


def latest_labels(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each track's newest `pet.sighting` label that is still active, by track id."""
    newest: dict[str, Mapping[str, Any]] = {}
    for r in sorted(rows, key=lambda r: float(r["ts_start"])):
        if isinstance(r.get("track_id"), str) and r.get("source") == "owner":
            newest[r["track_id"]] = r
    return {tid: dict(r) for tid, r in newest.items() if r.get("active") is True and isinstance(r.get("label"), str)}


def latest_nots(rows: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Each track's pets the owner said it isn't, from `pet.not` rows: the newest row of each (track, pet) wins."""
    newest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for r in sorted(rows, key=lambda r: float(r["ts_start"])):
        if isinstance(r.get("track_id"), str) and isinstance(r.get("pet"), str) and r.get("source") == "owner":
            newest[(r["track_id"], r["pet"])] = r
    out: dict[str, set[str]] = {}
    for (tid, pet), r in newest.items():
        if r.get("active") is True:
            out.setdefault(tid, set()).add(pet)
    return out


def _store(state: Any) -> knowledge.Store:
    from liberational.people import registry

    return knowledge.at(registry.knowledge_path(state))


def _query(state: Any, collection: str) -> list[knowledge.Record]:
    """A collection's records; none, without creating the store, when it doesn't exist yet."""
    from liberational.people import registry

    if not registry.knowledge_path(state).exists():
        return []
    return [r.record for r in _store(state).query(collection)]


def labels(state: Any) -> dict[str, dict[str, Any]]:
    """Every track's newest active owner label, however old, by track id."""
    rows = [{**r.value, "ts_start": r.start_ns / 1e9} for r in _query(state, LABELS)]
    return latest_labels(rows)


def nots(state: Any) -> dict[str, set[str]]:
    """Every track's pets the owner said it isn't, by track id."""
    return latest_nots({**r.value, "ts_start": r.start_ns / 1e9} for r in _query(state, NOTS))


def label_days() -> float:
    """How many days unlabeled sightings are listed for labeling; owner-labeled references have no time cutoff."""
    value = config.get("pets.label_days")
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError("pets.label_days must be finite and positive")
    return float(value)


def evidence_lock(state: str | Path) -> AbstractContextManager[None]:
    """Serialize owner labeling with its catalog reference checks."""
    return locking.locked(Path(state) / EVIDENCE_LOCK)


@dataclass
class _Kept:
    """Every recorded animal embedding of one segment store, an hour at a time: each UTC hour's rows, with the files
    they were read from (names, sizes, and mtimes), and the deletions log's when they were read."""

    deletions: tuple[int, int] | None
    hours: dict[str, tuple[tuple[tuple[str, int, int], ...], list[dict[str, Any]]]] = field(default_factory=dict)


_kept: dict[Path, _Kept] = {}
_kept_lock = threading.Lock()


def _files(base: Path) -> tuple[tuple[int, int] | None, dict[str, tuple[tuple[str, int, int], ...]]]:
    """(the deletions log's size and mtime, each UTC hour's embedding files with their sizes and mtimes): the store
    files each row under its own hour, named from it (YYYYMMDDHH...), and a condensed history beside its stream."""
    import os

    try:
        st = (base / ".deletions").stat()
        deletions: tuple[int, int] | None = (st.st_size, st.st_mtime_ns)
    except OSError:
        deletions = None
    hours: dict[str, list[tuple[str, int, int]]] = {}
    for stream in segment_log.ANIMAL_EMBEDDING_STREAMS:
        for name in (stream, segment_log.condensed(stream)):
            try:
                cams = [c for c in os.scandir(base / name) if c.is_dir()]
            except OSError:
                continue
            for cam in cams:
                for f in os.scandir(cam.path):
                    if f.name.startswith(".") or not f.name[:10].isdigit():
                        continue
                    try:
                        fs = f.stat()
                    except OSError:
                        continue
                    hours.setdefault(f.name[:10], []).append(
                        (f"{name}/{cam.name}/{f.name}", fs.st_size, fs.st_mtime_ns)
                    )
    return deletions, {h: tuple(sorted(files)) for h, files in hours.items()}


def _hour_rows(log: segment_log.Log, hour: str) -> list[dict[str, Any]]:
    """The embedding rows in the current embedding space that start in this UTC hour (a read overlapping it also
    finds a neighbor's), each vector as an array, which holds a tenth of what a list of floats does."""
    a = float(calendar.timegm(time.strptime(hour, "%Y%m%d%H")))
    space = embedding_model()
    rows = []
    for stream in segment_log.ANIMAL_EMBEDDING_STREAMS:
        for r in log.range(stream, a, a + 3600):
            if a <= float(r["ts_start"]) < a + 3600 and r.get("model") == space:
                vector = r.get("vector")
                rows.append(
                    {**r, "vector": np.asarray(vector, dtype=np.float32)} if isinstance(vector, list) else dict(r)
                )
    return rows


def embedding_rows(state: Any) -> list[dict[str, Any]]:
    """Every animal embedding the segment store keeps in the current embedding space, in hour order, read again only where its files changed (see
    the module's docs). Reader failures propagate, and a failed hour is read again next time: a partial reference set
    must never silently change recognition or its held-out check. The rows are shared: callers must not change them."""
    base = segment_log.root(state)
    deletions, files = _files(base)
    with _kept_lock:
        kept = _kept.get(base)
        if kept is None or kept.deletions != deletions:
            kept = _Kept(deletions)
        have = dict(kept.hours)
    log = segment_log.Log(base)
    for hour, stamp in files.items():
        if hour not in have or have[hour][0] != stamp:
            have[hour] = (stamp, _hour_rows(log, hour))
    fresh = _Kept(deletions, {h: have[h] for h in sorted(files)})
    with _kept_lock:
        _kept[base] = fresh
    return [r for _, rows in fresh.hours.values() for r in rows]


def _recorded(state: Any, now: float, labels: Mapping[str, Any], tracks: set[str] | None = None) -> list[Sighting]:
    """Complete recent and owner-reference sightings from every embedding the segment store keeps (only `tracks`,
    given them).

    Reference means use every surviving embedding of a labeled track. Reader failures propagate: a partial reference
    set must never silently change recognition or its held-out check.
    """
    cutoff = now - label_days() * 86400
    references = _current_references(state)
    return sightings(
        r
        for r in [*embedding_rows(state), *references]
        if (tracks is None or r.get("track_id") in tracks)
        and float(r["ts_start"]) <= now
        and (float(r["ts_start"]) >= cutoff or r.get("track_id") in labels)
    )


def recorded(state: Any, now: float | None = None) -> list[Sighting]:
    """The same sightings used by listing and labeling, including old owner references, for photos."""
    now = time.time() if now is None else now
    return _recorded(state, now, labels(state))


def _teach(
    found: Sequence[Sighting], active: Mapping[str, Mapping[str, Any]], pets: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, Mapping[str, Any]], set[str], list[tuple[Sighting, str]], dict[str, str | None]]:
    """What the owner's labels teach about these sightings: (pet names by id, the tracks labeled as a pet or "another
    animal", the ignored tracks, the labeled sightings with their class, and each species' one named pet or None).
    Corrects each sighting's species in place: a label's species is the owner's correction of the detector's,
    and a species no pet here is becomes any household animal."""
    household = set(offered(pets))
    for s in found:
        corrected = active.get(s.track_id, {}).get("species")
        species = corrected if isinstance(corrected, str) and corrected else s.species
        s.species = species if species in household else ANIMAL
    names = {str(p["id"]): str(p["name"]) for p in pets}
    named = {tid: row for tid, row in active.items() if row["label"] == OTHER or row["label"] in names}
    ignored = {tid for tid, row in active.items() if row["label"] == IGNORED}
    labeled = [(s, str(named[s.track_id]["label"])) for s in found if s.track_id in named]
    only: dict[str, str | None] = {}
    for p in pets:
        only[str(p["species"])] = None if str(p["species"]) in only else str(p["id"])
    return names, named, ignored, labeled, only


def _check(labeled: Sequence[tuple[Sighting, str]], species: str, only: Mapping[str, str | None]) -> dict[str, Any]:
    """A species' check: the one-pet gallery when it has exactly one named pet, else the held-out check."""
    return gallery(labeled, species, pet) if (pet := only.get(species)) is not None else held_out(labeled, species)


def _judge(
    s: Sighting,
    labeled: Sequence[tuple[Sighting, str]] | References,
    check: Mapping[str, Any],
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    """{"recognized": ...} when recognition is on for the sighting's species and names it, {"suggestion": ...} for a
    match it isn't trusted with yet (`clear`: true) or an unclear nearest one (`clear`: false), else {}. Never
    a pet in `exclude`."""
    threshold = check.get("threshold")
    match, sim, suggestion = recognize(s, labeled, MATCH_SIM if threshold is None else threshold, exclude)
    if match is not None and check["on"]:
        return {"recognized": {"label": match, "similarity": round(sim or 0.0, 3)}}
    if match is not None or suggestion is not None:
        return {
            "suggestion": {"label": match or suggestion, "similarity": round(sim or 0.0, 3), "clear": match is not None}
        }
    return {}


def _guess(row: Mapping[str, Any], names: Mapping[str, str]) -> dict[str, Any] | None:
    """A sighting's guess in the shape every naming list shares (as faces' and voices'): {id (the label to send
    back), name, score (0 to 100), band ("probable" for recognition's match or a clear suggestion, "maybe" for an
    unclear one), close (None: recognition gives no runner-up)}, or None."""
    found = row.get("recognized") or row.get("suggestion")
    if not found:
        return None
    sure = "recognized" in row or bool(found.get("clear"))
    label = str(found["label"])
    return {
        "id": label,
        "name": "another animal" if label == OTHER else names.get(label, "a removed pet"),
        "score": round(100 * float(found["similarity"])),
        "band": "probable" if sure else "maybe",
        "close": None,
    }


def group_guess(rows: Sequence[Mapping[str, Any]], names: Mapping[str, str]) -> dict[str, Any] | None:
    """A group's guess, in the sightings' shape: the label its probable sightings name most (the first named on
    a tie), else the one its maybe sightings name most; its score is the mean of those sightings' scores. What Pets'
    inbox worked out for itself before (PetInbox's guessOf)."""
    for band in ("probable", "maybe"):
        tally: dict[str, list[int]] = {}
        for r in rows:
            g = r.get("guess")
            if g and (band == "maybe" or g["band"] == "probable"):
                tally.setdefault(g["id"], []).append(g["score"])
        if tally:
            label, scores = max(tally.items(), key=lambda kv: len(kv[1]))
            first = next(r["guess"] for r in rows if r.get("guess") and r["guess"]["id"] == label)
            return {**first, "score": round(sum(scores) / len(scores)), "band": band}
    return None


def view(state: Any, pets: Sequence[Mapping[str, Any]], now: float | None = None) -> dict[str, Any]:
    """Pets' sightings for the dashboard: the sightings still waiting for a label, in groups, each with what recognition
    says (a match when it is on for the species, else a suggestion), and each species' held-out check. A labeled
    sighting leaves the groups and serves as recognition's reference."""
    now = time.time() if now is None else now
    active, denied = labels(state), nots(state)
    found = _recorded(state, now, active)
    names, named, ignored, labeled, only = _teach(found, active, pets)
    checks = {sp: _check(labeled, sp, only) for sp in sorted({s.species for s in found})}
    refs = References(labeled)

    def item(s: Sighting) -> dict[str, Any]:
        row: dict[str, Any] = {
            "track_id": s.track_id,
            "species": s.species,
            "cam": s.cam,
            "start": s.start,
            "end": s.end,
            "samples": len(s.vectors),
            "label": named[s.track_id]["label"] if s.track_id in named else None,
        }
        if row["label"] is None:
            row.update(_judge(s, refs, checks[s.species], denied.get(s.track_id, ())))
        row["guess"] = _guess(row, names)
        row["not"] = sorted(denied.get(s.track_id, ()))
        return row

    # Each named pet's newest sightings, labeled or recognized, for its card and its row's "last seen".
    seen: dict[str, list[dict[str, Any]]] = {pid: [] for pid in names}
    for s in found:
        row = item(s)
        if s.track_id in ignored:
            continue
        how = "labeled" if row["label"] else "recognized" if "recognized" in row else None
        pid = row["label"] or (row.get("recognized") or {}).get("label")
        if how and pid in seen:
            seen[pid].append({"track_id": s.track_id, "cam": s.cam, "start": s.start, "end": s.end, "how": how})
    for rows in seen.values():
        rows.sort(key=lambda r: -r["end"])
        del rows[SEEN_MAX:]
    listed = []
    for members in groups([s for s in found if s.track_id not in named and s.track_id not in ignored]):
        rows = [item(s) for s in members]
        listed.append(
            {
                "id": f"{members[0].species}:{members[0].track_id}",
                "species": members[0].species,
                "sightings": rows,
                "guess": group_guess(rows, names),
                # The pets every sighting in it was said not to be: a group's own "not", as a face group's.
                "not": sorted(set.intersection(*(set(r["not"]) for r in rows))) if rows else [],
            }
        )
    return {
        "seen": seen,
        "groups": listed,
        "count": len(listed),  # the groups waiting, as /faces/unknown/groups counts
        "recognition": [c for sp, c in checks.items() if sp != ANIMAL],
        "names": names,
        "label_days": label_days(),
    }


def label(
    state: Any, track_ids: Sequence[str], value: str | None, pets: Sequence[Mapping[str, Any]], now: float | None = None
) -> int:
    """Label these sightings as a named pet's id, OTHER, or None (clear). Returns how many were labeled. A pet must
    exist; one of another species corrects the sighting's species (the detector called a dog a cat), which the
    label keeps. Expired unlabeled detail can't be labeled."""
    now = time.time() if now is None else now
    if not isinstance(track_ids, list) or not track_ids or len(track_ids) > 500:
        raise ValueError("send 1 to 500 sighting ids")
    species_of = {str(p["id"]): str(p["species"]) for p in pets}
    if value is not None and value not in (OTHER, IGNORED) and value not in species_of:
        raise LookupError("pet not found")
    # Read before the lock: it only finds the sightings' cameras and spans, and other labels shouldn't queue behind it.
    # The lock keeps one label's write whole against another's.
    found = {s.track_id: s for s in _recorded(state, now, labels(state), set(track_ids))}
    missing = [t for t in track_ids if t not in found]
    if missing:
        raise LookupError(f"{len(missing)} sightings are no longer recorded")
    with evidence_lock(state):
        kept = {str(r.get("track_id")) for r in _current_references(state)}
        records = []
        for k, tid in enumerate(track_ids):
            s = found[tid]
            at = int((now + k * 1e-6) * 1e9)  # newest wins
            body = {
                "cam": s.cam,
                "track_id": tid,
                "species": species_of.get(value or "", s.species),  # a pet's species corrects the detector's
                "label": value or "",
                "active": value is not None,
                "source": "owner",
                "sighting_start": s.start,
                "sighting_end": s.end,
            }
            subjects = (value,) if value not in (None, OTHER, IGNORED) else ()
            records.append(knowledge.Record(str(uuid.uuid7()), LABELS, tid, at, at, at, body, subjects))
            if value not in (None, IGNORED) and tid not in kept:
                records.append(reference(s.mean, body, at))
        _store(state).put(records)
        return len(track_ids)


def embedding_model() -> str:
    """The embedding space references are in: the configured model and its crop (`animal_embedder.space`)."""
    from liberational.vision import animal_embedder

    return animal_embedder.space()


def _current_references(state: Any) -> list[dict[str, Any]]:
    """The `pet.reference` records in the current embedding space, as rows. One of another space (another model, or
    an earlier square crop) doesn't compare, so it's no example, and its label is stranded until restored."""
    space = embedding_model()
    return [r.value for r in _query(state, REFERENCES) if r.value.get("model") == space]


def reference(vector: np.ndarray, owner_label: Mapping[str, Any], at: int) -> knowledge.Record:
    """A labeled sighting's `pet.reference`: its unit embedding with the owner's label, keyed by its track, in
    the shape `sightings` reads as one more embedding row (and `pack-knowledge` validates), so the label keeps its
    example after a reset empties the segment store. `at` is the label's time in nanoseconds."""
    unit = np.asarray(vector, dtype=np.float32)
    unit = unit / max(float(np.linalg.norm(unit)), 1e-9)
    track = str(owner_label["track_id"])
    value = {
        "track_id": track,
        "species": str(owner_label["species"]),
        "cam": str(owner_label.get("cam") or ""),
        "ts_start": float(owner_label["sighting_start"]),
        "vector": [float(v) for v in unit],
        "dims": len(unit),
        "model": embedding_model(),
        "source_id": track,
        "label": str(owner_label["label"]),
        "owner_label": dict(owner_label),
    }
    return knowledge.Record(str(uuid.uuid7()), REFERENCES, track, at, at, at, value)


def keep_references(state: Any, records: Sequence[knowledge.Record]) -> int:
    """Writes these `reference` records under the evidence lock, except for a track that has one by now (labeled
    meanwhile, or a second run). Returns how many it wrote."""
    with evidence_lock(state):
        kept = {str(r.get("track_id")) for r in _current_references(state)}
        fresh = [r for r in records if r.key not in kept]
        if fresh:
            _store(state).put(fresh)
        return len(fresh)


def stranded(state: Any) -> list[dict[str, Any]]:
    """The active labels of a pet or "another animal" with no example left: no embedding of their track in the segment
    store and no `pet.reference`, in the current embedding space (labels made before `label` wrote references, after
    a reset, or after the space changed). Each is the label's record, and only one that says its species, camera,
    and start can become a reference."""
    held = {str(r.get("track_id")) for r in embedding_rows(state)}
    held |= {str(r.get("track_id")) for r in _current_references(state)}
    return [
        row
        for tid, row in labels(state).items()
        if row.get("label") not in ("", IGNORED)
        and tid not in held
        and isinstance(row.get("species"), str)
        and _finite(row.get("sighting_start")) is not None
    ]


def mark_not(
    state: Any,
    track_ids: Sequence[str],
    pet: str,
    pets: Sequence[Mapping[str, Any]],
    active: bool = True,
    now: float | None = None,
) -> int:
    """Say these sightings are not `pet` (a named pet's id), or take that back with `active` False. A sighting
    labeled as that pet loses the label in the same write. Returns how many were marked."""
    now = time.time() if now is None else now
    if not isinstance(track_ids, list) or not track_ids or len(track_ids) > 500:
        raise ValueError("send 1 to 500 sighting ids")
    if not isinstance(pet, str) or pet not in {str(p["id"]) for p in pets}:
        raise LookupError("pet not found")
    found = {s.track_id: s for s in _recorded(state, now, labels(state), set(track_ids))}  # before the lock
    missing = [t for t in track_ids if t not in found]
    if missing:
        raise LookupError(f"{len(missing)} sightings are no longer recorded")
    with evidence_lock(state):
        current = labels(state)  # read again under the lock: the label this may clear is the current one
        records = []
        for k, tid in enumerate(track_ids):
            s = found[tid]
            at = int((now + k * 1e-6) * 1e9)  # newest wins
            body = {"cam": s.cam, "track_id": tid, "pet": pet, "active": active, "source": "owner"}
            records.append(knowledge.Record(str(uuid.uuid7()), NOTS, f"{tid}:{pet}", at, at, at, body, (pet,)))
            if active and current.get(tid, {}).get("label") == pet:
                cleared = {
                    "cam": s.cam,
                    "track_id": tid,
                    "species": s.species,
                    "label": "",
                    "active": False,
                    "source": "owner",
                    "sighting_start": s.start,
                    "sighting_end": s.end,
                }
                records.append(knowledge.Record(str(uuid.uuid7()), LABELS, tid, at, at, at, cleared, ()))
        _store(state).put(records)
        return len(track_ids)


@dataclass
class _Gallery:
    """The owner-labeled reference sightings Review's boxes are recognized against, and their checks."""

    key: tuple[Any, ...]
    built: float  # time.monotonic() when read
    active: dict[str, dict[str, Any]]
    names: dict[str, str]
    household: set[str]
    labeled: list[tuple[Sighting, str]]
    checks: dict[str, dict[str, Any]]
    denied: dict[str, set[str]] = field(default_factory=dict)  # each track's pets the owner said it isn't

    @functools.cached_property
    def references(self) -> References:
        return References(self.labeled)


_galleries: dict[Path, _Gallery] = {}
_galleries_lock = threading.Lock()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _references(state: Any, wanted: Mapping[str, Mapping[str, Any]]) -> list[Sighting]:
    """The labeled tracks' sightings, from the kept embeddings of each label's own track and camera, plus their
    imported references. A label without a camera and start isn't one."""
    cams = {
        tid: row["cam"]
        for tid, row in wanted.items()
        if isinstance(row.get("cam"), str) and row["cam"] and _finite(row.get("sighting_start")) is not None
    }
    rows = [r for r in embedding_rows(state) if cams.get(str(r.get("track_id"))) == r.get("cam")]
    references = [r for r in _current_references(state) if r.get("track_id") in wanted]
    return sightings([*rows, *references])


def _reference_gallery(state: Any, pets: Sequence[Mapping[str, Any]]) -> _Gallery | None:
    """The cached reference gallery for this state directory, read again when the labels, references, or pets change,
    or after GALLERY_TTL_S. None without a knowledge store."""
    from liberational.people import registry

    path = registry.knowledge_path(state)
    if not path.exists():
        return None
    generation = _store(state).generation((LABELS, REFERENCES, NOTS))
    key = (generation, tuple(sorted((str(p["id"]), str(p["species"])) for p in pets)))
    with _galleries_lock:
        cached = _galleries.get(path)
        if cached is not None and cached.key == key and time.monotonic() - cached.built < GALLERY_TTL_S:
            return cached
    built = time.monotonic()
    active = labels(state)
    names, named, _, _, only = _teach([], active, pets)
    found = _references(state, named)
    _, _, _, labeled, _ = _teach(found, active, pets)
    checks = {sp: _check(labeled, sp, only) for sp in sorted({s.species for s, _ in labeled})}
    fresh = _Gallery(key, built, active, names, set(offered(pets)), labeled, checks, nots(state))
    with _galleries_lock:
        _galleries[path] = fresh
    return fresh


def resolve(
    state: Any, cam: str, start: float, end: float, tracks: Iterable[str], pets: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """The named pet each of these tracks is recognized as by appearance, from only this camera and window's
    embeddings: {track id: {"label": pet id, "similarity"}}. A track with any owner label, of a species whose check
    isn't on, or that recognition doesn't name as a pet (unknown, a suggestion, or "another animal") is left out."""
    built = _reference_gallery(state, pets)
    if built is None or not any(c["on"] for c in built.checks.values()):
        return {}
    wanted = {t for t in tracks if t not in built.active}  # an owner label wins, whatever it says
    if not wanted:
        return {}
    log = segment_log.Log(segment_log.root(state))
    space = embedding_model()
    rows = [
        r
        for stream in segment_log.ANIMAL_EMBEDDING_STREAMS
        for r in log.range(stream, start, end, cameras=[cam])
        if r.get("track_id") in wanted and r.get("model") == space
    ]
    out: dict[str, dict[str, Any]] = {}
    for s in sightings(rows):
        s.species = s.species if s.species in built.household else ANIMAL
        check = built.checks.get(s.species)
        if check is None or not check["on"]:
            continue
        recognized = _judge(s, built.references, check, built.denied.get(s.track_id, ())).get("recognized")
        if recognized is not None and recognized["label"] in built.names:
            out[s.track_id] = recognized
    return out


def resolve_many(
    state: Any,
    start: float,
    end: float,
    tracks_by_camera: Mapping[str, Iterable[str]],
    pets: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Like `resolve`, but for many cameras in one read and one gallery snapshot.

    Returns {"<cam>:<track id>": {"label": pet id, "similarity"}}.
    """
    built = _reference_gallery(state, pets)
    if built is None or not any(c["on"] for c in built.checks.values()):
        return {}
    wanted_by_camera = {
        cam: {track for track in tracks if track not in built.active} for cam, tracks in tracks_by_camera.items()
    }
    wanted_by_camera = {cam: tracks for cam, tracks in wanted_by_camera.items() if tracks}
    if not wanted_by_camera:
        return {}
    log = segment_log.Log(segment_log.root(state))
    space = embedding_model()
    rows_by_camera: dict[str, list[dict[str, Any]]] = {cam: [] for cam in wanted_by_camera}
    for stream in segment_log.ANIMAL_EMBEDDING_STREAMS:
        for row in log.range(stream, start, end, cameras=list(wanted_by_camera)):
            cam = row.get("cam")
            track = row.get("track_id")
            if (
                isinstance(cam, str)
                and isinstance(track, str)
                and cam in wanted_by_camera
                and track in wanted_by_camera[cam]
                and row.get("model") == space
            ):
                rows_by_camera[cam].append(row)
    out: dict[str, dict[str, Any]] = {}
    for cam, rows in rows_by_camera.items():
        for sighting in sightings(rows):
            sighting.species = sighting.species if sighting.species in built.household else ANIMAL
            check = built.checks.get(sighting.species)
            if check is None or not check["on"]:
                continue
            recognized = _judge(sighting, built.references, check, built.denied.get(sighting.track_id, ())).get(
                "recognized"
            )
            if recognized is not None and recognized["label"] in built.names:
                out[f"{cam}:{sighting.track_id}"] = recognized
    return out
