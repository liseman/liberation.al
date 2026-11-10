#!/usr/bin/env python3
"""Give the owner's pet labels examples in the current embedding space, without teaching the pets again.

A pet example (`pet.reference`) is a vector of one embedding space (`[models.hf.animal_embedding]`'s `space` and the
crop). After the model changes, or a reset drops a labeled track's embeddings, its label is stranded: it has no
example recognition can use (people/pet_recognition.py's `stranded`). This tool has two ways to give examples back.

**From kept photos** (the default): each stranded label whose photo is kept, in the knowledge store (`pet.photo`,
written with every label since the photo was kept) or still in `state/pet_shots`, is embedded again. A photo is the
track's largest box cut square with context (`animal_verifier.square`), so the animal sits in its middle: the tool
finds it again with see-actions' detector (RF-DETR Small, on the CPU), cuts the photo tight to that box as the
embedder does, and embeds it. A label without a photo, or whose photo the detector finds no animal in, is left as
it is: no example rather than a wrong one.

    uv run python tools/restore_pet_references.py --dry-run   # count only
    uv run python tools/restore_pet_references.py

**From what the old model was sure of** (`--carry-from <old space>`): for the examples taught before photos were
kept, which have nothing to embed again. It takes the unlabeled sightings that still have a saved photo
(`state/pet_shots`) and asks the old examples who each is: by the old model's vectors of the sighting when the
segment store still has them, else by embedding the photo with the old model (`--old-model`, a repository still
in the Hugging Face cache; without it those photos are passed over). A sighting the old examples name well above
recognition's own threshold (`--min-sim`, and `--margin` clear of every other class) is labeled as that pet for
the owner, and its photo is embedded by the current model as the pet's new example, with the photo kept. Those
labels say where they came from (`carried_from`), show on People's Pets like any label, and can be cleared there.
It prints how many examples each pet got, names the pets with fewer than recognition needs
(`pet_recognition.MIN_LABELED`) to teach again on the Pets page, and says whether each species' held-out check
passes on the new examples.

    uv run python tools/restore_pet_references.py --carry-from facebook/dinov3-vits16-pretrain-lvd1689m:tight \
        --old-model facebook/dinov3-vits16-pretrain-lvd1689m --dry-run
    uv run python tools/restore_pet_references.py --carry-from facebook/dinov3-vits16-pretrain-lvd1689m:tight \
        --old-model facebook/dinov3-vits16-pretrain-lvd1689m

The dry run names and counts but writes nothing, and works before the configured model is switched.

Run it by hand where the animal models are installed (the embedder loads on `vision.animal_models_device`; the
detector takes a few minutes on the CPU for a couple of thousand photos). It only adds records to the knowledge
store, under the pet evidence lock, and running it again finds nothing more to do.
"""

import argparse
import collections
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import knowledge, segment_log
from liberational.people import pet_recognition as recognition
from liberational.people import pets

BATCH = 64
MIN_SIM = 0.75  # carry-over: the old examples must name a sighting at least this surely (recognition's own was 0.65)
MARGIN = 0.10  # and this clear of every other class (recognition's own is 0.05)
PER_PET = 200  # carry-over: a pet's newest sightings at most


def decode(jpg: bytes | None) -> np.ndarray | None:
    """A saved photo's pixels (BGR), or None for no photo or one that doesn't decode."""
    import cv2

    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR) if jpg else None


def central(animals: np.ndarray) -> Sequence[float] | None:
    """The photo's own animal among the detector's (N x 6: x1, y1, x2, y2 normalized, score, class): the box whose
    centre is nearest the photo's, since the photo was cut around it. None when there are none."""
    if not len(animals):
        return None
    centres = (animals[:, :2] + animals[:, 2:4]) / 2
    return animals[int(np.argmin(np.abs(centres - 0.5).sum(axis=1)))][:4].tolist()


def detector() -> Any:
    """see-actions' detector on the CPU: `people(bgr tensor)`'s `animals` are the animal boxes."""
    import torch

    from liberational.see.gpu_pose import GpuPose

    models = Path(cfg.path("paths.models"))
    pose = models / "rtmpose-m_halpe26_256x192.onnx"
    return GpuPose(models / cfg.model_path("person_detector").name, pose, torch.device("cpu"))


def cut(photos: Mapping[str, bytes]) -> dict[str, np.ndarray]:
    """Each photo's own animal cut tight, by the photo's key, as the workers cut a box before embedding it. A photo
    that doesn't decode, or with no animal the detector finds, has none."""
    import torch

    from liberational.vision.animal_verifier import tight

    find = detector()
    crops: dict[str, np.ndarray] = {}
    for key, jpg in photos.items():
        image = decode(jpg)
        box = central(find.people(torch.from_numpy(image)).animals) if image is not None else None
        if image is not None and box is not None:
            crops[key] = tight(image, box)
    return crops


def embed(crops: Mapping[str, np.ndarray], repo: str | None = None) -> dict[str, np.ndarray]:
    """Each crop's embedding by its key: by the configured model (the current space), or by `repo`, an installed
    model of another space."""
    from liberational.vision.animal_embedder import DinoEmbedder

    embedder = DinoEmbedder(repo)
    keys = list(crops)
    out: dict[str, np.ndarray] = {}
    for first in range(0, len(keys), BATCH):
        batch = keys[first : first + BATCH]
        out.update(zip(batch, embedder.embed_crops([crops[key] for key in batch]), strict=True))
    return out


def restore(state: Path, dry_run: bool) -> None:
    """The default: every stranded label with a kept or saved photo gets its example in the current space."""
    stranded = recognition.stranded(state)
    kept = recognition.photos(state)
    found: dict[str, bytes] = {}
    for row in stranded:
        track = str(row["track_id"])
        if (jpg := kept.get(track) or recognition.saved_photo(state, track)) is not None:
            found[track] = jpg
    print(f"{len(stranded)} labels have no example; {len(found)} of them have a photo")
    if dry_run or not found:
        return
    vectors = embed(cut(found))
    now = time.time()
    records = []
    for k, row in enumerate(r for r in stranded if str(r["track_id"]) in vectors):
        track, at = str(row["track_id"]), int((now + k * 1e-6) * 1e9)
        records.append(recognition.reference(vectors[track], row, at))
        if track not in kept:  # from state/pet_shots: keep it with the taught state from now on
            records.append(recognition.photo(track, found[track], at))
    print(f"no animal found in {len(found) - len(vectors)} photos; those labels keep no example")
    print(f"restored {recognition.keep_references(state, records)} pet references")


def old_examples(
    state: Path, space: str, active: Mapping[str, Mapping[str, Any]], pet_ids: Iterable[str]
) -> list[tuple[recognition.Sighting, str]]:
    """The labeled sightings of embedding space `space`, each with its class: the `pet.reference` records of that
    space whose track still has an active label of a named pet or "another animal"."""
    classes = {*pet_ids, recognition.OTHER}
    rows = []
    for record in recognition._query(state, recognition.REFERENCES):
        value = record.value
        label = active.get(str(value.get("track_id")), {})
        if value.get("model") == space and label.get("label") in classes:
            rows.append({**value, "species": label.get("species") or value.get("species")})
    return [(s, str(active[s.track_id]["label"])) for s in recognition.sightings(rows)]


def old_sightings(state: Path, space: str, now: float) -> list[recognition.Sighting]:
    """The sightings the segment store still keeps in embedding space `space`, over the labelable window."""
    log = segment_log.at(state)
    start = now - recognition.label_days() * 86400
    rows = [
        r
        for stream in segment_log.ANIMAL_EMBEDDING_STREAMS
        for r in log.range(stream, start, now)
        if r.get("model") == space
    ]
    return recognition.sightings(rows)


def sure(
    query: recognition.Sighting,
    refs: recognition.References,
    min_sim: float,
    margin: float,
    exclude: Iterable[str] = (),
) -> str | None:
    """The class the examples name `query` as, or None unless it is clear: its nearest example of its species is at
    least `min_sim` alike, and `margin` clear of every other class's and every other species' nearest. The classes
    in `exclude` (the pets the owner said it isn't) take no part, as in `pet_recognition.recognize`."""
    found = refs.of(query.species)
    if found is None:
        return None
    _, labels, means = found
    excluded = set(exclude)
    sims = means @ query.mean
    best = {
        str(label): float(sims[labels == label].max())
        for label in dict.fromkeys(labels.tolist())
        if label not in excluded
    }
    if not best:
        return None
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    label, top = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else -1.0
    rivals = refs.besides(query.species)
    if rivals is not None and (keep := ~np.isin(rivals[0], list(excluded))).any():
        runner = max(runner, float((rivals[1][keep] @ query.mean).max()))
    return label if top >= min_sim and top - runner >= margin else None


def candidates(state: Path, taken: Mapping[str, Any]) -> dict[str, tuple[bytes, float]]:
    """The saved photos of tracks with no label (`taken`'s keys have one): each track's photo and when it was
    written, which for a photo a worker saved is during the sighting."""
    from liberational.people import face_shots, pet_shots

    found: dict[str, tuple[bytes, float]] = {}
    for folder in face_shots.hours(state, pet_shots.KIND):
        for path in folder.glob(f"*-{face_shots.VERSION}-r*.jpg"):
            track = path.name.split(f"-{face_shots.VERSION}-r")[0]
            if track in taken or track in found or not pets.TRACK.fullmatch(track):
                continue
            if (newest := pet_shots.newest(state, track)) is not None:
                found[track] = (newest.read_bytes(), newest.stat().st_mtime)
    return found


def carry(
    state: Path, space: str, old_model: str | None, min_sim: float, margin: float, per_pet: int, dry_run: bool
) -> None:
    """`--carry-from`: label the sightings the old space's examples surely name, with examples in the current
    space."""
    if space == recognition.embedding_model() and not dry_run:
        raise SystemExit(f"{space} is still the configured embedding space: switch [models.hf.animal_embedding] first")
    listing = pets.at(state).listing()
    names = {str(p["id"]): str(p["name"]) for p in listing}
    species_of = {str(p["id"]): str(p["species"]) for p in listing}
    active, denied = recognition.labels(state), recognition.nots(state)
    examples = old_examples(state, space, active, names)
    taught = collections.Counter(label for _, label in examples)
    refs = recognition.References(examples)
    now = time.time()
    photos = candidates(state, active)
    crops = cut({track: jpg for track, (jpg, _) in photos.items()})
    recorded = {s.track_id: s for s in old_sightings(state, space, now) if s.track_id in crops}
    rest = {track: crop for track, crop in crops.items() if track not in recorded}
    print(f"{len(examples)} examples of {space}; {len(photos)} unlabeled sightings have a saved photo")
    print(f"an animal was found in {len(crops)} photos; the old model's vectors are still stored for {len(recorded)}")
    if old_model is None:
        print(f"no --old-model: the other {len(rest)} photos are passed over")
        rest = {}
    for track, vector in embed(rest, old_model).items() if rest else ():
        species, at = track.split(":")[1], photos[track][1]
        recorded[track] = recognition.Sighting(track, species, "", at, at, [vector])
    named: dict[str, list[recognition.Sighting]] = {pid: [] for pid in names}
    for s in recorded.values():
        label = sure(s, refs, min_sim, margin, denied.get(s.track_id, ()))
        if label in named:
            named[label].append(s)
    chosen = {s.track_id: pid for pid, found in named.items() for s in sorted(found, key=lambda s: -s.end)[:per_pet]}
    print(f"named at {min_sim} or more, {margin} clear of any other: {len(chosen)} of {len(recorded)}")
    verb = "would get" if dry_run else "got"
    written = collections.Counter(chosen.values())
    if not dry_run:
        vectors = embed({track: crops[track] for track in chosen})
        records: dict[str, list[knowledge.Record]] = {}
        for k, (track, pid) in enumerate(chosen.items()):
            s, at = recorded[track], int((now + k * 1e-6) * 1e9)
            body = {
                "cam": s.cam,
                "track_id": track,
                "species": species_of[pid],
                "label": pid,
                "active": True,
                "source": "owner",  # the owner asked for the carry-over; `carried_from` says no one looked at this one
                "carried_from": space,
                "sighting_start": s.start,
                "sighting_end": s.end,
            }
            label = knowledge.Record(str(uuid.uuid7()), recognition.LABELS, track, at, at, at, body, (pid,))
            records[track] = [
                label,
                recognition.reference(vectors[track], body, at),
                recognition.photo(track, photos[track][0], at),
            ]
        written = collections.Counter(chosen[track] for track in keep(state, records))
    for pid in sorted(names, key=lambda p: names[p].casefold()):
        few = written[pid] < recognition.MIN_LABELED
        short = f"  <- fewer than {recognition.MIN_LABELED}: teach again on the Pets page" if few else ""
        print(f"  {names[pid]} ({species_of[pid]}): {taught[pid]} old examples, {verb} {written[pid]} new{short}")
    if not dry_run:
        for check in recognition.view(state, listing)["recognition"]:
            state_of = "on" if check["on"] else f"off ({check['reason']})"
            counts = f"{check['right']} right, {check['wrong']} wrong, {check['unknown']} unknown of {check['labeled']}"
            print(f"  recognition of {check['species']}s: {state_of}; held out: {counts}")


def keep(state: Path, records: Mapping[str, Sequence[knowledge.Record]]) -> list[str]:
    """Writes each track's records under the evidence lock, except for a track labeled meanwhile (or by an earlier
    run). Returns the tracks written."""
    with recognition.evidence_lock(state):
        active = recognition.labels(state)
        fresh = [track for track in records if track not in active]
        if fresh:
            recognition._store(state).put([r for track in fresh for r in records[track]])
        return fresh


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="count what would be done, and write nothing")
    parser.add_argument("--state", type=Path, help="the state directory (default: paths.state)")
    parser.add_argument("--carry-from", metavar="SPACE", help="carry the pets over from this earlier embedding space")
    parser.add_argument("--old-model", metavar="REPO", help="carry-over: the installed model of that earlier space")
    parser.add_argument("--min-sim", type=float, default=MIN_SIM, help=f"carry-over: how sure (default {MIN_SIM})")
    parser.add_argument("--margin", type=float, default=MARGIN, help=f"carry-over: how clear (default {MARGIN})")
    parser.add_argument("--per-pet", type=int, default=PER_PET, help=f"carry-over: sightings a pet (default {PER_PET})")
    args = parser.parse_args()
    state = args.state or Path(cfg.path("paths.state"))
    if args.carry_from:
        carry(state, args.carry_from, args.old_model, args.min_sim, args.margin, args.per_pet, args.dry_run)
    else:
        restore(state, args.dry_run)


if __name__ == "__main__":
    main()
