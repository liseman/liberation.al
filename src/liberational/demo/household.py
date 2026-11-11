"""Seed the photographable demo through the application's ordinary file and segment writers.

The generated media depicts only the procedural demo meshes. No camera, model, or remote asset is consulted.
"""

import asyncio
import datetime as dt
import io
import json
import math
import subprocess
import time
import uuid
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

from liberational import config as cfg
from liberational import segment_log as segments
from liberational.demo import dataset, scenes
from liberational.memory import events, places
from liberational.people import registry
from liberational.spatial import camera_clouds, cloud_scene, ply, positions


def video(picture: Path, clip: Path, seconds: int = 30) -> None:
    """Encode one generated still as a short seekable recording."""
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(picture),
            "-t",
            str(seconds),
            "-r",
            "5",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-preset",
            "fast",
            "-crf",
            "25",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "5",
            "-bf",
            "0",
            "-movflags",
            "+faststart",
            str(clip),
        ],
        check=True,
        timeout=60,
    )


def review_recordings(folder: Path, day: dt.date) -> list[dict[str, object]]:
    """Short views at the transcript's moments, registered through the same video index as real imports.

    Each minute gets a distinct hard link: the importer identifies a recording by its file path. Reusing one
    path at several timestamps would keep only its last timestamp. Render and encode each household scene once.
    """
    scenes_at = {
        7: ("kitchen", (0, 2), True),
        9: ("study", (0,), False),
        12: ("kitchen", (0, 1), False),
        13: ("kitchen", (), False),
        17: ("front-room", (0, 2), False),
        18: ("front-room", (0, 1, 2), True),
        19: ("front-room", (0, 1, 2), True),
        22: ("porch", (0,), True),
    }
    encoded: dict[tuple[str, tuple[int, ...], bool], Path] = {}
    index: list[dict[str, object]] = []
    for hour, minute in sorted({line.at[:2] for line in dataset.TRANSCRIPT}):
        scene = scenes_at[hour]
        camera, people, pet = scene
        if scene not in encoded:
            mesh = scenes.room(camera, occupied=False)
            for who, (x, y) in zip(people, ((1.45, 2.3), (2.5, 1.65), (3.35, 2.4)), strict=False):
                scenes.person(mesh, x, y, who)
            if pet:
                scenes.dog(mesh, 1.0, 1.0)
            jpg, _, _, _ = scenes.render(mesh)
            picture = folder / f"review-{hour}.jpg"
            picture.write_bytes(jpg)
            encoded[scene] = folder / f"review-{hour}.mp4"
            video(picture, encoded[scene], seconds=60)
        clip = folder / f"review-{hour}-{minute}.mp4"
        clip.hardlink_to(encoded[scene])
        index.append(
            {
                "cam": camera,
                "file": clip.name,
                "start": dataset._local(day, hour, minute),
                "fps": 5.0,
                "w": 960,
                "h": 540,
            }
        )
    return index


def media(state: Path, day: dt.date) -> None:
    """Render four distinct rooms and portraits, register short H.264 recordings, and keep each camera's cloud."""
    folder = state / "demo"
    folder.mkdir(exist_ok=True)
    index = []
    start = dataset._local(day, *dataset.MEDIA_TIME)
    clouds = camera_clouds.Store(camera_clouds.root(state))
    for camera in scenes.CAMERAS:
        jpg, points, colors, pose = scenes.render(scenes.room(camera))
        picture = folder / f"{camera}.jpg"
        picture.write_bytes(jpg)
        clip = folder / f"{camera}.mp4"
        video(picture, clip)
        index.append({"cam": camera, "file": clip.name, "start": start, "fps": 5.0, "w": 960, "h": 540})
        # Normalize the renderer's world coordinates exactly as real camera clouds are stored:
        # floor z=0, camera above the origin, forward along +y.
        camera_points = (points - pose[:3, 3]) @ pose[:3, :3]
        points, pose, _ = camera_clouds.level(camera_points, (pose[2, :3], float(pose[2, 3])))
        cloud = camera_clouds.Cloud(
            str(uuid.uuid4()),
            camera,
            start,
            len(points),
            1.0,
            "approximate",
            True,
            tuple(float(v) for v in pose.ravel()),
            960 * 0.72,
            960,
            540,
        )
        clouds.replace(
            camera_clouds.Made(
                cloud, ply.dumps(points, colors), np.asarray(Image.open(io.BytesIO(jpg)))[:, :, ::-1].copy()
            )
        )
    index.extend(review_recordings(folder, day))
    camera_pairs(state)
    (folder / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index))
    for n, name in enumerate(dataset.PEOPLE):
        gallery = cfg.path("paths.faces") / registry.photo_folder(name)
        gallery.mkdir(parents=True, exist_ok=True)
        for variant in range(3):
            jpg = scenes.portrait(n, variant)
            (gallery / f"{int(start * 1000) + variant}.jpg").write_bytes(jpg)
            (folder / f"person-{n}-{variant}.jpg").write_bytes(jpg)
    from liberational import knowledge
    from liberational.people import pet_recognition, pets

    pet = pets.at(state).listing()[0]
    pets.at(state).set_photo(pet["id"], f"demo:{dataset.PET_SPECIES}:1")
    knowledge.at(registry.knowledge_path(state)).put(
        [
            pet_recognition.photo(f"demo:{dataset.PET_SPECIES}:{k + 1}", scenes.pet_portrait(k), time.time_ns() + k)
            for k in range(dataset.EXAMPLES)
        ]
    )

    from liberational.vision import face_examples, faces

    for n, name in enumerate(dataset.PEOPLE):
        for k, example in enumerate(faces.examples(name)):
            picture = face_examples.picture(cfg.path("paths.faces"), example["key"])
            if picture is not None:
                who = (n + 1) % 3 if k == 0 else n  # the outlier stays visibly different
                picture.write_bytes((folder / f"person-{who}-{k % 3}.jpg").read_bytes())

    groups = [faces.new_group(np.stack(dataset._looks(n * 100, 512)[:3])) for n in range(2)]
    groups_data = []
    writers = segments.Writers(segments.root(state))
    try:
        for n, group in enumerate(groups):
            shots = []
            for v in range(3):
                sid, stamp = str(uuid.uuid4()), dataset._local(day, *dataset.MEDIA_TIME) - v * 5
                row = {
                    "id": sid,
                    "ts_start": stamp,
                    "ts_end": stamp + 1,
                    "track_id": f"demo-face-{n}",
                    "frame_id": "demo",
                    "bbox": [0.2, 0.1, 0.8, 0.9],
                    "det": 0.98,
                    "sim": 0.76,
                    "cluster": f"Face {group}",
                    "quality": 0.9,
                }
                writers.append(segments.FACE_MATCHES, dataset.HERE[dataset.PEOPLE[n]][0], [row])
                writers.append(
                    segments.FACE_EMBEDDINGS,
                    dataset.HERE[dataset.PEOPLE[n]][0],
                    [
                        {
                            "id": str(uuid.uuid4()),
                            "match_id": sid,
                            "ts_start": stamp,
                            "track_id": row["track_id"],
                            "vector": dataset._looks(n * 100, 512)[v],
                            "model": "demo",
                        }
                    ],
                )
                shots.append({"id": sid, "ts": stamp, "url": f"/demo/media/person-{n}-{v}.jpg"})
            groups_data.append({"id": group, "shots": shots})
        writers.sync()
    finally:
        writers.close()
    (folder / "groups.json").write_text(json.dumps(groups_data))
    unknown = cfg.path("paths.faces") / "_unknown"
    unknown.mkdir(exist_ok=True)
    for n, group in enumerate(groups):
        (unknown / f"{group}.jpg").write_bytes((folder / f"person-{n}-0.jpg").read_bytes())
    (folder / "day.json").write_text(
        json.dumps(
            {"day": day.isoformat(), "start": start, "end": start + 30, "moment": dataset._local(day, *dataset.MOMENT)}
        )
    )

    async def register() -> None:
        from continuum_py import Client

        from liberational import continuum_link

        async with await Client.connect(str(continuum_link.socket_path()), client_name="demo media") as client:
            await client.register_worker("demo-media", [])
            for camera in scenes.CAMERAS:
                await client.register_video_index(camera, str(folder))

    asyncio.run(register())


def camera_pairs(state: Path) -> None:
    """A simulated match from the known render geometry, never a model result.

    Kitchen and front-room see the same doorway. Their true poses refine the owner's rough kitchen spot;
    the other cameras keep their owner spots. Store ordinary anchors and pair evidence for the real viewer.
    """
    store = camera_clouds.Store(camera_clouds.root(state))
    spots = cloud_scene.spots_of(state)
    a, b = store.current("front-room"), store.current("kitchen")
    assert a is not None and b is not None

    def world(camera: str, pose: tuple[float, ...]) -> np.ndarray:
        spot = spots[camera].at
        assert spot is not None
        angle = spot.bearing
        transform = np.eye(4)
        transform[:3, :3] = [[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
        transform[:2, 3] = (0.2 if camera == "kitchen" else spot.x, spot.y)
        return transform @ np.asarray(pose).reshape(4, 4)

    relative = np.linalg.inv(world(a.camera, a.pose)) @ world(b.camera, b.pose)
    match = {
        "rotation": relative[:3, :3].ravel().tolist(),
        "shift": relative[:3, 3].tolist(),
        "match": [0.22, 0.18],
        "sure": [0.2, 0.18],
        "disagree": [0.0, 0.0, 0.0],
        "scale": 1.0,
    }
    for camera in scenes.CAMERAS:
        cloud = store.current(camera)
        assert cloud is not None
        cloud_scene.record(store, cloud, {"kitchen": match} if camera == "front-room" else {}, spots)


def activity(state: Path, day: dt.date) -> None:
    """Some observed clips wait for checking; the walk-by clips are counted by the normal activity reader."""
    start = dataset._local(day, *dataset.MEDIA_TIME)
    writers = segments.Writers(segments.root(state))
    ids = [str(uuid.uuid4()) for _ in scenes.CAMERAS]
    try:
        for n, (camera, name, kind, label) in enumerate(dataset.ACTIVITIES):
            at = start + n * 3
            row = {
                "id": ids[n],
                "ts_start": at,
                "ts_end": at + 8,
                "track_id": f"{camera}:demo",
                "camera_moved": False,
                "attributes": {"direction": "left_to_right", "supported": [[at, at + 8]], "observations": 40},
                "subject_type": "person",
                "kind": kind,
                "person": registry.at(state).person_id(name),
                "person_name": name,
            }
            writers.append(segments.WALKBY_SPANS if n < 2 else segments.ACTION_SPANS, camera, [row])
        for camera in ("front-room", "kitchen"):
            writers.append(
                segments.WALKBY_COVERAGE,
                camera,
                [
                    {
                        "id": str(uuid.uuid4()),
                        "ts_start": start,
                        "ts_end": start + 30,
                        "frames": 150,
                    }
                ],
            )
        writers.sync()
    finally:
        writers.close()
    for n, (_camera, _name, _kind, label) in enumerate(dataset.ACTIVITIES):
        if label is not None:
            events.name_activity(state, events.NameActivity(ids[n], start + n * 3, label))
    for n in range(2):
        events.feedback_activity(state, events.FeedbackActivity(ids[n], start + n * 3, "right"))


def paths(state: Path, day: dt.date) -> None:
    """Paths across the plan throughout the day, including a recent snapshot for the default Today view."""
    from liberational.people import pets

    place = places.places(state)[0]
    rows = []
    reg = registry.at(state)
    pet = pets.at(state).listing()[0]["id"]
    moment = dataset._local(day, *dataset.MOMENT)
    for name, journeys in dataset.ITINERARIES.items():
        entity = f"pet:{pet}" if name == dataset.PET else f"person:{reg.person_id(name)}"
        for j, (hour, minute, origin, destination) in enumerate(journeys):
            points = dataset.route(origin, destination)
            if j == len(journeys) - 1:
                points.append(dataset.HERE[name][2])
            expanded = []
            for a, b in pairwise(points):
                count = max(1, int(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 / 0.15))
                expanded.extend(
                    (a[0] + (b[0] - a[0]) * k / count, a[1] + (b[1] - a[1]) * k / count) for k in range(count)
                )
            expanded.append(points[-1])
            start = dataset._local(day, hour, minute)
            for k, (x, y) in enumerate(expanded):
                stamp = start + k * 60 / max(1, len(expanded) - 1)
                rows.append(
                    positions.Position(
                        str(uuid.uuid4()),
                        stamp,
                        stamp,
                        entity,
                        name,
                        f"demo-{name}-{j}",
                        "demo",
                        1.0,
                        place.id,
                        None,
                        None,
                        x,
                        y,
                        (
                            "kitchen"
                            if x < 4 and y < 3.5
                            else "study"
                            if x >= 4 and y > 5
                            else "front-room"
                            if x >= 4
                            else None
                        ),
                        None,
                        None,
                        place.plan_revision,
                    )
                )
            if j == len(journeys) - 1:
                # The still-visible person/pet remains at this endpoint at the replay moment.
                from dataclasses import replace

                rows.append(replace(rows[-1], id=str(uuid.uuid4()), t=moment, end=moment))
    positions.write(state, sorted(rows, key=lambda r: r.t))
    writers = segments.Writers(segments.root(state))
    try:
        for name in dataset.PEOPLE:
            journeys = dataset.ITINERARIES[name]
            for j, (hour, minute, _origin, destination) in enumerate(journeys):
                camera = {"kitchen": "kitchen", "living": "front-room", "study": "study"}.get(destination)
                if camera is None:
                    continue
                start = dataset._local(day, hour, minute) + 60
                end = dataset._local(day, *journeys[j + 1][:2]) if j + 1 < len(journeys) else moment
                writers.append(
                    segments.FACE_VISITS,
                    camera,
                    [
                        {
                            "id": str(uuid.uuid4()),
                            "ts_start": start,
                            "ts_end": end,
                            "name": name,
                            "best_sim": 0.9,
                            "samples": max(1, int((end - start) / 60)),
                            "shots": [],
                        }
                    ],
                )
        writers.sync()
    finally:
        writers.close()


def calls(day: dt.date) -> None:
    for n, name in enumerate(("Dana", "Sam")):
        folder = cfg.path("paths.calls") / day.isoformat() / f"demo-{n}"
        folder.mkdir(parents=True, exist_ok=True)
        start = dataset._local(day, 11 + n * 5, 20)
        (folder / "call.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "from": name,
                    "to": {"name": "Alice", "number": None},
                    "direction": "in",
                    "start": start,
                    "started": dt.datetime.fromtimestamp(start, dataset.history_qa.TZ).isoformat(),
                    "end": start + 145,
                    "duration_s": 145,
                    "durationSeconds": 145,
                    "turns": 2,
                    "speakers": {name: {}, "Alice": {}},
                    "outcome": "completed",
                    "number": "",
                    "summary": "Bringing groceries for lunch." if n == 0 else "Checking what we need for dinner.",
                }
            )
        )
        lines = (
            (
                "I'm stopping by with groceries. Do you need anything else?",
                "A loaf of bread would be lovely, thank you.",
            )
            if n == 0
            else (
                "I'm on my way home. Shall I pick up anything for dinner?",
                "We have pasta and tomatoes. See you soon.",
            )
        )
        (folder / "transcript.jsonl").write_text(
            "".join(
                json.dumps({"t": k * 12, "text": text, "speaker_label": name if k == 0 else "Alice"}) + "\n"
                for k, text in enumerate(lines)
            )
        )


def seed(state: Path, day: dt.date) -> None:
    from liberational.proximity import devices

    for label, target, name in (
        ("Alice's phone", "person", "Alice"),
        ("Sam's watch", "person", "Sam"),
        ("Key tag", "thing", "House keys"),
    ):
        devices.at().save("ble_name", label, target, name)
    media(state, day)
    activity(state, day)
    paths(state, day)
    calls(day)
