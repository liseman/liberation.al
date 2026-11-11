"""Seed the photographable demo through the application's ordinary file and segment writers.

The generated media depicts only the procedural demo meshes. No camera, model, or remote asset is consulted.
"""

import asyncio
import datetime as dt
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from liberational import config as cfg
from liberational import segment_log as segments
from liberational.demo import dataset, scenes
from liberational.memory import events, places
from liberational.people import registry
from liberational.spatial import camera_clouds, ply, positions


def media(state: Path, day: dt.date) -> None:
    """Render four distinct rooms and portraits, register short H.264 recordings, and keep each camera's cloud."""
    folder = state / "demo"
    folder.mkdir(exist_ok=True)
    index = []
    start = dataset._local(day, 18, 38)
    clouds = camera_clouds.Store(camera_clouds.root(state))
    for camera in scenes.CAMERAS:
        jpg, points, colors, pose = scenes.render(scenes.room(camera))
        picture = folder / f"{camera}.jpg"
        picture.write_bytes(jpg)
        clip = folder / f"{camera}.mp4"
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
                "30",
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
        index.append({"cam": camera, "file": clip.name, "start": start, "fps": 5.0, "w": 960, "h": 540})
        evening = folder / f"{camera}-evening.mp4"
        evening.unlink(missing_ok=True)
        os.link(clip, evening)
        index.append(
            {"cam": camera, "file": evening.name, "start": dataset._local(day, 22, 5), "fps": 5.0, "w": 960, "h": 540}
        )
        cloud = camera_clouds.Cloud(
            str(uuid.uuid4()),
            camera,
            time.time(),
            len(points),
            1.0,
            "height",
            1.0,
            True,
            tuple(float(v) for v in pose.ravel()),
            960 * 0.72,
            960,
            540,
        )
        clouds.replace(camera_clouds.Made(cloud, ply.dumps(points, colors)))
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
    import numpy as np

    from liberational.vision import face_examples, faces

    for n, name in enumerate(dataset.PEOPLE):
        for k, example in enumerate(faces.examples(name)):
            picture = face_examples.picture(cfg.path("paths.faces"), example["key"])
            if picture is not None:
                who = (n + 1) % 3 if k == 0 else n  # the outlier stays visibly different
                picture.write_bytes((folder / f"person-{who}-{k % 3}.jpg").read_bytes())

    groups = [faces.new_group(np.stack(dataset._looks(n * 100, 512)[:6])) for n in range(2)]
    groups_data = []
    writers = segments.Writers(segments.root(state))
    try:
        for n, group in enumerate(groups):
            shots = []
            for v in range(3):
                sid, stamp = str(uuid.uuid4()), time.time() - v * 120
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
                writers.append(segments.FACE_MATCHES, "front-room", [row])
                writers.append(
                    segments.FACE_EMBEDDINGS,
                    "front-room",
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
        (unknown / f"{group}.jpg").write_bytes(scenes.portrait(n))
    (folder / "day.json").write_text(json.dumps({"day": day.isoformat(), "start": start, "end": start + 30}))

    async def register() -> None:
        from continuum_py import Client

        from liberational import continuum_link

        async with await Client.connect(str(continuum_link.socket_path()), client_name="demo media") as client:
            await client.register_worker("demo-media", [])
            for camera in scenes.CAMERAS:
                await client.register_video_index(camera, str(folder))

    asyncio.run(register())


def activity(state: Path, day: dt.date) -> None:
    """Some observed clips wait for checking; the walk-by clips are counted by the normal activity reader."""
    start = dataset._local(day, 18, 38)
    writers = segments.Writers(segments.root(state))
    ids = [str(uuid.uuid4()) for _ in scenes.CAMERAS]
    try:
        for n, camera in enumerate(scenes.CAMERAS):
            at = start + n * 3
            row = {
                "id": ids[n],
                "ts_start": at,
                "ts_end": at + 8,
                "track_id": f"{camera}:demo",
                "camera_moved": False,
                "attributes": {"direction": "left_to_right", "supported": [[at, at + 8]], "observations": 40},
                "subject_type": "person",
                "kind": "read" if n == 3 else "stand_up",
                "person": registry.at(state).person_id(dataset.PEOPLE[n % 3]),
                "person_name": dataset.PEOPLE[n % 3],
            }
            writers.append(segments.WALKBY_SPANS if n < 2 else segments.ACTION_SPANS, camera, [row])
        writers.sync()
    finally:
        writers.close()
    events.name_activity(state, events.NameActivity(ids[2], start + 6, "Coming home"))
    for n in range(2):
        events.feedback_activity(state, events.FeedbackActivity(ids[n], start + n * 3, "right"))


def paths(state: Path, day: dt.date) -> None:
    """Paths across the plan throughout the day, including a recent snapshot for the default Today view."""
    place = places.places(state)[0]
    points = ((2.0, 1.8), (2.8, 1.8), (4.5, 1.8), (6.5, 2.8), (6.5, 4.4), (6.5, 6.4))
    rows = []
    reg = registry.at(state)
    for base in (dataset._local(day, 7), dataset._local(day, 12), dataset._local(day, 18), time.time() - 1800):
        for n, name in enumerate(dataset.PEOPLE):
            for k in range(60):
                a, b = points[(k // 12) % 5], points[(k // 12) % 5 + 1]
                portion = (k % 12) / 12
                x, y = a[0] + (b[0] - a[0]) * portion, a[1] + (b[1] - a[1]) * portion
                stamp = base + n * 120 + k * 4
                rows.append(
                    positions.Position(
                        str(uuid.uuid4()),
                        stamp,
                        stamp,
                        f"person:{reg.person_id(name)}",
                        name,
                        f"demo-{name}-{int(base)}",
                        "demo",
                        1.0,
                        place.id,
                        None,
                        None,
                        x + n * 0.12,
                        y + n * 0.12,
                        "front-room",
                        None,
                        None,
                        place.plan_revision,
                    )
                )
    positions.write(state, sorted(rows, key=lambda r: r.t))


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
                    "turns": 4,
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
    media(state, day)
    activity(state, day)
    paths(state, day)
    calls(day)
