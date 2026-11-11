"""The offline demo through its real launcher, HTTP adapter, and generated files."""

import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image

from liberational.demo import cli, dataset


def test_demo_denies_network_and_command_execution(tmp_path: Path) -> None:
    cli.prepare(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import socket, subprocess, sys
from liberational.demo.server import offline
sys.addaudithook(offline)
for action in (lambda: socket.create_connection(('127.0.0.1', 9)),
               lambda: socket.getaddrinfo('example.test', 443),
               lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'x', ('127.0.0.1', 9)),
               lambda: subprocess.run([sys.executable, '-c', 'pass'])):
    try:
        action()
    except PermissionError:
        pass
    else:
        raise AssertionError('offline boundary allowed an external operation')
with socket.socket(socket.AF_UNIX) as local:
    try:
        local.connect('missing-demo.sock')
    except FileNotFoundError:
        pass
""",
        ],
        env=cli.environment(tmp_path, cli.PORT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_demo_starts_with_media_and_answers_without_a_model(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    root = tmp_path / "house"
    base = f"http://127.0.0.1:{port}"
    log = tmp_path / "demo.log"

    def read(path: str, body: dict | None = None) -> bytes:
        data = None if body is None else json.dumps(body).encode()
        request = Request(base + path, data=data, headers={"Content-Type": "application/json", "Origin": base})
        with urlopen(request, timeout=20) as response:
            return response.read()

    with log.open("wb") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "liberational.demo.cli", "--dir", str(root), "--port", str(port)],
            env=dict(os.environ),
            stdout=output,
            stderr=output,
        )
        try:
            deadline = time.monotonic() + 300
            while True:
                assert process.poll() is None, log.read_text()
                try:
                    read("/health")
                    break
                except URLError:
                    assert time.monotonic() < deadline, log.read_text()
                    time.sleep(0.2)
            pictures = []
            for camera in ("front-room", "kitchen", "porch", "study"):
                jpg = read(f"/demo/media/{camera}.jpg")
                assert Image.open(io.BytesIO(jpg)).size == (960, 540)
                pictures.append(jpg)
                assert (root / "state" / "demo" / f"{camera}.mp4").stat().st_size > 1000
            assert len(set(pictures)) == 4
            people = json.loads(read("/people"))["people"]
            assert {p["name"]: p["here_now"]["cam"] for p in people} == {
                "Alice": "kitchen",
                "Dana": "front-room",
                "Sam": "front-room",
            }
            place = json.loads(read("/places"))["places"][0]["id"]
            scene = json.loads(read(f"/places/scene?place={place}"))
            assert len(scene["views"]) == 4
            assert all(v["cloud"] and v["points"] > 1000 for v in scene["views"])
            assert read(scene["map"]).startswith(b"ply\n")
            groups = json.loads(read("/faces/unknown/groups"))["groups"]
            assert len(groups) == 2
            strip = json.loads(read(f"/faces/unknown/{groups[0]['id']}/sightings"))
            assert len(strip["sightings"]) == strip["total"] == 3
            assert all(read(shot["url"]).startswith(b"\xff\xd8") for shot in strip["sightings"])
            sightings = json.loads(read("/pets/sightings"))
            assert sightings["photos_unavailable"] == sightings["videos_unavailable"] == 0
            assert all(row["video_status"] == "ready" for rows in sightings["seen"].values() for row in rows)
            assert all(len(g["shots"]) == g["count"] == 3 for g in groups)
            assert all(g["count"] == 3 for g in json.loads(read("/people"))["unknown"]["faces"])
            assert all(Image.open(io.BytesIO(read(g["shots"][0]["url"]))).size == (800, 800) for g in groups)
            assert len({read(shot["url"]) for g in groups for shot in g["shots"]}) == 6
            assert len(json.loads(read("/device-comings"))["devices"]) == 2
            assert len(json.loads(read("/live/happening"))["items"]) == 3
            assert json.loads(read("/review/summary"))["status"] == "ready"
            for kind in ("activity", "conversations", "whos_around"):
                deadline = time.monotonic() + 20
                while True:
                    report = json.loads(read(f"/reports/api/report?kind={kind}"))
                    if not report["pending"]:
                        break
                    assert time.monotonic() < deadline, report
                    time.sleep(0.1)
                assert report["first"] == report["last"]
                if kind == "activity":
                    assert sum(e["total"] for e in report["events"]) == 2
                    assert report["covered_days"]["events"] == 1
                elif kind == "conversations":
                    assert sum(p["lines"] for p in report["people"]) == len(dataset.TRANSCRIPT)
                else:
                    assert all(p["hours"] > 0 for p in report["people"])

            job = json.loads(read("/pack-ask/jobs", {"question": "What did we have for dinner?"}))
            assert job["status"] == "queued"
            result = json.loads(read(job["poll_url"]))
            assert result["status"] == "completed"
            assert "pasta" in result["result"]["answer"]
            pets = json.loads(read("/pets"))
            photo = pets["pets"][0]["photo"]
            assert Image.open(io.BytesIO(read(f"/pets/sightings/photo?track={photo}"))).size == (800, 800)
            assigned = json.loads(read("/people/assign", {"name": "Alice", "faces": [groups[0]["id"]]}))
            assert assigned["assigned"]["faces"] == 1
            assert assigned["failed"] == []
            assert json.loads(read("/faces/unknown/groups"))["count"] == 1
            assert not list((root / "models").glob("**/*"))
        finally:
            cli.stop(process)
    assert cli.port_free(port)


def test_demo_day_routes_cross_only_doorways() -> None:
    """Every boundary crossing in every itinerary must be through an actual opening of the shipped plan."""
    from itertools import pairwise

    doors = [point for kind, point, _width in dataset._OPENINGS if kind == "door"]
    paths = {}
    for name, journeys in dataset.ITINERARIES.items():
        paths[name] = [dataset.route(a, b) for _, _, a, b in journeys]
        for points in paths[name]:
            for a, b in pairwise(points):
                # Test the internal wall centre lines, limited to each wall's extent.
                for axis, wall, low, high in ((0, 4, 0, 8.5), (1, 3.5, 0, 4), (1, 5, 0, 9)):
                    if a[axis] == b[axis]:
                        continue
                    t = (wall - a[axis]) / (b[axis] - a[axis])
                    other = a[1 - axis] + t * (b[1 - axis] - a[1 - axis])
                    if 0 <= t <= 1 and low <= other <= high:
                        point = (wall, other) if axis == 0 else (other, wall)
                        assert any(abs(point[0] - d[0]) < 0.001 and abs(point[1] - d[1]) < 0.001 for d in doors), (
                            name,
                            a,
                            b,
                            point,
                        )
    assert len({str(path) for path in paths.values()}) == 4
    assert set(paths) == {*dataset.PEOPLE, dataset.PET}
