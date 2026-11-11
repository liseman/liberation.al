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

from liberational.demo import cli


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
            deadline = time.monotonic() + 150
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
            assert all(len(g["shots"]) == 3 for g in groups)
            assert len({read(shot["url"]) for g in groups for shot in g["shots"]}) == 6
            assert len(json.loads(read("/device-comings"))["devices"]) == 2
            assert len(json.loads(read("/live/happening"))["items"]) == 3
            assert json.loads(read("/review/summary"))["status"] == "ready"
            job = json.loads(read("/pack-ask/jobs", {"question": "What did we have for dinner?"}))
            assert job["status"] == "queued"
            result = json.loads(read(job["poll_url"]))
            assert result["status"] == "completed"
            assert "pasta" in result["result"]["answer"]
            pets = json.loads(read("/pets"))
            photo = pets["pets"][0]["photo"]
            assert Image.open(io.BytesIO(read(f"/pets/sightings/photo?track={photo}"))).size == (400, 400)
            assigned = json.loads(read("/people/assign", {"name": "Alice", "faces": [groups[0]["id"]]}))
            assert assigned["assigned"]["faces"] == 1
            assert assigned["failed"] == []
            assert json.loads(read("/faces/unknown/groups"))["count"] == 1
            assert not list((root / "models").glob("**/*"))
        finally:
            cli.stop(process)
    assert cli.port_free(port)
