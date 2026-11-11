"""The offline demo through its real launcher, HTTP adapter, and generated files."""

import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest
from PIL import Image
from websockets.sync.client import connect

from liberational import config as cfg
from liberational.demo import cli, dataset


def review_spans(port: int, start: float, end: float) -> list[tuple[float, float]]:
    """Use the gateway protocol Review actually uses, not the media files' presence on disk."""
    spans = []
    with connect(f"ws://127.0.0.1:{port}/ws", open_timeout=10) as ws:
        assert json.loads(ws.recv(timeout=10))["type"] == "hello"
        ws.send(json.dumps({"type": "streams"}))
        streams = json.loads(ws.recv(timeout=10))["streams"]
        for stream in streams:
            if not re.fullmatch(r".+\.video(?:_v1)?", stream["name"]):
                continue
            ws.send(
                json.dumps(
                    {
                        "type": "spans",
                        "stream": stream["id"],
                        "start_us": int(start * 1e6),
                        "end_us": int(end * 1e6),
                        "gap_us": 0,
                        "media_context": True,
                    }
                )
            )
            spans.extend((a / 1e6, b / 1e6) for a, b in json.loads(ws.recv(timeout=10))["spans"])
    return spans


def test_demo_denies_network_and_command_execution(tmp_path: Path) -> None:
    cli.prepare(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import socket, subprocess, sys
from pathlib import Path
from liberational import config as cfg
from liberational.demo.server import offline
sys.addaudithook(offline)
for action in (lambda: socket.create_connection(('127.0.0.1', 9)),
               lambda: socket.getaddrinfo('example.test', 443),
               lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'x', ('127.0.0.1', 9)),
               lambda: subprocess.run([sys.executable, '-c', 'pass']),
               lambda: cfg.local_config().parent.parent.joinpath('outside.txt').write_text('escape')):
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
    # The Python-only CI job does not build the native daemons. Exercise the real launcher wherever installed.
    missing = [name for name in ("continuum-broker", "continuum-gateway") if not cfg.program(name).is_file()]
    if missing or shutil.which("ffmpeg") is None:
        pytest.skip("full-launch integration needs installed Continuum daemons and FFmpeg")
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
            cameras = json.loads(read("/devices"))["cameras"]
            assert [c["stream"] for c in cameras] == ["front-room", "kitchen", "porch", "study"]
            pictures = []
            for camera in ("front-room", "kitchen", "porch", "study"):
                jpg = read(f"/demo/media/{camera}.jpg")
                assert Image.open(io.BytesIO(jpg)).size == (960, 540)
                pictures.append(jpg)
                assert (root / "state" / "demo" / f"{camera}.mp4").stat().st_size > 1000
            assert len(set(pictures)) == 4
            recording = json.loads((root / "state" / "demo" / "day.json").read_text())
            page = read("/").decode()
            gateway = re.search(r'gatewayPort\\?":\s*(\d+)', page)
            assert gateway is not None
            day = dataset.dt.date.fromisoformat(recording["day"])
            start = dataset._local(day, 0)
            spans = review_spans(int(gateway[1]), start, start + 86400)
            assert spans
            assert all(
                any(a <= dataset._local(day, *line.at[:2]) + line.at[2] < b for a, b in spans)
                for line in dataset.TRANSCRIPT
            )
            clouds = list((root / "state" / "camera_clouds").glob("*/cloud.json"))
            assert len(clouds) == 4
            assert all(json.loads(p.read_text())["at"] == recording["start"] for p in clouds)
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
            # Prepared reports stay readable, but creating new research never starts a worker or model.
            with pytest.raises(HTTPError) as refused:
                read("/pack-ask/jobs", {"question": "What changed in the garden?", "deep": True})
            assert refused.value.code == 409
            inbox = json.loads(read("/research/jobs"))
            assert [job["status"] for job in inbox["jobs"]] == ["queued", "done", "done", "done"]
            report_id = inbox["jobs"][1]["id"]
            assert "pasta" in json.loads(read(f"/research/jobs/{report_id}"))["result"]["answer"]
            assert json.loads(read(f"/research/jobs/{report_id}/read", {"read": True}))["ok"]
            saved = json.loads(read("/pack-ask/history"))["items"]
            assert saved[0]["question"] == "What did we have for dinner?"
            examples = json.loads(read("/people/examples?name=Alice"))["examples"]
            removed = json.loads(read("/people/examples/remove", {"name": "Alice", "keys": [examples[0]["key"]]}))
            assert removed["count"] == 6
            assert json.loads(read("/people/examples?name=Alice"))["count"] == 6
            zone = {"name": "Table", "polygon": [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8]]}
            assert json.loads(read("/zones", {"cam": "kitchen", "base": None, "zones": [zone]}))["ok"]
            assert json.loads(read("/zones"))["cameras"]["kitchen"]["zones"][0]["name"] == "Table"
            links = json.loads(read("/device-links"))["links"]
            assert len(links) == 3
            assert json.loads(read("/device-links/remove", {"id": links[0]["id"]}))["ok"]
            linked = json.loads(read("/device-links"))
            assert len(linked["links"]) == 2
            assert linked["unknown"] == 1
            assert len(linked["devices"]) == 3
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


@pytest.mark.parametrize("action", ["cameras", "remove", "zones", "settings", "research"])
def test_demo_settings_and_teaching_use_scratch_state(tmp_path: Path, action: str) -> None:
    root = tmp_path / "demo"
    cli.prepare(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, sys, threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from liberational import config as cfg
from liberational.demo import dataset, server
from liberational.dashboard.server import DashboardServer
from liberational.vision import faces
dataset.seed_faces(cfg.path('paths.faces'), 1000)
http = DashboardServer(('127.0.0.1', 0), server.Handler)
thread = threading.Thread(target=http.serve_forever, daemon=True)
thread.start()
base = 'http://127.0.0.1:' + str(http.server_port)
def read(path, body=None):
    request = Request(base + path, data=None if body is None else json.dumps(body).encode(),
                      headers={'Origin': base, 'Content-Type': 'application/json'})
    with urlopen(request, timeout=10) as reply:
        return json.load(reply)
try:
    if sys.argv[1] == 'cameras':
        cameras = read('/devices')['cameras']
        assert [c['stream'] for c in cameras] == ['front-room', 'kitchen', 'porch', 'study'], cameras
        assert all(c['present'] and c['selected'] and c['in_use'] for c in cameras)
    elif sys.argv[1] == 'research':
        with cfg.override({'ask.ollama_url': 'http://example.invalid'}):
            listing = read('/research/jobs')
            assert not listing['available']
            assert 'offline demo' in listing['reason']
            try:
                read('/pack-ask/jobs', {'question': 'How was the afternoon?', 'deep': True})
            except HTTPError as error:
                assert error.code == 409
                assert 'running model' in json.load(error)['error']
            else:
                raise AssertionError('The offline demo queued new research')
            assert read('/research/jobs')['jobs'] == []
    elif sys.argv[1] == 'zones':
        zone = {'name': 'Table', 'polygon': [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8]]}
        result = read('/zones', {'cam': 'kitchen', 'base': None, 'zones': [zone]})
        assert result['ok'], result
        assert read('/zones')['cameras']['kitchen']['zones'][0]['name'] == 'Table'
        assert (cfg.path('paths.state') / 'knowledge.sqlite3').exists()
    elif sys.argv[1] == 'settings':
        assert read('/config', {'values': {'vision.camera_names': ['kitchen: Counter']}})['ok']
        assert 'Counter' in cfg.local_config().read_text()
        assert read('/devices/camera', {'index': 1, 'flip': True})['cameras'][1]['flip']
        assert 'flip = true' in cfg.devices_file().read_text()
        assert read('/devices/sensitivity', {'mic': 'demo-mic', 'sensitivity': 45})['mics'][0]['sensitivity'] == 45
        assert read('/devices/auto_gain', {'mic': 'demo-mic', 'on': True})['mics'][0]['auto_gain']
    else:
        rows = read('/people/examples?name=Alice')['examples']
        assert len(rows) == 7
        result = read('/people/examples/remove', {'name': 'Alice', 'keys': [rows[0]['key']]})
        assert result['removed'] == 1, result
        assert len(faces.examples('Alice')) == 6
        assert faces.DB.resolve().is_relative_to(cfg.local_config().parent)
finally:
    http.shutdown()
    http.server_close()
    thread.join()
""",
            action,
        ],
        env=cli.environment(root, cli.PORT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == [root]
