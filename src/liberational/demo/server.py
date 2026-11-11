"""The demo's HTTP process: ordinary dashboard reads plus offline live and Ask fixtures.

Only pack-demo launches this module. It never starts dashboard warmers or hardware discovery. Outbound sockets
and subprocesses are refused, and writes that could trigger a service, model, call, or import are unavailable.
"""

import json
import mimetypes
import signal
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from liberational import config as cfg
from liberational.dashboard import server, transcript_lines
from liberational.demo import dataset, scenes

JOBS: dict[str, dict[str, Any]] = {}
STOP = threading.Event()


def offline(event: str, args: tuple[Any, ...]) -> None:
    if event in ("socket.connect", "socket.sendto", "socket.sendmsg") and args[0].family != socket.AF_UNIX:
        raise PermissionError("The demo never connects to a model or network service")
    if event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr") and args[0] not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        raise PermissionError("The demo never resolves external network names")
    if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"):
        raise PermissionError("The demo dashboard never starts external commands")


def answer(question: str) -> str:
    text = question.casefold()
    if "rex" in text or "dog" in text:
        return "Alice and Dana took Rex for a walk before the rain at lunchtime. Dana dried him off in the hall when they got back."
    if "dinner" in text or "eat" in text:
        return "Alice cooked pasta, Sam set the table, and Dana joined them for dinner. Afterwards they played a board game; Sam won."
    if "dana" in text or "visit" in text or "who" in text:
        return "Dana stopped by with groceries at lunchtime and returned for dinner. Alice and Sam were home in the evening."
    if "deadline" in text or "friday" in text:
        return "The project deadline moved to Friday. Dana is helping Alice with the slides."
    return "Alice worked in the study, Dana brought groceries, and Sam came home before dinner. They had pasta together and played a board game. This demo has prepared answers about dinner, visitors, Rex, and the project deadline."


def devices(path: str) -> dict[str, Any]:
    now = time.time()
    links = [
        {"id": str(n), "kind": "ble_name", "value": label, "target": target, "name": name}
        for n, (label, target, name) in enumerate(
            (("Alice's phone", "person", "Alice"), ("Sam's watch", "person", "Sam"), ("Key tag", "thing", "House keys"))
        )
    ]
    common = {
        "at": now,
        "start": now - 8 * 3600,
        "gap_seconds": 300,
        "caveat": "A device seen is a clue, not proof of who carried it.",
    }
    if path == "/device-presence":
        return {
            **common,
            "people": [
                {
                    "name": name,
                    "status": "recent",
                    "last_seen": now - 30,
                    "source": "bluetooth.sighting",
                    "place": "Demo house",
                    "links": 1,
                }
                for name in ("Alice", "Sam")
            ],
        }
    if path == "/device-whereabouts":
        return {
            **common,
            "days": 7,
            "links": [
                {
                    "target": l["target"],
                    "name": l["name"],
                    "links": 1,
                    "last_seen": now - 30,
                    "place": "Demo house",
                    "rssi": -48,
                    "source": "bluetooth.sighting",
                }
                for l in links
            ],
        }
    if path == "/device-comings":
        return {
            **common,
            "rows": 48,
            "devices": [
                {
                    "mac": f"demo-device-{n}",
                    "name": l["value"],
                    "place": "Demo house",
                    "linked": l,
                    "stays": [
                        {"start": now - (7 - n) * 3600, "end": now - (4 - n) * 3600},
                        {"start": now - (2 - n) * 3600, "end": now - 30},
                    ],
                }
                for n, l in enumerate(links[:2])
            ],
        }
    return {
        "devices": [
            {
                "source": "bluetooth.sighting",
                "identifier": l["value"],
                "name": l["value"],
                "place": "Demo house",
                "rssi": -48 - n * 5,
                "last_seen": now - 30,
                "sightings": 28 + n * 7,
                "selectors": [{"kind": l["kind"], "value": l["value"]}],
                "linked": l,
            }
            for n, l in enumerate(links)
        ],
        "links": links,
        "total": 3,
        "linked": 3,
        "unknown": 0,
        "hours": 24,
        "places": [],
    }


class Handler(server.H):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in server.APP_PATHS and (path == "/" or "text/html" in self.headers.get("Accept", "")):
            code, page = server.app_page()
            page = page.replace(b"</head>", b"<script>window.__PACK_DEMO__=true;</script></head>")
            return self._send(code, "text/html; charset=utf-8", page)
        if path.startswith("/demo/media/"):
            name = path.removeprefix("/demo/media/")
            allowed = {f"{c}.{ext}" for c in scenes.CAMERAS for ext in ("jpg", "mp4")}
            allowed |= {f"person-{n}-{v}.jpg" for n in range(3) for v in range(3)}
            if name not in allowed:
                return self._json({"error": "Unknown demo picture"}, 404)
            return self._media((self.state / "demo" / name).read_bytes(), mimetypes.guess_type(name)[0] or "image/jpeg")
        if path == "/people/video-thumb.jpg":
            camera = self._query().get("cam", "")
            if camera not in scenes.CAMERAS:
                return self._json({"error": "Unknown demo camera"}, 404)
            return self._media((self.state / "demo" / f"{camera}.jpg").read_bytes(), "image/jpeg")
        if path == "/events":
            now = time.time()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            entries = transcript_lines.merged_tail(80)
            payloads = {
                "hello": {"ts": now},
                "transcript": {"entries": entries, "reset": True},
                "scene": [
                    {"cam": c, "caption": text, "ts": now, "faces": []}
                    for c, text in zip(
                        scenes.CAMERAS,
                        (
                            "Alice is reading while Rex rests nearby.",
                            "Alice and Dana are preparing dinner.",
                            "Dana has arrived with groceries.",
                            "Alice is working at the desk.",
                        ),
                    )
                ],
                "ambient": {"devices": {dataset.MIC: {"name": dataset.MIC, "healthy": False}}},
                "vision": {
                    "ts": now,
                    "cameras": {c: {"state": "on", "size": [960, 540], "fps": 0} for c in scenes.CAMERAS},
                },
                "ready": {"ts": now},
            }
            try:
                for event, value in payloads.items():
                    self.wfile.write(f"event: {event}\ndata: {json.dumps(value)}\n\n".encode())
                self.wfile.flush()
                while not STOP.wait(10):
                    for row in payloads["scene"]:
                        row["ts"] = time.time()
                    payloads["vision"]["ts"] = time.time()
                    for event, value in (
                        ("scene", payloads["scene"]),
                        ("vision", payloads["vision"]),
                        ("ping", {"ts": time.time()}),
                    ):
                        self.wfile.write(f"event: {event}\ndata: {json.dumps(value)}\n\n".encode())
                    self.wfile.flush()
            except OSError:
                pass
            self.close_connection = True
            return
        if path == "/pets/sightings/photo":
            from liberational.people import pet_recognition

            jpg = pet_recognition.kept_photo(self.state, self._query().get("track", ""))
            if jpg is not None:
                return self._media(jpg, "image/jpeg")
        if path == "/pets":
            from liberational.dashboard import pets as pet_view

            code, view = pet_view.listing(self.state)
            if code == 200:
                view["here"] = {
                    p["id"]: {
                        "cam": "front-room",
                        "camera_name": "Front room",
                        "since": time.time() - 500,
                        "track": p["photo"],
                    }
                    for p in view["pets"]
                }
            return self._json(view, code)
        if path == "/people":
            from liberational.dashboard import roster_read

            view = roster_read.people_view(cold=True)
            view = {
                **view,
                "people": [
                    {**p, "here_now": {"cam": "front-room", "camera_name": "Front room", "since": time.time() - 300}}
                    for p in view.get("people", [])
                ],
            }
            return self._json(view)
        if path == "/live/happening":
            now = time.time()
            texts = (
                ("Dana arrived with groceries", "arrival", "porch", ["Dana"]),
                ("Alice put the kettle on", "activity", "kitchen", ["Alice"]),
                ("Sam set the table for dinner", "activity", "front-room", ["Sam"]),
            )
            return self._json(
                {
                    "items": [
                        {
                            "id": f"demo-{n}",
                            "ts": now - (3 - n) * 120,
                            "text": text,
                            "kind": kind,
                            "people": people,
                            "cam": cam,
                            "camera_name": cam.replace("-", " ").title(),
                        }
                        for n, (text, kind, cam, people) in enumerate(texts)
                    ],
                    "updated": now,
                }
            )
        if path == "/review/summary":
            import datetime as dt

            day = dt.date.fromisoformat(json.loads((self.state / "demo" / "day.json").read_text())["day"])
            start = dataset._local(day, 0)
            return self._json(
                {
                    "start": start,
                    "end": start + 86400,
                    "level": 3600,
                    "status": "ready",
                    "blocks": [
                        {
                            "start": dataset._local(day, entry.at),
                            "end": dataset._local(day, entry.at) + 3600,
                            "text": entry.text,
                            "tags": [],
                            "evidence": {"status": "unverified", "sources": []},
                        }
                        for entry in dataset.DAY
                    ],
                }
            )
        if path.startswith("/device-"):
            return self._json(devices(path))
        if path == "/config":
            return self._json(
                {"ok": True, "control": True, "sections": cfg.schema(), "needs_setup": False, "path": "pack.toml"}
            )
        if path == "/faces/unknown/groups":
            from liberational.vision import faces

            waiting = {int(row["id"]) for row in faces.unknown_groups()}
            ids = json.loads((self.state / "demo" / "groups.json").read_text())
            now = time.time()
            groups = [
                {
                    "id": ids[n]["id"],
                    "label": f"Unnamed face {n + 1}",
                    "count": 6 + n,
                    "last_seen": now - n * 600,
                    "guess": {"name": name, "score": 76 - n * 9, "band": "maybe", "close": None},
                    "score": 76 - n * 9,
                    "band": "maybe",
                    "reason": "Seen from a different angle",
                    "shots": ids[n]["shots"],
                    "shot_count": 3,
                }
                for n, name in enumerate(("Alice", "Dana"))
                if ids[n]["id"] in waiting
            ]
            return self._json({"groups": groups, "count": len(groups), "skipped": [], "pending": False})
        if path == "/voices/unknown/groups":
            return self._json({"groups": [], "count": 0, "skipped": [], "pending": False})
        if path.startswith("/pack-ask/jobs/"):
            job = JOBS.get(path.rsplit("/", 1)[-1])
            return self._json(job or {"error": "Question not found"}, 200 if job else 404)
        if path == "/pack-ask/history":
            return self._json({"items": [], "total": 0})
        if path == "/services":
            return self._json({"ok": True, "control": False, "platform": "demo", "services": []})
        if path == "/devices":
            return self._json({"ok": True, "control": False, "mics": [], "cameras": [], "notes": {}})
        if path in ("/usage", "/audio/levels"):
            return self._json({"demo": True})
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/pack-ask/jobs":
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                return self._json({"error": "Invalid question size"}, 400)
            body = json.loads(self.rfile.read(length))
            question = str(body.get("question", ""))
            now, key = time.time(), uuid.uuid4().hex
            job = {
                "id": key,
                "poll_url": f"/pack-ask/jobs/{key}",
                "mode": "local",
                "scope": "history",
                "question": question,
                "created_at": now,
                "started_at": now,
                "finished_at": now,
                "status": "completed",
                "result": {
                    "status": "ok",
                    "answer": answer(question),
                    "evidence": [],
                    "coverage": {},
                    "warnings": ["Prepared answer from the fictional demo household."],
                },
            }
            JOBS[key] = job
            return self._json({"id": key, "status": "queued", "poll_url": job["poll_url"]}, 202)
        # These writes only teach the isolated household. All operational actions stay unavailable.
        if self.path in {
            "/activity/name",
            "/activity/feedback",
            "/people/assign",
            "/people/assign-many",
            "/people/rename",
            "/people/describe",
            "/people/claim",
            "/people/add",
            "/people/photo",
            "/people/correct",
            "/people/undo",
            "/faces/not",
            "/faces/unknown/reject",
            "/faces/unknown/split",
            "/faces/unknown/hide",
            "/faces/unknown/restore",
            "/pets/save",
            "/pets/rename",
            "/pets/photo",
        }:
            return super().do_POST()
        return self._json({"error": "This action is unavailable in the offline demo."}, 403)


def main() -> None:
    from liberational.dashboard import presence

    sys.addaudithook(offline)
    pool = ThreadPoolExecutor(max_workers=1)
    presence._executor = pool
    port = int(sys.argv[1])
    http = server.DashboardServer(("127.0.0.1", port), Handler)

    def stop(*_: object) -> None:
        STOP.set()
        threading.Thread(target=http.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    try:
        http.serve_forever()
    finally:
        http.server_close()
        pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
