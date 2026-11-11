"""The demo's HTTP process: ordinary dashboard reads plus offline live and Ask fixtures.

Only pack-demo launches this module. It never starts dashboard warmers or hardware discovery. Outbound sockets
and subprocesses are refused. Ordinary teaching writes use the demo scratch state; hardware and model actions
explain what they need at their endpoint. The launcher resets the household on every start.
"""

import datetime as dt
import json
import mimetypes
import os
import signal
import socket
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from liberational import config as cfg
from liberational.dashboard import server, transcript_lines
from liberational.demo import dataset, scenes

JOBS: dict[str, dict[str, Any]] = {}
STOP = threading.Event()


def offline(event: str, args: tuple[Any, ...]) -> None:
    # Mutable files and SQLite databases belong only to this demo, including through symlinks.
    writing = event == "open" and bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    paths = args[:1] if writing or event in {"os.remove", "os.mkdir", "os.rmdir", "sqlite3.connect"} else ()
    if event in {"os.rename", "os.link", "os.symlink"}:
        paths = args[:2]
    root = cfg.local_config().parent.resolve()
    for path in paths:
        if (
            isinstance(path, (str, bytes, os.PathLike))
            and str(path) != ":memory:"
            and not Path(os.fsdecode(path)).resolve().is_relative_to(root)
        ):
            raise PermissionError("Demo changes must stay in its scratch directory")
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
        return "Alice is cooking pasta in the kitchen. Sam has set the table, and Dana has joined him and Rex in the living room. Dinner is planned for 18:38."
    if "dana" in text or "visit" in text or "who" in text:
        return "Dana stopped by with groceries at lunchtime and returned for dinner. Alice and Sam were home in the evening."
    if "deadline" in text or "friday" in text:
        return "The project deadline moved to Friday. Dana is helping Alice with the slides."
    if "today" not in text and "day" not in text and "happened" not in text:
        return "Answering a new question needs a running model. This offline demo has prepared answers about the household day, dinner, visitors, Rex, and the project deadline."
    return "Alice worked in the study, Dana brought groceries, and Sam came home before dinner. Alice is cooking pasta while Dana and Sam wait in the living room with Rex. This demo has prepared answers about dinner, visitors, Rex, and the project deadline."


def day() -> dt.date:
    return dt.date.fromisoformat(json.loads((cfg.path("paths.state") / "demo" / "day.json").read_text())["day"])


def clock() -> float:
    """Keep the offline replay at its photographed moment, even when opened on another day."""
    return dataset._local(day(), *dataset.MOMENT)


def here(name: str) -> dict[str, Any]:
    from liberational.vision import views

    camera, label, _, since = dataset.HERE[name]
    return {"cam": camera, "camera_name": views.names().get(camera, label), "since": dataset._local(day(), *since)}


def devices(path: str) -> dict[str, Any]:
    now = clock()
    from dataclasses import asdict

    from liberational.proximity import devices as proximity

    store = proximity.at()
    links = [asdict(link) for link in store.links()]
    observations = ["Alice's phone", "Sam's watch", "Key tag"]
    linked = {link["value"]: link for link in links}
    options = proximity.location_options()
    names = {option["id"]: option["name"] for option in options}
    located = {location.value: location.as_dict(names) for location in store.locations()}
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
                for name in dict.fromkeys(
                    link["name"] for link in links if link["target"] == "person" and link["value"] in observations
                )
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
                    "name": label,
                    "place": "Demo house",
                    "linked": linked.get(label),
                    "stays": [
                        {
                            "start": dataset._local(day(), 7),
                            "end": dataset._local(day(), 12, 45) if n == 0 else dataset._local(day(), 7, 40),
                        },
                        {"start": dataset._local(day(), 13 if n == 0 else 17), "end": now},
                    ],
                }
                for n, label in enumerate(observations[:2])
            ],
        }
    return {
        "devices": [
            {
                "source": "bluetooth.sighting",
                "identifier": label,
                "name": label,
                "place": "Demo house",
                "rssi": -48 - n * 5,
                "last_seen": now - 30,
                "sightings": 28 + n * 7,
                "selectors": [{"kind": "ble_name", "value": label}],
                "linked": linked.get(label),
                "location": located.get(label),
            }
            for n, label in enumerate(observations)
        ],
        "links": links,
        "total": len(observations),
        "linked": len(links),
        "unknown": len(observations) - len(linked.keys() & set(observations)),
        "hours": 24,
        "places": options,
        "locations": list(located.values()),
    }


def unavailable(path: str) -> str | None:
    """Only operations requiring hardware, a model, or a remote service are excluded."""
    if path.startswith(
        (
            "/services",
            "/devices/capture",
            "/devices/camera-controls",
            "/devices/mic",
            "/devices/privacy",
            "/objects/api/capture",
            "/network-cameras/",
        )
    ):
        return "The demo replays prepared pictures and sound. Connecting hardware or restarting capture needs a real install."
    if path in {"/calls/place", "/calls/hangup"}:
        return "Making a phone call needs a connected phone service. The demo includes two recorded example calls."
    if path.startswith("/playback/"):
        return "Playing through a speaker needs a real audio device. Use the recording player in this browser."
    if path in {
        "/research/jobs",
        "/clones/say",
        "/clones/build",
        "/clones/record",
        "/speakers/refine",
        "/people/search",
        "/objects/api/segment",
        "/objects/api/find",
    } or path.endswith(("/lookup", "/reference-url", "/reference-upload")):
        return "This action needs a running model or online search. The demo runs neither; you can still name and correct its saved examples."
    if path.startswith("/ingest/") and not path.endswith(("/items", "/remove", "/delete")):
        return "Importing new recordings needs the ingestion service. The demo's prepared recordings can be reviewed and labelled."
    return None


def camera_settings() -> dict[str, Any]:
    from liberational.audio import sensitivity
    from liberational.dashboard import cameras as camera_manifest
    from liberational.dashboard import devices as device_view

    return {
        "ok": True,
        "control": True,
        "mics": [
            {
                "id": dataset.MIC,
                "name": dataset.MIC,
                "stream": dataset.MIC,
                "present": True,
                "selected": True,
                "in_use": False,
                "host_api": "Prepared transcript",
                "level_dbfs": None,
                **device_view._sensitivity(dataset.MIC),
                **device_view._auto_gain(dataset.MIC, None),
            }
        ],
        "default_sensitivity": sensitivity.default(),
        "notes": {
            "cameras": "Prepared camera pictures. Names, zones, and saved preferences are editable; live capture needs a real camera.",
            "mics": "The demo transcript uses demo-mic. Live listening needs a real microphone.",
        },
        "cameras": [
            {
                "index": n,
                "name": camera,
                "path": None,
                "usb": None,
                "paths": [],
                "shared_usb": False,
                "present": True,
                "selected": True,
                "stream": camera,
                "in_use": True,
                "flip": bool(camera_manifest.video_settings(camera).get("flip", False)),
                "video": {
                    "width": 960,
                    "height": 540,
                    "fps": 5,
                    "format": "MJPEG",
                    "auto_exposure": True,
                    "exposure_target": 110,
                    **camera_manifest.video_settings(camera),
                },
                "exposure_supported": True,
                "exposure_note": "Preferences are saved; the prepared picture stays the same.",
                "modes": {
                    "status": "unavailable",
                    "choices": [],
                    "diagnostics": ["Prepared pictures have a fixed size"],
                },
            }
            for n, camera in enumerate(scenes.CAMERAS)
        ],
    }


class Handler(server.H):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path.startswith(("/ingest/api/google_photos/", "/devices/camera-controls")):
            return self._json(
                {"error": "This needs a connected device or online service; the demo uses prepared recordings."}, 409
            )
        if path in server.APP_PATHS and (path == "/" or "text/html" in self.headers.get("Accept", "")):
            code, page = server.app_page()
            # Date-based defaults open the archived demo day; timers and performance clocks remain ordinary.
            script = f"""<script>window.__PACK_DEMO__=true;
window.__PACK_DEMO_REVIEW__={dataset._local(day(), *dataset.MEDIA_TIME) + 2};
const NativeDate=Date;
window.Date=class extends NativeDate {{constructor(...args){{super(...(args.length?args:[{int(clock() * 1000)}]));}} static now(){{return {int(clock() * 1000)};}}}};
</script></head>"""
            page = page.replace(b"</head>", script.encode())
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
            now = clock()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            entries = [e for e in transcript_lines.merged_tail(80) if e.get("ts_start", 0) <= now]
            payloads = {
                "hello": {"ts": now},
                "transcript": {"entries": entries, "reset": True},
                "scene": [
                    {"cam": c, "caption": text, "ts": now, "faces": []}
                    for c, text in zip(
                        scenes.CAMERAS,
                        dataset.CAPTIONS,
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
                        row["ts"] = clock()
                    payloads["vision"]["ts"] = clock()
                    for event, value in (
                        ("scene", payloads["scene"]),
                        ("vision", payloads["vision"]),
                        ("ping", {"ts": clock()}),
                    ):
                        self.wfile.write(f"event: {event}\ndata: {json.dumps(value)}\n\n".encode())
                    self.wfile.flush()
            except OSError:
                pass
            self.close_connection = True
            return
        if path == "/pets/sightings":
            from liberational.dashboard import pets as pet_view

            code, view = pet_view.sightings(self.state)
            if code == 200:
                start = json.loads((self.state / "demo" / "day.json").read_text())["start"]
                for rows in view["seen"].values():
                    for n, row in enumerate(rows):
                        row.update(
                            cam="front-room",
                            start=start + n * 3,
                            end=start + n * 3 + 5,
                            photo_ready=True,
                            thumb_ready=True,
                            video_status="ready",
                            video_thumb="/demo/media/front-room.jpg",
                        )
                view.update(
                    seen_pending={},
                    seen_unavailable={},
                    photos_pending=0,
                    photos_unavailable=0,
                    videos_pending=0,
                    videos_unavailable=0,
                )
            return self._json(view, code)
        if path.startswith("/faces/unknown/") and path.endswith("/sightings"):
            from liberational.people import face_photos

            group = int(path.split("/")[3])
            q = self._query()
            rows, total = face_photos.page_of(
                group, max(0, int(q.get("offset", "0"))), min(60, max(1, int(q.get("limit", "60")))), self.state
            )
            known = {
                s["id"]: s for g in json.loads((self.state / "demo" / "groups.json").read_text()) for s in g["shots"]
            }
            return self._json(
                {
                    "sightings": [known[r["id"]] for r in rows if r["id"] in known],
                    "total": total,
                    "shots_pending": 0,
                    "shots_unavailable": 0,
                }
            )
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
                        **here(dataset.PET),
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
                    {**p, "here_now": here(p["name"]) if p["name"] in dataset.HERE else None}
                    for p in view.get("people", [])
                ],
            }
            return self._json(view)
        if path == "/live/happening":
            now = clock()
            return self._json(
                {
                    "items": [
                        {
                            "id": f"demo-{n}",
                            "ts": dataset._local(day(), *at),
                            "text": text,
                            "kind": kind,
                            "people": people,
                            "cam": cam,
                            "camera_name": cam.replace("-", " ").title(),
                        }
                        for n, (at, text, kind, cam, people) in enumerate(dataset.HAPPENING)
                    ],
                    "updated": now,
                }
            )
        if path == "/review/summary":
            import datetime as dt

            review_day = dt.date.fromisoformat(json.loads((self.state / "demo" / "day.json").read_text())["day"])
            start = dataset._local(review_day, 0)
            return self._json(
                {
                    "start": start,
                    "end": start + 86400,
                    "level": 3600,
                    "status": "ready",
                    "blocks": [
                        {
                            "start": dataset._local(review_day, entry.at),
                            "end": dataset._local(review_day, entry.at) + 3600,
                            "text": entry.text,
                            "tags": [],
                            "evidence": {"status": "unverified", "sources": []},
                        }
                        for entry in dataset.DAY
                    ],
                }
            )
        if path in {"/device-links", "/device-presence", "/device-whereabouts", "/device-comings"}:
            return self._json(devices(path))
        if path == "/config":
            return self._json(
                {"ok": True, "control": True, "sections": cfg.schema(), "needs_setup": False, "path": "pack.toml"}
            )
        if path == "/faces/unknown/groups":
            from liberational.vision import faces

            waiting = {int(row["id"]) for row in faces.unknown_groups()}
            ids = json.loads((self.state / "demo" / "groups.json").read_text())
            groups = [
                {
                    "id": ids[n]["id"],
                    "label": f"Unnamed face {n + 1}",
                    "count": len(ids[n]["shots"]),
                    "last_seen": max(s["ts"] for s in ids[n]["shots"]),
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
        if path == "/services":
            return self._json({"ok": True, "control": False, "platform": "demo", "services": []})
        if path == "/devices/privacy":
            from liberational.dashboard import privacy

            return self._json(
                {
                    "ok": True,
                    "control": True,
                    "ready": False,
                    "devices": [dict(row, state="on") for row in privacy.devices()],
                }
            )
        if path == "/devices":
            return self._json(camera_settings())
        if path in ("/usage", "/audio/levels"):
            return self._json({"demo": True})
        return super().do_GET()

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/pack-ask/jobs":
            try:
                body = self.demo_body()
                if body.get("deep") is True:
                    # Deep research goes the real way (memory/ask.py): queued for the Inbox where research can run,
                    # else refused with why. The demo runs no research worker, so a queued report stays waiting.
                    from liberational.memory import ask

                    return ask._deep_http(self, body, ask.validate_day(body.get("day")))
                question = str(body.get("question", "")).strip()
                if not question or len(question) > 8192:
                    raise ValueError("Ask a question of 1 to 8192 characters")
            except ValueError as error:
                return self._json({"error": str(error)}, 400)
            now, key = clock(), uuid.uuid7().hex
            job = {
                "id": key,
                "poll_url": f"/pack-ask/jobs/{key}",
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
            from liberational.memory import ask_history

            history = ask_history.at(self.state)
            history.start(
                question,
                exchange_id=key,
                surface=body.get("surface", "live"),
                day=body.get("day"),
                scope="history",
                created_at=now,
            )
            history.finish(key, job["result"], started_at=now, finished_at=now, status="completed")
            JOBS[key] = job
            return self._json({"id": key, "status": "queued", "poll_url": job["poll_url"]}, 202)
        reason = unavailable(path)
        if reason:
            return self._json({"error": reason}, 409)
        if path == "/config":
            return self.save_config()
        if path in {"/devices/sensitivity", "/devices/auto_gain"}:
            return self.save_microphone()
        if path == "/devices/camera":
            return self.save_camera()
        try:
            return super().do_POST()
        except PermissionError as error:
            return self._json({"error": str(error)}, 409)

    def demo_body(self) -> dict[str, Any]:
        if not self._can_control() or self.headers.get_content_type() != "application/json":
            raise ValueError("Use local, same-origin JSON")
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://{self.headers.get('Host')}", f"https://{self.headers.get('Host')}"):
            raise ValueError("Use local, same-origin JSON")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 65536:
            raise ValueError("Invalid request size")
        body = json.loads(self.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError("Send a JSON object")
        return body

    def save_config(self) -> None:
        from liberational.dashboard import settings

        owner, pinned = cfg.owner_name(), cfg.get("owner.speaker_label").strip()
        before = cfg.local_config().read_text()
        try:
            values = self.demo_body().get("values")
            if not isinstance(values, dict):
                raise ValueError("Send settings in values")
            for key, value in values.items():
                if (
                    (key.startswith("paths.") or key in {"owner.voice_ref", "agent.openclaw_config"})
                    and value
                    and not cfg.owner_file(str(value)).resolve().is_relative_to(cfg.local_config().parent)
                ):
                    raise ValueError("Demo file paths must stay in its scratch directory")
            written, overridden = cfg.save(values)
            if settings.owner_moved(owner, pinned):
                code, reply = settings._carry_owner(owner, cfg.owner_name())
                if code != 200:
                    settings._put_back(cfg.local_config(), before)
                    return self._json(reply, code)
        except ValueError as error:
            return self._json({"error": str(error)}, 400)
        return self._json(
            {"ok": True, "written": written, "overridden": overridden, "restarted": [], "restarting": False}
        )

    def save_microphone(self) -> None:
        from liberational.audio import gain, sensitivity

        try:
            body = self.demo_body()
            if "default" in body:
                value = body["default"]
                if type(value) is not int or not 0 <= value <= 100:
                    raise ValueError("Sensitivity must be a whole number from 0 to 100")
                cfg.save({"audio.sensitivity": value})
            elif body.get("mic") != dataset.MIC:
                raise ValueError("Choose the demo microphone")
            elif self.path == "/devices/auto_gain":
                if type(body.get("on")) is not bool:
                    raise ValueError("Automatic gain must be true or false")
                gain.save(dataset.MIC, body["on"])
            else:
                value = body.get("sensitivity")
                if value is not None and (type(value) is not int or not 0 <= value <= 100):
                    raise ValueError("Sensitivity must be a whole number from 0 to 100, or null")
                sensitivity.save(dataset.MIC, value)
        except ValueError as error:
            return self._json({"error": str(error)}, 400)
        return self._json(camera_settings())

    def save_camera(self) -> None:
        from liberational.dashboard import cameras as camera_manifest

        try:
            body = self.demo_body()
            index = body.pop("index", None)
            if type(index) is not int or not 0 <= index < len(scenes.CAMERAS):
                raise ValueError("Choose a demo camera")
            if "on" in body:
                raise ValueError(
                    "These cameras replay prepared pictures; starting or stopping capture needs a real camera"
                )
            if set(body) not in ({"flip"}, {"auto_exposure"}, {"exposure_target"}):
                raise ValueError("Changing capture resolution needs a real camera; demo pictures have a fixed size")
            if any(type(v) is not bool for k, v in body.items() if k != "exposure_target"):
                raise ValueError("Camera switches must be true or false")
            if "exposure_target" in body and (
                type(body["exposure_target"]) is not int or not 60 <= body["exposure_target"] <= 180
            ):
                raise ValueError("Target brightness must be between 60 and 180")
            camera_manifest.update_video(scenes.CAMERAS[index], body)
        except ValueError as error:
            return self._json({"error": str(error)}, 400)
        return self._json(camera_settings())


def main() -> None:
    from liberational.dashboard import presence

    sys.addaudithook(offline)
    from liberational.memory import reports

    reports.at(cfg.path("paths.state")).now = clock
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
