"""The Ingest tab's API (dashboard/ingest.py), served by the real dashboard handler, with the pipeline's registrar a
fake (test_ingest_broker.py covers the broker)."""

import http.client
import http.server
import json
import shutil
import urllib.parse
from types import SimpleNamespace

import pytest
from PIL import Image

from liberational import config as cfg
from liberational.dashboard import ingest, server
from liberational.ingest.pipeline import Pipeline
from liberational.ingest.sources.google_photos import GooglePhotosSource
from liberational.ingest.store import Store

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


class FakeRegistrar:
    def __init__(self):
        self.calls = []

    def video(self, stream, path, start, fps):
        self.calls.append(("video", stream))
        return f"chunk-{len(self.calls)}"

    def audio(self, stream, path, start):
        self.calls.append(("audio", stream))
        return f"chunk-{len(self.calls)}"


@pytest.fixture
def registrar():
    return FakeRegistrar()


@pytest.fixture
def base(serve, tmp_path, monkeypatch, registrar):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    pipe = Pipeline(Store(tmp_path / "ingest.sqlite3"), tmp_path / "media", registrar)
    monkeypatch.setattr(ingest, "_pipeline", pipe)
    monkeypatch.setattr(ingest, "WORKER", ingest.Worker())
    return serve(server.H)


def request(base, method, path, body=b"", headers=None):
    u = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    conn.request(method, path, body=body, headers=headers or {})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, json.loads(data) if data else None


def post_json(base, path, body, **headers):
    return request(base, "POST", path, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})


def photo_bytes(tmp_path, color=(200, 10, 10)) -> bytes:
    p = tmp_path / f"p{color}.jpg"
    Image.new("RGB", (32, 24), color).save(p)
    return p.read_bytes()


def items(base):
    ingest.WORKER.queue.join()
    return request(base, "GET", "/ingest/api/items")[1]["items"]


def test_the_view_lists_every_source_and_never_sends_a_wyze_secret(base):
    with cfg.override({"ingest": {"wyze_password": "hunter2", "wyze_cameras": ["front-door"]}}):
        code, view = request(base, "GET", "/ingest/api/items")
    assert code == 200
    assert all(isinstance(s["blocked"], bool) for s in view["sources"])  # a sign-in that can't work yet waits
    later = [s["name"] for s in view["sources"] if s["later"]]
    assert later == ["icloud_photos", "allsound", "thinking_hat"]  # no way in yet: Coming later
    assert [s["name"] for s in view["sources"]] == [
        "link",
        "wyze",
        "google_photos",
        "icloud_photos",
        "allsound",
        "thinking_hat",
    ]
    assert view["wyze"] == {
        "cameras": ["front-door"],
        "rtsp_url": "rtsp://127.0.0.1:8554",
        "segment_s": 300,
        "email": False,
        "password": True,
        "api_id": False,
        "api_key": False,
    }
    assert "hunter2" not in json.dumps(view)
    assert (view["items"], view["control"]) == ([], True)


@needs_ffmpeg
def test_an_upload_is_ingested_in_the_background_and_a_second_copy_is_a_duplicate(base, tmp_path, registrar):
    body = photo_bytes(tmp_path)
    code, reply = request(base, "POST", "/ingest/api/upload?name=beach%20day.jpg", body)
    assert (code, reply["new"], reply["item"]["name"], reply["item"]["status"]) == (
        200,
        True,
        "beach day.jpg",
        "queued",
    )
    [item] = items(base)
    assert (item["status"], item["kind"], item["source"], item["time_from"]) == ("done", "photo", "upload", "upload")
    assert [o["stream"] for o in item["outputs"]] == ["ingest-upload.video_v1"]
    assert "path" not in item["outputs"][0]  # no paths on disk in the view
    code, again = request(base, "POST", "/ingest/api/upload?name=copy.jpg", body)
    assert (code, again["new"], again["item"]["id"]) == (200, False, item["id"])
    assert registrar.calls == [("video", "ingest-upload")]


def test_an_item_says_what_reading_it_found_once_its_pass_is_done(base, tmp_path):
    """For the Ingest tab's "what came of it": the analysis ledger's row for each item."""
    request(base, "POST", "/ingest/api/upload?name=beach.jpg", photo_bytes(tmp_path))
    [item] = items(base)
    assert item["analysis"] == {"state": "waiting"}  # brought in, not read yet
    ledger = tmp_path / ingest.ANALYZED
    row = {"frames": 1, "captions": 2, "lines": 0, "errors": [], "at": 1.0, "state": "done", "attempts": 1}
    ledger.write_text(json.dumps({str(item["id"]): row}))
    [item] = items(base)
    assert item["analysis"] == {"state": "done", "frames": 1, "captions": 2, "lines": 0}


@pytest.mark.parametrize(
    ("status", "outputs", "row", "want"),
    [
        pytest.param("converting", [], None, None, id="still-coming-in"),
        pytest.param("failed", [], None, None, id="never-brought-in"),
        pytest.param("done", ["o"], {"state": "retry", "next_at": 9.0}, {"state": "retrying"}, id="retrying"),
        pytest.param(
            "done",
            ["o"],
            {"state": "failed", "errors": ["one", "the last"]},
            {"state": "failed", "error": "the last"},
            id="failed",
        ),
        pytest.param(
            "done",
            ["o"],
            {"frames": 3, "captions": 1, "lines": 7},
            {"state": "done", "frames": 3, "captions": 1, "lines": 7},
            id="done-without-a-state",  # rows from before states were kept
        ),
    ],
)
def test_what_analysis_found_in_an_item(status, outputs, row, want):
    item = SimpleNamespace(id=4, status=status, outputs=outputs)
    assert ingest.analysis(item, {"4": row} if row else {}) == want


def test_an_unreadable_ledger_reads_as_nothing_analyzed_yet(base, tmp_path):
    request(base, "POST", "/ingest/api/upload?name=beach.jpg", photo_bytes(tmp_path))
    (tmp_path / ingest.ANALYZED).write_text("{not json")
    [item] = items(base)
    assert item["analysis"] == {"state": "waiting"}


def test_uploads_are_same_origin_and_not_empty(base, tmp_path):
    code, _ = request(
        base, "POST", "/ingest/api/upload?name=a.jpg", photo_bytes(tmp_path), {"Origin": "https://evil.example"}
    )
    assert code == 403
    code, _ = request(base, "POST", "/ingest/api/upload?name=a.jpg", b"")
    assert code == 413
    assert items(base) == []


def test_an_upload_over_the_limit_is_refused_before_it_is_read(base, tmp_path):
    with cfg.override({"ingest": {"max_mb": 0}}):
        code, reply = request(base, "POST", "/ingest/api/upload?name=a.jpg", photo_bytes(tmp_path))
    assert code == 413 and "up to 0 MB" in reply["error"]


class Files(http.server.BaseHTTPRequestHandler):
    jpeg = b""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(self.jpeg)))
        self.end_headers()
        self.wfile.write(self.jpeg)

    def log_message(self, format, *args):
        pass


@needs_ffmpeg
def test_a_link_is_fetched_and_ingested(base, serve, tmp_path):
    Files.jpeg = photo_bytes(tmp_path, (1, 2, 3))
    files = serve(Files)
    code, reply = post_json(base, "/ingest/api/link", {"url": f"{files}/IMG_2.jpg"})
    assert code == 202 and reply["job"]["status"] in ("queued", "running")
    [item] = items(base)
    assert (item["source"], item["name"], item["status"]) == ("link", "IMG_2.jpg", "done")
    [job] = request(base, "GET", "/ingest/api/items")[1]["jobs"]
    assert (job["status"], job["item"]) == ("done", item["id"])


@pytest.mark.parametrize("url", ["", "file:///etc/passwd", "ftp://example.com/a.jpg"])
def test_a_link_that_isnt_http_is_refused(base, url):
    code, reply = post_json(base, "/ingest/api/link", {"url": url})
    assert code == 400 and "http" in reply["error"]


def test_a_poll_of_a_stub_source_fails_its_job_with_why(base):
    assert post_json(base, "/ingest/api/sources/nope/poll", {})[0] == 404
    code, _ = post_json(base, "/ingest/api/sources/thinking_hat/poll", {})
    assert code == 202
    ingest.WORKER.queue.join()
    [job] = request(base, "GET", "/ingest/api/items")[1]["jobs"]
    assert job["status"] == "failed" and "Thinking Hat" in job["error"]


def test_an_offline_source_fails_its_job_without_disabling_local_ingest(base, monkeypatch):
    def unavailable(self, state):
        raise OSError("source is offline")

    monkeypatch.setattr(GooglePhotosSource, "list_new", unavailable)
    assert post_json(base, "/ingest/api/sources/google_photos/poll", {})[0] == 202
    ingest.WORKER.queue.join()
    code, view = request(base, "GET", "/ingest/api/items")
    assert code == 200
    [job] = view["jobs"]
    assert job["status"] == "failed" and "offline" in job["error"]
    assert view["items"] == []


def test_the_google_sign_in_window_opens_only_for_this_machine(base, monkeypatch):
    opened = []
    monkeypatch.setattr(GooglePhotosSource, "login", lambda self: opened.append(self.profile))
    code, _ = post_json(base, "/ingest/api/google_photos/login", {}, **{"X-Forwarded-For": "100.64.0.9"})
    assert code == 403 and opened == []
    code, _ = post_json(base, "/ingest/api/google_photos/login", {})
    assert code == 202 and len(opened) == 1
    assert opened[0] == cfg.path("paths.ingest") / "google_photos" / "profile"


def test_google_photos_listing_before_sign_in_says_so(base):
    code, reply = request(base, "GET", "/ingest/api/google_photos/albums")
    assert code == 409 and "signed in" in reply["error"]


@pytest.mark.parametrize("listing", ["albums", "photos"])
def test_google_photos_connection_failures_return_json_and_leave_uploads_available(base, monkeypatch, listing):
    def unavailable(self, *args, **kwargs):
        raise OSError("connection failed: private-session-value")

    monkeypatch.setattr(GooglePhotosSource, listing, unavailable)
    code, reply = request(base, "GET", f"/ingest/api/google_photos/{listing}")
    assert code == 503
    assert "Google Photos" in reply["error"]
    assert "private-session-value" not in reply["error"]
    assert request(base, "GET", "/ingest/api/items")[0] == 200


@pytest.mark.parametrize(
    "body",
    [{}, {"items": []}, {"items": [{"key": "a", "name": "a.jpg"}]}, {"items": "all"}],
)
def test_a_google_import_needs_items_with_their_urls(base, body):
    code, _ = post_json(base, "/ingest/api/google_photos/import", body)
    assert code == 400


def test_a_retry_of_an_item_that_doesnt_exist_is_404(base):
    assert post_json(base, "/ingest/api/retry", {"id": 99})[0] == 404


def test_writes_take_json_only(base):
    code, _ = request(base, "POST", "/ingest/api/link", b"url=http://x", {"Content-Type": "text/plain"})
    assert code == 403


@pytest.fixture
def deletion_backend(monkeypatch):
    from types import SimpleNamespace

    from liberational.ingest import analyze, broker

    removed = []
    monkeypatch.setattr(analyze, "Ledger", lambda path: SimpleNamespace(remove=removed.append))
    return removed, broker


def test_delete_is_same_origin_and_missing_or_repeated_ids_are_safe(base, deletion_backend):
    assert post_json(base, "/ingest/api/delete", {"id": 1}, Origin="https://evil.example")[0] == 403
    assert post_json(base, "/ingest/api/delete", {"id": True})[0] == 400
    assert post_json(base, "/ingest/api/delete", {"id": "1"})[0] == 400
    assert post_json(base, "/ingest/api/delete", {"id": 1}) == (
        200,
        {"ok": True, "deleted": False, "pending": False},
    )


def test_delete_hides_item_and_removes_its_original_and_ledger(base, tmp_path, deletion_backend):
    removed, _ = deletion_backend
    p = ingest.pipeline()
    original = p.root / "originals" / "kept.wav"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"bytes")
    item, _ = p.store.add(
        sha256="abc", source="upload", name="kept.wav", original=str(original), stream="ingest-upload"
    )
    assert post_json(base, "/ingest/api/delete", {"id": item.id}) == (
        200,
        {"ok": True, "deleted": True, "pending": False},
    )
    assert not original.exists()
    assert p.store.get(item.id) is None
    assert removed == [item.id]
    assert request(base, "GET", "/ingest/api/items")[1]["items"] == []
    assert post_json(base, "/ingest/api/retry", {"id": item.id})[0] == 404
    assert post_json(base, "/ingest/api/delete", {"id": item.id})[1]["pending"] is False


def test_delete_failure_keeps_a_retryable_notice_instead_of_claiming_completion(
    base, tmp_path, deletion_backend, monkeypatch
):
    _, broker = deletion_backend
    p = ingest.pipeline()
    original = p.root / "originals" / "pending.wav"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"bytes")
    item, _ = p.store.add(
        sha256="abc", source="upload", name="pending.wav", original=str(original), stream="ingest-upload"
    )

    def unavailable(self, item):
        raise OSError("test broker unavailable")

    monkeypatch.setattr(broker.BrokerRegistrar, "purge_item", unavailable)
    code, reply = post_json(base, "/ingest/api/delete", {"id": item.id})
    assert code == 503
    assert "Retry" in reply["error"]
    view = request(base, "GET", "/ingest/api/items")[1]
    assert view["items"] == []
    assert view["deletions"] == [
        {
            "id": item.id,
            "name": "pending.wav",
            "detail": "Cleanup failed. Retry to finish deleting this item.",
            "blocked": False,
        }
    ]
    assert original.exists()


def test_cleanup_poll_only_advances_an_existing_deletion_and_finishes_after_ack(
    base, tmp_path, deletion_backend, monkeypatch
):
    _, broker = deletion_backend
    p = ingest.pipeline()
    original = p.root / "originals" / "waiting.wav"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"waiting")
    item, _ = p.store.add(
        sha256="waiting", source="upload", name="waiting.wav", original=str(original), stream="ingest-upload"
    )
    assert post_json(base, "/ingest/api/delete", {"id": item.id, "poll": True})[0] == 404
    assert not p.store.removed(item.id)
    state = {"acked": False, "purges": 0}

    def purge(self, item):
        state["purges"] += 1
        return "receipt"

    monkeypatch.setattr(broker.BrokerRegistrar, "purge_item", purge)
    monkeypatch.setattr(broker.BrokerRegistrar, "purge_finished", lambda self, purge_id: state["acked"])
    assert post_json(base, "/ingest/api/delete", {"id": item.id})[1]["pending"] is True
    assert post_json(base, "/ingest/api/delete", {"id": item.id, "poll": True})[1]["pending"] is True
    state["acked"] = True
    assert post_json(base, "/ingest/api/delete", {"id": item.id, "poll": True}) == (
        200,
        {"ok": True, "deleted": True, "pending": False},
    )
    assert state["purges"] == 1
    assert not original.exists()
    assert p.store.deletions() == []


def test_unsafe_cleanup_path_is_a_permanent_actionable_notice(base, tmp_path, deletion_backend):
    p = ingest.pipeline()
    outside = tmp_path / "unrelated.wav"
    outside.write_bytes(b"must survive")
    item, _ = p.store.add(
        sha256="outside", source="upload", name="unrelated.wav", original=str(outside), stream="ingest-upload"
    )
    code, reply = post_json(base, "/ingest/api/delete", {"id": item.id})
    assert code == 409
    assert reply["permanent"] is True
    assert "outside" in reply["error"]
    notices = request(base, "GET", "/ingest/api/items")[1]["deletions"]
    assert notices[0]["blocked"] is True
    assert outside.read_bytes() == b"must survive"
