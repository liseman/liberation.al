"""Object memory on Continuum (memory/objects.py, see/object_worker.py, dashboard/objects.py): real SIFT on a generated
texture, the objects as events in an in-memory log, the owner's crops on the in-memory vision log, and the worker's
matches through their segment codec into the same fake store."""

import base64
import io
import json
import time
from dataclasses import dataclass
from email.message import Message
from typing import Any

import cv2
import numpy as np
import pytest

from liberational import segment_log
from liberational.dashboard import objects as page
from liberational.domain.kernel import Event
from liberational.memory import objects
from liberational.people import registry
from liberational.see.object_worker import ObjectWorker, templates
from liberational.vision import segmentation

BBOX = [80 / 480, 60 / 360, 160 / 480, 160 / 360]  # the textured object in a frame, [x, y, w, h]


def texture() -> np.ndarray:
    rng = np.random.default_rng(123)
    t = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    for _ in range(25):
        cv2.circle(t, tuple(rng.integers(8, 152, 2).tolist()), int(rng.integers(3, 10)), (255, 255, 255), 2)
    return t


def frame(offset: int = 0, blank: bool = False) -> np.ndarray:
    image = np.full((360, 480, 3), 50, np.uint8)
    if not blank:
        image[60:220, 80 + offset : 240 + offset] = texture()
    return image


@dataclass
class Stored:
    event: Event
    position: int


class FakeLog:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def append(self, events: list[Event]) -> list[int]:
        self.events += events
        return list(range(len(self.events) - len(events) + 1, len(self.events) + 1))


class FakeRegistry:
    """The registry's events and log, for the objects stream only."""

    def __init__(self) -> None:
        self.log = FakeLog()

    def events(self, types: Any, start: float, end: float, streams: Any = ()) -> list[Stored]:
        wanted = {str(t) for t in types}
        return [
            Stored(e, i)
            for i, e in enumerate(self.log.events, 1)
            if str(e.type) in wanted and e.stream in streams and start <= e.occurred_at < end
        ]


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> FakeRegistry:
    fake = FakeRegistry()
    monkeypatch.setattr(registry, "at", lambda state=None: fake)
    monkeypatch.setattr(objects, "camera_names", lambda: ["cam0", "cam1"])
    return fake


@pytest.fixture
def store(tmp_path, log) -> objects.Store:
    return objects.Store(tmp_path)


def snapshot(image: np.ndarray | None = None, cam: str = "cam0", at: float | None = None) -> objects.Snapshot:
    data = cv2.imencode(".jpg", frame() if image is None else image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    return objects.Snapshots().add(cam, time.time() - 1 if at is None else at, data, "frame-9")


def owner_crops(state) -> list[dict]:
    """The owner's crops, as the knowledge store keeps them."""
    from liberational import knowledge
    from liberational.people import registry

    return [row.record.value for row in knowledge.Store(registry.knowledge_path(state)).query("object.sighting")]


def test_a_remembered_object_is_listed_found_by_alias_and_last_seen_where_it_was_cropped(store, vision, tmp_path):
    made = store.remember(name="wallet", snapshot=snapshot(), bbox=BBOX, aliases="purse, billfold", note="brown")
    assert (made["name"], made["aliases"], made["mode"]) == ("wallet", ["purse", "billfold"], "visual")
    assert made["last_seen"]["kind"] == "manual_label" and made["last_seen"]["camera"] == "cam0"
    (sighting,) = owner_crops(tmp_path)
    assert sighting["frame_id"] == "frame-9"  # the frame it was cropped from
    assert objects.decode(store.evidence(made["last_seen"]["evidence"])) is not None
    template = objects.decode(store.evidence(made["template"]))
    assert template is not None and abs(template.shape[0] - 160) <= 1 and abs(template.shape[1] - 160) <= 1
    answer = store.query("where did I leave my billfold?")
    assert answer["status"] == "last_seen" and "wallet" in answer["answer"]
    assert store.query("where is my umbrella?")["status"] == "unknown"


def test_a_textureless_crop_is_remembered_for_manual_use_only(store, vision):
    assert store.remember(name="mug", snapshot=snapshot(frame(blank=True)), bbox=BBOX)["mode"] == "manual_only"


@pytest.mark.parametrize(
    ("name", "bbox", "why"),
    [
        pytest.param("the man", BBOX, "people", id="a-person"),
        pytest.param("wallet", [0.9, 0.9, 0.2, 0.2], "outside", id="box-off-the-frame"),
        pytest.param("wallet", [0.1, 0.1, 0.01, 0.01], "24px", id="crop-too-small"),
        pytest.param("", BBOX, "name required", id="no-name"),
    ],
)
def test_a_bad_remember_is_refused(store, vision, name, bbox, why):
    with pytest.raises(ValueError, match=why):
        store.remember(name=name, snapshot=snapshot(), bbox=bbox)


def test_a_name_or_alias_is_remembered_once_and_a_crop_over_a_face_is_refused(store, vision, live_face_boxes):
    store.remember(name="wallet", snapshot=snapshot(), bbox=BBOX, aliases="purse")
    with pytest.raises(ValueError, match="already"):
        store.remember(name="Purse", snapshot=snapshot(), bbox=BBOX)
    at = time.time() - 1
    live_face_boxes.append(
        {"cam": "cam0", "start_ns": int(at * 1e9), "faces": [{"bbox": [0.2, 0.2, 0.4, 0.4], "name": "Sam"}]}
    )
    with pytest.raises(ValueError, match="face"):
        store.remember(name="keys", snapshot=snapshot(at=at), bbox=BBOX)


def test_a_forgotten_object_is_gone(store, vision):
    made = store.remember(name="wallet", snapshot=snapshot(), bbox=BBOX)
    store.forget(made["id"])
    assert store.list_objects() == []
    assert store.query("where is my wallet")["status"] == "unknown"


def test_a_renamed_object_keeps_its_template_other_names_and_sightings(store, vision):
    made = store.remember(name="wallet", snapshot=snapshot(), bbox=BBOX, aliases="purse")
    renamed = store.rename(made["id"], "  card holder  ")
    assert renamed["name"] == "card holder"
    assert renamed["aliases"] == ["purse"]  # its other names are left alone
    assert renamed["last_seen"]["captured_at"] == made["last_seen"]["captured_at"]
    assert objects.decode(store.evidence(renamed["template"])) is not None
    assert store.query("where is my card holder?")["status"] == "last_seen"
    assert store.query("where is my purse?")["status"] == "last_seen"
    assert store.query("where is my wallet?")["status"] == "unknown"  # the name it no longer goes by


@pytest.mark.parametrize("taken", ["Keys", "key ring"])
def test_a_rename_to_a_name_another_object_goes_by_is_refused(store, vision, taken):
    wallet = store.remember(name="wallet", snapshot=snapshot(), bbox=BBOX)
    store.remember(name="keys", snapshot=snapshot(), bbox=BBOX, aliases="key ring")
    with pytest.raises(ValueError, match="already"):
        store.rename(wallet["id"], taken)
    assert [o["name"] for o in store.list_objects()] == ["wallet", "keys"]


@pytest.mark.parametrize(
    ("name", "why"),
    [pytest.param("   ", "required", id="no-name"), pytest.param("the man", "people", id="a-person")],
)
def test_a_rename_to_a_name_the_store_would_not_remember_is_refused(store, vision, name, why):
    made = store.remember(name="wallet", snapshot=snapshot(), bbox=BBOX)
    with pytest.raises(ValueError, match=why):
        store.rename(made["id"], name)
    assert [o["name"] for o in store.list_objects()] == ["wallet"]


def test_a_rename_of_an_object_that_is_not_remembered_is_refused(store, vision):
    with pytest.raises(ValueError, match="no such object"):
        store.rename("not-an-object", "purse")


def test_a_removed_object_is_put_back_from_the_log_as_it_was(store, vision):
    made = store.remember(name="wallet", snapshot=snapshot(), bbox=BBOX)
    store.rename(made["id"], "card holder")
    store.forget(made["id"])
    assert store.list_objects() == []
    again = store.restore(made["id"])
    assert (again["id"], again["name"], again["template"]) == (made["id"], "card holder", made["template"])
    assert again["last_seen"]["captured_at"] == made["last_seen"]["captured_at"]
    assert objects.decode(store.evidence(again["template"])) is not None
    assert store.query("where is my card holder?")["status"] == "last_seen"
    with pytest.raises(ValueError, match="still remembered"):
        store.restore(made["id"])
    with pytest.raises(ValueError, match="no such object"):
        store.restore("never-remembered")


class FakeMatches:
    """Stands in for the segment writers (the native store's boundary): each record goes through its stream's typed
    columns and back, into the fake log the readers read, and counts as durable only once synced."""

    def __init__(self, vision) -> None:
        self.vision, self.pending = vision, []

    def append(self, stream, cam, records):
        spec = segment_log.STREAMS[stream]
        self.pending += [(stream, spec.to_record(spec.to_row(r), cam)) for r in records]

    def sync(self):
        for stream, record in self.pending:
            self.vision.records.setdefault(stream, []).append(record)
        self.pending.clear()
        return {}

    def close(self):
        pass


def worker(store: objects.Store, vision) -> ObjectWorker:
    matches = FakeMatches(vision)
    return ObjectWorker(client=None, cameras=[], store=store, matches=matches)  # ty: ignore[invalid-argument-type]


def visual(vision) -> list[dict[str, Any]]:
    return vision.records.get(segment_log.OBJECT_MATCHES, [])


def test_the_worker_commits_a_match_confirmed_in_a_second_frame_once_until_it_moves(store, vision):
    made = store.remember(name="wallet", snapshot=snapshot(at=time.time() - 300), bbox=BBOX)
    w = worker(store, vision)
    t = time.time() - 100  # sightings in the past, so "last seen" (up to now) has them
    assert w.observe("cam0", frame(), t, "f1", 1) == 0  # a first match waits for its confirmation
    assert w.observe("cam0", frame(), t + 1, "f1", 1) == 0  # the same frame again confirms nothing
    assert w.observe("cam0", frame(), t + 5, "f2", 2) == 1
    (seen,) = visual(vision)
    assert (seen["object_id"], seen["name"], seen["cam"], seen["frame_id"]) == (made["id"], "wallet", "cam0", "f2")
    assert seen["kind"] == "visual_match"
    assert owner_crops(store.state)[-1]["kind"] == "manual_label"  # only the owner's crop is taught state
    assert objects.consistent(seen["bbox"], BBOX)
    assert objects.decode(base64.b64decode(seen["evidence"])) is not None
    assert w.observe("cam0", frame(), t + 10, "f3", 3) == 0  # still there: not again within a minute
    assert w.observe("cam0", frame(offset=200), t + 15, "f4", 4) == 0  # moved: waits to be confirmed there
    assert w.observe("cam0", frame(offset=200), t + 20, "f5", 5) == 1
    assert w.observe("cam0", frame(blank=True), t + 25, "f6", 6) == 0
    assert store.list_objects()[0]["last_seen"]["kind"] == "visual_match"


def test_the_worker_reads_the_objects_again_and_skips_manual_only_ones(store, vision):
    w = worker(store, vision)
    t = time.time()
    assert w.observe("cam0", frame(), t, "f1", 1) == 0  # nothing remembered yet
    store.remember(name="mug", snapshot=snapshot(frame(blank=True)), bbox=BBOX)
    store.remember(name="wallet", snapshot=snapshot(), bbox=BBOX)
    w.read_at = -1e18  # as REFRESH_S later
    w.observe("cam0", frame(), t + 1, "f2", 2)
    assert w.observe("cam0", frame(), t + 2, "f3", 3) == 1
    assert [r["name"] for r in visual(vision)] == ["wallet"]


def test_the_page_captures_a_view_remembers_from_it_and_keeps_a_spoken_draft(store, vision, monkeypatch, sam_mask):
    monkeypatch.setattr(page, "_store", store)
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    vision.frame_now("cam0", frame(), time.time() - 1, "frame-7")
    snap = page.capture({"camera": "cam0", "draft": True, "name": "my wallet"})
    assert snap["camera"] == "cam0" and not snap["stale"]
    assert objects.decode(page.evidence(snap["evidence"])) is not None
    held = page.draft()
    assert held is not None and (held["name"], held["snapshot"]["token"]) == ("my wallet", snap["token"])
    selected = page.segment({"snapshot": snap["token"], "bbox": BBOX})
    made = page.enroll(
        {
            "name": "wallet",
            "camera": "cam0",
            "snapshot": snap["token"],
            "selection": selected["selection"],
            "bbox": BBOX,
        }
    )
    assert made["name"] == "wallet"
    status = page.status()
    assert status["objects"] == 1
    assert [c["camera"] for c in status["cameras"] if c["available"]] == ["cam0"]
    with pytest.raises(ValueError, match="one of"):
        page.capture({"camera": "cam9"})
    with pytest.raises(FileNotFoundError):
        page.enroll({"name": "keys", "camera": "cam0", "snapshot": "gone", "bbox": BBOX})


@pytest.mark.parametrize("path", ["/objects", "/objects/"])
def test_objects_page_uses_the_shared_dashboard_entry(dashboard_server, monkeypatch, path):

    monkeypatch.setattr(dashboard_server, "app_page", lambda: (200, b"configured dashboard shell"))

    class Handler:
        def __init__(self):
            self.path = path
            self.sent = None

        def _send(self, code, kind, body):
            self.sent = (code, kind, body)

    handler = Handler()
    assert page.handle(handler, "GET")
    assert handler.sent == (200, "text/html; charset=utf-8", b"configured dashboard shell")


@pytest.fixture
def sam_mask(monkeypatch):
    mask = np.zeros((360, 480), dtype=bool)
    mask[60:220, 80:240] = True
    monkeypatch.setattr(page, "_selections", {})
    monkeypatch.setattr(segmentation, "segment", lambda image, **prompt: segmentation.Result(mask, 0.9, "fixture/sam2"))
    return mask


def test_sam_selection_is_snapshot_bound_and_no_box_only_save_is_allowed(store, monkeypatch, sam_mask):
    monkeypatch.setattr(page, "_store", store)
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    first = page._snapshots.add("cam0", time.time(), snapshot().image)
    second = page._snapshots.add("cam0", time.time(), snapshot().image)
    with pytest.raises(ValueError, match="outline"):
        page.enroll({"name": "wallet", "camera": "cam0", "snapshot": first.token, "bbox": BBOX})
    chosen = page.segment({"snapshot": first.token, "bbox": BBOX})
    assert chosen["bbox"] == pytest.approx(BBOX)
    assert chosen["model"] == "fixture/sam2"
    assert page.evidence(chosen["mask"]).startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="another snapshot"):
        page.enroll({"name": "wallet", "camera": "cam0", "snapshot": second.token, "selection": chosen["selection"]})
    assert store.objects() == []


def test_sam_mask_is_saved_and_limits_object_matching_features(store, vision):
    mask = np.zeros((360, 480), dtype=np.uint8)
    cv2.circle(mask, (160, 140), 60, 1, -1)
    made = store.remember(
        name="wallet",
        snapshot=snapshot(),
        bbox=BBOX,
        mask=mask.astype(bool),
        segmentation_model="fixture/sam2",
        segmentation_score=0.9,
    )
    (row,) = store.objects()
    assert row["segmentation"]["model"] == "fixture/sam2"
    image = objects.decode(store.evidence(made["template"]))
    assert image is not None
    saved_mask = objects.template_mask(row, image)
    assert saved_mask is not None and saved_mask[0, 0] == 0 and saved_mask[80, 80] == 255
    assert image[0, 0].max() < 10
    (template,) = templates([row])
    keypoints = template.features[0]
    assert len(keypoints) >= 18
    assert all(saved_mask[round(k.pt[1]), round(k.pt[0])] != 0 for k in keypoints)
    assert len(base64.b64decode(row["template"])) + len(base64.b64decode(row["segmentation"]["mask"])) <= 40_000


def test_a_chosen_box_runs_sam2_once_and_its_selection_is_bound_to_the_snapshot(monkeypatch):
    from liberational.dashboard import objects as page
    from liberational.vision import segmentation

    snap = page._snapshots.add(
        "cam0", time.time(), objects.jpeg(np.zeros((100, 120, 3), dtype=np.uint8), max_bytes=200_000)
    )
    mask = np.zeros((100, 120), dtype=bool)
    mask[30:60, 30:60] = True
    calls = []
    monkeypatch.setattr(
        segmentation,
        "segment",
        lambda image, **prompt: calls.append(prompt) or segmentation.Result(mask, 0.9, "fixture/sam2"),
    )
    chosen = page.segment({"snapshot": snap.token, "bbox": [0.25, 0.3, 0.25, 0.3]})
    assert calls == [{"bbox": [0.25, 0.3, 0.25, 0.3], "point": None, "job": None}]
    assert chosen["bbox"] == pytest.approx([30 / 120, 0.3, 30 / 120, 0.3])
    assert np.array_equal(page._selection(chosen["selection"], snap.token).result.mask, mask)
    with pytest.raises(ValueError, match="another snapshot"):
        page._selection(chosen["selection"], "different")
    with pytest.raises(FileNotFoundError):
        page.segment({"snapshot": "missing", "bbox": [0, 0, 1, 1]})


def test_there_is_no_automatic_proposal_endpoint(monkeypatch):
    # Segmentation only ever runs for a box or point the owner chose.
    body = json.dumps({"snapshot": "any"}).encode()

    class Handler:
        path = "/objects/api/proposals"
        headers = Message()
        rfile = io.BytesIO(body)
        sent: Any = None

        def _json(self, data, status=200):
            self.sent = (status, data)

    Handler.headers["Content-Type"] = "application/json"
    Handler.headers["Content-Length"] = str(len(body))
    ran = []
    monkeypatch.setattr(segmentation, "_run", lambda *args: ran.append(args))
    handler = Handler()
    assert page.handle(handler, "POST")
    assert handler.sent == (404, {"error": "not found"})
    assert ran == []


def post_to(path: str, payload: dict[str, Any], origin: str | None = None) -> tuple[int, Any]:
    """A POST to an /objects route through the page's own handler, as the dashboard's H calls it."""
    body = json.dumps(payload).encode()

    class Handler:
        sent: Any = None

        def __init__(self) -> None:
            self.path = path
            self.headers = Message()
            self.headers["Content-Type"] = "application/json"
            self.headers["Content-Length"] = str(len(body))
            self.headers["Host"] = "127.0.0.1:8765"
            if origin:
                self.headers["Origin"] = origin
            self.rfile = io.BytesIO(body)

        def _json(self, data: Any, status: int = 200) -> None:
            self.sent = (status, data)

    handler = Handler()
    assert page.handle(handler, "POST")
    return handler.sent


def test_the_row_routes_rename_remove_and_put_back_an_object(store, vision, monkeypatch):
    monkeypatch.setattr(page, "_store", store)
    made = store.remember(name="wallet", snapshot=snapshot(), bbox=BBOX)
    assert post_to("/objects/api/rename", {"id": made["id"], "name": "card holder"}) == (
        200,
        {"object": {**made, "name": "card holder"}},
    )
    assert post_to("/objects/api/forget", {"id": made["id"]}) == (200, {"ok": True, "id": made["id"]})
    assert store.list_objects() == []
    assert post_to("/objects/api/restore", {"id": made["id"]}) == (200, {"object": {**made, "name": "card holder"}})
    assert store.list_objects()[0]["name"] == "card holder"


def test_a_refused_row_route_says_the_stores_reason_and_a_cross_origin_one_is_not_run(store, vision, monkeypatch):
    monkeypatch.setattr(page, "_store", store)
    made = store.remember(name="wallet", snapshot=snapshot(), bbox=BBOX)
    store.remember(name="keys", snapshot=snapshot(), bbox=BBOX)
    assert post_to("/objects/api/rename", {"id": made["id"], "name": "keys"}) == (
        400,
        {"error": "name or alias already enrolled"},
    )
    assert post_to("/objects/api/forget", {"id": "not-an-object"}) == (400, {"error": "no such object"})
    assert post_to("/objects/api/forget", {"id": made["id"]}, origin="http://example.test") == (
        403,
        {"error": "Cross-origin write rejected."},
    )
    assert [o["name"] for o in store.list_objects()] == ["wallet", "keys"]


def test_cancel_endpoint_prevents_a_delayed_segment_request_with_conflict_response():
    class Handler:
        def __init__(self, path, payload):
            body = json.dumps(payload).encode()
            self.path = path
            self.headers = Message()
            self.headers["Content-Type"] = "application/json"
            self.headers["Content-Length"] = str(len(body))
            self.rfile = io.BytesIO(body)
            self.sent = None

        def _json(self, data, status=200):
            self.sent = (status, data)

    cancel = Handler("/objects/api/cancel", {"job": "cancel-endpoint-delayed"})
    assert page.handle(cancel, "POST")
    assert cancel.sent == (200, {"ok": True, "cancelled": False})
    snap = page._snapshots.add(
        "cam0", time.time(), objects.jpeg(np.zeros((60, 80, 3), dtype=np.uint8), max_bytes=200_000)
    )
    delayed = Handler(
        "/objects/api/segment", {"snapshot": snap.token, "bbox": [0, 0, 1, 1], "job": "cancel-endpoint-delayed"}
    )
    assert page.handle(delayed, "POST")
    assert delayed.sent == (409, {"error": "The object selection was cancelled."})


def test_cancel_endpoint_is_served_through_the_dashboard_post_router(dashboard_server, monkeypatch):
    import http.client
    import http.server
    import json
    import threading

    from liberational.vision import segmentation

    calls = []
    monkeypatch.setattr(segmentation, "cancel", lambda job: calls.append(job) or False)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), dashboard_server.H)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=3)
    try:
        connection.request("POST", "/objects/api/cancel", '{"job":"fixture-job"}', {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"ok": True, "cancelled": False}
        assert calls == ["fixture-job"]
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_live_crop_import_preserves_pixels_without_overwriting_spoken_draft(store, monkeypatch):
    import base64

    monkeypatch.setattr(page, "_store", store)
    snapshots = objects.Snapshots()
    monkeypatch.setattr(page, "_snapshots", snapshots)
    spoken = snapshot()
    snapshots.draft = (spoken, "spoken object", time.time())
    raw = cv2.imencode(".jpg", np.full((40, 60, 3), [20, 80, 140], np.uint8))[1].tobytes()
    at = time.time() - 1
    result = page.import_view(
        {"camera": "cam0", "image": "data:image/jpeg;base64," + base64.b64encode(raw).decode(), "captured_at": at}
    )
    got = objects.decode(page.evidence(result["evidence"]))
    assert got is not None and got.shape == (40, 60, 3)
    assert tuple(got[20, 30]) == pytest.approx((20, 80, 140), abs=3)
    assert result["camera"] == "cam0" and result["captured_at"] == at
    assert snapshots.draft == (spoken, "spoken object", snapshots.draft[2])
    assert store.list_objects() == []


@pytest.mark.parametrize(
    "change",
    [
        {"camera": "unknown"},
        {"captured_at": 0},
        {"image": "data:image/jpeg;base64,???"},
        {"image": "data:image/png;base64,AAAA"},
    ],
)
def test_live_crop_import_rejects_invalid_input(change, log):
    import base64

    raw = cv2.imencode(".jpg", np.zeros((30, 40, 3), np.uint8))[1].tobytes()
    data = {
        "camera": "cam0",
        "captured_at": time.time(),
        "image": "data:image/jpeg;base64," + base64.b64encode(raw).decode(),
    }
    with pytest.raises(ValueError):
        page.import_view(data | change)


@pytest.mark.parametrize(("fields", "size"), [({}, 40000), ({"kind": "full_frame"}, 3_000_000)])
def test_import_route_accepts_bounded_image_envelopes_over_normal_json_limit(
    dashboard_server, monkeypatch, fields, size
):
    import http.client
    import http.server
    import json
    import threading

    calls = []
    monkeypatch.setattr(page, "import_view", lambda data: calls.append(data) or {"token": "live-crop"})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), dashboard_server.H)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=3)
    data = {"camera": "cam0", "image": "x" * size} | fields
    try:
        connection.request("POST", "/objects/api/import", json.dumps(data), {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"token": "live-crop"}
        assert calls == [data]
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_live_crop_can_be_remembered_without_sam_and_does_not_claim_segmentation(store, monkeypatch):
    monkeypatch.setattr(page, "_store", store)
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    raw = snapshot().image
    imported = page.import_view(
        {
            "camera": "cam0",
            "captured_at": time.time(),
            "image": "data:image/jpeg;base64," + base64.b64encode(raw).decode(),
        }
    )
    assert imported["source"] == "live_crop"
    made = page.enroll(
        {"name": "wallet", "camera": "cam0", "snapshot": imported["token"], "full_crop": True, "bbox": BBOX}
    )
    row = store.objects()[0]
    assert made["name"] == "wallet"
    assert not row.get("segmentation")
    template = objects.decode(base64.b64decode(row["template"]))
    original = objects.decode(raw)
    assert original is not None and template is not None
    assert template.shape[:2] == original.shape[:2]


def test_full_crop_is_bound_to_live_import_and_cannot_bypass_selection_validation(store, monkeypatch, sam_mask):
    monkeypatch.setattr(page, "_store", store)
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    ordinary = page._snapshots.add("cam0", time.time(), snapshot().image)
    with pytest.raises(ValueError, match="outline"):
        page.enroll({"name": "wallet", "camera": "cam0", "snapshot": ordinary.token, "full_crop": True})
    imported = page.import_view(
        {
            "camera": "cam0",
            "captured_at": time.time(),
            "image": "data:image/jpeg;base64," + base64.b64encode(snapshot().image).decode(),
        }
    )
    with pytest.raises(ValueError, match="camera mismatch"):
        page.enroll({"name": "wallet", "camera": "cam1", "snapshot": imported["token"], "full_crop": True})
    chosen = page.segment({"snapshot": ordinary.token, "bbox": BBOX})
    with pytest.raises(ValueError, match="another snapshot"):
        page.enroll(
            {
                "name": "wallet",
                "camera": "cam0",
                "snapshot": imported["token"],
                "selection": chosen["selection"],
                "full_crop": True,
            }
        )
    with pytest.raises(FileNotFoundError, match="selection"):
        page.enroll(
            {
                "name": "wallet",
                "camera": "cam0",
                "snapshot": imported["token"],
                "selection": "forged",
                "full_crop": True,
            }
        )
    held = page._snapshots.get(imported["token"])
    assert held is not None
    held.made -= objects.SNAPSHOT_TTL + 1
    with pytest.raises(FileNotFoundError, match="snapshot"):
        page.enroll({"name": "wallet", "camera": "cam0", "snapshot": imported["token"], "full_crop": True})
    assert store.objects() == []


def test_live_crop_selected_mask_wins_over_full_crop_opt_in(store, monkeypatch, sam_mask):
    monkeypatch.setattr(page, "_store", store)
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    imported = page.import_view(
        {
            "camera": "cam0",
            "captured_at": time.time(),
            "image": "data:image/jpeg;base64," + base64.b64encode(snapshot().image).decode(),
        }
    )
    selected = page.segment({"snapshot": imported["token"], "bbox": BBOX})
    page.enroll(
        {
            "name": "wallet",
            "camera": "cam0",
            "snapshot": imported["token"],
            "selection": selected["selection"],
            "full_crop": True,
        }
    )
    row = store.objects()[0]
    assert row["segmentation"]["model"] == "fixture/sam2"
    assert row["bbox"] == pytest.approx(BBOX)


def test_review_full_frame_import_keeps_native_jpeg_history_and_segments_the_whole_image(store, monkeypatch):
    monkeypatch.setattr(page, "_store", store)
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    raw = cv2.imencode(".jpg", np.full((900, 1600, 3), [20, 80, 140], np.uint8))[1].tobytes()
    captured_at = 1_700_000_000.125
    imported = page.import_view(
        {
            "kind": "full_frame",
            "camera": "cam1",
            "image": "data:image/jpeg;base64," + base64.b64encode(raw).decode(),
            "captured_at": captured_at,
        }
    )
    assert page.evidence(imported["evidence"]) == raw
    assert imported["source"] == "full_frame"
    assert (imported["camera"], imported["captured_at"]) == ("cam1", captured_at)
    assert page.draft() is None
    assert store.list_objects() == []
    calls = []
    whole = np.ones((900, 1600), dtype=bool)
    monkeypatch.setattr(
        segmentation, "segment", lambda image, **_: calls.append(image) or segmentation.Result(whole, 0.9, "f/sam2")
    )
    page.segment({"snapshot": imported["token"], "bbox": [0.1, 0.1, 0.5, 0.5]})
    assert calls == [raw]
    with pytest.raises(ValueError, match="outline"):
        page.enroll({"snapshot": imported["token"], "camera": "cam1", "name": "wallet", "full_crop": True})


@pytest.mark.parametrize("captured_at", [float("nan"), float("inf"), -1.0, 1e20, 10**400, True, "yesterday"])
def test_review_import_rejects_invalid_timestamps(log, captured_at):
    raw = snapshot().image
    with pytest.raises(ValueError, match="timestamp"):
        page.import_view(
            {
                "kind": "full_frame",
                "camera": "cam0",
                "captured_at": captured_at,
                "image": "data:image/jpeg;base64," + base64.b64encode(raw).decode(),
            }
        )


@pytest.mark.parametrize("size", [(8193, 1), (4001, 4000)])
def test_review_import_rejects_excessive_dimensions_before_decode(log, monkeypatch, size):
    from PIL import Image

    encoded = io.BytesIO()
    Image.new("L", size).save(encoded, format="JPEG")

    def unexpected_decode(raw):
        pytest.fail("oversized image reached the full decoder")

    monkeypatch.setattr(objects, "decode", unexpected_decode)
    with pytest.raises(ValueError, match="dimensions"):
        page.import_view(
            {
                "kind": "full_frame",
                "camera": "cam0",
                "captured_at": 1_700_000_000.0,
                "image": "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode(),
            }
        )


def test_live_import_still_rejects_native_frame_dimensions(log):
    raw = cv2.imencode(".jpg", np.zeros((900, 1600, 3), np.uint8))[1].tobytes()
    with pytest.raises(ValueError, match="dimensions"):
        page.import_view(
            {
                "kind": "crop",
                "camera": "cam0",
                "captured_at": time.time(),
                "image": "data:image/jpeg;base64," + base64.b64encode(raw).decode(),
            }
        )


@pytest.mark.parametrize(("kind", "size"), [("crop", 2_700_001), ("full_frame", 16_000_001)])
def test_import_rejects_oversized_encoded_image(log, kind, size):
    with pytest.raises(ValueError, match="bounded JPEG"):
        page.import_view(
            {
                "kind": kind,
                "camera": "cam0",
                "captured_at": time.time(),
                "image": "data:image/jpeg;base64," + "A" * size,
            }
        )


@pytest.fixture
def crop_snapshot(monkeypatch):
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    monkeypatch.setattr(page, "_selections", {})
    pixels = np.zeros((12, 16, 3), np.uint8)
    pixels[:, :, 0] = np.arange(16, dtype=np.uint8)[None, :] * 10
    pixels[:, :, 1] = np.arange(12, dtype=np.uint8)[:, None] * 10
    return page._snapshots.add("cam0", time.time(), snapshot(pixels).image)


@pytest.mark.parametrize(
    ("prompt", "local"),
    [
        ({"point": [0.4, 0.25]}, {"point": [0.5, 0.5], "bbox": None}),
        ({"bbox": [0.25, 0.1875, 0.1, 0.125]}, {"bbox": [0.125, 0.25, 0.25, 0.5], "point": None}),
    ],
)
def test_crop_segment_converts_frame_prompts_and_restores_frame_mask(monkeypatch, prompt, local):
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    snap = page._snapshots.add("cam0", time.time(), snapshot(np.zeros((80, 100, 3), np.uint8)).image)
    received = []

    def infer(image, **kwargs):
        decoded = objects.decode(image)
        assert decoded is not None
        received.append((decoded.shape[:2], kwargs))
        return segmentation.Result(np.ones(decoded.shape[:2], dtype=bool), 0.9, "fixture/sam2")

    monkeypatch.setattr(segmentation, "segment", infer)
    chosen = page.segment({"snapshot": snap.token, "crop": [20, 10, 40, 20], **prompt})
    assert received == [((512, 1024), {**local, "job": None})]
    assert chosen["bbox"] == pytest.approx([0.2, 0.125, 0.4, 0.25])
    expected = np.zeros((80, 100), dtype=bool)
    expected[10:30, 20:60] = True
    assert np.array_equal(page._selection(chosen["selection"]).result.mask, expected)


@pytest.mark.parametrize(
    "crop",
    [
        None,
        [],
        [0, 0, 2],
        [True, 0, 2, 2],
        [0.5, 0, 2, 2],
        [0, 0, float("nan"), 2],
        [-1, 0, 2, 2],
        [0, 0, 1, 2],
        [15, 0, 2, 2],
        [0, 11, 2, 2],
    ],
)
def test_invalid_pixel_crop_is_rejected_before_model_work(crop_snapshot, monkeypatch, crop):
    called = []
    monkeypatch.setattr(segmentation, "segment", lambda image, **_: called.append(image))
    with pytest.raises(ValueError, match="crop"):
        page.segment({"snapshot": crop_snapshot.token, "crop": crop, "bbox": [0, 0, 1, 1]})
    assert called == []


def test_direct_crop_uses_normal_template_and_visual_matching_path(store, monkeypatch, vision):
    monkeypatch.setattr(page, "_store", store)
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    snap = page._snapshots.add("cam0", time.time(), snapshot().image)
    made = page.enroll(
        {
            "name": "wallet",
            "camera": "cam0",
            "snapshot": snap.token,
            "crop": [80, 60, 160, 160],
            "direct_crop": True,
            "mask": "untrusted-client-mask",
            "bbox": [0, 0, 1, 1],
        }
    )
    (row,) = store.objects()
    assert row["bbox"] == pytest.approx(BBOX)
    assert "segmentation" not in row
    template = objects.decode(base64.b64decode(row["template"]))
    assert template is not None and template.shape[:2] == (160, 160)
    assert made["mode"] == "visual"
    (recognized,) = templates([row])
    assert len(recognized.features[0]) >= 18
    observer = worker(store, vision)
    now = time.time() - 10
    assert observer.observe("cam0", frame(), now, "crop-first", 1) == 0
    assert observer.observe("cam0", frame(), now + 5, "crop-second", 2) == 1
    assert visual(vision)[0]["object_id"] == made["id"]


def test_normalized_integer_crop_edges_do_not_add_a_pixel_through_float_noise(store):
    snap = snapshot(np.zeros((100, 100, 3), np.uint8))
    store.remember(name="wallet", snapshot=snap, bbox=[29 / 100, 29 / 100, 30 / 100, 30 / 100])
    (row,) = store.objects()
    template = objects.decode(base64.b64decode(row["template"]))
    assert template is not None and template.shape[:2] == (30, 30)


@pytest.mark.parametrize("endpoint", [page.segment, page.enroll])
def test_crop_validation_also_guards_segment_and_direct_save(crop_snapshot, endpoint):
    with pytest.raises(ValueError, match="crop"):
        endpoint(
            {
                "snapshot": crop_snapshot.token,
                "camera": "cam0",
                "crop": [15, 0, 2, 2],
                "direct_crop": True,
                "point": [0.5, 0.5],
            }
        )


@pytest.mark.parametrize("prompt", [{"point": [0, 0]}, {"bbox": [0, 0, 0.5, 0.5]}])
def test_crop_rejects_prompts_outside_rectangle_before_inference(crop_snapshot, monkeypatch, prompt):
    monkeypatch.setattr(segmentation, "segment", lambda *a, **k: pytest.fail("invalid prompt reached inference"))
    with pytest.raises(ValueError, match="crop"):
        page.segment({"snapshot": crop_snapshot.token, "crop": [4, 3, 8, 4], **prompt})


def test_direct_crop_preserves_camera_expiry_and_minimum_template_size(store, monkeypatch):
    monkeypatch.setattr(page, "_store", store)
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    snap = page._snapshots.add("cam0", time.time(), snapshot().image)
    data = {"snapshot": snap.token, "camera": "cam1", "name": "wallet", "crop": [80, 60, 160, 160], "direct_crop": True}
    with pytest.raises(ValueError, match="camera"):
        page.enroll(data)
    data.update(camera="cam0", crop=[80, 60, 23, 24])
    with pytest.raises(ValueError, match="24px"):
        page.enroll(data)
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    with pytest.raises(FileNotFoundError):
        page.enroll(data)
    assert store.objects() == []


def test_pixel_edge_still_floors_meaningfully_fractional_pixels():
    assert objects._pixel_edge(0.299, 100) == 29


def test_crop_bbox_exact_pixel_edges_are_valid_at_real_segmentation_boundary(monkeypatch):
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    snap = page._snapshots.add("cam0", time.time(), snapshot(np.zeros((100, 100, 3), np.uint8)).image)
    received = []

    def infer(image, *, bbox, point, job):
        # Real segmentation.segment validates normalized boxes with this shared validator.
        received.append(objects._box(bbox))
        decoded = objects.decode(image)
        assert decoded is not None
        return segmentation.Result(np.ones(decoded.shape[:2], dtype=bool), 0.9, "fixture/sam2")

    monkeypatch.setattr(segmentation, "segment", infer)
    chosen = page.segment({"snapshot": snap.token, "crop": [29, 29, 30, 30], "bbox": [0.29, 0.29, 0.30, 0.30]})
    assert len(received) == 1
    assert received[0] == pytest.approx((0.0, 0.0, 1.0, 1.0))
    assert chosen["bbox"] == pytest.approx([0.29, 0.29, 0.30, 0.30])


def native_metadata(captured_at):
    return {
        "capturedAtUs": round(captured_at * 1e6),
        "width": 640,
        "height": 360,
        "flip": False,
        "profileStatus": "absent",
        "nativeGeometry": {"width": 640, "height": 360},
    }


@pytest.mark.parametrize("change", [{"capturedAtUs": 1}, {"width": 641}, {"profileStatus": "unavailable"}])
def test_recorded_geometry_must_match_actual_imported_picture(store, monkeypatch, change):
    monkeypatch.setattr(page, "store", lambda: store)
    captured_at = round(time.time() - 3600, 6)
    encoded = cv2.imencode(".jpg", np.zeros((360, 640, 3), np.uint8))[1].tobytes()
    with pytest.raises(ValueError, match="geometry does not match"):
        page.import_view(
            {
                "kind": "full_frame",
                "camera": "cam0",
                "captured_at": captured_at,
                "image": "data:image/jpeg;base64," + base64.b64encode(encoded).decode(),
                "frame_metadata": native_metadata(captured_at) | change,
            }
        )


def test_legacy_import_keeps_decoded_coordinate_provenance(store, vision):
    snap = snapshot()
    snap.full_frame = True
    made = store.remember(name="wallet", snapshot=snap, bbox=BBOX)
    row = owner_crops(store.state)[-1]
    assert row["bbox_space"] == "decoded"
    assert made["last_seen"]["bbox"] == BBOX


def test_the_worker_follows_no_camera_until_an_object_is_taught_and_stops_when_every_one_is_forgotten(
    store, vision, monkeypatch
):
    import asyncio

    from liberational.see import object_worker
    from liberational.see.frames import Camera

    monkeypatch.setattr(object_worker, "REFRESH_S", 0.01)
    w = ObjectWorker(client=None, cameras=[Camera("cam0")], store=store, matches=FakeMatches(vision))  # ty: ignore[invalid-argument-type]
    following: list[str] = []

    async def camera(cam):
        following.append("start")
        try:
            await asyncio.Event().wait()  # follows frames until cancelled
        finally:
            following.append("stop")

    monkeypatch.setattr(w, "_camera", camera)

    async def scenario() -> list[str]:
        stop = asyncio.Event()
        running = asyncio.create_task(w.run(stop))
        await asyncio.sleep(0.1)
        seen = ["idle" if not following else "following"]  # nothing taught: no camera followed
        made = store.remember(name="wallet", snapshot=snapshot(), bbox=BBOX)
        while following != ["start"]:
            await asyncio.sleep(0.01)
        store.forget(made["id"])
        while following != ["start", "stop"]:
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(running, 5)
        return seen

    assert asyncio.run(asyncio.wait_for(scenario(), 10)) == ["idle"]
    assert following == ["start", "stop"]


def test_find_objects_runs_on_the_viewed_crop_and_each_mask_is_a_frame_selection(monkeypatch):
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    monkeypatch.setattr(page, "_selections", {})
    snap = page._snapshots.add("cam0", time.time(), snapshot(np.zeros((80, 100, 3), np.uint8)).image)
    clicked = page._remember_selection(snap.token, segmentation.Result(np.ones((80, 100), bool), 0.9, "fixture/sam2"))
    received = []

    def find_all(image, *, concept="", job=None):
        decoded = objects.decode(image)
        assert decoded is not None
        received.append((decoded.shape[:2], concept, job))
        left = np.zeros(decoded.shape[:2], dtype=bool)
        left[:, : decoded.shape[1] // 2] = True  # the crop's left half
        backend = segmentation.Backend("sam2", segmentation.GRID, "fixture/sam2", "/models/sam2")
        found = [segmentation.Result(left, 0.8, "fixture/sam2")] * segmentation.FIND_MAX
        return segmentation.Found(found, backend)

    monkeypatch.setattr(segmentation, "find_all", find_all)
    out = page.find({"snapshot": snap.token, "crop": [20, 10, 40, 20], "job": "j1", "concept": "  my wallet "})
    assert received == [((512, 1024), "my wallet", "j1")], "the crop, scaled up for SAM2, once, for the name typed"
    assert (out["backend"], out["note"]) == ("sam2", ""), "the answer names the model that ran"
    first = out["objects"][0]
    assert first["bbox"] == pytest.approx([0.2, 0.125, 0.2, 0.25])
    expected = np.zeros((80, 100), dtype=bool)
    expected[10:30, 20:40] = True
    assert np.array_equal(page._selection(first["selection"], snap.token).result.mask, expected)
    [polygon] = first["outline"]
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    assert (min(xs), max(xs), min(ys), max(ys)) == pytest.approx((0.2, 0.39, 0.125, 0.3625))
    assert len(out["objects"]) == segmentation.FIND_MAX and isinstance(out["ms"], int)
    assert page._selection(clicked["selection"], snap.token), "a run's masks don't push out the one clicked before"


def test_find_objects_without_a_crop_looks_at_the_whole_frame(monkeypatch):
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    monkeypatch.setattr(page, "_selections", {})
    snap = page._snapshots.add("cam0", time.time(), snapshot(np.zeros((80, 100, 3), np.uint8)).image)
    shapes = []

    def find_all(image, *, concept="", job=None):
        decoded = objects.decode(image)
        assert decoded is not None
        shapes.append((decoded.shape[:2], concept))
        backend = segmentation.Backend("edgetam", segmentation.TODAY, "fixture/edgetam", "/models/edgetam")
        return segmentation.Found(
            [], backend, "SAM 3 had no room on the GPU (it needs about 5 GB free), so EdgeTAM looked instead."
        )

    monkeypatch.setattr(segmentation, "find_all", find_all)
    found = page.find({"snapshot": snap.token})
    assert found["objects"] == [] and found["backend"] == "edgetam"
    assert found["note"].startswith("SAM 3 had no room on the GPU"), "a fallback says why it ran"
    assert shapes == [((80, 100), "")], "no name typed is a grid run, not an error"
    with pytest.raises(FileNotFoundError):
        page.find({"snapshot": "gone"})


def test_find_objects_refuses_a_name_that_is_not_text_or_is_too_long(monkeypatch):
    monkeypatch.setattr(page, "_snapshots", objects.Snapshots())
    snap = page._snapshots.add("cam0", time.time(), snapshot(np.zeros((80, 100, 3), np.uint8)).image)
    calls = []
    monkeypatch.setattr(segmentation, "find_all", lambda *args, **kw: calls.append(kw))
    with pytest.raises(ValueError, match="concept must be text"):
        page.find({"snapshot": snap.token, "concept": ["wallet"]})
    with pytest.raises(ValueError, match="at most 80 characters"):
        page.find({"snapshot": snap.token, "concept": "w" * (page.MAX_CONCEPT + 1)})
    assert calls == [], "a name is cleaned and checked before any model runs"
    assert page.concept({"concept": "  my \n wallet "}) == "my wallet"


def test_status_names_each_camera_by_its_place_and_falls_back_to_its_id(store, vision, monkeypatch):
    from liberational.vision import views

    monkeypatch.setattr(page, "_store", store)
    monkeypatch.setattr(views, "names", lambda: {"cam0": "living room"})
    vision.frame_now("cam0", frame(), time.time() - 1, "frame-7")
    cameras = page.status()["cameras"]
    assert [(c["camera"], c["name"], c["available"]) for c in cameras] == [
        ("cam0", "living room", True),
        ("cam1", "cam1", False),
    ]
