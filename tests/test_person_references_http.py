import http.client
import io
import json
from unittest.mock import create_autospec, patch

import numpy as np
import pytest
import requests
from PIL import Image

from liberational.dashboard import context, server
from liberational.people import references
from liberational.vision import faces


@pytest.fixture
def found(monkeypatch):
    """The faces `pack-faces reference` finds in a photo (run in process, never the model): set by a test."""
    faces_in: list[np.ndarray] = []

    def run(args):
        _cmd, name, path = args
        with patch.object(faces, "embed", autospec=True, return_value=[(np.zeros(4), v) for v in faces_in]):
            return True, json.dumps(faces.enroll_reference(name, path))

    monkeypatch.setattr(server, "_face_cmd", run)
    return faces_in


@pytest.fixture
def base(serve, tmp_path, monkeypatch, found):
    monkeypatch.setattr(context, "STATE", tmp_path)
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    monkeypatch.setattr(server, "people_view", lambda: {"people": [{"name": "Dana"}]})
    monkeypatch.setattr(faces, "DB", tmp_path / "db.npz")
    return serve(server.H)


def call(base, path, body=None, headers=None):
    connection = http.client.HTTPConnection(base.removeprefix("http://"), timeout=10)
    payload = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
    connection.request(
        "POST" if body is not None else "GET", path, payload, headers or {"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    raw = response.read()
    result = json.loads(raw) if response.getheader("Content-Type", "").startswith("application/json") else raw
    connection.close()
    return response.status, result


def test_upload_preview_attach_list_remove(base):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buf, "PNG")
    code, item = call(base, "/people/reference-upload", buf.getvalue(), {"Content-Type": "audio/mpeg"})
    assert code == 200
    assert item["kind"] == "image" and item["person"] is None
    code, raw = call(base, "/people/reference-media/" + item["id"])
    assert code == 200
    assert Image.open(io.BytesIO(raw)).size == (8, 8)
    code, error = call(base, "/people/reference-attach", {"id": item["id"], "person": "Unknown"})
    assert code == 400 and "target" in error["error"]
    code, attached = call(base, "/people/reference-attach", {"id": item["id"], "person": "Dana"})
    assert code == 200 and attached["enrollment"] == "reference only"
    assert len(call(base, "/people/references?person=Dana")[1]["references"]) == 1
    assert call(base, "/people/reference-remove", {"id": item["id"]})[0] == 200
    assert call(base, "/people/references?person=Dana")[1] == {"references": []}


def test_a_one_face_photo_teaches_its_person_and_removing_it_takes_the_row_back(base, found):
    # A single-face reference photo becomes one of the person's owner face rows
    face = np.zeros(512, np.float32)
    face[3] = 1.0
    found.append(face)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buf, "PNG")
    _, item = call(base, "/people/reference-upload", buf.getvalue(), {"Content-Type": "audio/mpeg"})
    code, attached = call(base, "/people/reference-attach", {"id": item["id"], "person": "Dana"})
    assert code == 200 and attached["enrollment"] == "face" and attached["face"]["result"] == "added"
    names, embs = faces.load_db()
    assert names == ["Dana"] and float(embs[0] @ face) == pytest.approx(1.0)
    assert call(base, "/people/reference-remove", {"id": item["id"]})[0] == 200
    assert faces.load_db()[0] == []


def test_oversize_upload_rejected_before_reading_body(base):
    c = http.client.HTTPConnection(base.removeprefix("http://"), timeout=5)
    c.putrequest("POST", "/people/reference-upload")
    c.putheader("Content-Length", str(references.MAX_BYTES + 1))
    c.endheaders()
    r = c.getresponse()
    assert r.status == 413
    assert "20 MB" in json.loads(r.read())["error"]
    c.close()


@pytest.mark.parametrize("path", ["/people/reference-url", "/people/reference-attach", "/people/search"])
def test_new_json_routes_reject_nonobjects_and_cross_origin(base, path):
    assert call(base, path, ["not", "object"])[0] == 400
    assert call(base, path, {}, {"Content-Type": "application/json", "Origin": "https://evil.example"})[0] == 403


def test_candidate_lookup_uses_grounded_provider_without_agent(monkeypatch):
    from liberational.people import name_lookup

    monkeypatch.setattr(server.cfg, "agent_configured", lambda: False)
    result = {"candidates": [{"name": "Dana", "detail": "Engineer", "url": "https://www.wikidata.org/wiki/Q123"}]}
    lookup = create_autospec(name_lookup.lookup, return_value=result)
    post = create_autospec(requests.post)
    monkeypatch.setattr(name_lookup, "lookup", lookup)
    monkeypatch.setattr(requests, "post", post)
    assert server.public_lookup("Dana", "Engineer", candidates=True) == result
    lookup.assert_called_once_with("Dana", "Engineer")
    post.assert_not_called()
