"""People's Training images routes (dashboard/face_examples.py): a person's face rows listed, their pictures served,
and rows removed only by someone who may teach."""

import http.client
import json

import numpy as np
import pytest

from liberational.vision import face_examples, faces
from liberational.vision.face_examples import Example


def unit(k):
    v = np.zeros(512, np.float32)
    v[k] = 1
    return v


@pytest.fixture
def api(serve_dashboard, tmp_path, monkeypatch):
    """Alice with two rows, the second a face that isn't hers with a picture; (request, the wrong row's key)."""
    monkeypatch.setattr(faces, "DB", tmp_path / "db.npz")
    monkeypatch.setattr(faces, "LEARNED_DB", tmp_path / "learned.npz")
    monkeypatch.setattr(faces, "NOTS_DB", tmp_path / "nots.npz")
    faces.save_db(["Alice"], unit(0)[None])
    faces.add_row("Alice", unit(7), Example("group", ref="group:4", jpg=b"\xff\xd8 not hers"))
    host = serve_dashboard().removeprefix("http://")

    def request(method, path, body=None):
        connection = http.client.HTTPConnection(host, timeout=5)
        connection.request(
            method, path, None if body is None else json.dumps(body), {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        return response.status, raw

    return request, face_examples.key(unit(7))


def test_the_listing_and_each_rows_picture_are_served(api):
    request, wrong = api
    status, raw = request("GET", "/people/examples?name=alice")
    reply = json.loads(raw)
    assert (status, reply["count"]) == (200, 2)
    assert {r["key"]: r["picture"] for r in reply["examples"]}[wrong] is True
    assert request("GET", f"/people/examples/photo?key={wrong}") == (200, b"\xff\xd8 not hers")
    assert request("GET", "/people/examples/photo?key=../../db.npz")[0] == 404


def test_removing_rows_answers_the_listing_as_it_is_now(api):
    request, wrong = api
    status, raw = request("POST", "/people/examples/remove", {"name": "Alice", "keys": [wrong]})
    reply = json.loads(raw)
    assert (status, reply["removed"], reply["count"]) == (200, 1, 1)
    assert faces.counts() == {"Alice": 1}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"name": "Alice", "keys": []}, id="no-keys"),
        pytest.param({"name": "Alice", "keys": ["../db"]}, id="not-a-key"),
        pytest.param({"name": "", "keys": ["0" * 24]}, id="no-name"),
    ],
)
def test_a_removal_that_names_nothing_is_refused(api, body):
    request, _ = api
    assert request("POST", "/people/examples/remove", body)[0] == 400
    assert faces.counts() == {"Alice": 2}


def test_a_viewer_who_may_not_teach_removes_nothing(api, dashboard_server, monkeypatch):
    request, wrong = api
    monkeypatch.setattr(dashboard_server.H, "_can_teach", lambda self, content_type="application/json": False)
    assert request("POST", "/people/examples/remove", {"name": "Alice", "keys": [wrong]})[0] == 403
    assert faces.counts() == {"Alice": 2}
