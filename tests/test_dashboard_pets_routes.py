"""Pet routes preserve the dashboard's same-origin and local-control boundaries."""

import http.client
import json

import pytest

from liberational.people import pets


class PetStore:
    def __init__(self):
        self.profiles = []

    def listing(self):
        return list(self.profiles)

    def save(self, name, species, pet_id=None, sex=None):
        profile = {"id": pet_id or "pet-id", "name": name, "species": species}
        if sex is not None:
            profile["sex"] = sex
        self.profiles = [profile]
        return profile

    def remove(self, pet_id):
        self.profiles = [p for p in self.profiles if p["id"] != pet_id]


@pytest.fixture
def pet_api(serve_dashboard, monkeypatch):
    store = PetStore()
    monkeypatch.setattr(pets, "at", lambda state: store)
    host = serve_dashboard().removeprefix("http://")

    def request(method, path, body=None, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin is not None:
            headers["Origin"] = origin
        connection = http.client.HTTPConnection(host, timeout=5)
        connection.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
        response = connection.getresponse()
        reply = response.status, json.loads(response.read())
        connection.close()
        return reply

    return request, store


def test_pet_http_routes_list_save_and_remove_only_owner_names(pet_api):
    request, store = pet_api
    assert request("GET", "/pets")[1]["pets"] == []
    status, reply = request("POST", "/pets/save", {"name": "Pip", "species": "cat"})
    assert status == 200
    assert reply["pet"] == {"id": "pet-id", "name": "Pip", "species": "cat"}
    assert request("GET", "/pets")[1]["pets"] == [reply["pet"]]
    assert request("POST", "/pets/remove", {"id": "pet-id"}) == (200, {"ok": True, "pets": []})
    assert store.profiles == []


@pytest.mark.parametrize("path", ["/pets/save", "/pets/remove"])
def test_pet_http_writes_reject_another_origin_without_touching_profiles(pet_api, path):
    request, store = pet_api
    status, reply = request("POST", path, {"name": "Pip", "species": "cat", "id": "pet-id"}, "http://other.invalid")
    assert status == 403
    assert "same-origin" in reply["error"]
    assert store.profiles == []


@pytest.mark.parametrize("path", ["/pets/save", "/pets/remove"])
def test_pet_http_writes_require_local_control(dashboard_server, pet_api, monkeypatch, path):
    request, store = pet_api
    monkeypatch.setattr(dashboard_server.H, "_can_control", lambda self: False)
    status, _ = request("POST", path, {"name": "Pip", "species": "cat", "id": "pet-id"})
    assert status == 403
    assert store.profiles == []


@pytest.fixture
def labels(monkeypatch):
    from liberational.people import pet_recognition

    calls = []
    monkeypatch.setattr(pet_recognition, "label", lambda state, tracks, value, pets: calls.append(tracks) or 1)
    return calls


def test_labeling_a_sighting_goes_through_from_this_machine(pet_api, labels):
    request, _ = pet_api
    assert request("POST", "/pets/sightings/label", {"track_ids": ["t1"], "label": "other"}) == (
        200,
        {"ok": True, "labeled": 1},
    )
    assert labels == [["t1"]]


def test_labeling_a_sighting_refuses_another_origin_or_a_remote_viewer(dashboard_server, pet_api, labels, monkeypatch):
    request, _ = pet_api
    body = {"track_ids": ["t1"], "label": "other"}
    assert request("POST", "/pets/sightings/label", body, "http://other.invalid")[0] == 403
    monkeypatch.setattr(dashboard_server.H, "_can_control", lambda self: False)
    assert request("POST", "/pets/sightings/label", body)[0] == 403
    assert labels == []


def test_removing_training_images_refuses_another_origin_or_a_remote_viewer(dashboard_server, pet_api, monkeypatch):
    from liberational.people import pet_recognition

    calls = []
    monkeypatch.setattr(
        pet_recognition, "remove_examples", lambda state, tracks, pet, pets: calls.append((tracks, pet)) or 1
    )
    request, _ = pet_api
    body = {"track_ids": ["t1"], "pet": "pet-id"}
    assert request("POST", "/pets/examples/remove", body, "http://other.invalid")[0] == 403
    assert request("POST", "/pets/examples/remove", body) == (200, {"ok": True, "removed": 1})
    monkeypatch.setattr(dashboard_server.H, "_can_control", lambda self: False)
    assert request("POST", "/pets/examples/remove", body)[0] == 403
    assert calls == [(["t1"], "pet-id")]


def test_blocking_and_listing_hidden_sightings_go_through_from_this_machine_only(
    dashboard_server, pet_api, monkeypatch
):
    from liberational.people import pet_recognition

    calls = []
    monkeypatch.setattr(
        pet_recognition, "block", lambda state, tracks, pets, active=True: calls.append((tracks, active)) or 1
    )
    monkeypatch.setattr(pet_recognition, "hidden", lambda state: {"ignored": [], "blocked": []})
    request, _ = pet_api
    assert request("POST", "/pets/sightings/block", {"track_ids": ["t1"]}) == (200, {"ok": True, "blocked": 1})
    assert request("POST", "/pets/sightings/block", {"track_ids": ["t1"], "block": False})[0] == 200
    assert calls == [(["t1"], True), (["t1"], False)]
    assert request("GET", "/pets/sightings/hidden") == (200, {"ignored": [], "blocked": []})
    assert request("POST", "/pets/sightings/block", {"track_ids": ["t1"], "block": "no"})[0] == 400
    assert request("POST", "/pets/sightings/block", {"track_ids": ["t1"]}, "http://other.invalid")[0] == 403
    monkeypatch.setattr(dashboard_server.H, "_can_control", lambda self: False)
    assert request("POST", "/pets/sightings/block", {"track_ids": ["t1"]})[0] == 403
    assert len(calls) == 2
