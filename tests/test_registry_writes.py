"""A registry write that fails is never reported as done: registry.recorded raises it, what it was for isn't
written, and a dashboard request answers with an error that says so."""

import http.client
import json

import numpy as np
import pytest

from liberational.people import contacts, registry
from liberational.vision import faces


def failing(*args, **kwargs):
    raise OSError("the event log is unavailable")


def test_a_failed_write_is_raised_as_write_failed():
    with pytest.raises(registry.WriteFailed, match="photo saved"), registry.recorded("photo saved"):
        raise OSError("down")


def test_a_read_with_a_fallback_still_goes_on():
    with registry.safely("a lookup"):
        raise OSError("down")


def test_adding_someone_the_registry_couldnt_record_leaves_the_people_list_as_it_was(tmp_path, monkeypatch):
    book = contacts.Book(tmp_path, tmp_path / "contacts.json", tmp_path / "people_added.json")
    monkeypatch.setattr(registry.Registry, "contact", failing)
    with pytest.raises(registry.WriteFailed):
        book.add("Dana", "+14155550123")
    assert not (tmp_path / "people_added.json").exists() or "Dana" not in json.loads(
        (tmp_path / "people_added.json").read_text()
    )


def test_naming_a_face_group_the_registry_couldnt_record_changes_no_face_file(tmp_path, monkeypatch):
    monkeypatch.setattr(faces, "DB", tmp_path / "db.npz")
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown_clusters.npz")
    faces._unk_save([3], np.eye(512, dtype=np.float32)[:1], [4], 4)
    monkeypatch.setattr(registry.Registry, "claim", failing)
    with pytest.raises(registry.WriteFailed):
        faces.name_cluster(3, "Dana")
    assert faces.load_db()[0] == [] and faces._unk_load()[0] == [3], "the group is still there, unnamed"


def test_a_request_whose_write_failed_answers_with_the_error(dashboard_server, serve_dashboard, monkeypatch):

    def post(self):
        with registry.recorded("photo moved"):
            raise OSError("down")

    monkeypatch.setattr(dashboard_server.H, "_post", post)
    c = http.client.HTTPConnection(serve_dashboard().removeprefix("http://"), timeout=5)
    c.request("POST", "/people/anything", body=b"{}", headers={"Content-Type": "application/json"})
    r = c.getresponse()
    body = json.loads(r.read())
    c.close()
    assert (r.status, body["ok"]) == (500, False) and "wasn't saved" in body["error"]
