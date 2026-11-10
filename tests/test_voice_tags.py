"""Tagging lines that have no voice: the dashboard queues "this line was <name>" (POST /speakers/tag), and the
microphone service labels the line (an owner tag in the event log; transcript.jsonl is never rewritten) and learns the
voice from its audio (Transcriber.process_tags)."""

import http.client
import itertools
import json
import time
import urllib.parse

import numpy as np
import pytest

import speech_fake
from liberational.dashboard import server
from liberational.domain import kernel
from liberational.listen import revisions
from liberational.people import registry

NOW = 1792800000.0
CHUNK = "audio/mic/2026-10-23/120000.flac"


def test_rejecting_a_turn_is_atomic_durable_and_undoes_together(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STATE", tmp_path)
    first = speech_fake.fake().say({"ts_start": 1000.0, "ts_end": 1003.0, "device": "mic", "text": "bad words"})
    second = speech_fake.fake().say({"ts_start": 1004.0, "ts_end": 1007.0, "device": "mic", "text": "more bad words"})
    keys = [registry.key_of(first), registry.key_of(second)]
    before = speech_fake.fake().all_lines()
    assert server.reject_transcript({"lines": [keys[0], "mic@9999.000"]})[0] == 404
    assert revisions.review_revision(tmp_path) == 0
    code, result = server.reject_transcript({"lines": keys})
    assert code == 200 and result["lines"] == keys
    assert server.heard_lines() == []
    assert speech_fake.fake().all_lines() == before
    assert server.reject_transcript({"lines": keys})[1]["undo"] == result["undo"]
    assert server.undo_transcript({"undo": result["undo"]}) == (200, {"ok": True, "lines": keys})
    assert [(row["id"], row["text"]) for row in server.heard_lines()] == [(row["id"], row["text"]) for row in before]
    assert server.undo_transcript({"undo": result["undo"]})[0] == 409


@pytest.mark.parametrize("body", [{}, {"lines": []}, {"lines": "mic@1.000"}, {"lines": [True]}, {"lines": [""]}])
def test_reject_transcript_validates_every_key_before_writing(tmp_path, monkeypatch, body):
    monkeypatch.setattr(server, "STATE", tmp_path)
    assert server.reject_transcript(body)[0] == 400
    assert not (tmp_path / "knowledge.sqlite3").exists()


def test_transcript_correction_route_requires_same_origin_json(tmp_path, monkeypatch, serve):
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    row = speech_fake.fake().say({"ts_start": 1000.0, "ts_end": 1003.0, "device": "mic", "text": "incorrect words"})
    host = urllib.parse.urlsplit(serve(server.H)).netloc

    def post(path, body, headers):
        connection = http.client.HTTPConnection(host, timeout=5)
        connection.request("POST", path, json.dumps(body), headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    request = {"lines": [registry.key_of(row)]}
    assert post("/transcript/reject", request, {"Content-Type": "text/plain"})[0] == 403
    assert (
        post("/transcript/reject", request, {"Content-Type": "application/json", "Origin": "http://evil.example"})[0]
        == 403
    )
    code, result = post("/transcript/reject", request, {"Content-Type": "application/json"})
    assert code == 200
    assert post("/transcript/undo", {"undo": result["undo"]}, {"Content-Type": "application/json"})[0] == 200


def test_transcript_correction_routes_refuse_no_control_requests_without_changing_state(tmp_path, monkeypatch, serve):
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    monkeypatch.setattr(server.H, "_can_control", lambda self: False)
    row = speech_fake.fake().say({"ts_start": 1000.0, "ts_end": 1003.0, "device": "mic", "text": "incorrect words"})
    key = registry.key_of(row)
    host = urllib.parse.urlsplit(serve(server.H)).netloc

    def post(path, body):
        connection = http.client.HTTPConnection(host, timeout=5)
        connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    assert post("/transcript/reject", {"lines": [key]}) == (403, {"ok": False, "error": "Controls unavailable"})
    assert server.heard_lines() == [row]
    assert revisions.review_revision(tmp_path) == 0
    assert not (tmp_path / "knowledge.sqlite3").exists()
    _, result = server.reject_transcript({"lines": [key]})
    before = revisions.review_revision(tmp_path)
    assert post("/transcript/undo", {"undo": result["undo"]}) == (403, {"ok": False, "error": "Controls unavailable"})
    assert server.heard_lines() == []
    assert revisions.review_revision(tmp_path) == before
    assert revisions.current_rejection(tmp_path, [key]) == result["undo"]


def unit(*v):
    a = np.asarray(v, np.float32)
    return a / np.linalg.norm(a)


@pytest.fixture
def tags(tmp_path, monkeypatch):
    path = tmp_path / "voice_tags.jsonl"
    monkeypatch.setattr(server, "VOICE_TAGS", path)
    return path


def test_queues_one_request_for_the_visible_turn(tags):
    code, body = server.request_voice_tags([1700000000.1234, "1700000005.5"], "  José  Núñez ")
    assert (code, body["ok"], body["queued"], body["name"]) == (200, True, 2, "José Núñez")
    reqs = [json.loads(line) for line in tags.read_text().splitlines()]
    assert len(reqs) == 1
    assert reqs[0]["lines"] == ["1700000000.123", "1700000005.500"]
    assert reqs[0]["name"] == "José Núñez"
    assert reqs[0]["requested_at"] and reqs[0]["group"]


def test_a_19_line_batch_writes_each_line_key_once(tags):
    keys = [f"mic@{1700000000 + i}.000" for i in range(19)]
    code, result = server.request_voice_tags(None, "Dana", lines=keys)
    assert code == 200 and result["queued"] == 19
    raw = tags.read_text()
    assert (len(raw.splitlines()), raw.count("mic@")) == (1, 19)
    assert json.loads(raw)["lines"] == keys


@pytest.mark.parametrize(
    ("ts", "name"),
    [
        pytest.param([1.0], "1abc", id="digit-first"),
        pytest.param([1.0], "", id="empty-name"),
        pytest.param([], "Dana", id="no-lines"),
        pytest.param("1.0", "Dana", id="string-not-list"),
        pytest.param(["x"], "Dana", id="non-numeric"),
        pytest.param([1.0] * 201, "Dana", id="over-200"),
    ],
)
def test_refuses_bad_names_and_lines(tags, ts, name):
    assert server.request_voice_tags(ts, name)[0] == 400
    assert not tags.exists()


def test_route_needs_same_origin_json(tags, serve, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    host = urllib.parse.urlsplit(serve(server.H)).netloc

    def post(**headers):
        c = http.client.HTTPConnection(host, timeout=5)
        c.request("POST", "/speakers/tag", json.dumps({"ts_starts": [1.0], "name": "Dana"}), headers)
        status = c.getresponse().status
        c.close()
        return status

    assert post(**{"Content-Type": "text/plain"}) == 403
    assert post(**{"Content-Type": "application/json", "Origin": "http://evil.example"}) == 403
    assert not tags.exists()
    assert post(**{"Content-Type": "application/json"}) == 200
    assert len(tags.read_text().splitlines()) == 1


@pytest.fixture
def ticking(monkeypatch):
    """The event log's clock steps a millisecond per event, so a tag request stamped with it falls strictly between
    the events around it."""
    t = itertools.count(time.time(), 0.001)
    monkeypatch.setattr(kernel, "now", lambda: next(t))


def test_queues_each_line_by_its_key(tags):
    code, body = server.request_voice_tags(None, "Dana", lines=["mic-b@1700000000.123", "1700000005.500"])
    assert (code, body["queued"]) == (200, 2)
    reqs = [json.loads(line) for line in tags.read_text().splitlines()]
    assert len(reqs) == 1
    assert reqs[0]["lines"] == ["mic-b@1700000000.123", "1700000005.500"]
