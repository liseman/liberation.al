"""dashboard.remote_teaching: password-checked browsers on other machines may teach (the Inbox and the People teaching
routes here), never change services, and never without a password or from another site."""

import base64
import http.client
import json
import urllib.parse

import pytest

from liberational.dashboard import server
from liberational.people import references


@pytest.fixture(params=["http", "https"])
def post(serve, monkeypatch, tmp_path, request):
    monkeypatch.setattr(server, "inbox_skip", lambda kind, ident, control: (200 if control else 403, {"ok": control}))
    base = serve(server.H)
    netloc = urllib.parse.urlsplit(base).netloc

    def go(path, *, remote=True, auth=True, origin=True, body=None, raw=None, content_type="application/json"):
        headers = {"Content-Type": content_type, "Host": netloc}
        if remote:
            headers["X-Forwarded-For"] = "100.64.0.9"
        if auth:
            headers["Authorization"] = "Bas" + "ic " + base64.b64encode(b"luca:pw").decode()
        if origin:
            headers["Origin"] = f"{request.param}://{netloc}"
        else:
            headers["Origin"] = "https://evil.example"
        conn = http.client.HTTPConnection(netloc, timeout=120)  # generous: devbox can be loaded
        data = raw if raw is not None else json.dumps(body or {"kind": "face", "id": 3})
        conn.request("POST", path, body=data, headers=headers)
        code = conn.getresponse().status
        conn.close()
        return code

    return go


def test_off_keeps_teaching_on_this_machine(post, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_teaching", lambda: False)
    assert post("/inbox/skip") == 403


def test_on_lets_a_password_checked_same_site_page_teach(post, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_teaching", lambda: True)
    assert post("/inbox/skip") == 200
    assert post("/inbox/skip", origin=False) == 403  # another site can't post it
    assert post("/inbox/skip", auth=False) == 401  # the password gate comes first


def test_on_still_keeps_services_on_this_machine(post, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_teaching", lambda: True)
    assert post("/services", body={"name": "vlm", "action": "off"}) == 403


def test_never_without_a_password(post, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    monkeypatch.setattr(server, "remote_teaching", lambda: True)
    assert post("/inbox/skip", auth=False) == 403


# Each route the handler gates itself (#2788), and the action behind it.
TEACHING_ACTIONS = {
    "/people/assign": "assign_to",
    "/people/assign-many": "assign_many",
    "/people/correct": "correct_card",
    "/people/undo": "undo_people",
    "/people/claim": "claim_identity",
}


@pytest.fixture
def taught(monkeypatch):
    """Stand-ins for the teaching actions: each records (action, args) and succeeds."""
    calls = []
    for action in TEACHING_ACTIONS.values():
        monkeypatch.setattr(
            server, action, lambda *args, action=action: calls.append((action, args)) or (200, {"ok": True})
        )
    return calls


@pytest.mark.parametrize("route", sorted(TEACHING_ACTIONS))
@pytest.mark.parametrize(
    "remote,teaching,status",
    [
        pytest.param(True, False, 403, id="remote-off-refused"),
        pytest.param(True, True, 200, id="remote-on-allowed"),
        pytest.param(False, False, 200, id="local-off-allowed"),
    ],
)
def test_every_teaching_route_follows_remote_teaching(post, taught, monkeypatch, route, remote, teaching, status):
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_teaching", lambda: teaching)
    assert post(route, remote=remote) == status
    assert [action for action, _ in taught] == ([TEACHING_ACTIONS[route]] if status == 200 else [])


def test_a_remote_claim_still_cannot_make_a_voice_the_owners(post, taught, monkeypatch):
    # Remote teaching lets a remote page claim, but the owner-voice check inside the claim stays on this machine.
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_teaching", lambda: True)
    assert post("/people/claim", body={"kind": "voice", "id": 3, "name": "Alice"}) == 200
    [(_, (kind, ident, name, control, *_))] = taught
    assert (kind, ident, name, control) == ("voice", 3, "Alice", False)


def test_a_refused_teaching_route_answers_like_a_refused_inbox_skip():
    code, body = server.teaching_refused()
    skip_code, skip_body = server.inbox_skip("face", 3, control=False)
    assert (code, body.keys(), body["ok"]) == (skip_code, skip_body.keys(), False)


# The teaching routes #2804 put on the same gate: each one's body, and what it calls once the gate lets it through.
MORE_ROUTES = {
    "/speakers/refine": ({"action": "assign", "line": "mic@1.0", "name": "Dana"}, "speakers_refine"),
    "/transcript/unlabel": ({"line": "mic@1.0"}, "_line_at"),
    "/people/add": ({"name": "Dana", "number": ""}, "add_person"),
    "/people/reference-url": ({"url": "https://example.test/dana.jpg"}, "fetch_media"),
    "/people/reference-attach": ({"id": "ref1", "person": "Dana"}, "attach"),
    "/people/reference-remove": ({"id": "ref1"}, "remove"),
    "/people/reference-upload": (None, "stage"),
}


@pytest.fixture
def taught_more(monkeypatch):
    """Stand-ins for what each of MORE_ROUTES calls past the gate: each records its name."""
    calls = []

    def record(name, reply):
        return lambda *args, **kwargs: calls.append(name) or reply

    monkeypatch.setattr(server, "speakers_refine", record("speakers_refine", (200, {"ok": True})))
    monkeypatch.setattr(server, "_line_at", record("_line_at", None))  # no such line: a 404 past the gate
    monkeypatch.setattr(server, "add_person", record("add_person", None))
    monkeypatch.setattr(server, "people_view", lambda: {"people": [{"name": "Dana"}]})
    monkeypatch.setattr(references, "fetch_media", record("fetch_media", b"jpeg"))
    monkeypatch.setattr(references, "stage", record("stage", {"id": "ref1"}))
    monkeypatch.setattr(references, "attach", record("attach", {"ok": True}))
    monkeypatch.setattr(references, "remove", record("remove", None))
    return calls


def send(post, route, **kwargs):
    body, _ = MORE_ROUTES[route]
    if route == "/people/reference-upload":
        return post(route, raw=b"jpeg bytes", content_type="application/octet-stream", **kwargs)
    return post(route, body=body, **kwargs)


@pytest.mark.parametrize("route", sorted(MORE_ROUTES))
@pytest.mark.parametrize(
    "remote,teaching,allowed",
    [
        pytest.param(True, False, False, id="remote-off-refused"),
        pytest.param(True, True, True, id="remote-on-allowed"),
        pytest.param(False, False, True, id="local-off-allowed"),
    ],
)
def test_the_other_teaching_routes_follow_remote_teaching(
    post, taught_more, monkeypatch, route, remote, teaching, allowed
):
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_teaching", lambda: teaching)
    code = send(post, route, remote=remote)
    # the first thing past the gate (reference-url then stages what it fetched)
    assert (code == 403, taught_more[:1]) == (not allowed, [MORE_ROUTES[route][1]] if allowed else [])


def test_a_remote_upload_must_not_be_a_type_another_site_can_send(post, taught_more, monkeypatch):
    # text/plain needs no preflight, so a cross-site page could send it; the upload's own type can't be.
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_teaching", lambda: True)
    assert post("/people/reference-upload", raw=b"jpeg bytes", content_type="text/plain") == 403
    assert taught_more == []


@pytest.mark.parametrize(
    "remote,teaching,control",
    [
        pytest.param(True, True, False, id="remote-on"),
        pytest.param(True, False, False, id="remote-off"),
        pytest.param(False, False, True, id="local"),
    ],
)
def test_this_is_me_stays_on_this_machine_even_with_remote_teaching(post, monkeypatch, remote, teaching, control):
    # Owner identity isn't teaching (#2804): This is me decides who the assistant answers as the owner.
    calls = []
    monkeypatch.setattr(
        server, "merge_into_owner", lambda name, voices, *, control: calls.append(control) or (200, {"ok": True})
    )
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_teaching", lambda: teaching)
    post("/people/me", body={"name": "Dana", "voices": [3]}, remote=remote)
    assert calls == [control]
