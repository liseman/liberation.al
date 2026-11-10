"""The dashboard's route table (#61): dashboard/routes.py, and the list of routes the server registers."""

import http.client
import json
import urllib.parse
from pathlib import Path

import pytest

from liberational.dashboard import routes, server

ROUTES = Path(__file__).parent / "data" / "dashboard_routes.txt"


@pytest.fixture
def table():
    return routes.Table()


def test_an_exact_path_is_found_for_its_method_only(table):
    @table.get("/pets")
    def pets(h):
        return "pets"

    route, _ = table.find("GET", "/pets")
    assert route.handler is pets
    assert table.find("POST", "/pets") is None
    assert table.find("GET", "/pets/") is None


def test_one_handler_can_take_several_paths(table):
    @table.post("/pets/save", "/pets/remove")
    def change(h, body):
        return None

    assert table.find("POST", "/pets/save")[0].handler is change
    assert table.find("POST", "/pets/remove")[0].handler is change


def test_a_pattern_must_match_the_whole_path_and_gives_its_match(table):
    @table.get_re(r"/clones/(\d{1,6})/lines")
    def lines(h, m):
        return None

    route, match = table.find("GET", "/clones/42/lines")
    assert (route.kind, match.group(1)) == ("pattern", "42")
    assert table.find("GET", "/clones/42/lines/more") is None
    assert table.find("GET", "/clones/abc/lines") is None


def test_an_exact_path_wins_over_a_catch_all_registered_after_it(table):
    @table.get("/people/video-thumb.jpg")
    def thumb(h):
        return None

    @table.get_when("/people/<name>", lambda p: p.startswith("/people/"))
    def person(h):
        return None

    assert table.find("GET", "/people/video-thumb.jpg")[0].handler is thumb
    assert table.find("GET", "/people/Alice")[0].handler is person


def test_patterns_and_tests_are_tried_in_the_order_registered(table):
    @table.get_re(r"/faces/shot/([0-9a-z]+)\.jpg")
    def shot(h, m):
        return None

    @table.get_when("/faces/shot/...", lambda p: p.startswith("/faces/shot/"))
    def missing(h):
        return None

    assert table.find("GET", "/faces/shot/abc.jpg")[0].handler is shot
    assert table.find("GET", "/faces/shot/abc.png")[0].handler is missing


def test_an_exact_path_an_earlier_catch_all_already_takes_is_refused(table):
    table.get_when("/people/<name>", lambda p: p.startswith("/people/"))(lambda h: None)
    with pytest.raises(ValueError, match="would be taken by /people/<name>"):
        table.get("/people/references")(lambda h: None)
    table.post("/people/references")(lambda h, body: None)  # another method is another route


def test_a_path_registered_twice_is_refused(table):
    table.get("/pets")(lambda h: None)
    with pytest.raises(ValueError, match="GET /pets is registered twice"):
        table.get("/pets")(lambda h: None)


def test_a_route_that_reads_its_own_body_says_so(table):
    table.post_raw("/clones/record")(lambda h: None)
    table.post("/clones/say")(lambda h, body: None)
    assert table.find("POST", "/clones/record")[0].raw is True
    assert table.find("POST", "/clones/say")[0].raw is False


def test_the_listing_names_each_route_and_the_module_that_answers_it(table):
    table.post("/zones")(lambda h, body: None)
    table.get_re(r"/clones/(\d+)/lines")(lambda h, m: None)
    assert table.listing() == [
        ("GET", r"/clones/(\d+)/lines", "test_dashboard_routes"),
        ("POST", "/zones", "test_dashboard_routes"),
    ]


def test_the_server_registers_the_same_routes_as_before():
    """Moving a route to its area's module mustn't add, drop, or rename one: the list is kept in tests/data, and a
    route added on purpose is added there too."""
    assert server.H is not None  # the server's import is what registers them
    listed = [f"{method} {path}" for method, path, _ in routes.listing()]
    assert listed == ROUTES.read_text().splitlines()


@pytest.fixture
def ask(serve, monkeypatch):
    """One request to a running dashboard: (status, content type, body)."""
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    netloc = urllib.parse.urlsplit(serve(server.H)).netloc

    def send(method, path, body=None):
        conn = http.client.HTTPConnection(netloc, timeout=5)
        conn.request(method, path, body=body, headers={"Content-Type": "application/json"} if body else {})
        response = conn.getresponse()
        reply = response.status, response.getheader("Content-Type"), response.read()
        conn.close()
        return reply

    return send


def test_a_get_nobody_registered_is_an_empty_404(ask):
    assert ask("GET", "/no/such/route") == (404, "text/plain", b"")


def test_a_post_nobody_registered_is_told_so_after_its_body_was_read(ask):
    status, _, body = ask("POST", "/no/such/route", b'{"a": 1}')
    assert (status, json.loads(body)) == (404, {"ok": False, "error": "unknown endpoint"})


def test_a_post_whose_body_is_not_json_is_refused_whether_or_not_its_path_is_known(ask):
    for path in ("/no/such/route", "/clones/say"):
        status, _, body = ask("POST", path, b"not json")
        assert (path, status, json.loads(body)) == (path, 400, {"ok": False, "error": "bad json"})


def test_a_registered_pattern_reaches_its_handler_with_the_match(ask):
    status, _, body = ask("GET", "/speaker/7/latest")
    assert (status, json.loads(body)) == (404, {"ok": False, "error": "no utterance with audio for that speaker"})


def test_a_delete_nobody_registered_is_refused(ask):
    status, _, body = ask("DELETE", "/people")
    assert (status, json.loads(body)) == (405, {"error": "Method not supported."})


def test_a_delete_reaches_asks_own_answer_for_its_paths(ask):
    status, _, body = ask("DELETE", "/pack-ask/jobs")
    assert (status, json.loads(body)) == (405, {"error": "Method not supported."})


@pytest.mark.parametrize(
    ("method", "path", "status", "reply"),
    [
        pytest.param(
            "POST", "/config/voices", 404, {"ok": False, "error": "unknown endpoint"}, id="read-only-settings"
        ),
        pytest.param("POST", "/devices", 404, {"ok": False, "error": "unknown endpoint"}, id="read-only-devices"),
        pytest.param(
            "POST", "/network-cameras", 404, {"ok": False, "error": "unknown endpoint"}, id="read-only-cameras"
        ),
    ],
)
def test_a_method_a_module_does_not_take_for_its_path_is_an_unknown_route(ask, method, path, status, reply):
    got, _, body = ask(method, path, b"{}")
    assert (got, json.loads(body)) == (status, reply)


def test_a_write_only_path_has_no_read(ask):
    assert ask("GET", "/devices/sensitivity") == (404, "text/plain", b"")
    assert ask("GET", "/frontend/log") == (404, "text/plain", b"")


def test_every_module_that_registers_routes_is_loaded_with_the_server():
    """A module the server no longer calls by name is still imported for its routes (routes.load)."""
    areas = {area for _, _, area in routes.listing()}
    assert {"settings", "devices", "network_cameras", "ingest", "reports", "frontend_log", "research"} <= areas
    assert "ask_routes" in areas


def test_a_catch_all_is_tried_after_a_narrower_route_registered_later(table):
    """The modules load in any order: a person's page mustn't take another module's /people/... route."""

    @table.get_last("/people/<name>", lambda p: p.startswith("/people/"))
    def person(h):
        return None

    @table.get_when("/people/reference-media/<id>", lambda p: p.startswith("/people/reference-media/"))
    def media(h):
        return None

    table.get("/people/video-thumb.jpg")(lambda h: None)  # nor does it stop an exact path being registered
    assert table.find("GET", "/people/reference-media/abc")[0].handler is media
    assert table.find("GET", "/people/Alice")[0].handler is person
    assert ("GET", "/people/<name>", "test_dashboard_routes") in table.listing()
