"""Dashboard pages: the app on its tab paths (built, or from Vite under --dev), /next redirecting to them, cache
headers, and the object memory page."""

import http.client
import http.server
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from liberational import config as cfg
from liberational.dashboard import assets, devices, server

INDEX = (
    "<!doctype html><html><head><script>window.NAMES={{names_json}};</script><title>{{assistant}}</title></head>"
    '<body><div id="root" data-owner="{{owner}}"></div>'
    '<script type="module" src="/static/app/assets/index-abc123.js"></script></body></html>'
)
VITE_CLIENT = '<script type="module" src="/static/app/@vite/client"></script>'
VITE_FILES = {
    "/static/app/index.html": ("text/html", INDEX.replace("<head>", "<head>" + VITE_CLIENT)),
    "/static/app/src/dashboard/main.tsx": ("text/javascript", "export const hot = 1"),
}


def get(base, path, headers=None):
    """(status, lowercased headers, body) of one GET."""
    c = http.client.HTTPConnection(base.removeprefix("http://"), timeout=5)
    c.request("GET", path, headers=headers or {})
    r = c.getresponse()
    out = r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    c.close()
    return out


def names(page):
    expression = page.split("window.NAMES=", 1)[1].split(";</script>", 1)[0]
    return json.loads(json.loads(expression.removeprefix("JSON.parse(").removesuffix(")")))


def test_shell_roundtrips_script_like_names_without_ending_its_script():
    owner = '</script><script>"synthetic owner"</script>'
    with cfg.override({"owner": {"name": owner}}):
        page = server.fill(INDEX)
    script = page.split("<script>", 1)[1].split("</script>", 1)[0]
    assert "<" not in script
    assert names(page)["owner"] == owner


def test_shell_uses_exact_native_camera_names(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(
        '[[device]]\nname="front-door"\nkind="rtsp"\n'
        '[[device]]\nname="CAM0"\nkind="camera"\n'
        '[[device]]\nname="mic"\nkind="microphone"\n'
    )
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    assert names(server.fill(INDEX))["cams"] == ["front-door", "CAM0"]


def test_shell_retains_disabled_camera_history_and_excludes_it_only_from_live(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(
        'device = [{name="front-door", kind="rtsp", enabled=false}, {name="desk", kind="camera"}]'
    )
    monkeypatch.setattr(server.cfg, "ROOT", tmp_path)
    data = names(server.fill(INDEX))
    assert data["cams"] == ["front-door", "desk"]
    assert data["activeCams"] == ["desk"]


def test_shell_has_no_phantom_camera_for_microphone_only_manifest(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text('[[device]]\nname="mic"\nkind="microphone"\n')
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    assert names(server.fill(INDEX))["cams"] == []


def test_shell_serializes_prototype_named_cameras_and_aliases_as_literal_keys(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to verify the shell's JavaScript boundary")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(
        'device=[{name="__proto__",kind="rtsp"},{name="constructor",kind="camera"},{name="toString",kind="camera"}]'
    )
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    with cfg.override({"vision": {"camera_names": ["__proto__: Front", "constructor: Rear", "toString: Side"]}}):
        script = server.fill(INDEX).split("<script>", 1)[1].split("</script>", 1)[0]
    result = subprocess.run(
        [node, "-e", "const window = {}; " + script + "; process.stdout.write(JSON.stringify(window.NAMES));"],
        capture_output=True,
        text=True,
        check=True,
    )
    shell = json.loads(result.stdout)
    assert shell["cams"] == ["__proto__", "constructor", "toString"]
    assert shell["views"] == {"__proto__": "Front", "constructor": "Rear", "toString": "Side"}


def test_shell_has_no_cameras_without_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    assert names(server.fill(INDEX))["cams"] == []


class Tree:
    """A temp static/ (the build) and frontend/ (its source)."""

    def __init__(self, root):
        self.static, self.app, self.frontend = root / "static", root / "static" / "app", root / "frontend"
        self.manifest = self.app / ".vite" / "manifest.json"
        for d in (self.app / ".vite", self.app / "assets", self.frontend / "src"):
            d.mkdir(parents=True)

    def build(self, at=1_000_000_000):
        (self.app / "index.html").write_text(INDEX)
        self.manifest.write_text("{}")
        os.utime(self.manifest, (at, at))


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Keep config defaults and import-time paths temporary until the HTTP servers have stopped."""
    state = tmp_path / "state"
    calls = tmp_path / "calls"
    for name, value in {
        "STATE": state,
        "CALLS_DIR": calls,
        "PEOPLE_ADDED": state / "people_added.json",
        "CONTACTS_FILE": state / "contacts.json",
        "INBOX_SKIPS": state / "inbox_skips.json",
    }.items():
        monkeypatch.setattr(server, name, value)
    with cfg.override(
        {
            "paths": {
                "state": str(state),
                "calls": str(calls),
                "faces": str(tmp_path / "faces"),
                "voices": str(tmp_path / "voices"),
            }
        }
    ):
        yield


@pytest.fixture
def tree(tmp_path, monkeypatch):
    t = Tree(tmp_path)
    for name, value in {"STATIC": t.static, "APP": t.app, "MANIFEST": t.manifest, "FRONTEND": t.frontend}.items():
        monkeypatch.setattr(assets, name, value)
    with cfg.override({"owner": {"name": "Alice & Co"}, "assistant": {"name": "Pax"}}):
        yield t


@pytest.fixture
def people_api_broker(request, monkeypatch):
    """Skip on the test thread, rather than discovering an absent or stale broker from an HTTP handler."""
    binary = os.environ.get("CONTINUUM_BROKER")
    if not binary or not Path(binary).is_file():
        pytest.skip("set CONTINUUM_BROKER to the current built broker for the People API route")
    socket = request.getfixturevalue("broker")
    monkeypatch.setenv("CONTINUUM_SOCKET", str(socket))


@pytest.fixture
def dashboard(serve, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    return serve(server.H)


def test_the_page_says_whether_hotkeys_are_on(tree):
    # Issue #64: dashboard.hotkeys, off by default, reaches the page as NAMES.hotkeys.
    tree.build()
    assert names(server.app_page()[1].decode())["hotkeys"] is False
    with cfg.override({"dashboard": {"hotkeys": True}}):
        assert names(server.app_page()[1].decode())["hotkeys"] is True


@pytest.mark.parametrize("path", server.APP_PATHS)
def test_tab_paths_serve_the_built_app_filled_in(tree, dashboard, path):
    tree.build()
    status, headers, body = get(dashboard, path, {"Accept": "text/html,*/*"})
    page = body.decode()
    assert (status, headers["content-type"], headers["cache-control"]) == (200, "text/html; charset=utf-8", "no-store")
    assert '<script type="module" src="/static/app/assets/index-abc123.js">' in page
    assert "<title>Pax</title>" in page
    assert 'data-owner="Alice &amp; Co"' in page
    assert (names(page)["owner"], names(page)["assistant"]) == ("Alice & Co", "Pax")
    assert "{{" not in page


def test_a_fetch_of_a_tab_path_gets_the_api(dashboard, monkeypatch, people_api_broker):
    from liberational.people import face_photos

    # This route test needs the real empty read path, without a detached photo worker outliving its settings.
    monkeypatch.setattr(face_photos, "refresh_async", face_photos.refresh)
    monkeypatch.setattr(server, "_faces_mod", lambda: SimpleNamespace(counts=dict, unknown_groups=list))
    _, headers, body = get(dashboard, "/people", {"Accept": "*/*"})  # fetch() from the People tab
    assert headers["content-type"] == "application/json"
    assert json.loads(body)["people"] == []


@pytest.mark.parametrize("accept", ["application/json", "*/*"])
def test_calls_tab_serves_html_navigation_and_preserves_the_json_api(tree, dashboard, accept):
    tree.build()
    status, headers, body = get(dashboard, "/calls", {"Accept": "text/html,*/*"})
    assert (status, headers["content-type"]) == (200, "text/html; charset=utf-8")
    assert b'<script type="module" src="/static/app/assets/index-abc123.js">' in body
    assert names(body.decode())["owner"] == "Alice & Co"

    status, headers, body = get(dashboard, "/calls", {"Accept": accept})
    assert (status, headers["content-type"], json.loads(body)) == (200, "application/json", [])


@pytest.mark.parametrize("accept", ["application/json", "*/*"])
def test_devices_tab_reloads_as_the_app_and_keeps_the_json_api(tree, dashboard, monkeypatch, accept):
    # #2816: devices.handle answered GET /devices before the page check, so reloading the Devices tab got JSON.
    tree.build()
    seen = {"control": False, "mics": [], "cameras": []}
    monkeypatch.setattr(devices, "view", create_autospec(devices.view, return_value=seen))  # it reads the hardware
    status, headers, body = get(dashboard, "/devices", {"Accept": "text/html,*/*"})
    assert (status, headers["content-type"]) == (200, "text/html; charset=utf-8")
    assert b'<script type="module" src="/static/app/assets/index-abc123.js">' in body

    status, headers, body = get(dashboard, "/devices", {"Accept": accept})
    assert (status, headers["content-type"], json.loads(body)) == (200, "application/json", seen)


def test_without_a_build_the_page_says_how_to_build_it(tree, dashboard):
    status, headers, body = get(dashboard, "/people", {"Accept": "text/html"})
    assert (status, headers["content-type"]) == (503, "text/html; charset=utf-8")
    assert b"mise run setup" in body


def test_a_build_older_than_the_frontend_says_so(tree, dashboard):
    tree.build(at=1_000_000_000)
    (tree.frontend / "src" / "main.tsx").write_text("// edited after the build")
    assert b"mise run frontend" in get(dashboard, "/")[2]
    os.utime(tree.manifest)  # rebuilt: now newer than every source
    assert b"mise run frontend" not in get(dashboard, "/")[2]


@pytest.mark.parametrize(
    ("path", "ctype", "cache"),
    [
        ("/static/app/assets/index-abc123.js", "text/javascript; charset=utf-8", assets.IMMUTABLE),
        ("/static/app/assets/index-abc123.js.map", "application/json; charset=utf-8", assets.IMMUTABLE),
        ("/static/app/stray.js", "text/javascript; charset=utf-8", "no-store"),
    ],
    ids=["hashed-js", "hashed-map", "unhashed"],
)
def test_only_hashed_assets_are_cached_for_good(tree, dashboard, path, ctype, cache):
    (tree.static / path.removeprefix("/static/")).write_text("export {}")
    status, headers, _ = get(dashboard, path)
    assert (status, headers["content-type"], headers["cache-control"]) == (200, ctype, cache)


def test_a_page_outside_the_build_is_never_read(tree):
    (tree.static / "secret.html").write_text("<p>not a page of the build</p>")
    assert assets.built_page("../secret.html") == (assets.NOT_BUILT.format(page="../secret.html"), False)


@pytest.fixture
def vite(serve, monkeypatch):
    """--dev with a stand-in for Vite's dev server; yields its port and the (path, Host) of every request."""
    seen = []

    class Vite(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            seen.append((self.path, self.headers["Host"]))
            ctype, body = VITE_FILES.get(self.path, ("text/plain", ""))
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

    port = int(serve(Vite).rsplit(":", 1)[1])
    monkeypatch.setattr(server, "DEV", True)
    with cfg.override({"ports": {"vite": port}}):
        yield port, seen


def test_in_dev_the_page_is_vites_filled_in(tree, dashboard, vite):
    port, seen = vite
    status, _, body = get(dashboard, "/review", {"Accept": "text/html"})
    page = body.decode()
    assert status == 200
    assert VITE_CLIENT in page  # Vite's client and refresh preamble, as it served them
    assert "<title>Pax</title>" in page
    assert ("/static/app/index.html", f"localhost:{port}") in seen  # the Host Vite's check expects: its own


def test_in_dev_the_frontends_modules_come_from_vite_through_the_dashboard(tree, dashboard, vite):
    status, headers, body = get(dashboard, "/static/app/src/dashboard/main.tsx")
    assert (status, body, headers["cache-control"]) == (200, b"export const hot = 1", "no-store")


def test_in_dev_without_vite_the_page_is_the_last_build(tree, dashboard, monkeypatch, capsys):
    tree.build()
    monkeypatch.setattr(server, "DEV", True)
    with socket.socket() as closed:  # bound but not listening, so a connection to it is refused
        closed.bind(("127.0.0.1", 0))
        with cfg.override({"ports": {"vite": closed.getsockname()[1]}}):
            status, _, body = get(dashboard, "/")
    assert status == 200
    assert b"<title>Pax</title>" in body
    assert b"@vite/client" not in body
    assert "mise run dev" in capsys.readouterr().err


def test_objects_is_the_built_page_from_the_dashboard_or_says_how_to_build_it(tree, dashboard):
    # The Objects page is the dashboard's own since object memory moved to Continuum (#778), and it is a tab of the
    # one app page, so it needs the app built like every other tab.
    status, _, body = get(dashboard, "/objects")
    assert status == 503
    assert b"mise run setup" in body
    tree.build()
    status, _, body = get(dashboard, "/objects/")
    assert status == 200
    assert "<title>Pax</title>" in body.decode()


def test_the_page_omits_the_retired_room_setting(tree):
    tree.build()
    with cfg.override({"vision": {"room": "old room"}}):
        assert "room" not in names(server.app_page()[1].decode())


def test_every_tab_the_app_routes_to_is_served_on_a_reload():
    # The app's tabs are real paths (frontend router.ts): a tab missing from APP_PATHS loads as a blank 404 page.
    import re

    router = (cfg.ROOT / "frontend" / "src" / "dashboard" / "router.ts").read_text()
    tabs = set(re.findall(r"\{ id: '\w+', path: '(/[\w-]*)'", router))
    assert tabs and "/inbox" in tabs
    assert tabs - {"/objects"} <= set(server.APP_PATHS)  # /objects has a page handler of its own
