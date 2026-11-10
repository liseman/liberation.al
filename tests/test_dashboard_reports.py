"""The Reports tab's API (dashboard/reports.py), served by the real dashboard handler over a fake Sources."""

import http.client
import json
import urllib.parse

import pytest

from liberational.dashboard import reports as page
from liberational.dashboard import server
from liberational.memory import reports
from test_reports import FakeSources


@pytest.fixture
def base(serve, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    view = reports.Reports(reports.Store(tmp_path / "reports.sqlite3"), FakeSources(), background=False)
    monkeypatch.setattr(reports, "at", lambda state: view)
    return serve(server.H)


def request(base, method, path, body=None, headers=None):
    u = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    data = json.dumps(body).encode() if body is not None else b""
    conn.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
    r = conn.getresponse()
    out = r.read()
    conn.close()
    return r.status, json.loads(out) if out else None


CHOSEN = [{"id": "conversations", "kind": "conversations", "arg": None, "title": "Conversations"}]


def test_layout_saved_from_the_page_is_what_every_later_read_gets(base):
    status, _ = request(base, "POST", "/reports/api/layout", {"reports": CHOSEN})
    assert status == 200
    status, got = request(base, "GET", "/reports/api/layout")
    assert got["reports"] == CHOSEN


def test_layout_refuses_a_cross_site_page(base):
    status, _ = request(base, "POST", "/reports/api/layout", {"reports": CHOSEN}, {"Origin": "http://evil.example"})
    assert status == 403
    _, got = request(base, "GET", "/reports/api/layout")
    assert [r["kind"] for r in got["reports"]] == ["whos_around", "conversations", "activity", "health"]
    assert got["built_in"] == ["whos_around", "conversations", "activity", "health"]
    assert got["kinds"]["walkbys"] == "Walk-bys by camera"


def test_layout_refuses_an_unknown_kind(base):
    status, body = request(base, "POST", "/reports/api/layout", {"reports": [{**CHOSEN[0], "kind": "nope"}]})
    assert status == 400 and "nope" in body["error"]


def test_report_of_an_unknown_kind_is_not_found(base):
    status, _ = request(base, "GET", "/reports/api/report?kind=nope")
    assert status == 404


def test_report_says_which_days_are_still_being_read(base):
    status, body = request(base, "GET", "/reports/api/report?kind=whos_around")
    assert status == 200
    assert body["pending"] == [body["last"]] and body["days"] == []


@pytest.mark.parametrize("kind,field", [("walkbys", "cameras"), ("person_activity", "people"), ("devices", "sources")])
def test_suggested_report_kinds_can_be_read_and_saved_without_becoming_built_ins(base, kind, field):
    status, body = request(base, "GET", f"/reports/api/report?kind={kind}")
    assert status == 200
    assert body[field] == []
    chosen = [{"id": "mine", "kind": kind, "arg": None, "title": "My report"}]
    status, saved = request(base, "POST", "/reports/api/layout", {"reports": chosen})
    assert status == 200
    assert saved["reports"] == chosen
    assert kind not in saved["built_in"]


def card(rid, kind, arg=None, title="A report", **extra):
    return {"id": rid, "kind": kind, "arg": arg, "title": title, **extra}


def test_saved_per_person_whos_around_cards_fold_into_the_filter_in_page_order(tmp_path):
    store = reports.Store(tmp_path / "reports.sqlite3")
    store.set_meta(
        "layout",
        {
            "reports": [
                card("whos_around", "whos_around"),
                card("conversations", "conversations"),
                card("whos_around:p-sam", "whos_around", "p-sam", "When Sam is around"),
                card("person_activity:p-sam", "person_activity", "p-sam"),
                card("whos_around:p-carly", "whos_around", "p-carly", "When Carly is around"),
            ],
            "known": ["whos_around", "conversations", "activity", "health"],
        },
    )
    got = page.layout(store)
    # Sam's activity card folds too, into an Activity by person card in its place.
    assert [r["id"] for r in got] == ["whos_around", "conversations", "person_activity"]
    assert got[0]["people"] == ["p-sam", "p-carly"] and got[0]["arg"] is None
    assert got[2] == card("person_activity", "person_activity", title="Activity by person", people=["p-sam"])


def test_a_person_card_kept_after_removing_whos_around_brings_it_back_in_that_place(tmp_path):
    store = reports.Store(tmp_path / "reports.sqlite3")
    page.save_layout(
        store,
        [
            card("conversations", "conversations"),
            card("whos_around:p-sam", "whos_around", "p-sam", "When Sam is around"),
        ],
    )
    got = page.layout(store)
    assert got == [
        card("conversations", "conversations"),
        card("whos_around", "whos_around", title="Who's around", people=["p-sam"]),
    ]


def test_the_filters_people_are_saved_and_read_back(base):
    chosen = [card("whos_around", "whos_around", people=["p-carly", "p-sam", "p-carly"])]
    status, saved = request(base, "POST", "/reports/api/layout", {"reports": chosen})
    assert status == 200
    _, got = request(base, "GET", "/reports/api/layout")
    assert got["reports"] == saved["reports"] == [card("whos_around", "whos_around", people=["p-carly", "p-sam"])]


@pytest.mark.parametrize(
    "people",
    [pytest.param("p-sam", id="not-a-list"), pytest.param([3], id="not-strings"), pytest.param([""], id="empty")],
)
def test_layout_refuses_malformed_people(base, people):
    status, _ = request(base, "POST", "/reports/api/layout", {"reports": [card("w", "whos_around", people=people)]})
    assert status == 400


def test_the_people_cap_counts_people_not_repeats(base):
    chosen = [card("whos_around", "whos_around", people=["p-sam"] * 30 + [f"p-{i}" for i in range(19)])]
    status, saved = request(base, "POST", "/reports/api/layout", {"reports": chosen})
    assert status == 200
    assert len(saved["reports"][0]["people"]) == 20


def test_layout_refuses_more_than_twenty_people(base):
    chosen = [card("whos_around", "whos_around", people=[f"p-{i}" for i in range(21)])]
    status, _ = request(base, "POST", "/reports/api/layout", {"reports": chosen})
    assert status == 400


def test_removing_whos_around_and_adding_it_back_keeps_its_pins(tmp_path):
    store = reports.Store(tmp_path / "reports.sqlite3")
    talk = card("conversations", "conversations")
    page.save_layout(store, [card("whos_around", "whos_around", people=["p-sam", "p-carly"]), talk])
    page.save_layout(store, [talk])
    back = page.save_layout(store, [talk, card("whos_around", "whos_around", title="Who's around")])
    assert back[1]["people"] == ["p-sam", "p-carly"]
    # Unpinned everyone, then off and back: nothing comes back, since the card had no pins when it went.
    page.save_layout(store, [talk, card("whos_around", "whos_around", title="Who's around")])
    page.save_layout(store, [talk])
    assert "people" not in page.save_layout(store, [talk, card("whos_around", "whos_around")])[1]


def test_camera_walkby_cards_fold_into_one_card_with_the_cameras_pinned(tmp_path):
    store = reports.Store(tmp_path / "reports.sqlite3")
    page.save_layout(
        store,
        [
            card("walkbys:living-room", "walkbys", "living-room", "Walk-bys on living-room"),
            card("conversations", "conversations"),
            card("walkbys:driveway", "walkbys", "driveway", "Walk-bys on driveway"),
        ],
    )
    assert page.layout(store) == [
        card("walkbys", "walkbys", title="Walk-bys by camera", cameras=["living-room", "driveway"]),
        card("conversations", "conversations"),
    ]


def test_removing_activity_by_person_and_adding_it_back_keeps_its_pins(tmp_path):
    store = reports.Store(tmp_path / "reports.sqlite3")
    talk = card("conversations", "conversations")
    page.save_layout(store, [card("person_activity", "person_activity", people=["p-sam"]), talk])
    page.save_layout(store, [talk])
    back = page.save_layout(store, [talk, card("person_activity", "person_activity")])
    assert back[1]["people"] == ["p-sam"]
