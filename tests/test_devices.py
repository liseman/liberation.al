"""Devices linked to people and things: links on the broker, stays, things' last sightings, presence
candidates, and Ask's answers. Every sighting here is synthetic: no board has recorded a real one yet."""

import threading
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest

import test_network_worker as net
from liberational import config as cfg
from liberational import segment_log
from liberational.dashboard import device_links, server
from liberational.memory import device_answers, places
from liberational.memory import history_qa as h
from liberational.proximity import ble, devices, vendors
from liberational.proximity.network import NetworkDevice
from liberational.proximity.network_worker import INVENTORY, OBSERVATIONS, NetworkWorker

NOW = 1_800_000_000.0
PHONE = "aa:bb:cc:dd:ee:01"  # Sam's phone on the home Wi-Fi
WATCH = "C0:FF:EE:00:00:01"  # Joe's watch, a static Bluetooth address
AIRPODS_DATA = {"76": "0719010f2022f58f01000045f2a1b3c4"}  # an exact payload, stable across address rotations


def sighting(at: float, identifier: str, *, sensor: str = "Desk", rssi: int = -60, data: dict | None = None) -> dict:
    return {
        "id": str(uuid.uuid7()),
        "start": at,
        "end": at + 5,
        "backend": "esp32",
        "adapter": "a1b2c3d4e5f6",
        "identifier_kind": "address",
        "identifier": identifier,
        "address_type": "random",
        "reports": 4,
        "rssi": {"samples": 4, "min": rssi - 5, "max": rssi, "mean": rssi - 2},
        "manufacturer_data": data or {},
        "service_data": {},
        "sensor_name": sensor,
        "payload_truncated": False,
        "payload_malformed": False,
        "payload_changes": 0,
    }


def coverage(at: float) -> dict:
    return {"id": str(uuid.uuid7()), "start": at, "end": at, "absence_is_not_inferred": True}


def network(at: float, mac: str, connected: bool | None = True) -> dict:
    return {
        "id": str(uuid.uuid7()),
        "start": at,
        "end": at,
        "mac": mac.upper(),  # the router's canonical form
        "hostname": "sams-phone",
        "ap_name": "Hall",
        "connected": connected,
    }


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reg = devices.Devices(segments=tmp_path / "segments", knowledge_db=tmp_path / "knowledge.sqlite3")
    monkeypatch.setattr(devices, "at", lambda state=None: reg)

    def record(stream: str, rows: list[dict[str, Any]]) -> None:
        """Rows written where their workers write them: the segment store."""
        writers = segment_log.Writers(reg.segments)
        if stream == devices.NET_OBSERVATIONS:
            filled = [
                {
                    "inventory_id": str(uuid.uuid7()),
                    "observer": "home",
                    "identifier_kind": "mac_address",
                    "identifier": r["mac"],
                    **r,
                }
                for r in rows
            ]
            writers.append(stream, "unifi", filled)
        else:
            writers.append(stream, "ble", rows)
        writers.close()

    # Questions route by who the configured state's registry knows ("is Dana home?"): its own, not the run's shared
    # one, where an earlier test's Dana made this one fail only when that test ran first.
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        yield reg, record


def test_a_link_is_revised_in_place_and_retired_without_touching_sightings(home):
    reg, _ = home
    first = reg.save("ble_identifier", WATCH, "person", "Joe")
    again = reg.save("ble_identifier", WATCH, "person", "Jordan")
    assert again.id == first.id and [x.name for x in reg.links()] == ["Jordan"], "one link per device, revised"
    reg.remove(first.id)
    assert reg.links() == []
    with pytest.raises(ValueError, match="12 hex digits"):
        reg.save("net_mac", "aa:bb", "person", "Sam")


def test_sightings_a_gap_apart_are_separate_stays():
    rows = [(NOW + t, NOW + t + 5, f"s{t}", "Desk" if t < 500 else "Kitchen") for t in (0, 300, 550, 2000)]
    found = devices.stays(rows, gap=600)
    assert [(s.start, s.end, s.places, s.sightings) for s in found] == [
        (NOW, NOW + 555, ("Desk", "Kitchen"), 3),
        (NOW + 2000, NOW + 2005, ("Kitchen",), 1),
    ]


def test_a_rotating_device_matches_by_its_payload_and_a_phone_by_its_mac():
    fingerprint = devices.ble.payload_fingerprint(sighting(NOW, "x", data=AIRPODS_DATA))
    assert fingerprint is not None
    pods = devices.Link("l1", "ble_fingerprint", fingerprint, "thing", "AirPods", "e1")
    assert devices.matches(pods, sighting(NOW, "11:22:33:44:55:66", data=AIRPODS_DATA), devices.BLE_SIGHTINGS)
    assert not devices.matches(pods, sighting(NOW, "11:22:33:44:55:66"), devices.BLE_SIGHTINGS)
    phone = devices.Link("l2", "net_mac", PHONE, "person", "Sam", "e2")
    assert devices.matches(phone, network(NOW, PHONE.upper()), devices.NET_OBSERVATIONS)


def ask(state: Path, question: str, now: float) -> dict[str, Any]:
    return h.collect(question, state, now)


def test_is_joe_home_reads_his_devices_and_says_when_it_cannot_tell(home, tmp_path):
    reg, record = home
    reg.save("ble_identifier", WATCH, "person", "Joe")
    now = NOW + 7200
    record(devices.BLE_SIGHTINGS, [sighting(now - 300, WATCH, sensor="Kitchen")])
    record(devices.BLE_COVERAGE, [coverage(now - 900), coverage(now - 30)])
    reply = ask(tmp_path, "is Joe home?", now)
    assert reply["paths"] == ["devices"]
    assert reply["answer"].startswith("Probably: Joe's device was seen 5 min ago by Kitchen, signal -60 dBm.")
    later = now + 3600  # no sighting and no sensor running since: unknown, not "left"
    assert ask(tmp_path, "is Joe home?", later)["answer"].startswith("I can't tell")
    record(devices.BLE_COVERAGE, [coverage(later - 10)])
    # Passive radio operation cannot certify that a quiet or rotating advertiser is absent.
    assert ask(tmp_path, "is Joe home?", later)["answer"].startswith("I can't tell")


def test_when_did_sam_get_here_reads_todays_stays(home, tmp_path):
    reg, record = home
    reg.save("net_mac", PHONE, "person", "Sam")
    morning = NOW - (NOW % 86400) + 15 * 3600  # some time today
    record(devices.NET_OBSERVATIONS, [network(morning + t, PHONE) for t in (0, 120, 240)])
    reply = ask(tmp_path, "when did Sam get here?", morning + 300)
    assert reply["paths"] == ["devices"]
    assert "Sam's device arrived at" in reply["answer"] and "heard by Hall" in reply["answer"]
    assert "not proof" in reply["answer"]


def test_where_did_i_last_have_my_airpods_names_the_sensor_and_signal(home, tmp_path):
    reg, record = home
    fingerprint = devices.ble.payload_fingerprint(sighting(NOW, "x", data=AIRPODS_DATA))
    reg.save("ble_fingerprint", str(fingerprint), "thing", "AirPods")
    record(
        devices.BLE_SIGHTINGS,
        [sighting(NOW - 3 * 86400, "11:00:00:00:00:01", sensor="Car", rssi=-71, data=AIRPODS_DATA)],
    )
    reply = ask(tmp_path, "where did I last have my AirPods?", NOW)
    assert reply["paths"] == ["devices"]
    assert "Your AirPods was last seen at" in reply["answer"] and "by Car, signal -71 dBm" in reply["answer"]


def test_unknown_devices_are_kept_and_counted(home):
    reg, record = home
    reg.save("ble_identifier", WATCH, "person", "Joe")
    record(
        devices.BLE_SIGHTINGS,
        [sighting(NOW, WATCH), sighting(NOW, "de:ad:be:ef:00:01"), sighting(NOW, "de:ad:be:ef:00:02")],
    )
    found = reg.summary(NOW - 60, NOW + 60)
    assert (found["total"], found["linked"], found["unknown"]) == (3, 1, 2)


def test_a_summary_row_says_how_to_link_it_and_which_link_it_has(home):
    reg, record = home
    joe = reg.save("ble_identifier", WATCH, "person", "Joe")
    keys = reg.save("ble_name", "Keys", "thing", "Keys")  # not heard in the window: still listed
    record(devices.BLE_SIGHTINGS, [sighting(NOW, WATCH, sensor="Hall", rssi=-48, data=AIRPODS_DATA)])
    record(devices.NET_OBSERVATIONS, [network(NOW, PHONE)])
    found = reg.summary(NOW - 60, NOW + 60)
    by = {d["source"]: d for d in found["devices"]}
    watch, phone = by[devices.BLE_SIGHTINGS], by[devices.NET_OBSERVATIONS]
    assert (watch["linked"], watch["place"], watch["rssi"]) == (
        {"id": joe.id, "target": "person", "name": "Joe"},
        "Hall",
        -48.0,
    )
    assert [s["kind"] for s in watch["selectors"]] == ["ble_identifier", "ble_fingerprint"]
    assert (phone["linked"], phone["selectors"]) == (None, [{"kind": "net_mac", "value": PHONE}])
    assert sorted(x["id"] for x in found["links"]) == sorted([joe.id, keys.id])


def test_a_summary_row_and_a_wifi_device_say_who_made_them(home, monkeypatch):
    """Known › Devices names a device nobody named by its maker: the Bluetooth company in its payload, the
    network address's IEEE block."""
    reg, record = home
    known = vendors.Registry(prefixes={6: {"001122": "Acme Radios"}}, companies={0x004C: "Apple, Inc."})
    monkeypatch.setattr(vendors, "registry", lambda: known)
    record(devices.BLE_SIGHTINGS, [sighting(NOW, "de:ad:be:ef:00:01", data=AIRPODS_DATA)])
    record(devices.NET_OBSERVATIONS, [network(NOW, "00:11:22:dd:ee:01")])  # a public address, unlike PHONE's
    by = {d["source"]: d for d in reg.summary(NOW - 60, NOW + 60)["devices"]}
    assert (by[devices.BLE_SIGHTINGS]["vendor"], by[devices.BLE_SIGHTINGS]["kind"]) == ("Apple", "AirPods or Beats")
    assert (by[devices.NET_OBSERVATIONS]["vendor"], by[devices.NET_OBSERVATIONS]["kind"]) == ("Acme Radios", None)
    assert [d["vendor"] for d in reg.comings(NOW - 60, NOW + 60)["devices"]] == ["Acme Radios"]


PLAN = 'plan "Home" walls 0.2/0.1\nroom r1 "Living room" other rect 0,0 4x6\nroom r2 "Kitchen" kitchen rect 4,0 3x3\n'


def a_place(state: Path) -> str:
    """A place with a two-room plan, as a walkthrough makes one."""
    return places.revise(state, None, producer="footprints-v1", plan=PLAN, now=NOW - 86400).place


def test_a_located_device_keeps_its_thing_link_and_lists_where_it_lives(home, tmp_path):
    """A device that lives in a room: a location beside its person or thing link, not instead of it."""
    reg, record = home
    tv = reg.save("ble_identifier", WATCH, "thing", "Living room TV")
    home_id = a_place(tmp_path)
    where = reg.locate("ble_identifier", WATCH, home_id, "Living room")
    record(devices.BLE_SIGHTINGS, [sighting(NOW, WATCH)])
    found = reg.summary(NOW - 60, NOW + 60)
    [row] = found["devices"]
    assert row["linked"] == {"id": tv.id, "target": "thing", "name": "Living room TV"}
    assert row["location"] == {"id": where.id, "place_id": home_id, "place": "Place 1", "room": "Living room"}
    assert [(x["id"], x["room"]) for x in found["locations"]] == [(where.id, "Living room")]
    assert [x.name for x in reg.links()] == ["Living room TV"]  # a location is no person or thing


def test_moving_a_device_revises_its_location_and_unlinking_retires_it(home, tmp_path):
    reg, record = home
    home_id = a_place(tmp_path)
    first = reg.locate("net_mac", PHONE, home_id, "Living room")
    moved = reg.locate("net_mac", PHONE.upper(), home_id, "Kitchen")
    assert moved.id == first.id
    assert [(x.room, x.value) for x in reg.locations()] == [("Kitchen", PHONE)]
    record(devices.NET_OBSERVATIONS, [network(NOW, PHONE)])
    [wifi] = reg.comings(NOW - 60, NOW + 60)["devices"]  # the Wi-Fi card says where it lives too
    assert wifi["location"] == {"id": first.id, "place_id": home_id, "place": "Place 1", "room": "Kitchen"}
    reg.unlocate(first.id)
    assert reg.locations() == []
    with pytest.raises(LookupError):
        reg.unlocate(first.id)


@pytest.mark.parametrize(
    ("room", "error"),
    [
        pytest.param("Garage", "no room", id="room-not-in-the-plan"),
        pytest.param("", None, id="the-whole-place"),
    ],
)
def test_a_location_must_be_a_place_and_one_of_its_rooms(home, tmp_path, room, error):
    reg, _ = home
    home_id = a_place(tmp_path)
    if error:
        with pytest.raises(ValueError, match=error):
            reg.locate("net_mac", PHONE, home_id, room)
    else:
        assert reg.locate("net_mac", PHONE, home_id, room).room is None
    with pytest.raises(ValueError, match="no place"):
        reg.locate("net_mac", PHONE, "01a1b7de-24d2-74d4-9f84-29cfbbc7d147", "")


def test_the_places_a_device_can_live_in_are_each_place_with_its_plans_rooms(home, tmp_path):
    home_id = a_place(tmp_path)
    places.name(tmp_path, home_id, "Home")
    assert devices.location_options(tmp_path) == [{"id": home_id, "name": "Home", "rooms": ["Living room", "Kitchen"]}]


def test_ask_says_where_a_located_thing_lives(home, tmp_path):
    reg, record = home
    reg.save("ble_identifier", WATCH, "thing", "TV")
    reg.locate("ble_identifier", WATCH, a_place(tmp_path), "Living room")
    record(devices.BLE_SIGHTINGS, [sighting(NOW - 60, WATCH)])
    reply = ask(tmp_path, "where is my TV?", NOW)
    assert "Your TV lives in the Living room (Place 1)" in reply["answer"]


@pytest.fixture
def http_(home, serve, monkeypatch, tmp_path):
    import http.client
    import json
    import urllib.parse

    from liberational.dashboard import server

    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    monkeypatch.setattr(device_links, "_held", {})  # an earlier test's listing, reused within the minute
    netloc = urllib.parse.urlsplit(serve(server.H)).netloc

    def send(method, path, body=None, origin=None):
        conn = http.client.HTTPConnection(netloc, timeout=10)
        headers = {"Content-Type": "application/json", **({"Origin": origin} if origin else {})}
        conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        response = conn.getresponse()
        value = response.status, json.loads(response.read())
        conn.close()
        return value

    return send


def test_the_dashboard_links_lists_and_unlinks_a_device(http_):
    code, made = http_(
        "POST", "/device-links", {"kind": "net_mac", "value": PHONE.upper(), "target": "person", "name": "Sam"}
    )
    assert (code, made["link"]["value"]) == (200, PHONE)
    code, listing = http_("GET", "/device-links")
    assert (code, [x["name"] for x in listing["links"]], listing["hours"]) == (200, ["Sam"], 24.0)
    code, _ = http_("POST", "/device-links/remove", {"id": made["link"]["id"]})
    assert code == 200
    assert http_("GET", "/device-links")[1]["links"] == []


def test_the_dashboard_locates_a_device_and_removes_its_location(http_, tmp_path):
    home_id = a_place(tmp_path)
    code, listing = http_("GET", "/device-links")
    assert (code, listing["places"]) == (200, [{"id": home_id, "name": "Place 1", "rooms": ["Living room", "Kitchen"]}])
    code, made = http_(
        "POST", "/device-locations", {"kind": "net_mac", "value": PHONE, "place": home_id, "room": "Kitchen"}
    )
    assert (code, made["location"]["room"], made["location"]["place"]) == (200, "Kitchen", "Place 1")
    assert [x["room"] for x in http_("GET", "/device-links")[1]["locations"]] == ["Kitchen"]
    code, _ = http_("POST", "/device-locations/remove", {"id": made["location"]["id"]})
    assert code == 200
    assert http_("GET", "/device-links")[1]["locations"] == []
    code, refused = http_(
        "POST", "/device-locations", {"kind": "net_mac", "value": PHONE, "place": home_id, "room": "Attic"}
    )
    assert code == 400 and "no room" in refused["error"]


@pytest.mark.parametrize(
    "method, path, body, code",
    [
        pytest.param(
            "POST",
            "/device-links",
            {"kind": "net_mac", "value": "aa:bb", "target": "person", "name": "Sam"},
            400,
            id="bad-mac",
        ),
        pytest.param(
            "POST",
            "/device-links",
            {"kind": "net_mac", "value": PHONE, "target": "pet", "name": "Rex"},
            400,
            id="bad-target",
        ),
        pytest.param("POST", "/device-links/remove", {"id": str(uuid.uuid4())}, 404, id="no-such-link"),
        pytest.param("GET", "/device-links?hours=0", None, 400, id="no-window"),
    ],
)
def test_the_dashboard_refuses_a_bad_device_link_request(http_, method, path, body, code):
    assert http_(method, path, body)[0] == code


def test_a_device_link_from_another_site_is_refused(http_):
    body = {"kind": "net_mac", "value": PHONE, "target": "person", "name": "Sam"}
    assert http_("POST", "/device-links", body, origin="http://evil.example")[0] == 403
    assert http_("GET", "/device-links")[1]["links"] == []


def test_voice_sends_a_linked_device_question_to_the_history_answer(home):
    reg, _ = home
    reg.save("net_mac", PHONE, "person", "Sam")
    assert h.is_history_question("is Sam home?")
    assert not h.is_history_question("is Dana home?")


def test_a_question_about_someone_with_no_linked_device_goes_its_usual_way(home, tmp_path):
    assert device_answers.asked("is Dana home?", tmp_path) is None
    assert device_answers.asked("where is the meeting?", tmp_path) is None


def test_a_router_cache_row_is_history_not_presence():
    phone = devices.Link("l2", "net_mac", PHONE, "person", "Sam", "e2")
    rows = {devices.NET_OBSERVATIONS: [network(NOW, PHONE, connected=None), network(NOW + 60, PHONE, connected=False)]}
    assert devices.seen([phone], rows, [], gap=600).stays == ()
    rows[devices.NET_OBSERVATIONS].append(network(NOW + 120, PHONE))
    assert [s.start for s in devices.seen([phone], rows, [], gap=600).stays] == [NOW + 120]


def test_the_network_workers_own_records_drive_a_stay_and_its_sensing():
    """The worker and this reader agree on the record shape: a field renamed on either side fails here."""

    def run(at: float, devices_: list) -> list:
        sink = net.Sink()
        inventory = net.snapshot(at=at, devices=devices_)
        NetworkWorker(net.Source(inventory), sink).process(inventory)
        return [{**r, "stream": s} for s, _, r in sink.rows]

    phone = NetworkDevice(
        "mac_address", PHONE.upper(), mac=PHONE.upper(), connected=True, hostname="sams-phone", access_point_name="Hall"
    )
    cached = NetworkDevice("mac_address", "02:00:00:00:00:09", mac="02:00:00:00:00:09", connected=None)
    # A poll holding a cached row (connection unknown) claims no coverage, so the router polls are separate.
    rows = run(NOW, [phone]) + run(NOW + 60, [phone]) + run(NOW + 120, [cached])
    observations = [r for r in rows if r["stream"] == OBSERVATIONS]
    inventory = [r for r in rows if r["stream"] == INVENTORY and r.get("complete") and r.get("authoritative")]
    assert inventory, "complete, authoritative polls are the network's sensing"
    link = devices.Link("l1", "net_mac", PHONE, "person", "Sam", "e1")
    found = devices.seen([link], {devices.NET_OBSERVATIONS: observations}, inventory, gap=600)
    assert [(s.start, s.end, s.places) for s in found.stays] == [(NOW, NOW + 60, ("Hall",))]
    assert found.sensing, "the polls say the router was being read"
    other = devices.Link("l2", "net_mac", "02:00:00:00:00:09", "person", "Dana", "e2")
    assert devices.seen([other], {devices.NET_OBSERVATIONS: observations}, inventory, gap=600).stays == ()


def test_links_live_in_the_knowledge_store_and_outlast_the_broker(tmp_path):
    db = tmp_path / "knowledge.sqlite3"
    first = devices.Devices(segments=tmp_path / "segments", knowledge_db=db)
    assert first.links() == []
    made = first.save("net_mac", PHONE, "person", "Sam")
    # Another process, after the broker's catalog was reset: the link is still there.
    again = devices.Devices(segments=tmp_path / "segments", knowledge_db=db)
    assert [(x.id, x.value, x.name) for x in again.links()] == [(made.id, PHONE, "Sam")]
    again.remove(made.id)
    assert first.links() == []


def test_home_status_keeps_unrelated_and_partial_source_coverage_unknown():
    phone = devices.Link("phone", "net_mac", PHONE, "person", "Alice", "link")
    radio = {"stream": devices.BLE_COVERAGE, **coverage(NOW - 1), "source": {"active_sensors": 0}}
    complete = {
        "stream": devices.NET_INVENTORY,
        "start": NOW - 2,
        "end": NOW - 2,
        "complete": True,
        "authoritative": True,
    }
    assert devices.home_status(devices.seen([phone], {}, [radio], 600), NOW, 600) == "unknown"
    assert devices.home_status(devices.seen([phone], {}, [complete], 600), NOW, 600) == "not_seen"
    partial = {**complete, "start": NOW - 1, "end": NOW - 1, "complete": False}
    assert devices.home_status(devices.seen([phone], {}, [complete, partial], 600), NOW, 600) == "unknown"


def test_one_persons_network_and_bluetooth_positives_combine_without_cross_source_negative_proof():
    links = [
        devices.Link("phone", "net_mac", PHONE, "person", "Alice", "a"),
        devices.Link("watch", "ble_identifier", WATCH, "person", "Alice", "b"),
    ]
    complete = {
        "stream": devices.NET_INVENTORY,
        "start": NOW - 1,
        "end": NOW - 1,
        "complete": True,
        "authoritative": True,
    }
    assert devices.home_status(devices.seen(links, {}, [complete], 600), NOW, 600) == "unknown"
    rows = {devices.NET_OBSERVATIONS: [network(NOW - 100, PHONE)], devices.BLE_SIGHTINGS: [sighting(NOW - 10, WATCH)]}
    result = devices.seen(links, rows, [], 600)
    assert len(result.stays) == 1 and result.stays[0].sightings == 2
    assert devices.home_status(result, NOW, 600) == "recent"


def test_am_i_home_resolves_the_configured_owner_with_an_explicit_device_link(home, tmp_path):

    reg, record = home
    reg.save("net_mac", PHONE, "person", "Alice")
    record(devices.NET_OBSERVATIONS, [network(NOW - 10, PHONE)])
    with cfg.override({"owner": {"name": "Alice"}}):
        reply = ask(tmp_path, "am I home?", NOW)
    assert reply["paths"] == ["devices"]
    assert reply["answer"].startswith("Probably: Alice's device was seen")


def test_home_question_with_unavailable_links_stays_on_the_unknown_device_path(monkeypatch, tmp_path):
    def unavailable(state=None):
        raise RuntimeError("synthetic unavailable store")

    monkeypatch.setattr(devices, "at", unavailable)
    reply = ask(tmp_path, "is Alice home?", NOW)
    assert reply["paths"] == ["devices"]
    assert reply["answer"] == "I can't tell: linked-device evidence is unavailable."
    assert reply["coverage"]["devices"]["read_error"] is True


def test_people_snapshot_groups_names_case_insensitively_and_reads_one_shared_window(tmp_path, monkeypatch):
    reg = devices.Devices(segments=tmp_path / "segments", knowledge_db=tmp_path / "knowledge.sqlite3")
    calls = []

    def links():
        calls.append("links")
        return [
            devices.Link("phone", "net_mac", PHONE, "person", "Alice", "a"),
            devices.Link("watch", "ble_identifier", WATCH, "person", "alice", "b"),
        ]

    def read(start, end):
        calls.append((start, end))
        return {devices.NET_OBSERVATIONS: [network(NOW - 10, PHONE)]}, []

    monkeypatch.setattr(reg, "links", links)
    monkeypatch.setattr(reg, "read", read)
    rows = reg.people(NOW - 600, NOW)
    assert [name for name, _ in rows] == ["Alice"]
    assert len(rows[0][1].links) == 2
    assert calls == ["links", (NOW - 600, NOW)]


def test_comings_lists_each_wifi_device_with_its_stays_today(home):
    """Known › Devices' "comes and goes": a stay a run of connected rows no more than devices.gap_s apart."""
    reg, record = home
    gap = devices.gap_s()
    reg.save("net_mac", PHONE, "person", "Sam")
    other = "aa:bb:cc:dd:ee:02"
    record(
        devices.NET_OBSERVATIONS,
        [
            network(NOW - 5000, PHONE),
            network(NOW - 5000 + gap / 2, PHONE),  # the same stay
            network(NOW - 1000, PHONE),  # back after more than the gap: a second stay
            network(NOW - 500, other),
            network(NOW - 400, other, connected=None),  # a cache row, not presence: ignored
        ],
    )
    found = reg.comings(NOW - 6000, NOW)
    assert found["rows"] == 5
    by = {d["mac"]: d for d in found["devices"]}
    phone = by[PHONE.upper()]
    assert phone["stays"] == [
        {"start": NOW - 5000, "end": NOW - 5000 + gap / 2},
        {"start": NOW - 1000, "end": NOW - 1000},
    ]
    assert (phone["linked"]["name"], phone["place"], phone["name"]) == ("Sam", "Hall", "sams-phone")
    assert by[other.upper()]["stays"] == [{"start": NOW - 500, "end": NOW - 500}]
    assert [d["mac"] for d in found["devices"]] == [other.upper(), PHONE.upper()]  # newest stay first


@pytest.mark.parametrize(
    ("entry", "want"),
    [
        pytest.param("aa:bb:cc:dd:ee:ff: Living room", {"aa:bb:cc:dd:ee:ff": "Living room"}, id="as-written"),
        pytest.param("AA-BB-CC-DD-EE-FF = Hall  way", {"aa:bb:cc:dd:ee:ff": "Hall way"}, id="dashes-equals-spaces"),
    ],
)
def test_an_access_point_name_is_a_mac_and_a_name(entry, want):
    assert devices.parse_access_points([entry]) == want


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("Living room", id="no-mac"),
        pytest.param("aa:bb:cc:dd:ee: Hall", id="short-mac"),
        pytest.param("aa:bb:cc:dd:ee:ff:", id="no-name"),
        pytest.param("aa:bb:cc:dd:ee:ff: " + "x" * 41, id="too-long"),
    ],
)
def test_a_bad_access_point_name_is_refused(entry):
    with pytest.raises(ValueError):
        devices.parse_access_points([entry])


def test_comings_and_whereabouts_show_an_access_point_by_the_owners_name(home):
    """Known › Devices showed an access point by its MAC when the router gave it no name."""
    reg, record = home
    reg.save("net_mac", PHONE, "person", "Sam")
    named, unnamed = "aa:bb:cc:00:00:01", "aa:bb:cc:00:00:02"
    record(
        devices.NET_OBSERVATIONS,
        [
            {**network(NOW - 500, PHONE), "ap_id": named.upper(), "ap_name": None},
            {**network(NOW - 400, "aa:bb:cc:dd:ee:09"), "ap_id": unnamed, "ap_name": None},
        ],
    )
    with cfg.override({"devices": {"access_points": [f"{named}: Upstairs"]}}):
        by = {d["mac"]: d for d in reg.comings(NOW - 600, NOW)["devices"]}
        assert (by[PHONE.upper()]["place"], by[PHONE.upper()]["place_id"]) == ("Upstairs", named)
        assert (by["AA:BB:CC:DD:EE:09"]["place"], by["AA:BB:CC:DD:EE:09"]["place_id"]) == (unnamed, unnamed)
        assert reg.comings(NOW - 600, NOW)["access_points"] == {named: "Upstairs"}
        (sam,) = reg.whereabouts(NOW, 1)
        assert (sam["place"], sam["place_id"]) == ("Upstairs", named)
    # With no name of the owner's, the router's name for it still shows.
    record(devices.NET_OBSERVATIONS, [{**network(NOW - 100, PHONE), "ap_id": named}])
    assert reg.comings(NOW - 600, NOW)["devices"][0]["place"] == "Hall"


def test_whereabouts_says_where_each_link_was_last_seen_looking_back_by_day(home):
    reg, record = home
    fingerprint = devices.ble.payload_fingerprint(sighting(NOW, "x", data=AIRPODS_DATA))
    reg.save("ble_fingerprint", str(fingerprint), "thing", "AirPods")
    reg.save("ble_name", "Keys", "thing", "Keys")  # never heard
    reg.save("ble_identifier", WATCH, "person", "Joe")
    record(
        devices.BLE_SIGHTINGS,
        [
            sighting(NOW - 3 * 86400, "11:00:00:00:00:01", sensor="Car", rssi=-71, data=AIRPODS_DATA),
            sighting(NOW - 40, WATCH, sensor="Desk", rssi=-50),
        ],
    )
    rows = reg.whereabouts(NOW, 30)
    assert [(r["target"], r["name"]) for r in rows] == [("thing", "AirPods"), ("thing", "Keys"), ("person", "Joe")]
    airpods, keys, joe = rows
    assert (airpods["last_seen"], airpods["place"], airpods["rssi"]) == (NOW - 3 * 86400 + 5, "Car", -71.0)
    assert (keys["last_seen"], keys["place"]) == (None, None)
    assert (joe["last_seen"], joe["place"]) == (NOW - 35, "Desk")
    assert reg.whereabouts(NOW, 2)[0]["last_seen"] is None, "past the lookback: not seen"


def test_the_dashboard_names_an_access_point_and_shows_it_at_once(http_, tmp_path, monkeypatch):
    local = tmp_path / "pack.toml"
    local.write_text(cfg.local_config().read_text())
    monkeypatch.setenv("PACK_CONFIG", str(local))  # the save writes this test's own pack.toml
    cfg.load.cache_clear()
    code, body = http_("POST", "/device-access-point", {"mac": "AA-BB-CC-00-00-01", "name": " Up  stairs "})
    assert (code, body) == (200, {"ok": True, "access_points": {"aa:bb:cc:00:00:01": "Up stairs"}})
    assert 'access_points = ["aa:bb:cc:00:00:01: Up stairs"]' in local.read_text()
    code, today = http_("GET", "/device-comings")
    assert (code, today["access_points"]) == (200, {"aa:bb:cc:00:00:01": "Up stairs"})
    code, where = http_("GET", "/device-whereabouts")
    assert (code, where["access_points"]) == (200, {"aa:bb:cc:00:00:01": "Up stairs"})
    assert http_("POST", "/device-access-point", {"mac": "aa:bb:cc:00:00:01", "name": ""})[1]["access_points"] == {}
    assert http_("POST", "/device-access-point", {"mac": "not a mac", "name": "Hall"})[0] == 400
    cfg.load.cache_clear()


def test_settings_saves_access_point_names_normalized_and_refuses_a_bad_one(tmp_path, monkeypatch):
    local = tmp_path / "pack.toml"
    local.write_text(cfg.local_config().read_text())
    monkeypatch.setenv("PACK_CONFIG", str(local))
    cfg.load.cache_clear()
    with pytest.raises(cfg.SettingError, match="devices.access_points"):
        cfg.save({"devices.access_points": ["Living room"]})
    cfg.save({"devices.access_points": ["AA-BB-CC-00-00-01: Hall"]})
    assert 'access_points = ["aa:bb:cc:00:00:01: Hall"]' in local.read_text()
    cfg.load.cache_clear()


def test_the_dashboard_serves_whereabouts_and_comings_and_forgets_them_when_a_link_changes(http_):
    code, before = http_("GET", "/device-whereabouts")
    assert (code, before["links"]) == (200, [])
    http_("POST", "/device-links", {"kind": "net_mac", "value": PHONE, "target": "person", "name": "Sam"})
    code, after = http_("GET", "/device-whereabouts")
    assert (code, [r["name"] for r in after["links"]]) == (200, ["Sam"]), "a new link shows at once"
    code, today = http_("GET", "/device-comings")
    assert (code, today["devices"]) == (200, [])
    assert today["start"] <= today["at"]


PAST = 1_700_002_800.0  # an hour's start in the past, so the hours after it are settled and kept


@pytest.fixture
def ranges(monkeypatch):
    """Each (stream, start, end) the segment store is asked to read."""
    reads = []
    real = segment_log.Log.range

    def counted(self, stream, start, end, cameras=None):
        reads.append((stream, start, end))
        return real(self, stream, start, end, cameras)

    monkeypatch.setattr(segment_log.Log, "range", counted)
    return reads


def test_a_settled_hour_is_tallied_once_and_a_row_across_an_hours_end_counts_once(home, ranges):
    reg, record = home
    start, end = PAST - 1800, PAST + 3 * 3600  # from mid-hour
    record(
        devices.BLE_SIGHTINGS,
        [sighting(PAST + 600, WATCH), sighting(PAST + 4200, WATCH), sighting(PAST + 3598, "de:ad:be:ef:00:01")],
    )
    first = reg.summary(start, end)
    assert {d["identifier"]: d["sightings"] for d in first["devices"]} == {WATCH: 2, "de:ad:be:ef:00:01": 1}
    ranges.clear()
    assert reg.summary(start, end) == first
    assert {(a, b) for _, a, b in ranges} == {(start, PAST)}  # only the part hour; the whole ones were kept


def test_a_row_written_late_into_a_kept_hour_shows(home):
    reg, record = home
    start, end = PAST - 1800, PAST + 3 * 3600
    record(devices.BLE_SIGHTINGS, [sighting(PAST + 600, WATCH)])
    assert reg.summary(start, end)["total"] == 1
    record(devices.BLE_SIGHTINGS, [sighting(PAST + 1200, "de:ad:be:ef:00:02")])
    assert reg.summary(start, end)["total"] == 2


def test_a_device_linked_after_its_hours_were_kept_shows_its_link(home):
    reg, record = home
    start, end = PAST - 1800, PAST + 3 * 3600
    record(devices.BLE_SIGHTINGS, [sighting(PAST + 600, "de:ad:be:ef:00:03", data=AIRPODS_DATA)])
    [row] = reg.summary(start, end)["devices"]
    fingerprint = next(s["value"] for s in row["selectors"] if s["kind"] == "ble_fingerprint")
    pods = reg.save("ble_fingerprint", fingerprint, "thing", "AirPods")
    [row] = reg.summary(start, end)["devices"]
    assert row["linked"] == {"id": pods.id, "target": "thing", "name": "AirPods"}


@pytest.mark.parametrize(
    ("kind", "value", "source", "record"),
    [
        pytest.param("ble_identifier", WATCH, devices.BLE_SIGHTINGS, sighting(NOW, WATCH), id="identifier"),
        pytest.param("ble_identifier", WATCH, devices.BLE_SIGHTINGS, sighting(NOW, "x"), id="other-identifier"),
        pytest.param(
            "ble_name", "Keys", devices.BLE_SIGHTINGS, {**sighting(NOW, "x"), "local_name": "Keys"}, id="name"
        ),
        pytest.param("ble_name", "Keys", devices.BLE_SIGHTINGS, sighting(NOW, "x"), id="no-name"),
        pytest.param(
            "ble_fingerprint", "f" * 64, devices.BLE_SIGHTINGS, sighting(NOW, "x", data=AIRPODS_DATA), id="other-print"
        ),
        pytest.param("net_mac", PHONE, devices.NET_OBSERVATIONS, network(NOW, PHONE), id="mac"),
        pytest.param("net_mac", PHONE, devices.NET_OBSERVATIONS, network(NOW, "aa:bb:cc:dd:ee:02"), id="other-mac"),
        pytest.param("ble_identifier", PHONE, devices.NET_OBSERVATIONS, network(NOW, PHONE), id="wrong-source"),
    ],
)
def test_a_signature_matches_a_link_exactly_when_its_sighting_does(kind, value, source, record):
    link = devices.Link("l", kind, value, "thing", "Thing", "e")
    assert devices.signature_matches(link, devices.signature(record, source), source) == devices.matches(
        link, record, source
    )


def test_a_payload_fingerprint_link_matches_by_signature():
    record = sighting(NOW, "x", data=AIRPODS_DATA)
    fingerprint = ble.payload_fingerprint(dict(record))
    assert fingerprint is not None
    link = devices.Link("l", "ble_fingerprint", fingerprint, "thing", "AirPods", "e")
    sig = devices.signature(record, devices.BLE_SIGHTINGS)
    assert devices.signature_matches(link, sig, devices.BLE_SIGHTINGS) is True


@pytest.fixture
def slow_summary(monkeypatch):
    """devices.at().summary in place, held until `go` is set; `calls` counts the reads."""
    go, started, calls = threading.Event(), threading.Event(), []

    class Reg:
        def summary(self, start, end):
            calls.append((start, end))
            started.set()
            go.wait(5)
            return {"devices": [], "total": len(calls), "linked": 0, "unknown": 0, "links": [], "locations": []}

    monkeypatch.setattr(device_links.devices, "at", lambda: Reg())
    monkeypatch.setattr(device_links.devices, "location_options", lambda state=None: [])
    monkeypatch.setattr(device_links, "_held", {})
    monkeypatch.setattr(device_links, "_reading", {})
    return go, started, calls


def test_requests_arriving_together_share_one_read_and_the_next_reuses_it(slow_summary):
    go, started, calls = slow_summary
    replies = []
    threads = [threading.Thread(target=lambda: replies.append(device_links.listing("", now=NOW))) for _ in range(3)]
    for t in threads:
        t.start()
    started.wait(5)
    go.set()
    for t in threads:
        t.join(5)
    assert len(calls) == 1
    want = {"devices": [], "total": 1, "linked": 0, "unknown": 0, "links": [], "locations": [], "places": []}
    assert replies == [(200, {**want, "hours": 24.0})] * 3
    assert device_links.listing("", now=NOW + 30)[1]["total"] == 1  # reused within the minute
    assert device_links.listing("", now=NOW + device_links.REUSE_S + 1)[1]["total"] == 2


def test_a_read_that_began_before_a_link_changed_is_not_kept(slow_summary):
    go, started, _ = slow_summary
    reader = threading.Thread(target=lambda: device_links.listing("", now=NOW))
    reader.start()
    started.wait(5)
    device_links._forget()  # a link saved or removed while the read was running
    go.set()
    reader.join(5)
    assert device_links.listing("", now=NOW + 1)[1]["total"] == 2  # read again, with the new links


def test_the_dashboard_reads_the_device_list_once_at_start_at_low_priority(monkeypatch):
    listing = create_autospec(device_links.listing, return_value=(200, {}))
    priority = create_autospec(server._low_priority)
    monkeypatch.setattr(device_links, "listing", listing)
    monkeypatch.setattr(server, "_low_priority", priority)
    server._warm_devices()
    priority.assert_called_once_with(19)
    listing.assert_called_once_with("", state=server.STATE)


def test_a_failed_device_warm_up_is_logged_not_raised(monkeypatch):
    logged = []

    def unreachable(hours):
        raise RuntimeError("no broker")

    monkeypatch.setattr(device_links, "listing", unreachable)
    monkeypatch.setattr(server, "_low_priority", lambda priority=15: None)
    monkeypatch.setattr(server, "log", lambda *a, **k: logged.append(a))
    server._warm_devices()
    assert logged and "warm-up failed" in logged[0][0]


@pytest.fixture
def cached(tmp_path):
    """Two processes' worth of Devices over one state directory and one tally cache: `fresh()` is a new one."""

    def fresh():
        return devices.Devices(
            segments=tmp_path / "segments",
            knowledge_db=tmp_path / "knowledge.sqlite3",
            tally_cache=tmp_path / devices.TALLY_CACHE,
        )

    return fresh


def write_ble(tmp_path, rows):
    writers = segment_log.Writers(tmp_path / "segments")
    writers.append(devices.BLE_SIGHTINGS, "ble", rows)
    writers.close()


def test_a_restart_answers_from_the_saved_hours_without_reading_them(tmp_path, cached, ranges):
    start, end = PAST - 1800, PAST + 3 * 3600
    write_ble(tmp_path, [sighting(PAST + 600, WATCH), sighting(PAST + 4200, WATCH)])
    first = cached().summary(start, end)
    ranges.clear()
    assert cached().summary(start, end) == first  # a new process
    assert {(a, b) for _, a, b in ranges} == {(start, PAST)}  # only the part hour: the whole ones were saved


def test_a_saved_hour_the_store_changed_since_is_tallied_again(tmp_path, cached):
    start, end = PAST - 1800, PAST + 3 * 3600
    write_ble(tmp_path, [sighting(PAST + 600, WATCH)])
    assert cached().summary(start, end)["total"] == 1
    write_ble(tmp_path, [sighting(PAST + 1200, "de:ad:be:ef:00:04")])  # written late, after the save
    assert cached().summary(start, end)["total"] == 2


@pytest.mark.parametrize(
    "saved", [b"not json", b'{"version": 0, "hours": []}', b'{"version": 1, "hours": [{"source": 1}]}']
)
def test_an_unreadable_or_old_cache_is_rebuilt_from_the_store(tmp_path, cached, saved):
    write_ble(tmp_path, [sighting(PAST + 600, WATCH)])
    (tmp_path / devices.TALLY_CACHE).parent.mkdir(parents=True)
    (tmp_path / devices.TALLY_CACHE).write_bytes(saved)
    assert cached().summary(PAST - 1800, PAST + 3 * 3600)["total"] == 1


def test_a_tally_that_cant_be_saved_still_answers(tmp_path, cached):
    row = sighting(PAST + 600, WATCH)
    # A NaN sends the record through json, which keeps an int past 64 bits that orjson can't write back.
    write_ble(tmp_path, [{**row, "reports": 2**70, "rssi": {**row["rssi"], "mean": float("nan")}}])
    assert cached().summary(PAST - 1800, PAST + 3 * 3600)["total"] == 1
    assert not (tmp_path / devices.TALLY_CACHE).exists()
    assert list((tmp_path / devices.TALLY_CACHE).parent.iterdir()) == []  # no temporary file left behind
