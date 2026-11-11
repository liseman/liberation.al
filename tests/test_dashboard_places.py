"""Known › Places' reads (dashboard/places.py): the places listing and one place's positions."""

import http.client
import json
import socket
import threading
from dataclasses import replace
from urllib.parse import urlsplit

import pytest

import plans
from liberational import segment_log as S
from liberational.dashboard import context
from liberational.dashboard import places as view
from liberational.memory import places
from liberational.spatial import plan_scene, positions

T = 1_762_500_000.0
PLAN = plans.scene('plan "Home" walls 0.2/0.1\nroom den "Den" living rect 0,0 4x6\n')


def test_the_listing_gives_each_place_its_plan_revision_and_last_visit(tmp_path, vision):
    home = places.revise(tmp_path, None, producer="footprints-v1", plan=PLAN, now=T)
    bare = places.revise(tmp_path, None, producer="cameras-v1", now=T + 10)
    for start in (T + 100, T + 900):
        vision.add(
            S.PLACE_VISITS, ts_start=start, ts_end=start + 50, cam="rig", place=home.place, producer="p", basis="wifi"
        )
    code, reply = view.listing(tmp_path, now=T + 3600)
    assert code == 200
    rows = {r["id"]: r for r in reply["places"]}
    assert rows[home.place]["plan"] == PLAN and rows[home.place]["plan_source"] == "machine"
    assert rows[home.place]["revision"] == {"id": home.id, "at": T, "producer": "footprints-v1"}
    assert rows[home.place]["last_visit"] == T + 950
    assert (rows[bare.place]["plan"], rows[bare.place]["last_visit"]) == (None, None)
    assert rows[bare.place]["name"].startswith("Place ")  # a placeholder until the owner names it


def seen(state, at, place, end=None):
    """A record in the position layer (spatial/positions.py), as the placer writes one for a place's camera."""
    row = position(at, x=1.0, y=1.0, end=end)
    positions.write(state, [replace(row, id=S.new_id(), place=place)])


def test_a_place_only_its_cameras_watch_lists_when_anyone_was_last_seen_there(tmp_path, mapped):
    # A visit is the owner's (the rig was there), so a place only fixed cameras watch has none: its activity is
    # listed apart from visits.
    watched, _ = mapped
    other = places.revise(tmp_path, None, producer="cameras-v1", now=T + 10).place
    seen(tmp_path, T + 100, watched, end=T + 120)
    seen(tmp_path, T + 300, other)
    rows = {r["id"]: r for r in view.listing(tmp_path, now=T + 3600)[1]["places"]}
    assert (rows[watched]["last_activity"], rows[watched]["last_visit"]) == (T + 120, None)
    assert rows[other]["last_activity"] == T + 300


@pytest.mark.parametrize(("ago", "found"), [(3 * 86400, True), (8 * 86400, False)], ids=["this-week", "older"])
def test_last_activity_looks_back_a_week(tmp_path, mapped, ago, found):
    watched, _ = mapped
    now = T + 30 * 86400
    seen(tmp_path, now - ago, watched)
    row = view.listing(tmp_path, now=now)[1]["places"][0]
    assert row["last_activity"] == (now - ago if found else None)


def test_a_place_found_in_a_newer_pass_isnt_read_again_in_the_older_ones(tmp_path, mapped, monkeypatch):
    # A busy place is read for the hour it was found in, not the week a place nobody was seen in still needs.
    watched, _ = mapped
    never = places.revise(tmp_path, None, producer="mapper v1", now=T + 10).place
    now = T + 30 * 86400
    seen(tmp_path, now - 60, watched)
    asked = []
    scan = positions._scan
    monkeypatch.setattr(
        positions,
        "_scan",
        lambda state, lo, hi, slots, **k: asked.append((slots[0], now - lo)) or scan(state, lo, hi, slots, **k),
    )
    rows = {r["id"]: r for r in view.listing(tmp_path, now=now)[1]["places"]}
    assert (rows[watched]["last_activity"], rows[never]["last_activity"]) == (now - 60, None)
    mine, theirs = (positions.layer_slot(p, positions.FIXED_CAMERA) for p in (watched, never))
    assert [back for slot, back in asked if slot == mine] == [3600]
    assert [back for slot, back in asked if slot == theirs] == [3600, 86400, 7 * 86400]


def test_an_owner_corrected_plan_is_the_one_listed(tmp_path, vision):
    home = places.revise(tmp_path, None, producer="footprints-v1", plan=PLAN, now=T)
    mine = PLAN.replace("4x6", "4x5")
    places.correct_plan(tmp_path, home.place, mine, now=T + 5)
    row = view.listing(tmp_path, now=T + 60)[1]["places"][0]
    assert (row["plan"], row["plan_source"]) == (mine, "owner")


def position(
    t,
    *,
    entity=None,
    name=None,
    track="cam0:7",
    x=None,
    y=None,
    room=None,
    room_id=None,
    end=None,
    source="fixed-camera",
    in_frame=True,
    path=None,
    uncertain=False,
):
    return positions.Position(
        id=f"p{t}", t=t, end=t if end is None else end, entity=entity, name=name, track=track, source=source,
        confidence=0.8, place="home", room=room, room_id=room_id, x=x, y=y, camera="cam0", u=1.0, v=2.0, revision="r1",
        in_frame=in_frame, path=path, uncertain=uncertain,
    )  # fmt: skip


class Reader:
    """A stand-in for spatial/positions.py's reader (the stream it reads is its own tests' business): records what
    it was asked, answers `rows`."""

    def __init__(self) -> None:
        self.rows: list[positions.Position] = []
        self.asked: list[tuple] = []
        self.refuse = ""  # a ValueError to raise, as the reader does for a frame it can't use

    def __call__(self, state, start, end, place=None, entity=None, every=None, frame_revision=None, cancelled=None):
        self.asked.append((start, end, place, entity, every, frame_revision))
        if cancelled is not None and cancelled():
            raise positions.Cancelled
        if self.refuse:
            raise ValueError(self.refuse)
        return self.rows


@pytest.fixture
def reader(monkeypatch):
    fake = Reader()
    monkeypatch.setattr(positions, "positions", fake)
    return fake


@pytest.fixture
def home(tmp_path, vision):
    """A place with a machine plan: (its id, the revision its plan is drawn in)."""
    rev = places.revise(tmp_path, None, producer="footprints-v1", plan=PLAN, now=T)
    return rev.place, rev.id


def window(place, span=3600.0):
    return {"place": place, "start": str(T), "end": str(T + span)}


def test_an_unnamed_face_groups_tracks_are_one_person_in_the_list(tmp_path, reader, home):
    # "face:3" is someone, if unnamed: its own entry, not folded in with the strangers
    reader.rows = [
        position(T + 1, entity="face:3", name="Face 3", track="cam0:4", x=1.0, y=2.0, room="Den", room_id="den"),
        position(T + 2, entity="face:3", name="Face 3", track="porch:7", x=1.5, y=2.5, room="Den", room_id="den"),
    ]
    place, _frame = home
    code, reply = view.positions_view(window(place), tmp_path)
    assert code == 200
    assert reply["entities"] == [{"id": "face:3", "name": "Face 3", "kind": "person"}]
    assert sorted(t["track"] for t in reply["tracks"]) == ["cam0:4", "porch:7"]


def test_one_entitys_tracks_the_reader_fused_are_one_path(tmp_path, reader, home):
    # cam0:4 alone, then a fused point (no track of its own), then porch:7 alone: one path, in time order (a
    # step of half a metre a second apart, so none is thinned away as standing still)
    joined = "cam0:4+porch:7"
    reader.rows = [
        position(T + 1, entity="person:dana", name="Dana", track="cam0:4", x=1.0, y=2.0, path=joined),
        position(T + 2, entity="person:dana", name="Dana", track=None, x=1.5, y=2.0, path=joined),
        position(T + 3, entity="person:dana", name="Dana", track="porch:7", x=2.0, y=2.0, path=joined),
        position(T + 4, entity="person:dana", name="Dana", track="den:2", x=3.0, y=1.0),  # not joined: its own
    ]
    reply = view.positions_view(window(home[0]), tmp_path)[1]
    assert [(t["track"], [p["x"] for p in t["points"]]) for t in reply["tracks"]] == [
        (joined, [1.0, 1.5, 2.0]),
        ("den:2", [3.0]),
    ]


def test_an_uncertain_point_is_left_off_its_path_and_an_unnamed_joined_path_is_its_own_entry(tmp_path, reader, home):
    # A foot on an occluding edge no other camera settled can be metres off
    joined = "cam0:4+porch:7"
    reader.rows = [
        position(T + 1, entity="person:dana", name="Dana", track="cam0:2", x=1.0, y=2.0),
        position(T + 2, entity="person:dana", name="Dana", track="cam0:2", x=4.0, y=6.0, uncertain=True),
        position(T + 3, entity="person:dana", name="Dana", track="cam0:2", x=1.2, y=2.2),
        position(T + 4, track=None, x=2.0, y=2.0, path=joined),  # an unnamed person both cameras saw, fused
        position(T + 5, track="cam0:4", x=2.1, y=2.1, path=joined),
    ]
    reply = view.positions_view(window(home[0]), tmp_path)[1]
    assert [(t["entity"], t["track"], [p["x"] for p in t["points"]]) for t in reply["tracks"]] == [
        ("person:dana", "cam0:2", [1.0, 1.2]),
        (joined, joined, [2.0, 2.1]),
    ]


def test_positions_group_by_who_put_points_on_tracks_and_rooms_in_spans(tmp_path, reader, home):
    reader.rows = [
        position(T + 1, entity="person:dana", name="Dana", x=1.0, y=2.0, room="Den", room_id="den"),
        position(T + 2, entity="person:dana", name="Dana", x=1.5, y=2.5, room="Den", room_id="den"),
        position(T + 3, track="cam0:9", x=3.0, y=1.0, room="Den", room_id="den"),  # not known yet: by track
        position(T + 4, entity="pet:miso", name="Miso", room="Bedroom", end=T + 600, source="zone"),
    ]
    place, frame = home
    code, reply = view.positions_view(window(place), tmp_path)
    assert code == 200
    # an hour over 2,000 points is one every 1.8 s, so the reader gives every row; and the points are put in the frame
    # of the plan the tab draws
    assert reader.asked == [(T, T + 3600, place, None, None, frame)]
    assert reply["entities"] == [
        {"id": "person:dana", "name": "Dana", "kind": "person"},
        {"id": "cam0:9", "name": None, "kind": "unknown"},
        {"id": "pet:miso", "name": "Miso", "kind": "pet"},
    ]
    dana = reply["tracks"][0]
    assert dana["entity"] == "person:dana" and [(p["x"], p["y"], p["room"]) for p in dana["points"]] == [
        (1.0, 2.0, "den"),
        (1.5, 2.5, "den"),
    ]
    assert reply["rooms"] == [
        {"entity": "pet:miso", "room": None, "name": "Bedroom", "start": T + 4, "end": T + 600, "confidence": 0.8, "source": "zone"}
    ]  # fmt: skip


def test_a_long_track_is_thinned_evenly_keeping_its_ends(tmp_path, reader, home):
    reader.rows = [position(T + i, entity="person:dana", x=float(i), y=0.0) for i in range(view.MAX_POINTS * 3)]
    points = view.positions_view(window(home[0], 86400), tmp_path)[1]["tracks"][0]["points"]
    assert reader.asked[0][4] == pytest.approx(2.16)  # the reader thins a day to a row every 2.16 s, for `moved`
    assert len(points) == view.MAX_POINTS  # and what it still gives is capped
    assert (points[0]["t"], points[-1]["t"]) == (T, T + view.MAX_POINTS * 3 - 1)


def test_a_days_path_keeps_a_walks_shape_but_one_point_every_43_s_standing_still(tmp_path, reader, home):
    # 100 s shuffling on the spot (5 cm of foot jitter), then a 20 s walk at half a metre a second: on a Day's plan the
    # walk is every step, not a line between two points 43.2 s apart
    still = [position(T + i, entity="person:dana", x=0.05 * (i % 2), y=0.0) for i in range(100)]
    walk = [position(T + 100 + j, entity="person:dana", x=0.5 * j, y=0.0) for j in range(1, 21)]
    reader.rows = still + walk
    points = view.positions_view(window(home[0], 86400), tmp_path)[1]["tracks"][0]["points"]
    assert [p["t"] - T for p in points] == [0, 44, 88, *range(101, 121)]


@pytest.mark.parametrize(
    "query",
    [
        {"start": str(T), "end": str(T + 60)},  # no place
        {"place": "home", "start": str(T + 60), "end": str(T)},  # ends before it starts
        {"place": "home", "start": str(T), "end": str(T + 32 * 86400)},  # longer than 31 days
        {"place": "home", "start": "1e300", "end": "2e300"},
        {"place": "home", "start": "nan", "end": str(T)},
        {"place": "home", "start": "soon", "end": str(T)},
    ],
)
def test_a_bad_positions_query_is_a_400(tmp_path, reader, query):
    code, reply = view.positions_view(query, tmp_path)
    assert code == 400 and "error" in reply
    assert reader.asked == []


def test_thinning_leaves_a_short_list_alone():
    assert view.thin([1, 2, 3], 5) == [1, 2, 3]
    assert view.thin(list(range(10)), 2) == [0, 9]


def test_the_listing_route_answers_json_and_a_page_load_gets_the_app(serve_dashboard):
    host = urlsplit(serve_dashboard()).netloc

    def get(path, accept):
        conn = http.client.HTTPConnection(host, timeout=120)  # generous: a loaded machine is slow, not broken
        conn.request("GET", path, headers={"Accept": accept})
        r = conn.getresponse()
        body = r.read()
        conn.close()
        return r.status, r.getheader("Content-Type") or "", body

    status, kind, body = get("/places", "application/json")
    assert (status, kind.split(";")[0]) == (200, "application/json")
    assert json.loads(body) == {"places": []}
    assert get("/places", "text/html")[1].startswith("text/html")


def test_a_positions_read_stops_without_an_answer_when_the_client_hangs_up(
    tmp_path, reader, home, serve_dashboard, monkeypatch
):
    monkeypatch.setattr(context, "STATE", tmp_path)
    url = urlsplit(serve_dashboard())
    asked, hung_up, tick = threading.Event(), threading.Event(), threading.Event()

    def read(state, start, end, *, cancelled, **kw):
        asked.set()
        for _ in range(3000):  # the read asks as it goes, for 30 s at most
            if cancelled():
                hung_up.set()
                break
            tick.wait(0.01)
        raise positions.Cancelled

    monkeypatch.setattr(positions, "positions", read)
    conn = socket.create_connection((url.hostname, url.port), timeout=30)
    query = f"place={home[0]}&start={T}&end={T + 3600}"
    conn.sendall(f"GET /places/positions?{query} HTTP/1.1\r\nHost: x\r\nAccept: application/json\r\n\r\n".encode())
    assert asked.wait(30)
    assert not hung_up.is_set()
    conn.close()
    assert hung_up.wait(30)


def test_a_client_still_waiting_isnt_taken_for_gone(tmp_path, reader, home, serve_dashboard, monkeypatch):
    monkeypatch.setattr(context, "STATE", tmp_path)
    host = urlsplit(serve_dashboard()).netloc
    checks = []

    def read(state, start, end, *, cancelled, **kw):
        checks.append(cancelled())
        return []

    monkeypatch.setattr(positions, "positions", read)
    conn = http.client.HTTPConnection(host, timeout=120)
    conn.request(
        "GET", f"/places/positions?place={home[0]}&start={T}&end={T + 3600}", headers={"Accept": "application/json"}
    )
    reply = conn.getresponse()
    body = json.loads(reply.read())
    conn.close()
    assert (reply.status, body["tracks"], checks) == (200, [], [False])


def test_points_with_no_entity_or_track_are_left_off_the_paths(tmp_path, reader, home):
    reader.rows = [
        position(T + 1, track=None, x=1.0, y=1.0),
        position(T + 2, track=None, x=6.0, y=4.0),  # joined, these would zig-zag between two people
        position(T + 3, track=None, room="Den", end=T + 60, source="zone"),
    ]
    reply = view.positions_view(window(home[0]), tmp_path)[1]
    assert reply["tracks"] == []
    assert [(r["entity"], r["name"]) for r in reply["rooms"]] == [("someone", "Den")]


def test_a_point_that_cant_be_put_in_the_drawn_plans_frame_is_left_out(tmp_path, reader, home):
    reader.rows = [
        position(T + 1, entity="person:dana", x=1.0, y=2.0),
        position(T + 2, entity="person:dana", x=9.0, y=9.0, in_frame=False),  # would land off the drawn plan
        position(T + 3, entity="person:dana", room="Den", end=T + 60, source="zone", in_frame=False),
    ]
    reply = view.positions_view(window(home[0]), tmp_path)[1]
    assert [p["t"] for p in reply["tracks"][0]["points"]] == [T + 1]
    assert [r["name"] for r in reply["rooms"]] == ["Den"]  # a room-only span is kept either way


def test_an_unknown_place_is_a_404_and_a_frame_the_reader_refuses_a_400(tmp_path, reader, home):
    assert view.positions_view(window("no-such-place"), tmp_path)[0] == 404
    reader.refuse = "no revision r9 of that place"
    code, reply = view.positions_view(window(home[0]), tmp_path)
    assert (code, reply["error"]) == (400, "no revision r9 of that place")


def test_the_listing_names_the_revision_the_plan_is_drawn_in(tmp_path, home):
    row = view.listing(tmp_path, now=T + 60)[1]["places"][0]
    assert row["plan_revision"] == home[1]


def test_a_correction_holds_over_the_machine_plan_until_withdrawn(tmp_path, home):
    place = home[0]
    mine = PLAN.replace('"Den"', '"Living room"')
    code, reply = view.correct({"place": place, "plan": mine}, True, tmp_path, now=T + 10)
    assert (code, reply["plan"], reply["plan_source"]) == (200, mine, "owner")
    row = view.listing(tmp_path, now=T + 60)[1]["places"][0]
    # the owner's plan is the place's only plan: the machine's isn't offered beside it
    assert (row["plan"], row["plan_source"], row["machine_plan"]) == (mine, "owner", None)
    code, reply = view.correct({"place": place, "plan": "  "}, True, tmp_path, now=T + 20)
    assert (code, reply["plan"], reply["plan_source"]) == (200, PLAN, "machine")
    assert view.listing(tmp_path, now=T + 60)[1]["places"][0]["machine_plan"] is None


@pytest.mark.parametrize(
    "body,control,code",
    [
        pytest.param({"place": "p", "plan": PLAN}, False, 403, id="remote-teaching-off"),
        pytest.param({"place": "no-such-place", "plan": PLAN}, True, 404, id="unknown-place"),
        pytest.param({"place": None, "plan": PLAN}, True, 400, id="no-place"),
        pytest.param({"plan": 3}, True, 400, id="plan-not-text"),
        pytest.param(
            {"plan": plan_scene.dump(plan_scene.build(walls=[((0.0, 0.0), (4.0, 0.0), 0.2)]))}, True, 400, id="no-room"
        ),
        pytest.param({"plan": 'plan "Home"\nroom den "Den" living rect 0,0 4x6\n'}, True, 400, id="the-old-format"),
        pytest.param({"plan": PLAN[:-1] + " " * view.MAX_SCENE_BYTES + "}"}, True, 400, id="too-long"),
    ],
)
def test_a_correction_that_isnt_one_changes_nothing(tmp_path, home, body, control, code):
    body = {"place": home[0], **body}
    assert view.correct(body, control, tmp_path, now=T + 10)[0] == code
    assert view.listing(tmp_path, now=T + 60)[1]["places"][0]["plan_source"] == "machine"


def test_correcting_again_after_a_newer_revision_keeps_the_frame_the_text_is_in(tmp_path, home):
    place, first = home
    mine = PLAN.replace('"Den"', '"Study"')
    assert view.correct({"place": place, "plan": mine, "base": first}, True, tmp_path, now=T + 10)[0] == 200
    newer = places.revise(tmp_path, place, producer="footprints-v1", plan=PLAN.replace("4x6", "4x7"), now=T + 100)
    row = view.listing(tmp_path, now=T + 200)[1]["places"][0]
    assert (row["plan_revision"], row["machine_newer"]) == (first, False), "no machine plan is offered over the owner's"
    # The owner corrects their own plan again: the text is still in the first revision's frame, not the newer one's.
    again = mine.replace('"Study"', '"Office"')
    assert (
        view.correct({"place": place, "plan": again, "base": row["plan_revision"]}, True, tmp_path, now=T + 300)[0]
        == 200
    )
    row = view.listing(tmp_path, now=T + 400)[1]["places"][0]
    assert (row["plan"], row["plan_revision"]) == (again, first)
    assert row["plan_revision"] != newer.id


def test_a_correction_without_a_base_is_drawn_in_the_newest_frame(tmp_path, home):
    place = home[0]
    newer = places.revise(tmp_path, place, producer="footprints-v1", plan=PLAN, now=T + 100)
    assert view.correct({"place": place, "plan": PLAN}, True, tmp_path, now=T + 200)[0] == 200
    assert view.listing(tmp_path, now=T + 300)[1]["places"][0]["plan_revision"] == newer.id


def test_a_base_that_isnt_one_of_the_places_revisions_is_a_400(tmp_path, home):
    code, reply = view.correct({"place": home[0], "plan": PLAN, "base": "not-a-revision"}, True, tmp_path, now=T + 10)
    assert code == 400 and "not a revision" in reply["error"]
    assert view.listing(tmp_path, now=T + 60)[1]["places"][0]["plan_source"] == "machine"


def test_a_stored_base_no_longer_in_the_chain_falls_back_to_the_frame_current_when_saved(tmp_path, home):
    place, first = home
    # A correction whose stored base isn't among the place's revisions (pruned, or a record from elsewhere): the
    # frame falls back to the revision current when it was saved, rather than raising.
    places._put(tmp_path, places.PLAN, place, {"place": place, "plan": PLAN, "base": "gone", "source": "owner"}, T + 10)
    places.revise(tmp_path, place, producer="footprints-v1", plan=PLAN.replace("4x6", "4x7"), now=T + 100)
    row = view.listing(tmp_path, now=T + 200)[1]["places"][0]
    assert (row["plan_source"], row["plan_revision"]) == ("owner", first)


TWO_ROOMS = PLAN + 'room hall "Hall" hall rect 4,0 2x6\n'


def test_restoring_a_version_saves_it_again_on_top_and_restoring_the_one_before_undoes_it(tmp_path, home):
    place = home[0]
    hand = PLAN.replace('"Den"', '"Study"')
    view.correct({"place": place, "plan": hand}, True, tmp_path, now=T + 10)
    view.correct({"place": place, "plan": TWO_ROOMS}, True, tmp_path, now=T + 20)
    first, second = view.plan_versions(tmp_path, place)[1]["versions"]
    assert [(v["rooms"], v["current"]) for v in (first, second)] == [(1, False), (2, True)]

    code, reply = view.restore({"place": place, "record": first["id"]}, True, tmp_path, now=T + 30)
    assert (code, reply["plan"], reply["plan_source"]) == (200, hand, "owner")
    # appended, not rewound: three versions now, and the one it replaced is still there to restore
    versions = view.plan_versions(tmp_path, place)[1]["versions"]
    assert [(v["rooms"], v["current"]) for v in versions] == [(1, False), (2, False), (1, True)]
    code, reply = view.restore({"place": place, "record": second["id"]}, True, tmp_path, now=T + 40)
    assert (code, reply["plan"]) == (200, TWO_ROOMS)


def test_each_version_says_what_changed_from_the_one_before_and_which_one_holds(tmp_path, home):
    place = home[0]
    spotted = TWO_ROOMS + "x-camera porch at 1,1 yaw 90\n"
    view.correct({"place": place, "plan": PLAN.replace('"Den"', '"Study"')}, True, tmp_path, now=T + 10)
    view.correct({"place": place, "plan": spotted}, True, tmp_path, now=T + 20)
    view.correct({"place": place, "plan": "  "}, True, tmp_path, now=T + 30)
    versions = view.plan_versions(tmp_path, place)[1]["versions"]
    assert [v["summary"] for v in versions] == [
        "1 room",
        "Added Hall, renamed Study to Den, added camera porch",
        "Plan withdrawn",
    ]
    assert [v["current"] for v in versions] == [False, False, False], "the machine's plan holds after a withdrawal"
    view.correct({"place": place, "plan": spotted}, True, tmp_path, now=T + 40)
    versions = view.plan_versions(tmp_path, place)[1]["versions"]
    assert (versions[-1]["summary"], versions[-1]["current"]) == ("2 rooms, 1 camera spot", True)


ROOM = plans.scene('plan "x"\nroom den "Den" living rect 0,0 4x6\n')


@pytest.mark.parametrize(
    ("before", "after", "said"),
    [
        pytest.param(None, ROOM, "1 room", id="first"),
        pytest.param(ROOM, ROOM, "No change to rooms, openings, or cameras", id="same"),
        pytest.param(ROOM, ROOM.replace("4x6", "4x5"), "Reshaped Den", id="reshaped"),
        pytest.param(
            ROOM,
            plans.scene('plan "x"\nroom den "Den" living poly 4,6 0,6 0,0 4,0\n'),
            "No change to rooms, openings, or cameras",
            id="same-outline-from-another-corner",
        ),
        pytest.param(ROOM, ROOM + "door exterior>den at 0,3 w0.9\n", "Added 1 door", id="door"),
        pytest.param(
            ROOM + "window exterior>den at 2,0 w1\nwindow exterior>den at 2,6 w1\n",
            ROOM,
            "Removed 2 windows",
            id="windows",
        ),
        pytest.param(
            ROOM + "x-camera porch at 1,1 yaw 90\n",
            ROOM + "x-camera porch at 2,1 yaw 90\n",
            "Moved camera porch",
            id="moved",
        ),
        pytest.param(
            ROOM + "x-camera porch at 1,1 yaw 350\n",
            ROOM + "x-camera porch at 1,1 yaw 10\n",
            "Turned camera porch",
            id="turned",
        ),
        pytest.param(
            ROOM + "x-camera porch at 1,1 yaw 359.8\n",
            ROOM + "x-camera porch at 1,1 yaw 0.1\n",
            "No change to rooms, openings, or cameras",
            id="a-third-of-a-degree-across-north",
        ),
        pytest.param(
            ROOM + "x-camera porch at 1,1\n",
            ROOM + "x-camera porch at 1,1 height 2\n",
            "Changed camera porch",
            id="height",
        ),
        pytest.param(ROOM + "x-camera porch at 1,1\n", ROOM, "Removed camera porch", id="spot-removed"),
        pytest.param(
            ROOM,
            plans.scene(
                'plan "x"\nroom a "A" other rect 0,0 1x1\nroom b "B" other rect 1,0 1x1\nroom c "C" other rect 2,0 1x1\n'
                'room d "D" other rect 3,0 1x1\n'
            ),
            "Added A, added B, added C, and 2 more",
            id="more-than-three",
        ),
        pytest.param("{broken", ROOM, "1 room", id="after-an-unreadable-one"),
        pytest.param(ROOM, "{broken", "A plan that can't be read", id="unreadable"),
        pytest.param(ROOM, "", "Plan withdrawn", id="withdrawn"),
    ],
)
def test_a_versions_summary_names_what_changed(before, after, said):
    assert view.plan_change(before, after) == said


def test_a_version_in_the_old_format_is_not_listed_and_cant_be_restored(tmp_path, home):
    # a plan saved before scenes is still in the store, where nothing is ever rewritten; it is no longer read
    place = home[0]
    places.correct_plan(tmp_path, place, 'plan "Home"\nroom den "Den" living rect 0,0 4x6\n', now=T + 10)
    view.correct({"place": place, "plan": TWO_ROOMS}, True, tmp_path, now=T + 20)
    old = places.plan_history(tmp_path, place)[0]
    versions = view.plan_versions(tmp_path, place)[1]["versions"]
    assert [(v["rooms"], v["current"], v["summary"]) for v in versions] == [(2, True, "2 rooms")]
    code, reply = view.restore({"place": place, "record": old.id}, True, tmp_path, now=T + 30)
    assert code == 422 and "old format" in reply["error"]
    assert len(places.plan_history(tmp_path, place)) == 2, "a refused restore writes nothing"


def test_no_version_is_current_while_an_old_format_plan_is_the_newest(tmp_path, home):
    place = home[0]
    view.correct({"place": place, "plan": TWO_ROOMS}, True, tmp_path, now=T + 10)
    places.correct_plan(tmp_path, place, 'plan "Home"\nroom den "Den" living rect 0,0 4x6\n', now=T + 20)
    assert [v["current"] for v in view.plan_versions(tmp_path, place)[1]["versions"]] == [False]


def test_restoring_a_plan_drawn_on_its_page_keeps_its_page(tmp_path, paged):
    place, frame = paged
    record = places.plan_history(tmp_path, place)[-1].id
    view.correct({"place": place, "plan": PAGE.replace("4x6", "4x5")}, True, tmp_path, now=T + 40)
    assert view.restore({"place": place, "record": record}, True, tmp_path, now=T + 50)[0] == 200
    held = places.get(tmp_path, place)
    assert held is not None and held.view is not None
    assert (held.plan, held.view.plan, held.plan_revision) == (IN_FRAME, PAGE, frame)


def test_a_withdrawal_is_a_version_too(tmp_path, home):
    place = home[0]
    view.correct({"place": place, "plan": TWO_ROOMS}, True, tmp_path, now=T + 10)
    view.correct({"place": place, "plan": ""}, True, tmp_path, now=T + 20)
    versions = view.plan_versions(tmp_path, place)[1]["versions"]
    assert [(v["withdrawn"], v["current"]) for v in versions] == [(False, False), (True, False)]


def test_a_place_never_corrected_has_no_versions(tmp_path, home):
    assert view.plan_versions(tmp_path, home[0]) == (200, {"place": home[0], "versions": []})
    assert view.plan_versions(tmp_path, "no-such-place")[0] == 404


@pytest.mark.parametrize(
    "body,control,code",
    [
        pytest.param({}, False, 403, id="remote-teaching-off"),
        pytest.param({"record": "not-a-record"}, True, 404, id="unknown-record"),
        pytest.param({"place": "no-such-place"}, True, 404, id="unknown-place"),
        pytest.param({"record": None}, True, 400, id="no-record"),
    ],
)
def test_a_restore_that_isnt_one_changes_nothing(tmp_path, home, body, control, code):
    view.correct({"place": home[0], "plan": TWO_ROOMS}, True, tmp_path, now=T + 10)
    first = view.plan_versions(tmp_path, home[0])[1]["versions"][0]["id"]
    assert view.restore({"place": home[0], "record": first, **body}, control, tmp_path, now=T + 20)[0] == code
    assert len(view.plan_versions(tmp_path, home[0])[1]["versions"]) == 1


CLOUD = b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n" + bytes(range(256)) * 4
FLOOR = (0.0, 0.0, 1.0, -0.01)
TRANSFORM = (1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, 0.0)
CAM = places.CameraPose(
    pose=tuple(float(i == j) for i in range(4) for j in range(4)), focal=300.0, width=512, height=288
)


@pytest.fixture
def mapped(tmp_path, vision):
    """A place with a cloud, a floor, a plan transform, one camera, and a plan: (its id, the revision)."""
    rev = places.revise(
        tmp_path, None, producer="mapper v1", plan=PLAN, map_bytes=CLOUD, map_format="ply", floor=FLOOR,
        plan_transform=TRANSFORM, cameras={"cam0": CAM}, now=T,
    )  # fmt: skip
    return rev.place, rev.id


def test_the_scene_gives_the_newest_maps_frame_cameras_and_plan(tmp_path, mapped):
    place, rev = mapped
    code, reply = view.scene(tmp_path, place)
    assert code == 200
    assert (reply["revision"], reply["map_format"], reply["map_bytes"]) == (rev, "ply", len(CLOUD))
    assert reply["map"] == f"/places/map?place={place}&revision={rev}"
    assert (reply["floor"], reply["plan_transform"]) == (list(FLOOR), list(TRANSFORM))
    assert reply["cameras"]["cam0"]["focal"] == 300.0 and len(reply["cameras"]["cam0"]["pose"]) == 16
    assert (reply["plan"], reply["plan_source"]) == (PLAN, "machine")
    assert (reply["splat"], reply["splat_format"], reply["splat_bytes"]) == (None, None, None)
    # no splat, and no pass has tried this map yet: the page says so, and that none is scheduled
    assert (reply["splat_state"], reply["splat_at"]) == ("none", None)
    assert reply["splat_wait"] == {"reason": "pass", "tried": None, "pass_at": None}


def test_a_newer_map_in_the_same_frame_shows_the_last_splat_and_one_in_another_frame_names_it_stale(tmp_path, mapped):
    # A later revision that kept the map's frame has no splat of its own, so the last splat still fits.
    place, rev_id = mapped
    first = places.revisions(tmp_path, place)[-1]
    places.write_splat(tmp_path, first, master=b"ply", web=b"sog-bytes")
    same = places.revise(
        tmp_path, place, producer="mapper v1", map_bytes=CLOUD, map_format="ply", frame=first.frame, now=T + 10
    )
    reply = view.scene(tmp_path, place)[1]
    assert (reply["revision"], reply["splat_revision"]) == (same.id, rev_id)
    assert reply["splat"] == f"/places/splat?place={place}&revision={rev_id}&format=sog"
    assert reply["splat_stale"] is None
    # offered, and labelled as an earlier map's: when that map was made, and why this one has none of its own
    assert (reply["splat_state"], reply["splat_at"], reply["splat_wait"]["reason"]) == ("earlier", T, "pass")
    moved = places.revise(tmp_path, place, producer="mapper v1", map_bytes=CLOUD, map_format="ply", now=T + 20)
    reply = view.scene(tmp_path, place)[1]
    assert (reply["revision"], reply["splat"], reply["splat_revision"]) == (moved.id, None, None)
    assert reply["splat_stale"] == {"revision": rev_id, "at": T}
    assert (reply["splat_state"], reply["splat_at"], reply["splat_wait"]["reason"]) == ("stale", None, "pass")


def test_the_scene_names_the_revisions_splat_once_it_has_one(tmp_path, mapped):
    place, rev_id = mapped
    places.write_splat(tmp_path, places.revisions(tmp_path, place)[-1], master=b"ply", web=b"sog-bytes")
    reply = view.scene(tmp_path, place)[1]
    assert reply["splat"] == f"/places/splat?place={place}&revision={rev_id}&format=sog"
    assert (reply["splat_format"], reply["splat_bytes"]) == ("sog", len(b"sog-bytes"))
    assert (reply["splat_state"], reply["splat_at"], reply["splat_wait"]) == ("current", T, None)


@pytest.mark.parametrize("reason", ["few_frames", "failed", "no_runtime"])
def test_the_scene_gives_the_splat_passs_own_reason_once_it_has_tried_this_map(tmp_path, mapped, reason):
    place, _ = mapped
    rev = places.revisions(tmp_path, place)[-1]
    places.write_splat_note(tmp_path, rev, reason, T + 5)
    wait = view.scene(tmp_path, place)[1]["splat_wait"]
    assert (wait["reason"], wait["tried"]) == (reason, T + 5)
    # a newer map hasn't been tried: the note was about the one before it
    places.revise(tmp_path, place, producer="mapper v1", map_bytes=CLOUD, map_format="ply", now=T + 10)
    wait = view.scene(tmp_path, place)[1]["splat_wait"]
    assert (wait["reason"], wait["tried"]) == ("pass", None)


def test_a_map_from_before_maps_said_where_their_points_came_from_has_no_sources(tmp_path, mapped):
    assert view.scene(tmp_path, mapped[0])[1]["sources"] is None


def test_the_scene_says_each_camera_and_views_share_of_the_map_and_the_splat(tmp_path, mapped):
    place, rev_id = mapped
    rev = places.revisions(tmp_path, place)[-1]
    views = [{"id": "a", "camera": "porch", "t": 1.0}, {"id": "b", "camera": "hall", "t": 2.0}]
    views.append({"id": "c", "camera": "porch", "t": 3.0})
    places.write_sources(tmp_path, rev, {"views": views, "points": [10, 20, 30], "gaussians": [1, 2, 3]})
    sources = view.scene(tmp_path, place)[1]["sources"]
    # no splat yet: its Gaussians aren't counted, and there's no views file to read
    assert sources["splat_views"] is None
    assert sources["cameras"] == [
        {"name": "hall", "points": 20, "gaussians": None, "views": [1]},
        {"name": "porch", "points": 40, "gaussians": None, "views": [0, 2]},
    ]
    places.write_splat(tmp_path, rev, master=b"ply", web=b"sog-bytes", views=b"\x00\x00\x01\x00")
    sources = view.scene(tmp_path, place)[1]["sources"]
    assert sources["splat_views"] == f"/places/splat?place={place}&revision={rev_id}&format=views"
    assert [(v["camera"], v["points"], v["gaussians"]) for v in sources["views"]] == [
        ("porch", 10, 1),
        ("hall", 20, 2),
        ("porch", 30, 3),
    ]
    assert [(c["name"], c["gaussians"]) for c in sources["cameras"]] == [("hall", 2), ("porch", 4)]


def test_a_map_with_no_cameras_says_the_pass_has_nothing_to_make_a_splat_from(tmp_path, mapped):
    place, _ = mapped
    places.revise(tmp_path, place, producer="mapper v1", map_bytes=CLOUD, map_format="ply", cameras={}, now=T + 10)
    reply = view.scene(tmp_path, place)[1]
    assert (reply["splat_state"], reply["splat_wait"]["reason"]) == ("none", "no_cameras")


def test_the_scene_says_what_its_splat_was_trained_from_as_the_trainer_named_it(tmp_path, mapped):
    place, _ = mapped
    first = places.revisions(tmp_path, place)[-1]
    places.write_splat(tmp_path, first, master=b"ply", web=b"sog-bytes")
    assert view.scene(tmp_path, place)[1]["splat_from"] is None  # a splat nobody said the source of
    later = places.revise(tmp_path, place, producer="mapper v1", frame=first.frame, now=T + 10)
    places.write_splat(tmp_path, later, master=b"ply", web=b"sog-bytes", source="cameras")
    reply = view.scene(tmp_path, place)[1]
    assert (reply["splat_revision"], reply["splat_from"]) == (later.id, "cameras")


def test_an_owner_plan_drawn_on_a_map_with_no_plan_of_its_own_is_outlined_in_the_floors_frame(tmp_path, mapped):
    # An imported plan drawn in a fused map's floor frame (the map has no plan file, so no plan transform)
    place, _ = mapped
    bare = places.revise(tmp_path, place, producer="mapper v1", map_bytes=CLOUD, map_format="ply", now=T + 5)
    places.revise(tmp_path, place, producer="mapper v1", floor=(0.0, 0.0, 1.0, 0.0), now=T + 6)
    level = places.latest(tmp_path, place)
    assert bare.plan_transform is None and level is not None
    mine = PLAN.replace('"Den"', '"Lounge"')
    view.correct({"place": place, "plan": mine, "base": level.id}, True, tmp_path, now=T + 10)
    reply = view.scene(tmp_path, place)[1]
    assert (reply["plan"], reply["plan_source"]) == (mine, "owner")
    assert reply["plan_transform"] == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_a_place_without_a_map_has_no_scene_and_an_unknown_one_is_a_404(tmp_path, home):
    code, reply = view.scene(tmp_path, home[0])
    assert (code, reply["place"], reply["revision"]) == (200, home[0], None)
    assert view.scene(tmp_path, "no-such-place")[0] == 404
    assert view.map_file(tmp_path, home[0], home[1]) is None


def test_the_map_is_served_in_ranges_cached_for_good_and_404_for_another_revision(
    tmp_path, mapped, serve_dashboard, monkeypatch
):
    place, rev = mapped
    monkeypatch.setattr(context, "STATE", tmp_path)
    host = urlsplit(serve_dashboard()).netloc

    def get(path, **headers):
        conn = http.client.HTTPConnection(host, timeout=120)  # generous: a loaded machine is slow, not broken
        conn.request("GET", path, headers=headers)
        r = conn.getresponse()
        body = r.read()
        conn.close()
        return r, body

    url = f"/places/map?place={place}&revision={rev}"
    r, body = get(url)
    assert (r.status, body) == (200, CLOUD)
    assert r.getheader("ETag") == f'"{rev}"' and "immutable" in (r.getheader("Cache-Control") or "")
    r, body = get(url, Range="bytes=10-19")
    assert (r.status, body, r.getheader("Content-Range")) == (206, CLOUD[10:20], f"bytes 10-19/{len(CLOUD)}")
    assert get(url, **{"If-None-Match": f'"{rev}"'})[0].status == 304
    assert get(f"/places/map?place={place}&revision=not-a-revision")[0].status == 404
    splat = f"/places/splat?place={place}&revision={rev}&format=sog"
    assert get(splat)[0].status == 404  # none made yet
    places.write_splat(tmp_path, places.revisions(tmp_path, place)[-1], master=b"ply", web=b"sog-bytes")
    r, body = get(splat)
    # the ETag names the format too: a copy kept from another encoding of this revision's splat doesn't match it
    assert (r.status, body, r.getheader("ETag")) == (200, b"sog-bytes", f'"{rev}.sog"')
    assert "immutable" in (r.getheader("Cache-Control") or "")
    assert get(splat, **{"If-None-Match": f'"{rev}.sog"'})[0].status == 304
    assert get(splat, **{"If-None-Match": f'"{rev}"'})[0].status == 200
    assert get(f"/places/splat?place={place}&revision={rev}&format=spz")[0].status == 404
    views = f"/places/splat?place={place}&revision={rev}&format=views"
    assert get(views)[0].status == 404  # that splat doesn't say which view each Gaussian came from


def test_a_splats_views_are_served_cached_for_good(tmp_path, mapped, serve_dashboard, monkeypatch):
    place, rev = mapped
    monkeypatch.setattr(context, "STATE", tmp_path)
    places.write_splat(tmp_path, places.revisions(tmp_path, place)[-1], master=b"ply", web=b"sog", views=b"\x01\x00")
    conn = http.client.HTTPConnection(urlsplit(serve_dashboard()).netloc, timeout=120)
    conn.request("GET", f"/places/splat?place={place}&revision={rev}&format=views")
    r = conn.getresponse()
    assert (r.status, r.read(), r.getheader("ETag")) == (200, b"\x01\x00", f'"{rev}.views"')
    conn.close()


def test_one_persons_tracks_on_two_cameras_stay_two_paths(tmp_path, reader, home):
    reader.rows = [
        position(T + 1, entity="person:dana", name="Dana", track="cam0:7", x=1.0, y=1.0),
        position(T + 1, entity="person:dana", name="Dana", track="cam1:3", x=2.0, y=1.5),  # a metre off, same second
        position(T + 2, entity="person:dana", name="Dana", track="cam0:7", x=1.2, y=1.0),
    ]
    reply = view.positions_view(window(home[0]), tmp_path)[1]
    assert reply["entities"] == [{"id": "person:dana", "name": "Dana", "kind": "person"}]
    assert [(t["entity"], t["track"], [p["x"] for p in t["points"]]) for t in reply["tracks"]] == [
        ("person:dana", "cam0:7", [1.0, 1.2]),
        ("person:dana", "cam1:3", [2.0]),
    ]


# An owner plan drawn on its own page: the page's x and y become the plan frame's y + 10 and x, the page
# mirrored and given a quarter turn, as an imported plan's fit does.
PAGE = plans.scene('plan "Home" walls 0.2/0.1\nroom den "Den" living rect 0,0 4x6\n')
TURNED = {"turn": 1.5707963267948966, "shift": [10.0, 0.0], "mirror": True}
IN_FRAME = plans.scene('plan "Home" walls 0.2/0.1\nroom den "Den" living poly 10,0 16,0 16,4 10,4\n')


@pytest.fixture
def paged(tmp_path, home):
    place, frame = home
    places.correct_plan(tmp_path, place, IN_FRAME, now=T + 1, base=frame, view={"plan": PAGE, **TURNED})
    return place, frame


def test_a_plan_drawn_on_its_own_page_is_listed_as_drawn_and_kept_in_its_frame_underneath(tmp_path, paged):
    place, _ = paged
    row = view.listing(tmp_path, now=T + 3600)[1]["places"][0]
    assert (row["plan"], row["plan_source"]) == (PAGE, "owner")
    held = places.get(tmp_path, place)
    assert held is not None and held.plan == IN_FRAME, "positions and the 3D scene still read the plan frame"


def test_positions_are_turned_onto_the_page_the_plan_is_drawn_on(tmp_path, reader, paged):
    reader.rows = [position(T + 1, entity="person:dana", x=12.0, y=1.0)]  # page (1, 2): mirrored, then turned
    reply = view.positions_view(window(paged[0]), tmp_path)[1]
    assert [(p["x"], p["y"]) for p in reply["tracks"][0]["points"]] == [(1.0, 2.0)]


def test_an_edit_of_a_plan_drawn_on_its_page_is_moved_into_the_plan_frame(tmp_path, paged):
    place, frame = paged
    edited = PAGE.replace("4x6", "4x5")
    code, reply = view.correct({"place": place, "plan": edited, "base": "ignored"}, True, tmp_path, now=T + 10)
    assert (code, reply["plan"]) == (200, edited)
    held = places.get(tmp_path, place)
    assert held is not None and held.view is not None and held.view.plan == edited
    assert held.plan_revision == frame, "still drawn in the frame the page was registered to"
    (room,) = plan_scene.rooms_of(held.plan or "")
    assert set(room[2]) == {(10.0, 0.0), (10.0, 4.0), (15.0, 4.0), (15.0, 0.0)}


def test_a_view_that_isnt_one_is_refused(tmp_path, home):
    place, frame = home
    with pytest.raises(ValueError, match="view"):
        places.correct_plan(tmp_path, place, IN_FRAME, base=frame, view={"plan": PAGE, "turn": "a quarter"})


def test_the_positions_reply_counts_the_cameras_with_a_pose_in_the_place(tmp_path, reader, home, mapped):
    # `home` has a plan and no camera in its map, so nobody can be placed there; `mapped` registers one
    unposed, posed = home[0], mapped[0]
    assert view.positions_view(window(unposed), tmp_path)[1]["cameras"] == 0
    assert view.positions_view(window(posed), tmp_path)[1]["cameras"] == 1


# ---- the learned map's state (#259): what the 3D view says about learning, from the map step's summary


def told(monkeypatch, **summary):
    """Stand in for the map step's summary (spatial/surroundings_map.py's `summary`), in the shape it gives."""
    whole = {"on": False, "sampling": [], "kept": 0, "waiting": 0, "running": False, "places": {}, **summary}
    monkeypatch.setattr(view, "_summary", lambda state: whole)


def learned(state, place=None, now=T):
    """A revision as the map step writes one, with its splat: (the place, the revision)."""
    first = places.latest(state, place) if place else None
    rev = places.revise(
        state, place, producer="surroundings v1", map_bytes=CLOUD, map_format="ply",
        frame=first.frame if first else None, now=now,
    )  # fmt: skip
    places.write_splat(state, rev, master=b"ply", web=b"sog-bytes", source="surroundings")
    return rev.place, rev


@pytest.mark.parametrize(
    ("summary", "learning"),
    [
        pytest.param({}, {"on": False, "cameras": 0, "improving": False}, id="off"),
        pytest.param(
            {"on": True, "sampling": ["cam0", "cam1"]}, {"on": True, "cameras": 2, "improving": False}, id="on"
        ),
        pytest.param(
            {"on": True, "sampling": ["cam0"], "waiting": 3},
            {"on": True, "cameras": 1, "improving": True},
            id="on-with-frames-waiting",
        ),
    ],
)
def test_a_place_with_no_map_still_says_whether_learning_is_on(tmp_path, vision, monkeypatch, summary, learning):
    place = places.revise(tmp_path, None, producer="owner plan", plan=PLAN, now=T).place
    told(monkeypatch, **summary)
    reply = view.scene(tmp_path, place)[1]
    assert reply == {"place": place, "revision": None, "learning": {**learning, "views": None, "improved": None}}


def test_a_learned_map_says_its_views_and_when_it_last_improved_not_another_places(tmp_path, vision, monkeypatch):
    place, rev = learned(tmp_path)
    other = {"revision": "r-other", "views": 99, "rounds": 9, "improved": T + 500}
    mine = {"revision": rev.id, "views": 42, "rounds": 3, "improved": T - 60}
    told(monkeypatch, places={"another-place": other, place: mine})
    reply = view.scene(tmp_path, place)[1]
    assert reply["learning"] == {"on": False, "cameras": 0, "improving": False, "views": 42, "improved": T - 60}
    # learning is off: the splat is this map's own, and nothing is being added to it
    assert (reply["splat_state"], reply["splat_from"], reply["splat_wait"]) == ("current", "surroundings", None)


@pytest.mark.parametrize(
    ("summary", "state"),
    [
        pytest.param({"on": True, "running": True}, "improving", id="a-step-is-running"),
        pytest.param({"on": True, "waiting": 2}, "improving", id="frames-are-waiting"),
        pytest.param({"on": True}, "current", id="on-with-nothing-to-add"),
        pytest.param({"on": False, "waiting": 2}, "current", id="off-with-frames-left-waiting"),
    ],
)
def test_a_learned_splat_is_improving_only_while_learning_is_adding_to_it(
    tmp_path, vision, monkeypatch, summary, state
):
    place, _ = learned(tmp_path)
    told(monkeypatch, **summary)
    assert view.scene(tmp_path, place)[1]["splat_state"] == state


def test_a_splat_made_some_other_way_is_never_called_improving(tmp_path, mapped, monkeypatch):
    place, _ = mapped
    places.write_splat(tmp_path, places.revisions(tmp_path, place)[-1], master=b"ply", web=b"sog-bytes")
    told(monkeypatch, on=True, running=True)
    assert view.scene(tmp_path, place)[1]["splat_state"] == "current"


def test_a_newer_learned_splat_is_the_one_the_scene_names(tmp_path, vision, monkeypatch):
    told(monkeypatch, on=True)
    place, first = learned(tmp_path)
    assert view.scene(tmp_path, place)[1]["splat_revision"] == first.id
    _, second = learned(tmp_path, place, now=T + 30)
    reply = view.scene(tmp_path, place)[1]
    assert (reply["revision"], reply["splat_revision"]) == (second.id, second.id)
    assert reply["splat"] == f"/places/splat?place={place}&revision={second.id}&format=sog"


def test_a_revision_that_kept_the_map_shows_that_maps_splat_as_its_own(tmp_path, vision, monkeypatch):
    # the floor or a camera set on a learned map is a revision with no map of its own: the splat is still this map's
    told(monkeypatch)
    place, rev = learned(tmp_path)
    floored = places.revise(tmp_path, place, producer="floor v1", floor=FLOOR, now=T + 10)
    reply = view.scene(tmp_path, place)[1]
    assert (reply["revision"], reply["splat_revision"]) == (floored.id, rev.id)
    assert (reply["splat_state"], reply["splat_wait"]) == ("current", None)


def test_a_learned_map_without_its_splat_yet_says_its_on_its_way(tmp_path, vision, monkeypatch):
    told(monkeypatch, on=True, running=True)
    rev = places.revise(tmp_path, None, producer="surroundings v1", map_bytes=CLOUD, map_format="ply", now=T)
    reply = view.scene(tmp_path, rev.place)[1]
    assert (reply["splat_state"], reply["splat_wait"]["reason"]) == ("none", "learned")


def test_a_learned_map_on_a_place_with_an_owners_plan_still_serves_the_owners_plan(tmp_path, monkeypatch):
    told(monkeypatch)
    # A place as a reset leaves it: listed with the owner's plan and no map. Then the cameras found its first map.
    kept = places.revise(tmp_path, None, producer="places reset v1", now=T - 100)
    mine = PLAN.replace('"Den"', '"Lounge"')
    places.correct_plan(tmp_path, kept.place, mine, now=T - 90)
    _, rev = learned(tmp_path, kept.place)
    reply = view.scene(tmp_path, kept.place)[1]
    assert (reply["revision"], reply["plan"], reply["plan_source"]) == (rev.id, mine, "owner")
    # not lined up with the map yet: nothing to lay the plan on it by, and why
    assert (reply["plan_aligned"], reply["plan_transform"]) == (False, None)
    assert (reply["plan_lineup"]["state"], reply["plan_lineup"]["reason"]) == ("unaligned", "no_floor")
    row = view.listing(tmp_path, now=T + 60)[1]["places"][0]
    assert (row["plan"], row["plan_source"], row["machine_plan"], row["machine_newer"]) == (mine, "owner", None, False)


def test_a_machine_plan_made_after_the_owners_is_never_served_in_its_place(tmp_path, mapped):
    place, rev = mapped
    mine = PLAN.replace('"Den"', '"Lounge"')
    view.correct({"place": place, "plan": mine, "base": rev}, True, tmp_path, now=T + 10)
    places.revise(
        tmp_path, place, producer="mapper v1", plan=PLAN.replace("4x6", "4x5"), map_bytes=CLOUD, map_format="ply",
        now=T + 100,
    )  # fmt: skip
    reply = view.scene(tmp_path, place)[1]
    assert (reply["plan"], reply["plan_source"], reply["plan_transform"]) == (mine, "owner", None)
    assert reply["plan_lineup"] is None, "a map with a plan of the machine's isn't one the line-up works on"


def test_an_owners_plan_lined_up_with_the_map_is_laid_on_it(tmp_path, mapped):
    place, rev = mapped
    mine = PLAN.replace('"Den"', '"Lounge"')
    view.correct({"place": place, "plan": mine, "base": rev}, True, tmp_path, now=T + 10)
    reply = view.scene(tmp_path, place)[1]
    assert (reply["plan"], reply["plan_source"], reply["plan_aligned"]) == (mine, "owner", True)
    assert reply["plan_transform"] == list(TRANSFORM)
