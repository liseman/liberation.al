"""Known › Places' reads (dashboard/places.py, #2848 and #2831): the places listing and one place's positions."""

import http.client
import json
from urllib.parse import urlsplit

import numpy as np
import pytest

from liberational import segment_log as S
from liberational.dashboard import places as view
from liberational.dashboard import server
from liberational.memory import places
from liberational.spatial import floorplan, ply, positions

T = 1_762_500_000.0
PLAN = 'plan "Home" walls 0.2/0.1\nroom den "Den" living rect 0,0 4x6\n'


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


def seen(vision, at, *, camera=None, place=None, end=None):
    """A position sample as spatial/positions.py writes it (#2905)."""
    vision.add(
        S.POSITIONS, ts_start=at, ts_end=at if end is None else end, cam=camera or "rig", camera=camera, place=place,
        source="fixed-camera", confidence=0.8,
    )  # fmt: skip


def test_a_place_only_its_cameras_watch_lists_when_anyone_was_last_seen_there(tmp_path, mapped, vision):
    # #2905: a visit is the owner's (the rig was there), so a place only fixed cameras watch had none and read "No
    # visits yet" with hundreds of tracks on its plan.
    watched, _ = mapped  # registers cam0
    tagged = places.revise(tmp_path, None, producer="cameras-v1", now=T + 10).place
    seen(vision, T + 100, camera="cam0", end=T + 120)
    seen(vision, T + 500, camera="cam9")  # a camera no place registered: nobody's activity
    seen(vision, T + 300, place=tagged)
    rows = {r["id"]: r for r in view.listing(tmp_path, now=T + 3600)[1]["places"]}
    assert (rows[watched]["last_activity"], rows[watched]["last_visit"]) == (T + 120, None)
    assert rows[tagged]["last_activity"] == T + 300


@pytest.mark.parametrize(("ago", "found"), [(3 * 86400, True), (8 * 86400, False)], ids=["this-week", "older"])
def test_last_activity_looks_back_a_week(tmp_path, mapped, vision, ago, found):
    now = T + 30 * 86400  # mapped registered cam0
    seen(vision, now - ago, camera="cam0")
    row = view.listing(tmp_path, now=now)[1]["places"][0]
    assert row["last_activity"] == (now - ago if found else None)


def test_a_place_found_in_a_newer_pass_has_its_cameras_left_out_of_the_older_ones(
    tmp_path, mapped, vision, monkeypatch
):
    # A busy place's cameras are read for the hour it was found in, not the week the other places still need.
    watched, _ = mapped  # registers cam0
    never = places.revise(tmp_path, None, producer="walkthrough-v1", now=T + 10).place  # no cameras: never found
    now = T + 30 * 86400
    seen(vision, now - 60, camera="cam0")
    asked = []
    read = positions.samples
    monkeypatch.setattr(
        positions, "samples", lambda *a, cameras=None: asked.append(cameras) or read(*a, cameras=cameras)
    )
    rows = {r["id"]: r for r in view.listing(tmp_path, now=now)[1]["places"]}
    assert (rows[watched]["last_activity"], rows[never]["last_activity"]) == (now - 60, None)
    assert asked == [["cam0", positions.RIG], [positions.RIG], [positions.RIG]]


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

    def __call__(self, state, start, end, place=None, entity=None, every=None, frame_revision=None):
        self.asked.append((start, end, place, entity, every, frame_revision))
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
    # #2981: "face:3" is someone, if unnamed: its own entry, not folded in with the strangers
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
    # #2981: cam0:4 alone, then a fused point (no track of its own), then porch:7 alone: one path, in time order (a
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
    # #3070: a foot on an occluding edge no other camera settled can be metres off
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
    # walk is every step, not a line between two points 43.2 s apart (#3296)
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


def test_the_listing_route_answers_json_and_a_page_load_gets_the_app(serve):
    host = urlsplit(serve(server.H)).netloc

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
    assert (row["plan"], row["plan_source"], row["machine_plan"]) == (
        mine,
        "owner",
        PLAN,
    )  # the machine's kept, to compare
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
        pytest.param({"plan": 'plan "x"\ndoor a>b w1\n'}, True, 400, id="no-room"),
        pytest.param({"plan": PLAN + "#" * view.MAX_PLAN_BYTES}, True, 400, id="too-long"),
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
    assert (row["plan_revision"], row["machine_newer"]) == (first, True)
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


def test_revert_puts_back_the_previous_owner_plan_and_a_second_revert_undoes_it(tmp_path, home):
    place = home[0]
    hand = PLAN.replace('"Den"', '"Study"')
    view.correct({"place": place, "plan": hand}, True, tmp_path, now=T + 10)
    view.correct({"place": place, "plan": TWO_ROOMS}, True, tmp_path, now=T + 20)
    code, reply = view.plan_versions(tmp_path, place)
    assert code == 200
    assert [v["rooms"] for v in reply["versions"]] == [1, 2]
    previous = reply["previous"]
    assert (previous["id"], previous["rooms"], previous["at"]) == (reply["versions"][0]["id"], 1, T + 10)

    code, reply = view.restore({"place": place, "record": previous["id"]}, True, tmp_path, now=T + 30)
    assert (code, reply["plan"], reply["plan_source"]) == (200, hand, "owner")
    # appended, not rewound: three records now, and the previous one is the two-room plan it replaced
    reply = view.plan_versions(tmp_path, place)[1]
    assert [v["rooms"] for v in reply["versions"]] == [1, 2, 1]
    assert (reply["previous"]["rooms"], reply["previous"]["at"]) == (2, T + 20)


def test_the_only_owner_plan_has_nothing_earlier_to_revert_to(tmp_path, home):
    view.correct({"place": home[0], "plan": TWO_ROOMS}, True, tmp_path, now=T + 10)
    assert view.plan_versions(tmp_path, home[0])[1]["previous"] is None


def test_an_earlier_owner_plan_with_the_same_text_is_skipped(tmp_path, home):
    place = home[0]
    view.correct({"place": place, "plan": PLAN.replace('"Den"', '"Study"')}, True, tmp_path, now=T + 10)
    view.correct({"place": place, "plan": TWO_ROOMS}, True, tmp_path, now=T + 20)
    view.correct({"place": place, "plan": TWO_ROOMS}, True, tmp_path, now=T + 30)
    assert view.plan_versions(tmp_path, place)[1]["previous"]["at"] == T + 10


def test_a_re_registration_of_the_same_plan_isnt_a_version_to_revert_to(tmp_path, home):
    # Place 1: a hand plan, then an imported one, then that imported plan registered again into a newer frame, which
    # moves its corners by millimetres. Revert goes back to the hand plan, not to the same plan in the older frame.
    place = home[0]
    view.correct({"place": place, "plan": PLAN.replace('"Den"', '"Study"')}, True, tmp_path, now=T + 10)
    view.correct({"place": place, "plan": TWO_ROOMS}, True, tmp_path, now=T + 20)
    view.correct({"place": place, "plan": TWO_ROOMS.replace("4,0 2x6", "4.003,0 2x6")}, True, tmp_path, now=T + 30)
    assert view.plan_versions(tmp_path, place)[1]["previous"]["at"] == T + 10


@pytest.mark.parametrize(
    "a,b,same",
    [
        pytest.param("rect 0,0 4x6", "rect 0.004,-0.003 4x6", True, id="millimetres"),
        pytest.param("rect 0,0 4x6", "rect 0.05,0 4x6", False, id="one-snap-step"),
        pytest.param('room den "Den" rect 0,0 4x6', 'room den "Study" rect 0,0 4x6', False, id="renamed"),
        pytest.param("room a2 rect 0,0 4x6", "room a3 rect 0,0 4x6", False, id="numbered-id"),
        pytest.param("rect 0,0 4x6", "rect 0,0 4x6\ndoor a>b w1", False, id="door-added"),
        pytest.param(
            "rect 0,0 4x6\nx-camera a at 1,1 yaw 7", "rect 0,0 4x6\nx-camera a at 1,1 yaw 90", True, id="camera"
        ),
    ],
)
def test_same_plan_allows_only_millimetres(a, b, same):
    assert view.same_plan(a, b) is same


def test_with_the_machines_plan_holding_revert_offers_the_newest_owner_plan(tmp_path, home):
    place = home[0]
    view.correct({"place": place, "plan": TWO_ROOMS}, True, tmp_path, now=T + 10)
    view.correct({"place": place, "plan": ""}, True, tmp_path, now=T + 20)  # withdrawn, as the old bar did
    reply = view.plan_versions(tmp_path, place)[1]
    assert [v["withdrawn"] for v in reply["versions"]] == [False, True]
    assert (reply["previous"]["rooms"], reply["previous"]["at"]) == (2, T + 10)


def test_a_place_never_corrected_has_no_versions(tmp_path, home):
    assert view.plan_versions(tmp_path, home[0]) == (200, {"place": home[0], "versions": [], "previous": None})
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
        tmp_path, None, producer="tier1 fuse v1", plan=PLAN, map_bytes=CLOUD, map_format="ply", floor=FLOOR,
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


def test_a_newer_map_in_the_same_frame_shows_the_last_splat_and_one_in_another_frame_names_it_stale(tmp_path, mapped):
    # #2877: a nightly fusion with too few lit frames for a splat kept the map's frame, so the last splat still fits.
    place, rev_id = mapped
    first = places.revisions(tmp_path, place)[-1]
    places.write_splat(tmp_path, first, master=b"ply", web=b"spz-bytes")
    same = places.revise(
        tmp_path, place, producer="tier1 fuse v1", map_bytes=CLOUD, map_format="ply", frame=first.frame, now=T + 10
    )
    reply = view.scene(tmp_path, place)[1]
    assert (reply["revision"], reply["splat_revision"]) == (same.id, rev_id)
    assert reply["splat"] == f"/places/splat?place={place}&revision={rev_id}" and reply["splat_stale"] is None
    moved = places.revise(tmp_path, place, producer="tier1 fuse v1", map_bytes=CLOUD, map_format="ply", now=T + 20)
    reply = view.scene(tmp_path, place)[1]
    assert (reply["revision"], reply["splat"], reply["splat_revision"]) == (moved.id, None, None)
    assert reply["splat_stale"] == {"revision": rev_id, "at": T}


def test_the_scene_names_the_revisions_splat_once_it_has_one(tmp_path, mapped):
    place, rev_id = mapped
    places.write_splat(tmp_path, places.revisions(tmp_path, place)[-1], master=b"ply", web=b"spz-bytes")
    reply = view.scene(tmp_path, place)[1]
    assert reply["splat"] == f"/places/splat?place={place}&revision={rev_id}"
    assert (reply["splat_format"], reply["splat_bytes"]) == ("spz", len(b"spz-bytes"))


def test_the_scene_carries_the_owners_plan_only_in_the_frame_it_was_drawn_in(tmp_path, mapped):
    place, rev = mapped
    mine = PLAN.replace('"Den"', '"Living room"')
    view.correct({"place": place, "plan": mine, "base": rev}, True, tmp_path, now=T + 10)
    assert view.scene(tmp_path, place)[1]["plan"] == mine
    # A new map and plan (a new frame): the 3D view shows them, so the outline is the machine's plan in that frame.
    places.revise(
        tmp_path,
        place,
        producer="tier1 fuse v1",
        plan=PLAN.replace("4x6", "4x5"),
        map_bytes=CLOUD,
        map_format="ply",
        now=T + 100,
    )
    reply = view.scene(tmp_path, place)[1]
    assert (reply["plan_source"], "4x5" in reply["plan"]) == ("machine", True)


def test_an_owner_plan_drawn_on_a_map_with_no_plan_of_its_own_is_outlined_in_the_floors_frame(tmp_path, mapped):
    # #3276: an imported plan drawn in a fused map's floor frame (the map has no plan file, so no plan transform)
    place, _ = mapped
    bare = places.revise(tmp_path, place, producer="tier1 fuse v1", map_bytes=CLOUD, map_format="ply", now=T + 5)
    places.revise(tmp_path, place, producer="tier1 fuse v1", floor=(0.0, 0.0, 1.0, 0.0), now=T + 6)
    level = places.latest(tmp_path, place)
    assert bare.plan_transform is None and level is not None
    mine = PLAN.replace('"Den"', '"Lounge"')
    view.correct({"place": place, "plan": mine, "base": level.id}, True, tmp_path, now=T + 10)
    reply = view.scene(tmp_path, place)[1]
    assert (reply["plan"], reply["plan_source"]) == (mine, "owner")
    assert reply["plan_transform"] == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_a_place_without_a_map_has_no_scene_and_an_unknown_one_is_a_404(tmp_path, home):
    assert view.scene(tmp_path, home[0]) == (200, {"place": home[0], "revision": None})
    assert view.scene(tmp_path, "no-such-place")[0] == 404
    assert view.map_file(tmp_path, home[0], home[1]) is None


def test_the_map_is_served_in_ranges_cached_for_good_and_404_for_another_revision(tmp_path, mapped, serve, monkeypatch):
    place, rev = mapped
    monkeypatch.setattr(server, "STATE", tmp_path)
    host = urlsplit(serve(server.H)).netloc

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
    assert get(f"/places/splat?place={place}&revision={rev}")[0].status == 404  # none made yet
    places.write_splat(tmp_path, places.revisions(tmp_path, place)[-1], master=b"ply", web=b"spz-bytes")
    r, body = get(f"/places/splat?place={place}&revision={rev}")
    assert (r.status, body, r.getheader("ETag")) == (200, b"spz-bytes", f'"{rev}"')
    assert "immutable" in (r.getheader("Cache-Control") or "")


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


# An owner plan drawn on its own page (#3276): the page's x and y become the plan frame's y + 10 and x, the page
# mirrored and given a quarter turn, as an imported plan's fit does.
PAGE = 'plan "Home" walls 0.2/0.1\nroom den "Den" living rect 0,0 4x6\n'
TURNED = {"turn": 1.5707963267948966, "shift": [10.0, 0.0], "mirror": True}
IN_FRAME = 'plan "Home" walls 0.2/0.1\nroom den "Den" living poly 10,0 16,0 16,4 10,4\n'


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
    (room,) = floorplan.rooms_from_dsl(held.plan or "")
    assert set(room[2]) == {(10.0, 0.0), (10.0, 4.0), (15.0, 4.0), (15.0, 0.0)}


def test_a_view_that_isnt_one_is_refused(tmp_path, home):
    place, frame = home
    with pytest.raises(ValueError, match="view"):
        places.correct_plan(tmp_path, place, IN_FRAME, base=frame, view={"plan": PAGE, "turn": "a quarter"})


def floor_points(x0, x1, y0, y1, step=0.05):
    """A floor of map points over plan [x0, x1) x [y0, y1) under TRANSFORM (plan = map + (2, 3))."""
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.c_[xs.ravel() - 2.0, ys.ravel() - 3.0, np.zeros(xs.size)]


def walked_place(tmp_path, walked_den, walked_hall, *, producer="tier1 fuse v1", signals=None):
    den, hall = floor_points(0, 4, 0, 6), floor_points(4, 6, 0, 6)
    walks = np.r_[np.full(len(den), int(walked_den)), np.full(len(hall), int(walked_hall))]
    cloud = ply.dumps(np.r_[den, hall], counts={"seen": np.ones(len(walks), np.int64), "walked": walks})
    rev = places.revise(
        tmp_path, None, producer=producer, plan=TWO_ROOMS, map_bytes=cloud, map_format="ply", floor=FLOOR,
        plan_transform=TRANSFORM, signals={"walkthrough": ["walk-a"]} if signals is None else signals, now=T,
    )  # fmt: skip
    return rev.place


def listed(tmp_path, place):
    return {r["id"]: r for r in view.listing(tmp_path, now=T + 60)[1]["places"]}[place]


def test_a_place_never_walked_lists_no_walks_and_every_room_unwalked(tmp_path, vision):
    row = listed(tmp_path, walked_place(tmp_path, False, False, signals={}))
    assert row["walks"] == 0
    assert row["unwalked_rooms"] == [{"id": "den", "name": "Den"}, {"id": "hall", "name": "Hall"}]


def test_a_walked_place_lists_the_rooms_its_walks_didnt_cover(tmp_path, vision):
    row = listed(tmp_path, walked_place(tmp_path, True, False))
    assert row["walks"] == 1
    assert row["unwalked_rooms"] == [{"id": "hall", "name": "Hall"}]


def test_a_place_walked_everywhere_has_no_unwalked_rooms(tmp_path, vision):
    assert listed(tmp_path, walked_place(tmp_path, True, True))["unwalked_rooms"] == []


def test_a_walked_place_without_a_fused_map_cant_say_which_rooms(tmp_path, vision):
    row = listed(tmp_path, walked_place(tmp_path, True, False, producer="cut3r walkthrough"))
    assert row["walks"] == 1 and row["unwalked_rooms"] is None


def test_a_place_without_a_plan_cant_say_which_rooms(tmp_path, vision):
    rev = places.revise(tmp_path, None, producer="cameras-v1", now=T)
    assert listed(tmp_path, rev.place)["unwalked_rooms"] is None
