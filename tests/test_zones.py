"""Owner-drawn zones on a camera's view: checking a zone, saving a camera's set, and its audit trail."""

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from liberational import knowledge
from liberational.dashboard import zones as routes
from liberational.see import zones

T = 1_800_000_000.0
BOX = [[0.1, 0.6], [0.3, 0.6], [0.3, 0.9], [0.1, 0.9]]  # a litter box in the lower left


def stored(state):
    """The zone sets as the knowledge store keeps them."""
    from liberational.people import registry

    return knowledge.Store(registry.knowledge_path(state)).query(zones.COLLECTION)


def litter(**kw):
    return {"name": "Litter box", "polygon": BOX, **kw}


def test_a_new_camera_saves_its_first_set_and_reads_it_back(tmp_path, vision):
    saved = zones.define(tmp_path, "cam0", zones.parse_zones([litter()]), None, now=T)
    now = zones.current(tmp_path)
    assert list(now) == ["cam0"]
    assert [(z.name, z.polygon) for z in now["cam0"].zones] == [
        ("Litter box", ((0.1, 0.6), (0.3, 0.6), (0.3, 0.9), (0.1, 0.9)))
    ]
    assert now["cam0"].id == saved.id
    (row,) = stored(tmp_path)
    assert (row.record.id, row.record.key, row.record.value["source"]) == (saved.id, "cam0", "owner")


def test_an_edit_keeps_the_zone_id_and_the_old_version_stays_in_history(tmp_path, vision):
    first = zones.define(tmp_path, "cam0", zones.parse_zones([litter()]), None, now=T)
    zone = first.zones[0]
    renamed = zones.parse_zones([{"id": zone.id, "name": "Cat box", "polygon": BOX}])
    second = zones.define(tmp_path, "cam0", renamed, first.id, now=T + 60)
    assert second.zones[0].id == zone.id
    assert [[z.name for z in d.zones] for d in zones.history(tmp_path, "cam0")] == [
        ["Litter box"],
        ["Cat box"],
    ]


def test_a_save_from_a_stale_version_is_refused_and_writes_nothing(tmp_path, vision):
    first = zones.define(tmp_path, "cam0", zones.parse_zones([litter()]), None, now=T)
    zones.define(tmp_path, "cam0", (), first.id, now=T + 10)
    with pytest.raises(zones.StaleBase):
        zones.define(tmp_path, "cam0", zones.parse_zones([litter()]), first.id, now=T + 20)
    assert len(stored(tmp_path)) == 2
    assert zones.current(tmp_path)["cam0"].zones == ()


def test_reading_zones_of_a_state_with_no_knowledge_store_finds_none_and_creates_nothing(tmp_path, vision):
    from liberational.people import registry

    assert zones.current(tmp_path) == {}
    assert zones.history(tmp_path, "cam0") == []
    assert not registry.knowledge_path(tmp_path).exists()


def test_a_set_imported_from_the_broker_export_is_the_cameras_current_set(tmp_path, vision):
    # pack-knowledge import writes the broker's zone.defined records as they were exported: the payload id, the
    # camera as key, the header time, and the body; the zones read after the catalog reset are those.
    from liberational.people import registry

    body = {"cam": "garden", "zones": [{"id": "zone-1a2b3c", **litter()}], "source": "owner"}
    at = round(T * 1e9)
    record = knowledge.Record("0190a5b8-0000-7000-8000-00000000000a", zones.COLLECTION, "garden", at, at, at, body)
    knowledge.Store(registry.knowledge_path(tmp_path)).put([record])
    ((cam, now),) = zones.current(tmp_path).items()
    assert (cam, now.id, now.at, [z.id for z in now.zones]) == ("garden", record.id, T, ["zone-1a2b3c"])
    edited = zones.define(tmp_path, "garden", (), now.id, now=T + 5)  # and the next save edits from it
    assert zones.current(tmp_path)["garden"].id == edited.id


def test_cameras_keep_separate_sets(tmp_path, vision):
    zones.define(tmp_path, "cam0", zones.parse_zones([litter()]), None, now=T)
    zones.define(tmp_path, "cam1", zones.parse_zones([litter(name="Food bowl")]), None, now=T + 1)
    now = zones.current(tmp_path)
    assert {cam: [z.name for z in d.zones] for cam, d in now.items()} == {"cam0": ["Litter box"], "cam1": ["Food bowl"]}


def test_a_version_saved_after_a_clock_step_back_still_comes_last(tmp_path, vision):
    first = zones.define(tmp_path, "cam0", zones.parse_zones([litter()]), None, now=T)
    zones.define(tmp_path, "cam0", (), first.id, now=T - 5)
    assert zones.current(tmp_path)["cam0"].zones == ()


@pytest.mark.parametrize(
    ("zone", "message"),
    [
        pytest.param({"polygon": BOX}, "needs a name", id="no-name"),
        pytest.param(litter(name="  "), "needs a name", id="blank-name"),
        pytest.param(litter(name="x" * 61), "at most 60", id="long-name"),
        pytest.param(litter(polygon=BOX[:2]), "3 to 32 corners", id="two-corners"),
        pytest.param(litter(polygon=[[0.1, 0.1]] * 33), "3 to 32 corners", id="too-many-corners"),
        pytest.param(litter(polygon=[[0.1, 0.1], [1.2, 0.1], [0.5, 0.5]]), "0 to 1", id="off-frame"),
        pytest.param(litter(polygon=[[0.1, 0.1], [True, 0.1], [0.5, 0.5]]), "numbers", id="bool-corner"),
        pytest.param(litter(polygon=[[0.1, 0.1], [float("nan"), 0.1], [0.5, 0.5]]), "numbers", id="nan-corner"),
        pytest.param(litter(polygon=[[0.1, 0.1], [0.105, 0.1], [0.105, 0.105]]), "too small", id="tiny"),
        pytest.param(litter(polygon=[[0.1, 0.1], [0.5, 0.5], [0.5, 0.1], [0.1, 0.5]]), "crosses itself", id="bowtie"),
        pytest.param(litter(id="not-ours"), "zone id", id="foreign-id"),
    ],
)
def test_parse_zone_refuses(zone, message):
    with pytest.raises(ValueError, match=message):
        zones.parse_zone(zone)


def test_two_zones_on_a_camera_cannot_share_a_name_ignoring_case():
    with pytest.raises(ValueError, match="share a name"):
        zones.parse_zones([litter(), litter(name="LITTER BOX")])


def test_a_camera_has_at_most_sixteen_zones():
    many = [litter(name=f"Zone {i}") for i in range(17)]
    with pytest.raises(ValueError, match="at most 16"):
        zones.parse_zones(many)


@pytest.mark.parametrize(
    ("x", "y", "inside"),
    [(0.2, 0.75, True), (0.05, 0.75, False), (0.2, 0.5, False), (0.35, 0.95, False)],
)
def test_contains_a_point_in_the_box(x, y, inside):
    assert zones.contains(zones.parse_zone(litter()).polygon, x, y) is inside


def test_contains_follows_a_concave_zone():
    ell = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.2], [0.2, 0.2], [0.2, 0.5], [0.1, 0.5]]  # an L
    polygon = zones.parse_zone(litter(polygon=ell)).polygon
    assert (zones.contains(polygon, 0.15, 0.4), zones.contains(polygon, 0.4, 0.4)) == (True, False)


COORD = st.floats(min_value=0.05, max_value=0.95)


@given(st.tuples(COORD, COORD), st.floats(0.01, 0.04), st.integers(3, 12), st.floats(0, 6.28))
def test_a_regular_polygon_contains_its_centre_and_not_points_outside_its_circle(centre, radius, sides, turn):
    import math

    cx, cy = centre
    r = min(radius, cx, cy, 1 - cx, 1 - cy)
    corners = [
        [cx + r * math.cos(turn + 2 * math.pi * k / sides), cy + r * math.sin(turn + 2 * math.pi * k / sides)]
        for k in range(sides)
    ]
    polygon = tuple((float(x), float(y)) for x, y in corners)
    assert zones.simple(polygon)
    assert zones.contains(polygon, cx, cy)
    assert not zones.contains(polygon, cx + r * 1.01, cy)


def test_the_routes_save_list_and_show_history(tmp_path, vision):
    code, reply = routes.save({"cam": "cam0", "zones": [litter()], "base": None}, True, tmp_path)
    assert (code, reply["ok"]) == (200, True)
    base = reply["definition"]["id"]
    assert routes.listing(tmp_path)[1]["cameras"]["cam0"]["zones"][0]["name"] == "Litter box"
    assert routes.save({"cam": "cam0", "zones": [], "base": base}, True, tmp_path)[0] == 200
    code, reply = routes.history("cam0", tmp_path)
    assert (code, [len(v["zones"]) for v in reply["versions"]]) == (200, [0, 1])


@pytest.mark.parametrize(
    ("body", "control", "code"),
    [
        pytest.param({"cam": "cam0", "zones": [litter()], "base": None}, False, 403, id="not-this-machine"),
        pytest.param(None, True, 400, id="not-an-object"),
        pytest.param({"zones": [litter()], "base": None}, True, 400, id="no-cam"),
        pytest.param({"cam": "cam0", "zones": [litter(name="")], "base": None}, True, 400, id="bad-zone"),
        pytest.param({"cam": "cam0", "zones": [litter()], "base": "someone-elses"}, True, 409, id="stale"),
    ],
)
def test_save_refuses_and_writes_nothing(tmp_path, vision, body, control, code):
    assert routes.save(body, control, tmp_path)[0] == code
    assert not vision.records


def test_history_needs_a_camera(tmp_path, vision):
    assert routes.history("", tmp_path)[0] == 400


@pytest.fixture
def http_(serve_dashboard, monkeypatch, tmp_path, vision):
    import http.client
    import json
    import urllib.parse

    from liberational.dashboard import context

    monkeypatch.setattr(context, "STATE", tmp_path)
    netloc = urllib.parse.urlsplit(serve_dashboard()).netloc

    def send(method, path, body=None, origin=None):
        conn = http.client.HTTPConnection(netloc, timeout=5)
        headers = {"Content-Type": "application/json", **({"Origin": origin} if origin else {})}
        conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        response = conn.getresponse()
        value = response.status, json.loads(response.read())
        conn.close()
        return value

    return send


def test_the_dashboard_serves_zone_saves_listing_and_history(http_):
    code, reply = http_("POST", "/zones", {"cam": "cam0", "zones": [litter()], "base": None})
    assert (code, reply["ok"]) == (200, True)
    code, listing = http_("GET", "/zones")
    assert (code, [z["name"] for z in listing["cameras"]["cam0"]["zones"]]) == (200, ["Litter box"])
    code, history = http_("GET", "/zones/history?cam=cam0")
    assert (code, [v["id"] for v in history["versions"]]) == (200, [reply["definition"]["id"]])


def test_the_dashboard_refuses_a_zone_save_from_another_site(http_, vision):
    body = {"cam": "cam0", "zones": [litter()], "base": None}
    assert http_("POST", "/zones", body, origin="http://elsewhere.example")[0] == 403
    assert not vision.records


# Visits: the pure pipeline, fed frames as the worker would.

IN = [0.18, 0.7, 0.22, 0.8]  # a box whose bottom centre (0.2, 0.8) is in the litter box
OUT = [0.58, 0.2, 0.62, 0.4]  # bottom centre (0.6, 0.4): elsewhere


def defined(at=T, zone_list=None):
    zs = zones.parse_zones(zone_list or [{"id": "zone-litter", "name": "Litter box", "polygon": BOX}])
    return zones.Definition(f"def-{at}", "cam0", zs, at)


def piped(definition=None):
    p = zones.ZonePipeline("cam0")
    p.define(definition or defined())
    return p


def frame(p, ts, *subjects, kind="animal", ok=True, label="cat"):
    """A frame at T + ts holding (track, box) subjects."""
    record = {
        "id": f"f-{ts}",
        "ts_start": T + ts,
        "boxes": [box for _, box in subjects],
        "track_ids": [track for track, _ in subjects],
        "ok": ok,
    }
    return p.frame(kind, record, [label] * len(subjects))


def run(p, frames):
    """Feed (ts, subjects...) frames; return every output."""
    out = []
    for ts, *subjects in frames:
        out.extend(frame(p, ts, *subjects))
    return out


def visits(out):
    return [o.record for o in out if o.kind == "visit"]


def every(start, stop, step=0.5):
    n = round((stop - start) / step)
    return [start + i * step for i in range(n + 1)]


def test_a_cat_walks_in_stays_ten_seconds_and_walks_out():
    p = piped()
    out = run(
        p, [(0, ("t1", OUT)), *[(t, ("t1", IN)) for t in every(1, 11)], *[(t, ("t1", OUT)) for t in every(11.5, 14)]]
    )
    assert [
        (v["ts_start"] - T, v["ts_end"] - T, v["dwell_s"], v["entry_seen"], v["exit_seen"], v["ended"])
        for v in visits(out)
    ] == [(1.0, 11.0, 10.0, True, True, "left")]
    v = visits(out)[0]
    assert (v["zone"], v["zone_name"], v["label"], v["subject_kind"], v["candidate"]) == (
        "zone-litter",
        "Litter box",
        "cat",
        "animal",
        True,
    )
    assert next(o for o in out if o.kind == "visit").parents[0] == "def-1800000000.0"


def test_a_frame_of_jitter_on_the_edge_does_not_end_the_visit():
    p = piped()
    out = run(
        p,
        [
            (0, ("t1", IN)),
            (0.5, ("t1", OUT)),
            (1, ("t1", IN)),
            (1.5, ("t1", IN)),
            *[(t, ("t1", OUT)) for t in every(2, 4)],
        ],
    )
    assert [(v["ts_start"] - T, v["ts_end"] - T) for v in visits(out)] == [(0.0, 1.5)]


def test_a_cat_hidden_inside_a_hooded_box_is_still_visiting():
    p = piped()
    out = run(
        p,
        [
            (0, ("t1", IN)),
            *[(t,) for t in every(0.5, 60)],
            (60.5, ("t1", IN)),
            *[(t, ("t1", OUT)) for t in every(61, 63)],
        ],
    )
    assert [(v["ts_start"] - T, v["ts_end"] - T, v["ended"]) for v in visits(out)] == [(0.0, 60.5, "left")]


def test_a_subject_never_seen_again_is_lost_at_its_last_inside_frame():
    p = piped()
    out = run(p, [(0, ("t1", IN)), *[(t,) for t in every(0.5, 121)]])
    assert [(v["ts_end"] - T, v["exit_seen"], v["ended"]) for v in visits(out)] == [(0.0, False, "lost")]


def test_first_seen_inside_means_the_entry_was_not_seen():
    p = piped()
    out = run(p, [(0, ("t1", IN)), *[(t, ("t1", OUT)) for t in every(0.5, 3)]])
    assert visits(out)[0]["entry_seen"] is False


def test_a_camera_gap_censors_open_visits_and_ends_the_coverage_stretch():
    p = piped()
    out = run(p, [(0, ("t1", IN)), (5, ("t1", IN)), (30, ("t1", IN))])
    assert [(v["ts_end"] - T, v["ended"]) for v in visits(out)] == [(5.0, "censored")]
    covered = [o.record for o in out if o.kind == "coverage"]
    assert [(c["ts_start"] - T, round(c["ts_end"] - T, 3), c["frames"]) for c in covered] == [(0.0, 5.001, 2)]


def test_a_failed_frame_or_an_untracked_box_in_a_zone_is_not_coverage():
    p = piped()
    run(p, [(0, ("t1", OUT))])
    frame(p, 0.5, ("t1", OUT), ok=False)
    frame(p, 1.0, (None, IN))
    frame(p, 1.5, (None, OUT))  # untracked, but nowhere near a zone: still coverage
    assert p.snapshot()["kinds"]["animal"]["stretch"]["frames"] == 2


def test_saving_new_zones_ends_open_visits():
    p = piped()
    run(p, [(0, ("t1", IN)), (1, ("t1", IN))])
    out = p.define(defined(at=T + 2))
    assert [(v["ended"], v["definition"]) for v in visits(out)] == [("zones_changed", "def-1800000000.0")]


def test_coverage_waits_for_the_earliest_open_visit():
    p = piped()
    out = run(
        p, [*[(t, ("t1", OUT)) for t in every(0, 59.5)], (60, ("t1", IN)), *[(t, ("t1", IN)) for t in every(60.5, 200)]]
    )
    covered = [o.record for o in out if o.kind == "coverage"]
    assert [(c["ts_start"] - T, c["ts_end"] - T) for c in covered] == [(0.0, 60.0)]
    assert visits(out) == []  # still in the box at 200 s: neither the visit nor coverage past its start is out


def test_people_and_animals_keep_separate_visits_and_coverage():
    p = piped()
    frame(p, 0, ("p1", IN), kind="person", label="person")
    frame(p, 0, ("t1", OUT))
    state = p.snapshot()["kinds"]
    assert (len(state["person"]["visits"]), len(state["animal"]["visits"])) == (1, 0)


def test_a_camera_without_zones_produces_nothing():
    p = zones.ZonePipeline("cam0")
    assert frame(p, 0, ("t1", IN)) == []
    assert p.snapshot()["kinds"]["animal"]["stretch"] is None


POS = st.sampled_from([IN, OUT, None])  # None: not detected in this frame


@st.composite
def histories(draw):
    """Up to 80 frames 0.5 to 15 s apart, each with up to two tracks somewhere or not detected."""
    ts, frames = 0.0, []
    for _ in range(draw(st.integers(1, 80))):
        ts += draw(st.sampled_from([0.5, 0.5, 0.5, 1.0, 3.0, 15.0]))
        subjects = [(t, b) for t in ("t1", "t2") if (b := draw(POS)) is not None]
        frames.append((ts, *subjects))
    return frames


@given(histories(), st.data())
def test_a_snapshot_taken_anywhere_resumes_to_the_same_records(frames, data):
    whole = piped()
    expected = run(whole, frames)
    split = data.draw(st.integers(0, len(frames)))
    first = piped()
    before = run(first, frames[:split])
    resumed = zones.ZonePipeline("cam0", state=first.snapshot())
    assert before + run(resumed, frames[split:]) == expected


@given(histories())
def test_visits_and_coverage_are_well_formed(frames):
    out = run(piped(), frames)
    for v in visits(out):
        assert v["ts_start"] <= v["ts_end"]
        assert v["dwell_s"] == pytest.approx(v["ts_end"] - v["ts_start"], abs=1e-3)
    by_track: dict[str, list] = {}
    for v in visits(out):
        by_track.setdefault(v["track_id"], []).append(v)
    for found in by_track.values():  # one track is in one zone once at a time
        spans = sorted((v["ts_start"], v["ts_end"]) for v in found)
        assert all(a[1] < b[0] for a, b in pairwise(spans))
    covered = sorted((o.record["ts_start"], o.record["ts_end"]) for o in out if o.kind == "coverage")
    assert all(a[1] <= b[0] for a, b in pairwise(covered))  # stretches never overlap
