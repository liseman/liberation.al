"""v1's visit rule (see.faces.VisitBuilder): one named person's matches on one camera, closed after a 10 s gap."""

from liberational.see.faces import VisitBuilder


def match(ts: float, name: str | None, cam: str = "cam0", sim: float = 0.6) -> dict:
    return {"ts_start": ts, "cam": cam, "name": name, "sim": sim}


def test_matches_join_into_a_visit_until_a_gap_over_10_seconds() -> None:
    v = VisitBuilder()
    assert v.feed(match(100.0, "Sam", sim=0.6)) == []
    assert v.feed(match(108.0, "Sam", sim=0.9)) == []
    assert v.feed(match(118.0, "Sam")) == []  # 10 s after the last: still the same visit
    closed = v.feed(match(128.5, "Sam"))
    assert closed == [{"ts_start": 100.0, "ts_end": 118.0, "name": "Sam", "cam": "cam0", "best_sim": 0.9, "samples": 3}]


def test_visits_are_per_person_and_camera_and_unnamed_matches_make_none() -> None:
    v = VisitBuilder()
    for m in [match(1.0, "Sam"), match(2.0, "Sam", cam="cam1"), match(3.0, "Ann"), match(4.0, None)]:
        v.feed(m)
    closed = v.close_before(20.0)
    assert sorted((r["name"], r["cam"]) for r in closed) == [("Ann", "cam0"), ("Sam", "cam0"), ("Sam", "cam1")]
    assert v.close_before(40.0) == []


def test_close_before_leaves_a_visit_still_within_its_gap_open() -> None:
    v = VisitBuilder()
    v.feed(match(100.0, "Sam"))
    assert v.close_before(110.0) == []
    assert [r["ts_end"] for r in v.close_before(110.5)] == [100.0]


def test_a_visit_keeps_its_three_best_recorded_faces_as_shots() -> None:
    v = VisitBuilder()

    def shot(n: int, quality: float) -> dict:
        return {"detection": f"d{n}", "ts": 100.0 + n, "bbox": [0.1, 0.1, 0.2, 0.2], "quality": quality}

    for n, quality in enumerate([0.3, 0.9, 0.1, 0.7, 0.5]):
        v.feed(match(100.0 + n, "Sam"), shot(n, quality))
    v.feed(match(105.0, "Sam"))  # a frame whose faces weren't recorded: no shot
    (visit,) = v.close_before(200.0)
    assert [s["detection"] for s in visit["shots"]] == ["d1", "d3", "d4"]


def test_a_visit_with_no_recorded_face_has_no_shots() -> None:
    v = VisitBuilder()
    v.feed(match(100.0, "Sam"))
    (visit,) = v.close_before(200.0)
    assert "shots" not in visit
