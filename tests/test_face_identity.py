"""Last seen per person and camera from see-actions' face visits, read through the registry (people/face_identity.py)."""

from liberational import segment_log
from liberational.knowledge_eventlog import SQLiteEventLog
from liberational.people import face_identity, registry


def test_last_seen_resolves_all_visits_from_one_current_registry_read(tmp_path, vision, monkeypatch):
    reg = registry.at(tmp_path)
    for name in ("Alice", "Dana"):
        reg.ensure(name, "test")
        visit(vision, name, 100, 110)
        visit(vision, name, 200, 210)
    assert isinstance(reg.log, SQLiteEventLog)
    # Registry.refresh() takes its snapshot from identity_version(), the identity streams' generation.
    original = reg.log.identity_version
    reads = []

    def identity_version():
        reads.append(None)
        return original()

    monkeypatch.setattr(reg.log, "identity_version", identity_version)
    assert {r["name"] for r in face_identity.last_seen(tmp_path)["matches"]} == {"Alice", "Dana"}
    assert len(reads) == 1, "one identity snapshot must serve the whole visit projection"


def test_warm_visits_keep_both_holders_of_a_reused_name_and_new_registry_names(tmp_path, vision, monkeypatch):
    from liberational.domain import kernel

    monkeypatch.setattr(kernel, "now", lambda: 100.0)
    reg = registry.at(tmp_path)
    reg.ensure("Alice", "test")
    visit(vision, "Alice", 150, 160)
    visit(vision, "Dana", 150, 170)
    assert [r["name"] for r in face_identity.last_seen(tmp_path)["matches"]] == ["Alice"]
    monkeypatch.setattr(kernel, "now", lambda: 200.0)
    reg.rename("Alice", "Alicia", "test")
    reg.ensure("Alice", "test")
    reg.ensure("Dana", "test")
    visit(vision, "Alice", 250, 260)
    assert [(r["name"], r["ts"]) for r in face_identity.last_seen(tmp_path)["matches"]] == [
        ("Alice", 260.0),
        ("Dana", 170.0),
        ("Alicia", 160.0),
    ]


def test_current_visits_do_not_show_future_rows_and_tail_matches_expire(tmp_path, vision, monkeypatch):
    visit(vision, "Alice", 100, 110)
    visit(vision, "Dana", 3100, 3110)
    vision.add(segment_log.FACE_MATCHES, ts_start=2990, name="Alice", cam="cam0", sim=0.8)
    monkeypatch.setattr(face_identity.time, "time", lambda: 3000.0)
    assert [(r["name"], r["ts"]) for r in face_identity.last_seen(tmp_path)["matches"]] == [("Alice", 2990.0)]
    monkeypatch.setattr(face_identity.time, "time", lambda: 3120.0)
    assert [(r["name"], r["ts"]) for r in face_identity.last_seen(tmp_path)["matches"]] == [
        ("Dana", 3110.0),
        ("Alice", 110.0),
    ]


def test_historical_name_resolution_uses_clamped_end_after_current_read(tmp_path, vision, monkeypatch):
    from liberational.domain import kernel

    monkeypatch.setattr(kernel, "now", lambda: 100.0)
    reg = registry.at(tmp_path)
    reg.ensure("Alice", "test")
    visit(vision, "Alice", 150, 400)
    monkeypatch.setattr(kernel, "now", lambda: 200.0)
    reg.rename("Alice", "Alicia", "test")
    monkeypatch.setattr(kernel, "now", lambda: 300.0)
    reg.ensure("Alice", "test")
    assert [r["name"] for r in face_identity.last_seen(tmp_path)["matches"]] == ["Alice"]
    assert face_identity.last_seen(tmp_path, end=250)["matches"] == [
        {"name": "Alicia", "cam": "cam0", "ts": 250, "sim": 0.9}
    ]


def visit(vision, name, start, end, cam="cam0", sim=0.9):
    return vision.add(segment_log.FACE_VISITS, ts_start=start, ts_end=end, name=name, cam=cam, best_sim=sim, samples=3)


def test_the_newest_visit_per_person_and_camera_newest_first(tmp_path, vision):
    reg = registry.at(tmp_path)
    for name in ("Alice", "Dana"):
        reg.ensure(name, "test")
    visit(vision, "Alice", 100, 110, sim=0.7)
    visit(vision, "Alice", 200, 230, sim=0.8)
    visit(vision, "Alice", 150, 160, cam="cam1")
    visit(vision, "Dana", 300, 305, sim=0.95)
    got = face_identity.last_seen(tmp_path)
    assert got["pending"] is False
    assert got["matches"] == [
        {"name": "Dana", "cam": "cam0", "ts": 305.0, "sim": 0.95},
        {"name": "Alice", "cam": "cam0", "ts": 230.0, "sim": 0.8},
        {"name": "Alice", "cam": "cam1", "ts": 160.0, "sim": 0.9},
    ]


def test_a_window_and_cameras_narrow_it_and_a_visit_running_past_the_end_counts_to_the_end(tmp_path, vision):
    registry.at(tmp_path).ensure("Alice", "test")
    visit(vision, "Alice", 100, 110)
    visit(vision, "Alice", 200, 260)
    visit(vision, "Alice", 205, 215, cam="cam1")
    assert [r["ts"] for r in face_identity.last_seen(tmp_path, start=0, end=150)["matches"]] == [110.0]
    assert face_identity.last_seen(tmp_path, start=150, end=250)["matches"] == [
        {"name": "Alice", "cam": "cam0", "ts": 250.0, "sim": 0.9},
        {"name": "Alice", "cam": "cam1", "ts": 215.0, "sim": 0.9},
    ]
    assert [r["cam"] for r in face_identity.last_seen(tmp_path, cameras=["cam1"])["matches"]] == ["cam1"]
    assert face_identity.last_seen(tmp_path, start=300)["matches"] == []


def test_a_renamed_person_shows_under_the_new_name_and_a_forgotten_one_drops_out(tmp_path, vision, monkeypatch):
    from liberational.domain import kernel

    monkeypatch.setattr(kernel, "now", lambda: 100.0)  # the registry's events happen at 100
    reg = registry.at(tmp_path)
    for name in ("Alice", "Dana"):
        reg.ensure(name, "test")
    visit(vision, "Alice", 150, 160)  # matched as Alice, before the rename
    visit(vision, "Dana", 150, 170)
    visit(vision, "Zed", 150, 180)  # a name the registry never had
    monkeypatch.setattr(kernel, "now", lambda: 200.0)
    reg.rename("Alice", "Alicia", "dashboard")
    reg.forget("Dana", "dashboard")
    assert face_identity.last_seen(tmp_path)["matches"] == [{"name": "Alicia", "cam": "cam0", "ts": 160.0, "sim": 0.9}]
    reg.ensure("Alice", "dashboard")  # someone new takes the old name: the old visit stays Alicia's
    assert [r["name"] for r in face_identity.last_seen(tmp_path)["matches"]] == ["Alicia"]


def test_without_a_registry_the_names_are_as_matched(tmp_path, vision, monkeypatch):
    monkeypatch.setattr(registry, "log_exists", lambda state=None: False)
    visit(vision, "Alice", 100, 110)
    assert [r["name"] for r in face_identity.last_seen(tmp_path)["matches"]] == ["Alice"]


def test_someone_still_in_view_is_seen_now_from_the_matches_not_their_last_closed_visit(tmp_path, vision):
    """A visit is published once its run ends, so an hour of continuous matches has no visit yet:
    the newest matches within TAIL_S of the window's end stand for the open run."""
    registry.at(tmp_path).ensure("Alice", "test")
    visit(vision, "Alice", 100, 110)  # her last closed visit
    for ts in (3000, 3010, 3020):  # still in view: matched every ten seconds, no visit published
        vision.add(segment_log.FACE_MATCHES, ts_start=ts, cam="cam0", name="Alice", sim=0.8, detection=f"d-{ts}")
    vision.add(
        segment_log.FACE_MATCHES, ts_start=3015, cam="cam0", cluster="Face 3", sim=0.3, detection="d-x"
    )  # nobody
    got = face_identity.last_seen(tmp_path, end=3030)
    assert got["matches"] == [{"name": "Alice", "cam": "cam0", "ts": 3020.0, "sim": 0.8}]
    # Matches older than the tail are the visits' business: a window ending long after them reads the visit.
    assert [r["ts"] for r in face_identity.last_seen(tmp_path, end=3020 + face_identity.TAIL_S + 1)["matches"]] == [
        110.0
    ]
