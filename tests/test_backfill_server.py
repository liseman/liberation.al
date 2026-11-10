"""The dashboard prefers who the face worker found (its visits on the vision log) for history_at and last seen."""

from liberational.dashboard import context, live_status, people_list


def test_recorded_visits_come_first_and_give_last_seen(tmp_path, monkeypatch, vision):
    monkeypatch.setattr(context, "STATE", tmp_path)
    vision.visit("Alice", 1000, 1100, cam="cam0", best_sim=0.8, samples=9)
    vision.visit("Bob", 5000, 5100, cam="cam1", best_sim=0.7, samples=4)
    faces = [{"name": "Face 3", "sim": 0.0, "bearing_deg": 10}, {"name": "Alice", "sim": 0.5, "bearing_deg": 2}]
    live = {"ts": 1050, "window": 30, "cams": {"cam0": {"ts": 1049, "dt": -1, "caption": "a desk", "faces": faces}}}
    out = live_status._with_recorded_people(live, 1050, 30)
    faces = out["cams"]["cam0"]["faces"]
    assert [(f["name"], f.get("source")) for f in faces] == [("Alice", "recordings"), ("Face 3", None)]  # one Alice
    assert "cam1" not in out["cams"]  # Bob was seen at another time
    assert people_list._face_last_seen_at().get("bob") == (5100, "cam1")  # the camera it was seen on, for People
