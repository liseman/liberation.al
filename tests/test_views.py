import tomllib

import pytest

from liberational import config as cfg
from liberational.vision import views


def test_named_camera_places_survive_parsing():
    assert views.parse(["entryway: Front door", "rear-2: Back yard"]) == {
        "entryway": "Front door",
        "rear-2": "Back yard",
    }


def test_configured_ids_win_over_legacy_aliases():
    assert views.parse(["0: zero", "cam00: padded", "CAM0: upper"], ["0", "cam00", "CAM0"]) == {
        "0": "zero",
        "cam00": "padded",
        "CAM0": "upper",
    }
    assert views.parse(["Cam 0: hall", "cam02: den"]) == {"cam0": "hall", "cam2": "den"}


@pytest.mark.parametrize("name", ["", "bad/name", "two words", "café", "cam0\n"])
def test_camera_id_rejects_unsafe_tokens(name):
    assert not views.valid_camera_id(name)


def test_manifest_exact_camera_names_and_missing(tmp_path):
    path = tmp_path / "devices.toml"
    assert views.manifest_camera_names(path) is None
    path.write_text(
        'device = [{name="entryway", kind="rtsp"}, {name="0", kind="camera"}, '
        '{name="cam00", kind="camera"}, {name="CAM0", kind="camera"}, '
        '{name="mic", kind="microphone"}]'
    )
    assert views.manifest_camera_names(path) == ["entryway", "0", "cam00", "CAM0"]


def test_disabled_camera_keeps_history_identity_but_is_absent_from_live_shell(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    path = tmp_path / "config" / "devices.toml"
    path.write_text('device = [{name="entryway", kind="rtsp", enabled=false}, {name="desk", kind="camera"}]')
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    assert views.camera_ids() == ["entryway", "desk"]
    assert views.camera_ids(active_only=True) == ["desk"]


@pytest.mark.parametrize(
    "text",
    ["device = [", 'device = "wrong"', 'device = [{kind="rtsp"}]', 'device = [{name="bad/name", kind="camera"}]'],
)
def test_invalid_manifest_is_actionable(tmp_path, text):
    path = tmp_path / "devices.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="devices.toml"):
        views.manifest_camera_names(path)


def test_camera_ids_come_only_from_the_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    path = tmp_path / "config" / "devices.toml"
    assert views.camera_ids() == []  # no manifest, no cameras
    path.write_text("device = []")
    assert views.camera_ids() == []
    path.write_text('device = [{name="cam00", kind="rtsp"}]')
    assert views.camera_ids() == ["cam00"]
    with cfg.override({"vision": {"camera_names": ["cam00: Den"]}}):
        assert views.names() == {"cam00": "Den"}


@pytest.mark.parametrize("name", ["entryway", "rear-2", "0", "cam00", "CAM0", "__proto__"])
def test_camera_id_accepts_exact_native_tokens(name):
    assert views.valid_camera_id(name)


def test_place_save_preserves_configured_ambiguous_tokens(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "config").mkdir(parents=True)
    (root / "config" / "devices.toml").write_text(
        'device = [{name="0", kind="rtsp"}, {name="cam00", kind="camera"}, {name="CAM0", kind="camera"}]'
    )
    monkeypatch.setattr(cfg, "ROOT", root)
    pack = tmp_path / "pack.toml"
    monkeypatch.setattr(cfg, "local_config", lambda: pack)
    cfg.save({"vision.camera_names": "0: Zero, cam00: Padded, CAM0: Upper, entryway: Front"})
    assert tomllib.loads(pack.read_text())["vision"]["camera_names"] == [
        "0: Zero",
        "cam00: Padded",
        "CAM0: Upper",
        "entryway: Front",
    ]


@pytest.mark.parametrize(
    "question,want",
    [
        ("camera 0 at noon", ["0"]),
        ("who was here at 0:30 in the last 2 hours", []),
        ("camera entryway-extra now", ["entryway-extra"]),
        ("camera entryway-extension now", []),
        ("what did entryway show yesterday", ["entryway"]),
        ("what happened on entryway yesterday", ["entryway"]),
        ("entryway yesterday", []),
        ("cam00 last hour", ["cam00"]),
        ("CAM0 last hour", ["CAM0"]),
    ],
)
def test_configured_camera_mentions_are_exact_and_contextual(tmp_path, monkeypatch, question, want):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(
        'device=[{name="entryway",kind="rtsp"},{name="entryway-extra",kind="camera"},'
        '{name="0",kind="camera"},{name="cam00",kind="camera"},{name="CAM0",kind="camera"}]'
    )
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    cameras, remaining = views.configured_camera_mentions(question)
    assert cameras == want
    assert len(remaining) == len(question)


def test_configured_camera_reference_is_masked_before_time_parsing(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text('device=[{name="year",kind="rtsp"}]')
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    cameras, remaining = views.configured_camera_mentions("what did camera year show yesterday")
    assert cameras == ["year"]
    assert "year" not in remaining and "yesterday" in remaining


@pytest.fixture
def named_camera_manifest(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(
        'device=[{name="garden",kind="rtsp"},{name="reading-room",kind="camera"},'
        '{name="back_porch",kind="camera"},{name="0",kind="camera"}]'
    )
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    with cfg.override({"vision": {"camera_names": ["garden: Green room", "reading-room: Library"]}}):
        yield


@pytest.mark.parametrize(
    "question,want",
    [
        ("what did the garden camera show today", ["garden"]),
        ("what happened in the garden today", []),
        ("what happened in the garden camera today", ["garden"]),
        ("what happened on the Garden today", ["garden"]),
        ("what did camera GARDEN show today", ["garden"]),
        ("what happened from the reading room camera today", ["reading-room"]),
        ("what happened in the back porch today", []),
        ("what happened in the green room today", []),
        ("what happened on the green room today", ["garden"]),
        ("what did the library camera show today", ["reading-room"]),
        ("garden today", []),
        ("what happened in the kitchen today", []),
        ("what happened in the last 0 hours at 0:30", []),
        ("what did camera 0 show today", ["0"]),
    ],
)
def test_natural_configured_camera_selectors(named_camera_manifest, question, want):
    selected, remaining = views.configured_camera_mentions(question)
    assert selected == want
    assert len(remaining) == len(question)
    assert "today" in remaining or not want


def test_ambiguous_camera_name_requires_actual_camera_id(named_camera_manifest):
    with (
        cfg.override({"vision": {"camera_names": ["garden: Library", "reading-room: Library"]}}),
        pytest.raises(ValueError, match="[Aa]mbiguous.*camera"),
    ):
        views.configured_camera_mentions("what did the library camera show today")


def test_explicit_native_id_wins_over_a_conflicting_camera_name(named_camera_manifest):
    with cfg.override({"vision": {"camera_names": ["reading-room: garden"]}}):
        assert views.configured_camera_mentions("what did camera garden show today")[0] == ["garden"]
        with pytest.raises(ValueError, match="[Aa]mbiguous.*camera"):
            views.configured_camera_mentions("what happened on the garden today")


def test_case_colliding_native_ids_need_exact_explicit_id(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(
        'device=[{name="Garden",kind="camera"},{name="garden",kind="camera"}]'
    )
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    assert views.configured_camera_mentions("camera Garden today")[0] == ["Garden"]
    assert views.configured_camera_mentions("camera garden today")[0] == ["garden"]
    with pytest.raises(ValueError, match="[Aa]mbiguous.*camera"):
        views.configured_camera_mentions("on the GARDEN today")


@pytest.mark.parametrize("question", ["what happened in the morning", "what happened in the night"])
def test_a_daypart_camera_name_does_not_select_a_camera(named_camera_manifest, question):
    with cfg.override({"vision": {"camera_names": ["garden: Morning", "reading-room: Night"]}}):
        assert views.configured_camera_mentions(question)[0] == []
        assert views.configured_camera_mentions("what did the night camera show")[0] == ["reading-room"]


def test_longer_spoken_camera_id_owns_its_full_phrase(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(
        'device=[{name="garden",kind="camera"},{name="garden-room",kind="camera"}]'
    )
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    assert views.configured_camera_mentions("camera garden room today")[0] == ["garden-room"]
    assert views.configured_camera_mentions("on the garden room today")[0] == ["garden-room"]


@pytest.mark.parametrize(
    "question",
    [
        "What did Dana say about camping in the garden yesterday?",
        "What did Dana discuss about planting in the green room yesterday?",
        "What did Dana talk about reading in the reading room yesterday?",
    ],
)
def test_a_room_in_a_speech_topic_does_not_select_a_camera(named_camera_manifest, question):
    selected, remaining = views.configured_camera_mentions(question)
    assert selected == []
    assert remaining == question


@pytest.mark.parametrize(
    "question",
    [
        "What did Dana say about the garden camera yesterday?",
        "What did Dana say about camping on camera garden yesterday?",
        "What did Dana say about camping in the garden camera yesterday?",
    ],
)
def test_explicit_camera_inside_speech_topic_still_selects(named_camera_manifest, question):
    assert views.configured_camera_mentions(question)[0] == ["garden"]
