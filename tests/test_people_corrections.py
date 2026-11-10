"""People's corrections and "pick what's theirs": GET /inbox/unnamed, POST /people/assign, /people/correct,
and /people/undo, with the sighting corrections they rest on (vision/sightings.py: the owner's claims in the people
registry on the face worker's detections), the card's choices (people/card_prefs.py), and the face rows they add and
take back (vision/faces.py). A temp state dir, the voice CLI run in process, made-up names, and synthetic faces on the
vision log; never the real data/."""

import datetime as dt
import json
from unittest.mock import patch

import numpy as np
import pytest

from liberational import config as cfg
from liberational.dashboard import (
    call_log,
    context,
    inbox_state,
    known_voices,
    people_list,
    people_reads,
    people_writes,
    photo_files,
    server,
    speaker_lines,
    transcript_lines,
)
from liberational.domain.identity import Basis
from liberational.people import card_prefs, face_shots, registry, video_thumbs
from liberational.vision import faces as face_db
from liberational.vision import sightings as st
from test_people_unified import pack, ran, write_lines  # noqa: F401  # fixtures, used by name
from vision_fake import FakeVisionLog
from voices import write_voices

DAY = dt.datetime.combine(dt.date.today() - dt.timedelta(days=6), dt.time(12)).timestamp()  # within SHOT_DAYS
E = np.eye(512, dtype=np.float32)  # E[k]: a face, unlike every other k
S: dict[int, str] = {}  # the fixture's sightings: its number for each, as the tests name them -> its id


def sight(log: FakeVisionLog, n: int, ts: float, name: str | None, quality: float = 0.5, cam: str = "cam0", group=None):
    """Sighting n: a face the face worker matched to `name` (None: unnamed, in face group `group`)."""
    cluster = f"Face {group}" if group is not None else None
    S[n] = log.face(ts, cam, name, 0.8 if name else None, quality=quality, cluster=cluster)["id"]
    face_shots.write(cfg.path("paths.state"), S[n], "0", b"ready-jpeg")
    video_thumbs.write(cfg.path("paths.state"), cam, ts, "0", b"ready-scene-jpeg")


@pytest.fixture
def state(tmp_path, monkeypatch, pack, ran, vision):  # noqa: F811
    """The dashboard on its own state dir and face database: Kara (face E[1], three shots in one visit), Bob (face
    E[2]), unknown face group 7 (E[3]) seen twice, an unnamed clip of face group 9 on cam1, Kara's two lines, and
    unnamed voice 2 with a line."""
    s = tmp_path / "state"
    s.mkdir()
    for name, path in {}.items():
        monkeypatch.setattr(server, name, s / path)
    monkeypatch.setattr(speaker_lines, "VOICE_TAGS", s / "voice_tags.jsonl")
    monkeypatch.setattr(speaker_lines, "REFINE_LOG", s / "refine_log.jsonl")
    monkeypatch.setattr(people_list, "PEOPLE_ADDED", s / "people_added.json")
    monkeypatch.setattr(inbox_state, "INBOX_SKIPS", s / "inbox_skips.json")
    monkeypatch.setattr(people_list, "CONTACTS_FILE", s / "contacts.json")
    monkeypatch.setattr(call_log, "CALLS_DIR", s / "calls")
    monkeypatch.setattr(context, "STATE", s)
    monkeypatch.setattr(face_db, "DB", tmp_path / "faces" / "db.npz")
    monkeypatch.setattr(face_db, "UNK_DB", tmp_path / "faces" / "unknown.npz")
    monkeypatch.setattr(people_list, "_face_last_seen_at", dict)
    with cfg.override({"paths": {"state": str(s), "faces": str(tmp_path / "faces")}, "backfill": {"visit_gap_s": 30}}):
        face_db.save_db(["Kara", "Bob"], E[1:3])
        face_db._unk_save([7], E[3:4], [2], 8)
        S.clear()
        sight(vision, 1, DAY, "Kara", 0.9)
        sight(vision, 2, DAY + 5, "Kara", 0.6)
        sight(vision, 3, DAY + 10, "Kara", 0.3)
        sight(vision, 4, DAY + 400, "Bob")
        sight(vision, 5, DAY + 1000, None, cam="cam1", group=7)
        sight(vision, 6, DAY + 2000, None, 0.3, cam="cam1", group=9)
        sight(vision, 7, DAY + 2010, None, 0.9, cam="cam1", group=9)
        vision.visit("Kara", DAY, DAY + 10, cam="cam0", best_sim=0.8, samples=3)
        vision.visit("Bob", DAY + 400, DAY + 400, cam="cam0", best_sim=0.8)
        write_voices(s, {1: {"label": "Kara", "n": 2, "last_seen": 10.0}, 2: {"n": 3, "last_seen": 20.0}})
        write_lines(
            s,
            {
                "ts_start": DAY + 1,
                "ts_end": DAY + 2,
                "text": "Hi there.",
                "speaker": "Kara",
                "speaker_id": 1,
                "device": "Desk Mic",
                "audio": {"stream": "mic.audio", "start_ns": 0, "end_ns": 1000000000},
            },
            {
                "ts_start": DAY + 3,
                "ts_end": DAY + 4,
                "text": "Who's that?",
                "speaker": "Speaker 2",
                "speaker_id": 2,
                "speaker_sim": 0.7,
                "audio": {"stream": "mic.audio", "start_ns": 0, "end_ns": 1000000000},
            },
        )
        yield s


def card(name):
    out = people_reads.person_card(name)
    assert out is not None
    return out


def kara_shots(c):
    return [s["id"] for s in c["shots"]]


def names_of(state, *ns):
    """Who sightings ns (the fixture's numbers) are now."""
    named = {f["id"]: f["name"] for f in st.faces(DAY - 1, DAY + 3000, state=state)}
    return [named[S[n]] for n in ns]


# ---- the sightings ----
def test_a_correction_stands_over_the_match_and_comes_undone(state):
    prior = st.correct([S[2]], "Luca", state)  # the face worker matched it to Kara, but the owner said it's Luca
    assert names_of(state, 2) == ["Luca"]
    st.uncorrect(prior, state)
    assert names_of(state, 2) == ["Kara"]  # the match again


def test_unnamed_clips_are_stretches_nobody_named_is_in(state):
    clips = st.unnamed_clips(30, state=state)
    assert [(k["id"], k["ids"], k["thumb"]) for k in clips] == [
        (f"cam1@{DAY + 2000:.2f}", [S[6], S[7]], S[7]),  # the better shot shows
        (f"cam1@{DAY + 1000:.2f}", [S[5]], S[5]),
    ]


# ---- the pickers ----
def test_the_unnamed_list_has_faces_voices_and_clips(state):
    out = people_reads.people_unnamed()
    assert out["faces"] == [
        {
            "id": 7,
            "photos": [f"/faces/shot/{S[5]}.jpg"],
            "count": 2,
            "last_seen": DAY + 1000,
            "cam": "cam1",
            "camera_name": "cam1",
        }
    ]
    assert [(v["id"], v["label"], v["sample"]["text"]) for v in out["voices"]] == [
        (2, "Unnamed voice A", "Who's that?")
    ]
    assert [v["id"] for v in out["videos"]] == [f"cam1@{DAY + 2000:.2f}", f"cam1@{DAY + 1000:.2f}"]


def test_assigning_picks_names_them_all_and_one_undo_puts_them_back(state):
    code, out = people_writes.assign_to(
        {"name": "Sam", "faces": [7], "voices": [2], "videos": [f"cam1@{DAY + 2000:.2f}"]}
    )
    assert code == 200, out
    assert (out["name"], out["assigned"], out["failed"]) == ("Sam", {"faces": 1, "voices": 1, "videos": 1}, [])
    assert "Sam" in [p["name"] for p in out["people"]]
    assert face_db.unknown_groups() == []
    assert [str(n) for n in face_db.load_db()[0]].count("Sam") == 1  # group 7's face
    assert known_voices._speakers()["speakers"][1]["label"] == "Sam"
    assert names_of(state, 6, 7) == ["Sam", "Sam"]

    code, back = people_writes.undo_people({"undo": out["undo"]})
    assert code == 200 and back["ok"], back
    assert face_db.unknown_groups() == [{"id": 7, "count": 2}]
    assert [str(n) for n in face_db.load_db()[0]] == ["Kara", "Bob"]
    assert known_voices._speakers()["speakers"][1]["label"] == "Speaker 2"
    assert names_of(state, 6, 7) == [None, None]
    assert people_writes.undo_people({"undo": out["undo"]})[0] == 409


@pytest.mark.parametrize("kind, ident", [("face", 7), ("voice", 2)])
def test_claim_undo_returns_the_group_to_the_unnamed_list(state, kind, ident):
    code, out = people_writes.claim_identity(kind, ident, "Sam", control=True)
    assert code == 200, out
    assert out["undo"]
    assert ident not in [item["id"] for item in people_reads.people_unnamed()[kind + "s"]]

    code, back = people_writes.undo_people({"undo": out["undo"]})
    assert code == 200 and back["ok"], back
    assert ident in [item["id"] for item in people_reads.people_unnamed()[kind + "s"]]
    assert people_writes.undo_people({"undo": out["undo"]})[0] == 409


def test_claim_undo_preserves_an_existing_persons_other_voice(state):
    before = registry.at(state).person_of(registry.voiceprint(1)).person
    code, out = people_writes.claim_identity("voice", 2, "Kara", control=True)
    assert code == 200, out
    assert [voice["label"] for voice in known_voices._speakers()["speakers"]] == ["Kara", "Kara"]

    code, back = people_writes.undo_people({"undo": out["undo"]})
    assert code == 200 and back["ok"], back
    assert [voice["label"] for voice in known_voices._speakers()["speakers"]] == ["Kara", "Speaker 2"]
    assert registry.at(state).person_of(registry.voiceprint(1)).person == before


def test_claim_undo_restores_face_gallery_and_the_unknown_photo(state):
    n = face_db.MAX_PER_NAME
    face_db.save_db(["Kara"] * n, E[100 : 100 + n])
    photo_files._unknown_photo(7).parent.mkdir(parents=True, exist_ok=True)
    photo_files._unknown_photo(7).write_bytes(b"unknown crop")
    photo = photo_files._photo_dir("Kara") / "1000000000000.jpg"
    photo.parent.mkdir(parents=True, exist_ok=True)
    photo.write_bytes(b"taught crop")
    reg = registry.at(state)
    reg.claim(registry.photo(photo.stem), "Kara", Basis.OWNER_TAG, "test")
    prior_person = reg.person_of(registry.photo(photo.stem)).person

    code, out = people_writes.claim_identity("face", 7, "Kara", control=True)
    assert code == 200, out
    assert not photo_files._unknown_photo(7).exists()
    code, back = people_writes.undo_people({"undo": out["undo"]})
    assert code == 200 and back["ok"], back
    assert face_db.unknown_groups() == [{"id": 7, "count": 2}]
    names, embeddings = face_db.load_db()
    assert list(names) == ["Kara"] * n
    assert sorted(int(np.argmax(e)) for e in embeddings) == list(range(100, 100 + n))
    assert photo_files._unknown_photo(7).read_bytes() == b"unknown crop"
    assert photo_files._photos("Kara") == [photo]
    assert photo.read_bytes() == b"taught crop"
    assert reg.person_of(registry.photo(photo.stem)).person == prior_person


@pytest.mark.parametrize("selected", [False, True])
def test_claim_undo_keeps_every_previously_taught_photo_at_the_photo_limit(state, selected):
    folder = photo_files._photo_dir("Kara")
    folder.mkdir(parents=True, exist_ok=True)
    prior = {f"{1000000000000 + i}.jpg": f"taught {i}".encode() for i in range(people_writes.PHOTOS_PER_PERSON)}
    for name, data in prior.items():
        (folder / name).write_bytes(data)
    photo_files._unknown_photo(7).parent.mkdir(parents=True, exist_ok=True)
    photo_files._unknown_photo(7).write_bytes(b"unknown crop")
    code, out = people_writes._claim("face", 7, "Kara", True, b"selected crop" if selected else None)
    assert code == 200, out

    code, back = people_writes.undo_people({"undo": out["undo"]})
    assert code == 200 and back["ok"], back
    assert {photo.name: photo.read_bytes() for photo in folder.glob("*.jpg")} == prior
    assert photo_files._unknown_photo(7).read_bytes() == b"unknown crop"


def test_undoing_a_new_owner_voice_preserves_the_existing_owner_voice(state):
    write_voices(state, {1: "Alice"})
    before = registry.at(state).person_of(registry.voiceprint(1)).person
    code, out = people_writes.claim_identity("voice", 2, "Alice", control=True)
    assert code == 200, out
    assert registry.owner_voice(state, 1)
    assert registry.owner_voice(state, 2)

    code, back = people_writes.undo_people({"undo": out["undo"]})
    assert code == 200 and back["ok"], back
    assert registry.owner_voice(state, 1)
    assert not registry.owner_voice(state, 2)
    assert registry.at(state).person_of(registry.voiceprint(1)).person == before


def test_repeating_the_owner_voice_claim_does_not_offer_to_undo_earlier_teaching(state):
    write_voices(state, {1: "Alice"})
    code, out = people_writes.claim_identity("voice", 1, "Alice", control=True)
    assert code == 200, out
    assert not out.get("undo")
    assert registry.owner_voice(state, 1)


def test_repeating_a_voice_claim_does_not_offer_to_undo_earlier_teaching(state):
    code, out = people_writes.claim_identity("voice", 1, "Kara", control=True)
    assert code == 200, out
    assert not out.get("undo")
    assert known_voices._speakers()["speakers"][0]["label"] == "Kara"


def test_what_cannot_be_assigned_is_listed_and_the_rest_stands(state):
    code, out = people_writes.assign_to({"name": "Sam", "faces": [7, 99], "voices": [1], "videos": ["cam9@1.00"]})
    assert code == 200
    assert out["assigned"] == {"faces": 1, "voices": 0, "videos": 0}
    assert [(f["kind"], f["id"]) for f in out["failed"]] == [("face", 99), ("voice", 1), ("video", "cam9@1.00")]
    assert "Kara's already" in out["failed"][1]["error"]
    code, out = people_writes.assign_to({"name": "Sam", "faces": [99]})
    assert (code, out["ok"]) == (400, False)
    assert people_writes.assign_to({"name": "Sam"})[0] == 400
    assert people_writes.assign_to({"name": "4chan", "faces": [7]})[0] == 400


def test_assigning_to_several_people_at_once_is_one_undo(state):
    code, out = people_writes.assign_many({"items": [{"name": "Sam", "faces": [7]}, {"name": "Dana", "voices": [2]}]})
    assert code == 200, out
    assert out["assigned"] == [
        {"name": "Sam", "faces": 1, "voices": 0, "videos": 0},
        {"name": "Dana", "faces": 0, "voices": 1, "videos": 0},
    ]
    assert out["failed"] == []
    assert {"Sam", "Dana"} <= {p["name"] for p in out["people"]}
    assert face_db.unknown_groups() == []
    assert known_voices._speakers()["speakers"][1]["label"] == "Dana"

    code, back = people_writes.undo_people({"undo": out["undo"]})
    assert code == 200 and back["ok"], back
    assert face_db.unknown_groups() == [{"id": 7, "count": 2}]
    assert known_voices._speakers()["speakers"][1]["label"] == "Speaker 2"
    assert people_writes.undo_people({"undo": out["undo"]})[0] == 409


def test_assigning_to_several_lists_what_failed_by_name_and_the_rest_stands(state):
    code, out = people_writes.assign_many(
        {"items": [{"name": "Sam", "faces": [7, 99]}, {"name": "Dana", "voices": [1]}, {"name": "4chan", "faces": [7]}]}
    )
    assert code == 200, out
    assert out["assigned"] == [{"name": "Sam", "faces": 1, "voices": 0, "videos": 0}]
    assert [(f["name"], f.get("kind"), f.get("id")) for f in out["failed"]] == [
        ("Sam", "face", 99),
        ("Dana", "voice", 1),
        ("4chan", None, None),
    ]
    assert "Dana" not in [p["name"] for p in out["people"]]  # nothing was theirs, so they weren't added


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="no-items"),
        pytest.param({"items": []}, id="empty"),
        pytest.param({"items": ["Sam"]}, id="not-objects"),
        pytest.param({"items": [{"name": "Sam", "faces": [99]}]}, id="nothing-given"),
    ],
)
def test_assigning_to_several_with_nothing_to_give_is_a_400(state, body):
    assert people_writes.assign_many(body)[0] == 400


# ---- the card's menu ----
def test_not_kara_moves_a_shot_to_someone_new_and_undo_brings_it_back(state):
    code, out = people_writes.correct_card({"name": "kara", "kind": "shot", "id": S[1], "action": "move", "to": "Luca"})
    assert code == 200, out
    assert out["to"] == "Luca"
    assert out["card"]["photo"]["id"] == S[2]  # the next best fills in
    assert S[1] not in kara_shots(out["card"])
    assert card("Luca")["photo"]["id"] == S[1]
    code, back = people_writes.undo_people({"undo": out["undo"]})
    assert code == 200 and back["ok"], back
    assert back["card"]["photo"]["id"] == S[1]
    assert [str(n) for n in face_db.load_db()[0]] == ["Kara", "Bob"]  # no face row was added


def test_not_sure_just_removes_it_from_kara(state):
    code, out = people_writes.correct_card({"name": "Kara", "kind": "shot", "id": S[1], "action": "remove"})
    assert code == 200
    assert out["card"]["photo"]["id"] == S[2]
    assert "to" not in out
    assert names_of(state, 1) == [None]  # it stays off her card, though the face worker matched it to her


def correct(**body):
    code, out = people_writes.correct_card(body)
    assert code == 200, out
    return out


def test_use_as_main_photo_and_hide_from_the_card(state):
    out = correct(name="Kara", kind="shot", id=S[3], action="main")
    assert out["card"]["photo"]["id"] == S[3]
    assert people_list.people_roster()[0]["photo"] == f"/faces/shot/{S[3]}.jpg"  # the list shows the same face
    people_writes.undo_people({"undo": out["undo"]})
    assert card("Kara")["photo"]["id"] == S[1]
    assert correct(name="Kara", kind="shot", id=S[1], action="hide")["card"]["photo"]["id"] == S[2]
    out = correct(name="Kara", kind="voice", id=f"desk-mic@{DAY + 1:.3f}", action="hide")
    assert out["card"]["voices"] == []
    people_writes.undo_people({"undo": out["undo"]})
    assert [v["text"] for v in card("Kara")["voices"]] == ["Hi there."]


def test_not_kara_on_a_video_moves_the_whole_visit(state):
    vid = card("Kara")["videos"][0]["id"]
    assert vid == f"cam0@{DAY:.2f}"
    code, out = people_writes.correct_card({"name": "Kara", "kind": "video", "id": vid, "action": "move", "to": "Bob"})
    assert code == 200, out
    assert out["card"]["photo"] is None and out["card"]["shots"] == []
    assert sorted(s["id"] for s in card("Bob")["shots"]) == sorted([S[1], S[4]])  # S[2] and S[3] look like S[1]
    people_writes.undo_people({"undo": out["undo"]})
    assert card("Kara")["photo"]["id"] == S[1]


def test_not_kara_on_a_box_in_review_unnames_its_whole_track_and_undo_puts_it_back(state, vision):
    on_track = [vision.face(DAY + 100 + i, "cam0", "Kara", 0.8, track_id="12")["id"] for i in range(3)]
    other = vision.face(DAY + 104, "cam0", "Kara", 0.8, track_id="13")["id"]  # another person's track, same moment
    now = lambda: {f["id"]: f["name"] for f in st.faces(DAY + 90, DAY + 110, state=state)}
    out = correct(name="Kara", kind="track", id=f"cam0:12@{DAY + 101:.2f}", action="remove")
    assert [now()[i] for i in on_track] == [None, None, None]
    assert now()[other] == "Kara"
    people_writes.undo_people({"undo": out["undo"]})
    assert [now()[i] for i in [*on_track, other]] == ["Kara"] * 4


def test_not_kara_on_a_voice_line_relabels_it(state):
    line = f"desk-mic@{DAY + 1:.3f}"
    code, out = people_writes.correct_card({"name": "Kara", "kind": "voice", "id": line, "action": "move", "to": "Bob"})
    assert code == 200, out
    assert out["card"]["voices"] == []
    assert [v["line"] for v in card("Bob")["voices"]] == [line]
    people_writes.undo_people({"undo": out["undo"]})
    assert [v["line"] for v in card("Kara")["voices"]] == [line]


@pytest.mark.parametrize(
    "action",
    [pytest.param({"action": "remove"}, id="remove"), pytest.param({"action": "move", "to": "Bob"}, id="move")],
)
def test_not_kara_on_a_voice_line_keeps_it_as_her_negative_until_the_undo(state, action):
    from liberational import segment_log
    from liberational.listen import voices

    key = f"desk-mic@{DAY + 1:.3f}"
    line = people_writes._line_by_key(key)
    assert line is not None
    prints = voices.Voiceprints(state / "voiceprints.json", lambda: 1)
    m = voices.VoiceMatcher(prints, voices.RegistryPeople(state), None, segments=segment_log.root(state))  # ty: ignore[invalid-argument-type]
    vector = np.zeros(segment_log.VOICE_DIM, np.float32)
    vector[0] = 1
    m.file(vector, 1, line["id"], int(line["ts_start"] * 1e9), int((line.get("ts_end") or line["ts_start"] + 2) * 1e9))
    m.close()
    code, out = people_writes.correct_card({"name": "Kara", "kind": "voice", "id": key, **action})
    assert code == 200, out
    kara = str(registry.at(state).person_id("Kara"))
    assert [(n.person, n.key) for n in voices.confirmed_negatives(state)] == [(kara, f"utterance:{key}")]
    people_writes.undo_people({"undo": out["undo"]})
    assert voices.confirmed_negatives(state) == []


def _speaker_of(key: str) -> str | None:
    line = people_writes._line_by_key(key)
    assert line is not None, key
    return transcript_lines._view().line(line).get("speaker")


@pytest.mark.parametrize(
    ("action", "after"),
    [
        pytest.param({"action": "move", "to": "Bob"}, "Bob", id="move"),
        pytest.param({"action": "remove"}, "unknown", id="remove"),
    ],
)
def test_an_unmatched_voices_line_is_moved_or_removed_and_undone(state, action, after):
    key = registry.key_of({"ts_start": DAY + 3, "ts_end": DAY + 4, "speaker_id": 2})
    assert _speaker_of(key) == "Speaker 2"
    code, out = people_writes.correct_card({"name": "Speaker 2", "kind": "voice", "id": key, **action})
    assert code == 200, out
    assert out["card"] is None  # no person, so no card
    assert _speaker_of(key) == after
    code, undone = people_writes.undo_people({"undo": out["undo"]})
    assert code == 200 and undone["ok"], undone
    assert undone["card"] is None
    assert _speaker_of(key) == "Speaker 2"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"kind": "voice", "id": f"desk-mic@{DAY + 1:.3f}", "action": "remove"}, id="kates-line"),
        pytest.param({"kind": "shot", "id": 6, "action": "remove"}, id="not-a-voice-line"),
        pytest.param({"kind": "voice", "id": "x@1.000", "action": "hide"}, id="hide"),
    ],
)
def test_an_unmatched_voice_corrects_only_its_own_voice_lines(state, body):
    if body["kind"] == "shot":
        body = {**body, "id": S[body["id"]]}
    code, out = people_writes.correct_card({"name": "Speaker 2", **body})
    assert code == 404, out
    assert _speaker_of(f"desk-mic@{DAY + 1:.3f}") == "Kara"


def test_voice_correction_and_undo_keep_the_exact_microphone_key(state):
    write_lines(
        state,
        {"ts_start": DAY + 1, "text": "Other recording", "speaker": "Kara", "speaker_id": 1, "device": "Other Mic"},
    )
    first = f"desk-mic@{DAY + 1:.3f}"
    other = f"other-mic@{DAY + 1:.3f}"
    code, result = people_writes.correct_card(
        {"name": "Kara", "kind": "voice", "id": first, "action": "move", "to": "Bob"}
    )
    assert code == 200, result
    assert [v["line"] for v in card("Bob")["voices"]] == [first]
    assert speaker_lines.speakers_refine("assign", {"line": other, "name": "Bob"})[0] == 200
    code, undone = people_writes.undo_people({"undo": result["undo"]})
    assert code == 200 and undone["ok"], undone
    assert [v["line"] for v in card("Kara")["voices"]] == [first]
    assert [v["line"] for v in card("Bob")["voices"]] == [other]


def test_failed_undo_retries_only_failed_operations(state):
    card_prefs.set_main(state, "Kara", S[2])
    card_prefs.hide(state, "Kara", "shot", S[3])
    token = card_prefs.record(
        state,
        "correct",
        "Kara",
        [
            {"op": "hide", "name": "Kara", "kind": "shot", "id": S[3]},
            {"op": "main", "name": "Kara", "before": S[1]},
        ],
    )
    replace = card_prefs.os.replace

    def cannot_restore_main(src, dst):
        if dst == state / card_prefs.PREFS and json.loads(src.read_text())["kara"].get("main") == S[1]:
            raise OSError("temporary write failure")
        return replace(src, dst)

    with patch.object(card_prefs.os, "replace", autospec=True, side_effect=cannot_restore_main):
        code, failed = people_writes.undo_people({"undo": token})
    assert (code, failed["ok"]) == (200, False)
    assert failed["errors"] == ["temporary write failure"]
    assert card_prefs.get(state, "Kara")["main"] == S[2]
    assert card_prefs.get(state, "Kara")["hidden"]["shot"] == []

    # A new owner choice after the successful unhide must survive the retry.
    card_prefs.hide(state, "Kara", "shot", S[3])
    code, retried = people_writes.undo_people({"undo": token})
    assert (code, retried["ok"]) == (200, True)
    assert retried["errors"] == []
    assert card_prefs.get(state, "Kara")["main"] == S[1]
    assert card_prefs.get(state, "Kara")["hidden"]["shot"] == [S[3]]
    assert people_writes.undo_people({"undo": token})[0] == 409


@pytest.mark.parametrize(
    ("body", "code"),
    [
        pytest.param({"name": "Nobody", "kind": "shot", "id": 1, "action": "remove"}, 404, id="no-such-person"),
        pytest.param({"name": "Kara", "kind": "shot", "id": 4, "action": "remove"}, 404, id="bobs-shot"),
        pytest.param({"name": "Kara", "kind": "shot", "id": "nope", "action": "remove"}, 404, id="no-such-shot"),
        pytest.param({"name": "Kara", "kind": "shot", "id": "../x", "action": "hide"}, 400, id="not-an-id"),
        pytest.param({"name": "Kara", "kind": "video", "id": 1, "action": "main"}, 400, id="main-on-a-video"),
        pytest.param({"name": "Kara", "kind": "shot", "id": 1, "action": "move", "to": "Kara"}, 400, id="to-herself"),
        pytest.param({"name": "Kara", "kind": "voice", "id": "x@1.000", "action": "hide"}, 404, id="no-such-line"),
        pytest.param({"name": "Kara", "kind": "shot", "id": True, "action": "hide"}, 400, id="bool-id"),
        pytest.param(
            {"name": "Kara", "kind": "track", "id": f"cam0:1@{DAY:.2f}", "action": "main"}, 400, id="main-on-a-track"
        ),
        pytest.param({"name": "Kara", "kind": "track", "id": "cam0/1", "action": "remove"}, 400, id="not-a-track-id"),
        pytest.param(
            {"name": "Kara", "kind": "track", "id": f"cam0:9@{DAY:.2f}", "action": "remove"}, 404, id="no-such-track"
        ),
    ],
)
def test_corrections_that_cant_be_made(state, body, code):
    if body["kind"] == "shot" and type(body["id"]) is int:
        body = {**body, "id": S[body["id"]]}  # the fixture's sighting by its number
    assert people_writes.correct_card(body)[0] == code


def test_an_unknown_undo_token_is_a_404(state):
    assert people_writes.undo_people({"undo": "nope"})[0] == 404


# ---- rename and delete carry the choices ----
def test_a_rename_carries_corrections_and_choices(state):
    correct(name="Kara", kind="shot", id=S[3], action="main")
    correct(name="Bob", kind="shot", id=S[4], action="move", to="Luca")
    assert people_writes.rename_person("Luca", "Lucas", control=True)[0] == 200
    assert names_of(state, 4) == ["Lucas"]  # the owner's claim is on the person, whatever they're called
    assert people_writes.rename_person("Kara", "Katherine", control=True)[0] == 200
    assert card_prefs.get(state, "Katherine")["main"] == S[3]


def test_card_prefs_merge_on_rename_and_go_on_delete(tmp_path):
    card_prefs.set_main(tmp_path, "Kara", "d-3")
    card_prefs.hide(tmp_path, "Kara", "shot", 1)
    card_prefs.hide(tmp_path, "Kat", "shot", 2)
    card_prefs.set_main(tmp_path, "Kat", "d-9")
    card_prefs.rename(tmp_path, "Kara", "Kat")  # a merge: Kat keeps her main photo, and the hidden shots join
    assert card_prefs.get(tmp_path, "kat") == {"main": "d-9", "hidden": {"shot": [2, 1], "video": [], "voice": []}}
    card_prefs.forget(tmp_path, "Kat")
    assert card_prefs.get(tmp_path, "Kat") == {"main": None, "hidden": {"shot": [], "video": [], "voice": []}}


def test_an_undo_record_is_used_once(tmp_path):
    token = card_prefs.record(tmp_path, "correct", "Kara", [{"op": "main", "name": "Kara", "before": None}])
    first = card_prefs.take(tmp_path, token)
    assert first is not None
    assert first["ops"] == [{"op": "main", "name": "Kara", "before": None}]
    assert card_prefs.take(tmp_path, token) is None
    assert card_prefs.known(tmp_path, token)


def test_a_face_row_added_and_taken_back_restores_the_one_it_pushed_out(tmp_path, monkeypatch):
    monkeypatch.setattr(face_db, "DB", tmp_path / "db.npz")
    n = face_db.MAX_PER_NAME
    face_db.save_db(["Kara"] * n, E[:n])
    dropped = face_db.add_row("Kara", E[100] * 3)  # normalized as it goes in
    assert dropped == pytest.approx(E[0].tolist())
    assert face_db.remove_row("kara", E[100], dropped)
    names, embs = face_db.load_db()
    assert len(names) == n
    assert int(np.argmax(embs[0])) == 0  # the oldest row is back
    assert not face_db.remove_row("Kara", E[100])


def test_the_unnamed_list_is_not_a_person_named_unnamed(state):
    assert people_writes.add_person("Unnamed") is None
    assert card("Unnamed")["name"] == "Unnamed"
    assert set(json.loads(json.dumps(people_reads.people_unnamed()))) == {"faces", "voices", "videos"}


# ---- from the local review ----
def test_a_voice_for_someone_who_has_one_joins_it_and_its_undo_leaves_their_voice_named(state):
    code, out = people_writes.assign_to({"name": "Kara", "voices": [2]})  # from any machine: nothing restarts
    assert code == 200, out
    code, back = people_writes.undo_people({"undo": out["undo"]})
    assert (code, back["ok"]) == (200, False)
    assert "can't be undone" in back["errors"][0]
    assert "Kara" in [s["label"] for s in known_voices._speakers()["speakers"]]  # Kara keeps her voice


def test_undoing_a_video_move_puts_the_event_log_back(state):
    reg = registry.at(state)
    vid = card("Kara")["videos"][0]["id"]
    out = correct(name="Kara", kind="video", id=vid, action="move", to="Bob")
    assert reg.person_of(registry.detection(S[2])).person == reg.person_id("Bob")  # the owner's word, in the log
    assert names_of(state, 1, 2, 3) == ["Bob", "Bob", "Bob"]
    code, back = people_writes.undo_people({"undo": out["undo"]})
    assert (code, back["ok"]) == (200, True), back
    assert names_of(state, 1, 2, 3) == ["Kara", "Kara", "Kara"]
    assert reg.person_of(registry.detection(S[2])).person is None  # the face worker's match again


def test_undoing_a_face_group_puts_back_its_photo_and_the_row_it_pushed_out(state):
    n = face_db.MAX_PER_NAME
    face_db.save_db(["Kara"] * n, E[100 : 100 + n])
    photo_files._unknown_photo(7).parent.mkdir(parents=True, exist_ok=True)
    photo_files._unknown_photo(7).write_bytes(b"jpg")
    code, out = people_writes.assign_to({"name": "Kara", "faces": [7]})
    assert code == 200, out
    assert not photo_files._unknown_photo(7).exists()  # the group's crop became Kara's photo
    code, back = people_writes.undo_people({"undo": out["undo"]})
    assert back["ok"], back
    names, embs = face_db.load_db()
    assert len(names) == n
    assert sorted(int(np.argmax(e)) for e in embs) == list(range(100, 100 + n))  # the oldest row is back
    assert photo_files._unknown_photo(7).read_bytes() == b"jpg"
    assert face_db.unknown_groups() == [{"id": 7, "count": 2}]


def test_an_item_that_fails_halfway_leaves_nothing_behind(state, monkeypatch):
    rename = people_writes._rename_sightings

    def broken(ids, name, ops):
        rename(ids, name, ops)  # the sightings are named, then the next write fails
        raise RuntimeError("disk full")

    monkeypatch.setattr(people_writes, "_rename_sightings", broken)
    code, out = people_writes.assign_to({"name": "Sam", "videos": [f"cam1@{DAY + 2000:.2f}"]})
    assert code == 400
    assert out["failed"][0]["error"] == "disk full"
    assert names_of(state, 6, 7) == [None, None]  # the clip's sightings were put back
    assert "Sam" not in [p["name"] for p in people_list.people_roster()]  # and nobody was added for nothing


def test_two_strangers_in_one_stretch_are_two_clips(state, vision):
    sight(vision, 8, DAY + 2005, None, cam="cam1", group=10)  # someone else, between the two shots of group 9
    clips = st.unnamed_clips(30, state=state)
    assert [k["ids"] for k in clips[:2]] == [[S[6], S[7]], [S[8]]]  # newest end first


def test_a_move_that_finds_nothing_adds_nobody(state):
    assert (
        people_writes.correct_card({"name": "Kara", "kind": "shot", "id": S[4], "action": "move", "to": "Zed"})[0]
        == 404
    )
    assert "Zed" not in [p["name"] for p in people_list.people_roster()]


def test_deleting_who_a_shot_went_to_keeps_it_off_the_first_card(state):
    correct(name="Kara", kind="shot", id=S[1], action="move", to="Luca")
    assert people_writes.delete_person("Luca", control=True)[0] == 200
    assert names_of(state, 1) != ["Kara"]  # still not Kara
    assert S[1] not in kara_shots(card("Kara"))


def test_hidden_shots_and_videos_leave_the_card(state):
    correct(name="Kara", kind="shot", id=S[2], action="hide")
    assert S[2] not in kara_shots(card("Kara"))
    vid = card("Kara")["videos"][0]["id"]
    correct(name="Kara", kind="video", id=vid, action="hide")
    assert card("Kara")["videos"] == []


@pytest.mark.parametrize("cam", ["front-door", "FaceCam_A", "7North", "cam00", "0"])
def test_named_camera_visit_can_be_hidden(state, vision, cam):
    sight(vision, 20, DAY + 3000, "Kara", cam=cam)
    vision.visit("Kara", DAY + 3000, DAY + 3000, cam=cam)
    clip = next(v for v in card("Kara")["videos"] if v["cam"] == cam)
    correct(name="Kara", kind="video", id=clip["id"], action="hide")
    assert all(v["id"] != clip["id"] for v in card("Kara")["videos"])


def test_newest_audio_lines_follow_new_speech_and_purges(state):
    from liberational import speech_log
    from speech_fake import fake

    assert people_reads._newest_lines()[2]["text"] == "Who's that?"
    fake().say(
        {
            "ts_start": DAY + 100,
            "text": "A newer line.",
            "speaker_id": 2,
            "speaker": "Speaker 2",
            "speaker_sim": 0.7,
            "audio": {"stream": "mic.audio", "start_ns": 0, "end_ns": 1000000000},
        }
    )
    assert people_reads._newest_lines()[2]["text"] == "A newer line."
    fake().purge(speech_log.UTTERANCES, lambda e: e["speaker_id"] == 2)
    assert 2 not in people_reads._newest_lines()


def test_face_history_failure_keeps_voice_samples_and_marks_the_response_incomplete(state, monkeypatch):
    def unavailable(*args, **kwargs):
        raise TimeoutError("face history unavailable")

    monkeypatch.setattr(st, "faces", unavailable)
    out = people_reads.people_unnamed()
    assert [v["sample"]["text"] for v in out["voices"]] == ["Who's that?"]
    assert [f["id"] for f in out["faces"]] == [7]
    assert out["videos"] == []
    assert out["incomplete"] == "Could not load face photos and video clips."


def test_warm_unnamed_picks_follow_group_claims_corrections_new_matches_and_purges(state, vision):
    target = f"cam1@{DAY + 2000:.2f}"
    grouped = f"cam1@{DAY + 1000:.2f}"

    def ids():
        return {v["id"] for v in people_reads.people_unnamed()["videos"]}

    assert ids() == {target, grouped}
    reg = registry.at(state)
    reg.claim(registry.face_group(9), "Sam", Basis.INTRODUCTION, "test")
    assert ids() == {grouped}, "a new group claim applies without new detections"
    reg.rename("Sam", "Dana", "test")
    assert ids() == {grouped}, "renamed known faces still exclude their camera run"
    prior = st.correct([S[5]], "Kara", state)
    assert ids() == set(), "a corrected named face excludes the previously unnamed clip"
    st.uncorrect(prior, state)
    assert ids() == {grouped}
    added = vision.face(DAY + 1000.1, "cam1", "Kara", 0.9)
    assert ids() == set(), "a new named match in the clip excludes it"
    vision.records["face.match"].remove(added)
    assert ids() == {grouped}, "purging that match restores the remaining evidence"
    vision.records["face.match"] = [r for r in vision.records["face.match"] if r["id"] != S[5]]
    assert ids() == set(), "purged detections cannot remain selectable as video"


def test_naming_a_voice_teaches_listen_its_line_and_the_undo_takes_it_back(state):
    # The claim confirms the voice's lines (here its one line) as the person's, as /speakers/tag would
    from liberational.listen import voices

    with patch.object(voices, "confirm_lines", autospec=True, return_value=1) as confirm:
        code, out = people_writes.claim_identity("voice", 2, "Sam", control=True)
    assert code == 200, out
    ((_, lines), _) = confirm.call_args
    assert [line["text"] for line in lines] == ["Who's that?"]
    line = registry.utterance_of(lines[0])
    decision = registry.at(state).person_of(line)
    assert decision.winner is not None and decision.winner.basis == Basis.OWNER_TAG
    assert decision.person == registry.at(state).person_id("Sam")

    code, back = people_writes.undo_people({"undo": out["undo"]})
    assert code == 200 and back["ok"], back
    assert registry.at(state).person_of(line).person is None


def test_labeling_a_line_as_someone_teaches_listen_that_line(state):
    from liberational.listen import voices

    with patch.object(voices, "confirm_lines", autospec=True, return_value=1) as confirm:
        code, out = speaker_lines.speakers_refine("assign", {"line": registry.line_key(DAY + 3), "name": "Sam"})
    assert code == 200, out
    ((_, lines), _) = confirm.call_args
    assert [line["text"] for line in lines] == ["Who's that?"]
