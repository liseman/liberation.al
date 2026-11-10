"""Step 7 of docs/DOMAIN-MODEL.md: encounters from face visits and transcript lines (memory/timeline.py), and the
history answers and day summaries that use them. Everything runs on temp dirs, with made-up people."""

import datetime as dt
import json
from unittest.mock import create_autospec

import pytest

import speech_fake
from liberational import config as cfg
from liberational import memory_log, segment_log, speech_log
from liberational.dashboard import presence
from liberational.domain.identity import Basis
from liberational.domain.memory import Encounter, Modality
from liberational.listen import revisions
from liberational.memory import ask, history_qa, journal, journal_pages, recall, timeline
from liberational.people import forgotten, registry
from liberational.proximity import devices
from voices import write_voices

SEEN, HEARD = Modality.SEEN, Modality.HEARD
NOW = dt.datetime(2026, 10, 23, 12, tzinfo=dt.UTC).timestamp()
T = NOW - 3600  # an hour ago, the same day in UTC
OFF = {"timeline": {"enabled": False}}


def line(ts, speaker, sid, sim=0.7, text=None, **extra):
    return {"ts_start": ts, "ts_end": ts + 2, "speaker": speaker, "speaker_id": sid, "speaker_sim": sim,
            "text": text or f"line at {ts}", **extra}  # fmt: skip


class Home:
    """A state dir with three voices (Alice, Dana, and an unnamed one), their person ids in the log, and no visits."""

    def __init__(self, state, vision):
        self.state = state
        self.vision = vision
        write_voices(state, {1: "Alice", 2: "Dana", 7: None})
        self.reg = registry.at(state)
        self.lines([])

    def lines(self, rows, append=False):
        speech = speech_fake.fake()
        if not append and speech.lines.get(speech_log.UTTERANCES):
            speech.purge(speech_log.UTTERANCES, lambda e: True)
        for e in rows:
            speech.say(e)

    def visit(self, name, start, end, cam="cam0", sim=0.8):
        self.vision.visit(name, start, end, cam, best_sim=sim, samples=4)

    def no_visits(self):
        self.vision.records.pop(segment_log.FACE_VISITS, None)

    def pid(self, name):
        return self.reg.person_id(name)

    def load(self):
        tl = timeline.load(self.state)
        assert tl is not None
        return tl

    def spans(self, name):
        return [(e.start, e.end, e.seen_or_heard) for e in self.load().encounters(self.pid(name))]

    def names(self):
        tl = self.load()
        return [tl.name(e.person) for e in tl.entries]

    def ask(self, q):
        return history_qa.collect(q, self.state, NOW)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    # load() drops every cached state once more than 8 are kept, so states other tests loaded could empty it
    # between two loads here.
    monkeypatch.setattr(timeline, "_cache", {})
    monkeypatch.setattr(timeline, "_stores", {})


@pytest.fixture
def home(tmp_path, vision):
    with cfg.override({"timeline": {"enabled": True, "gap_s": 120.0}}):
        yield Home(tmp_path / "state", vision)


# ---- projection --------------------------------------------------------------------------------------------------


def test_gaps_under_the_limit_join_and_longer_ones_split(home):
    home.lines([line(T, "Alice", 1), line(T + 100, "Alice", 1), line(T + 400, "Alice", 1)])
    assert home.spans("Alice") == [(T, T + 102, {HEARD}), (T + 400, T + 402, {HEARD})]
    with cfg.override({"timeline": {"gap_s": 300.0}}):  # a longer limit joins all three
        assert home.spans("Alice") == [(T, T + 402, {HEARD})]


def test_rejected_speech_leaves_cached_encounters_and_undo_restores_it(home):
    home.lines([line(T, "Alice", 1, text="wrong words"), line(T + 100, "Alice", 1, text="kept words")])
    assert home.spans("Alice") == [(T, T + 102, {HEARD})]
    first = speech_fake.fake().all_lines()[0]
    token = revisions.reject_lines(home.state, [first])
    assert home.spans("Alice") == [(T + 100, T + 102, {HEARD})]
    evidence = json.dumps(home.ask("what did Alice say today"))
    assert "wrong words" not in evidence
    assert "kept words" in evidence
    revisions.undo_correction(home.state, token)
    assert home.spans("Alice") == [(T, T + 102, {HEARD})]


def test_seen_and_heard_join_into_one_encounter(home):
    home.visit("Alice", T, T + 60)
    home.lines([line(T + 150, "Alice", 1)])  # 90 s after the visit ends
    home.visit("Dana", T + 1000, T + 1010)
    tl = home.load()
    (alice,) = tl.encounters(home.pid("Alice"))
    assert alice.encounter == Encounter(home.pid("Alice"), T, T + 152, frozenset({SEEN, HEARD}))
    assert [v.cam for v in alice.visits] == ["cam0"]
    assert [ln.start for ln in alice.lines] == [T + 150]
    assert home.spans("Dana") == [(T + 1000, T + 1010, {SEEN})]


def test_visits_on_two_cameras_are_one_encounter(home):
    home.visit("Alice", T, T + 30, cam="cam0")
    home.visit("Alice", T + 20, T + 90, cam="cam2")
    (e,) = home.load().encounters(home.pid("Alice"))
    assert (e.start, e.end, sorted(v.cam for v in e.visits)) == (T, T + 90, ["cam0", "cam2"])


def test_lines_nobody_said_the_assistant_and_playback_are_not_presence(home):
    home.lines(
        [
            line(T, "Speaker 7", 7),  # an unnamed voice
            line(T + 10, "unknown", None),
            line(T + 20, "Alice", 1, playback=True),
            line(T + 25, "Alice", 1, media=True),  # heard from the TV (#2468)
            line(T + 30, "Alice", 1, assistant=True),
            line(T + 40, "Alice", 1),  # inside a window when the pack was speaking
        ]
    )
    (home.state / "playback_windows.json").write_text(json.dumps([{"id": "x", "start": T + 39, "end": T + 45}]))
    assert home.names() == []


def test_names_come_from_the_registry_and_strangers_have_no_encounters(home):
    home.visit("Zed", T, T + 10)  # a name on a visit the registry has never heard of
    home.visit("alice", T, T + 10)  # the stored spelling differs; the person does not
    assert home.names() == ["Alice"]


def test_no_event_log_means_no_timeline(tmp_path, speech):
    state = tmp_path / "bare"
    state.mkdir()
    speech.say(line(T, "Alice", 1))
    with cfg.override({"timeline": {"enabled": True}}):
        assert timeline.load(state) is None
        assert timeline.present(0, NOW, state) == []


# ---- identity ----------------------------------------------------------------------------------------------------


def test_a_relabel_in_the_log_moves_a_line_to_someone_else(home):
    home.lines([line(T, "Alice", 1)])
    home.reg.claim(registry.utterance(T), "Dana", Basis.OWNER_TAG, "dashboard")
    assert home.spans("Alice") == []
    assert home.spans("Dana") == [(T, T + 2, {HEARD})]


def test_unlabeling_a_line_in_the_log_takes_it_from_its_voice(home):
    home.lines([line(T, "Speaker 1", 1)])  # follows voice 1, which is Alice
    assert home.names() == ["Alice"]
    home.reg.detach(T, "unknown", "dashboard", via=registry.voiceprint(1))
    assert home.names() == []


def test_a_rename_keeps_the_person_and_shows_the_new_name(home):
    home.visit("Alice", T, T + 10)  # a visit keeps the name it was matched under
    home.lines([line(T + 20, "Alice", 1)])
    pid = home.pid("Alice")
    home.reg.rename("Alice", "Alicia", "dashboard")
    tl = home.load()
    (e,) = tl.entries
    assert (e.person, tl.name(e.person), e.seen_or_heard) == (pid, "Alicia", {SEEN, HEARD})


def test_someone_new_with_an_old_name_does_not_get_the_renamed_persons_lines(home):
    home.lines([line(T, "Dana", 2, text="hello there")])  # heard as Dana; a rename never rewrites the line
    dee = home.pid("Dana")
    home.reg.rename("Dana", "Dee", "dashboard")
    home.reg.ensure("Dana", "dashboard")  # People: Add, someone new
    tl = home.load()
    (e,) = tl.entries
    assert (e.person, tl.name(e.person)) == (dee, "Dee")


def test_a_merge_joins_two_people(home):
    home.visit("Dana", T, T + 10)
    home.lines([line(T + 30, "Alice", 1)])
    home.reg.merge("Dana", "Alice", "dashboard")
    tl = home.load()
    (e,) = tl.entries
    assert (tl.name(e.person), e.seen_or_heard) == ("Alice", {SEEN, HEARD})


def test_forgetting_a_person_leaves_no_encounter_of_theirs(home):
    home.visit("Dana", T, T + 10)
    home.lines([line(T + 20, "Dana", 2), line(T + 40, "Speaker 7", 7)])
    home.reg.claim(registry.utterance(T + 40), "Dana", Basis.OWNER_TAG, "dashboard")  # only in the log
    home.visit("Alice", T, T + 10)
    assert len(home.spans("Dana")) == 1
    home.reg.forget("Dana", "dashboard")  # what People: Delete does to the log; the visit row still says Dana
    assert home.names() == ["Alice"]


# ---- queries -----------------------------------------------------------------------------------------------------


@pytest.fixture
def met(home):
    """Alice seen then heard at T, Dana heard at T + 600 and T + 2000: three encounters."""
    home.visit("Alice", T, T + 60)
    home.lines([line(T + 100, "Alice", 1), line(T + 600, "Dana", 2), line(T + 2000, "Dana", 2)])
    return home


def test_who_was_present_in_a_range(met):
    tl = met.load()
    assert {p.name: p.seen_or_heard for p in tl.present(T, T + 700)} == {"Alice": {SEEN, HEARD}, "Dana": {HEARD}}
    assert [p.name for p in tl.present(T + 1000, T + 1500)] == []
    assert [p.name for p in tl.present(T + 1999, T + 3000)] == ["Dana"]
    assert [p.name for p in timeline.present(T + 50, T + 55, met.state)] == ["Alice"]


def test_encounters_by_person_and_time(met):
    dana = met.pid("Dana")
    assert [e.start for e in timeline.encounters(dana, state=met.state)] == [T + 600, T + 2000]
    assert [e.start for e in timeline.encounters(dana, T + 1000, T + 3000, met.state)] == [T + 2000]
    assert len(timeline.encounters(None, 0, NOW, met.state)) == 3


def test_a_long_encounter_overlapping_the_start_of_a_range_is_found(met):
    met.visit("Dana", T - 5000, T + 5000)
    assert "Dana" in [p.name for p in met.load().present(T + 4000, T + 4100)]


def test_last_encounter(met):
    tl, dana = met.load(), met.pid("Dana")
    last, earlier = tl.last(dana), tl.last(dana, 0, T + 1000)
    assert last is not None and earlier is not None
    assert (last.start, earlier.start) == (T + 2000, T + 600)
    assert tl.last(met.pid("Alice"), T + 1000) is None


# ---- the cache ---------------------------------------------------------------------------------------------------


def test_unchanged_sources_reuse_the_timeline(home):
    home.lines([line(T, "Alice", 1)])
    home.load()  # opening the log the first time may create its write-ahead log
    assert timeline.load(home.state) is home.load()


def test_a_label_only_in_the_log_names_the_line_and_invalidates_the_cache(home):
    home.lines([line(T, "Speaker 7", 7)])
    before = speech_fake.fake().range(0, NOW)
    timeline.load(home.state)
    assert home.names() == []
    stamp = timeline._signature(home.state)
    # Another process labels the line: no file changes, only where the event log stands.
    other = registry.Registry(registry.at(home.state).log)
    other.claim(registry.utterance(T), "Dana", Basis.OWNER_TAG, "dashboard")
    after = timeline._signature(home.state)
    assert after[:-1] == stamp[:-1] and after[-1] != stamp[-1], (stamp, after)
    assert home.spans("Dana") == [(T, T + 2, {HEARD})]
    assert speech_fake.fake().range(0, NOW) == before  # the label is only an event


def test_naming_a_voice_reads_every_line_again(home):
    home.lines([line(T, "Speaker 7", 7)])
    assert home.names() == []
    write_voices(home.state, {7: "Dana"})  # Dana has voice 2 already; voice 7 is hers too
    assert home.spans("Dana") == [(T, T + 2, {HEARD})]


def test_new_lines_are_read_without_reading_the_old_ones_again(home, monkeypatch):
    home.lines([line(T, "Alice", 1)])
    home.load()
    # The private resolver is the only place to see how many lines a load reads.
    resolve, read = timeline.TodayStores._resolve, []

    def spy(view, raw):
        read.append(len(raw))
        return resolve(view, raw)

    monkeypatch.setattr(timeline.TodayStores, "_resolve", staticmethod(spy))
    home.lines([line(T + 1000, "Dana", 2)], append=True)
    assert home.names() == ["Alice", "Dana"]
    assert [n for n in read if n] == [1]


def test_a_purge_reads_every_line_again(home, speech):
    home.lines([line(T, "Alice", 1), line(T + 1000, "Dana", 2)])
    assert home.names() == ["Alice", "Dana"]
    speech.purge(speech_log.UTTERANCES, lambda e: e["speaker_id"] == 2)  # forgetting Dana purges her lines
    assert home.names() == ["Alice"]


# ---- history answers (history_qa with the timeline on and off) -------------------------------------------------


@pytest.fixture
def history(home, monkeypatch):
    """history_qa in UTC, with Alice seen at T for two minutes and Dana heard five minutes after T."""
    monkeypatch.setattr(history_qa, "TZ", dt.UTC)
    home.visit("Alice", T, T + 120)
    home.lines([line(T + 300, "Dana", 2, text="Morning, is the kettle on?")])
    return home


def at(ts):
    return f"{dt.datetime.fromtimestamp(ts, dt.UTC):%H:%M on %Y-%m-%d}"


def test_who_was_here_counts_people_only_heard(history):
    r = history.ask("who was here today")
    assert "Alice (seen)" in r["answer"] and "Dana (heard)" in r["answer"]
    assert [e["source"] for e in r["evidence"]] == ["camera", "transcript"]
    assert (r["evidence"][1]["id"], r["evidence"][1]["quote"]) == ("T2", "Morning, is the kettle on?")
    assert "Evidence: [C1], [T2]." in r["answer"]
    assert r["evidence"][1]["href"] == f"/review#day=2026-10-23&t={T + 300}"


def test_who_was_here_counts_a_face_the_registry_doesnt_know(history):
    """A registry that lost its people (a wipe that kept the face gallery) has no person for a visit's name; the
    cameras still recognized them, so they're seen, not nobody."""
    history.visit("Carly", T + 30, T + 40)
    r = history.ask("who was here today")
    assert r["answer"].startswith("Alice (seen), Dana (heard), Carly (seen): recognized today."), r["answer"]
    assert [e["quote"].split(" seen on ")[0] for e in r["evidence"] if e["source"] == "camera"] == ["Carly", "Alice"]
    history.no_visits()
    history.lines([])
    history.visit("Carly", T, T + 10)
    assert history.ask(f"who was here at {at(T + 5)}")["answer"].startswith("Carly (seen): recognized around")


@pytest.mark.parametrize(
    ("window", "words"),
    [
        ("today", "today"),
        ("yesterday", "yesterday"),
        ("last week", "last week"),
        ("2026-10-20", "on 2026-10-20"),
        ("last 2 hours", "in the last 2 hours"),
        ("the morning of 2026-10-30", "in the morning of 2026-10-30"),
        ("all available retained history", "in all available retained history"),
    ],
)
def test_a_window_reads_as_words(window, words):
    assert history_qa._during(window) == words


def test_who_was_here_at_a_time(history):
    answer = history.ask(f"who was here at {at(T + 60)}")["answer"]
    assert "Alice (seen)" in answer and "Dana (heard)" in answer  # Dana 4 minutes later, inside the 5 minute window
    assert history.ask(f"who was here at {at(T - 3 * 3600)}")["answer"].startswith("Nobody was recognized")


def test_who_was_here_says_how_from_the_part_inside_the_window(history):
    """An encounter seen until 09:58 and heard from 09:59 is only heard in a window from 10:15 to 10:25, so the
    answer and a day page's People present say "heard", as the evidence shows."""
    history.no_visits()
    t0 = dt.datetime(2026, 10, 23, 9, 0, tzinfo=dt.UTC).timestamp()
    history.visit("Dana", t0, t0 + 58 * 60)
    history.lines([line(t0 + 59 * 60 + 60 * i, "Dana", 2, text=f"hello {i}") for i in range(32)])
    assert history.spans("Dana") == [(t0, t0 + 90 * 60 + 2, {SEEN, HEARD})]  # one encounter
    r = history.ask("who was here at 10:20 on 2026-10-23")
    assert r["answer"] == (
        "Dana (heard): recognized around 10:20. Voice matching is uncertain, not proof of identity. Evidence: [T1]."
    )
    assert {e["source"] for e in r["evidence"]} == {"transcript"}
    rows = journal_pages.people_present(history.state, t0 + 75 * 60, t0 + 85 * 60)
    assert [(p["name"], p["seen_or_heard"]) for p in rows or []] == [("Dana", ["heard"])]


def test_who_was_on_a_camera_keeps_to_that_camera(history):
    """A camera asked about keeps to its own visits: nobody seen elsewhere, and nobody only heard."""
    history.visit("Dana", T + 200, T + 260, cam="cam2")
    r = history.ask("who was on camera 2 today")
    other = history.ask("who did I see on cam2 today")
    assert (other["answer"], other["evidence"], "paths" in other) == (r["answer"], r["evidence"], False)
    assert r["answer"] == "Dana: recognized today. Face matching is uncertain, not proof of identity. Evidence: [C1]."
    assert r["evidence"][0]["href"] == f"/review#day=2026-10-23&t={T + 200}&cam=cam2"
    assert [(e["source"], e["camera"]) for e in r["evidence"]] == [("camera", "cam2")]
    assert history.ask("who was on camera 3 today")["answer"].startswith("Nobody was recognized")
    # Alice is within 5 minutes of T - 150, on cam0, and Dana is not.
    assert "Alice" in history.ask(f"who was on camera 0 at {at(T - 150)}")["answer"]
    assert history.ask(f"who was on camera 2 at {at(T - 150)}")["answer"].startswith("Nobody was recognized")


def test_who_was_on_a_camera_follows_a_rename(history):
    """A visit keeps the name it was seen under until the next rematch; the answer names the person now."""
    history.reg.rename("Alice", "Alicia", "dashboard")
    r = history.ask("who was on camera 0 today")
    assert "Alicia" in r["answer"]
    assert [e["quote"].split(" seen on ")[0] for e in r["evidence"]] == ["Alicia"]


def test_who_was_on_a_camera_names_a_newcomers_rematched_visit_by_them(history):
    """A rematch names every stored sighting by the face gallery now, even frames from before someone was
    tagged. So a visit named Alice after a newcomer took that name is the newcomer's, not the renamed Alicia's."""
    history.no_visits()
    history.reg.rename("Alice", "Alicia", "dashboard")
    write_voices(history.state, {1: "Alicia", 2: "Dana", 3: "Alice"})  # a new voice, tagged as the newcomer
    history.visit("Alice", T, T + 60)  # the newcomer, seen before they were tagged, named by the rematch
    on = history.ask("who was on camera 0 today")["answer"]
    with cfg.override(OFF):
        assert history.ask("who was here today")["answer"] == on
    assert "Alice" in on and "Alicia" not in on


def test_last_seen_on_a_camera_matches_by_person(history, vision):
    """The visit and camera-row fallbacks find a person by who they are: a visit by who holds its name now (a
    rematch renames it), a live camera row by who had the face's name when it was written (it is never
    rewritten). So a renamed person keeps both, and someone new who takes the old name gets no camera row."""
    vision.caption(T + 10, "cam1", "A person at a desk", faces=[{"name": "Alice", "sim": 0.7}])
    history.reg.rename("Alice", "Alicia", "dashboard")  # the visit keeps "Alice" until the next rematch
    answer = history.ask("when was Alicia last on camera0 today")["answer"]
    assert answer.startswith("Alicia was last seen at ") and " on cam0" in answer
    visits, _ = recall.run("visits", {"start": T - 60, "end": NOW, "person": "Alicia"}, "q", history.state, [])
    assert [(v["camera"], v["quote"].split(" seen on ")[0]) for v in visits] == [("cam0", "Alicia")]
    assert history.ask("when was Alicia last on camera1 today")["answer"].startswith("Alicia was last recognized")
    history.reg.claim(registry.voiceprint(3), "Alice", Basis.OWNER_TAG, "test")  # someone new is called Alice
    assert history.ask("when was Alice last on camera1 today")["answer"].startswith("No supported")
    assert history.ask("when was Alicia last on camera1 today")["answer"].startswith("Alicia was last recognized")


def test_last_seen_on_a_camera_skips_a_forgotten_persons_rows_for_a_newcomer(history, vision):
    """A live camera row keeps the face name it was written with. When that name meant someone since deleted, the
    row is nobody's: someone new who takes the name does not get it."""
    history.reg.ensure("Gus", "test")
    vision.caption(T + 10, "cam1", "A person at a desk", faces=[{"name": "Gus", "sim": 0.7}])
    forgotten.remember(["Gus"], history.state)  # People: Delete
    history.reg.forget("Gus", "dashboard")
    history.reg.ensure("Gus", "dashboard")  # someone new is called Gus
    assert history.ask("when was Gus last on camera1 today")["answer"].startswith("No supported")


def test_last_seen_and_last_heard(history):
    r = history.ask("when was Dana last seen")
    assert r["answer"].startswith("2026-") and ": Dana was last heard." in r["answer"] and "[T1]" in r["answer"]
    assert r["evidence"][0]["quote"] == "Morning, is the kettle on?"
    r = history.ask("when did I last see Alice")
    assert (
        r["answer"].startswith("2026-")
        and ": Alice was last seen" in r["answer"]
        and " on cam0" in r["answer"]
        and "[C1]" in r["answer"]
    )
    assert r["evidence"][0]["camera"] == "cam0"


def test_last_seen_and_heard_cites_both(history):
    history.lines([line(T + 130, "Alice", 1, text="Back in a minute.")], append=True)
    r = history.ask("when did you last see Alice?")
    assert ": Alice was last seen and heard" in r["answer"] and "[C1][T2]" in r["answer"]
    assert [e["id"] for e in r["evidence"]] == ["C1", "T2"]


def test_last_seen_keeps_to_the_window_asked_about(history):
    midnight = dt.datetime(2026, 10, 23, tzinfo=dt.UTC).timestamp()
    history.visit("Alice", midnight - 90, midnight - 30)  # seen yesterday at 23:58
    history.lines([line(midnight + 30, "Alice", 1, text="Happy new day.")], append=True)  # heard today: one encounter
    assert (midnight - 90, midnight + 32, {SEEN, HEARD}) in history.spans("Alice")  # it does cross midnight
    r = history.ask("when was Alice last seen yesterday")
    assert r["answer"].startswith("2026-") and ": Alice was last seen" in r["answer"]  # seen, not "seen and heard"
    assert "2026-10-22" in r["answer"] and "2026-10-23" not in r["answer"]
    assert "Happy new day" not in json.dumps(r["evidence"])


@pytest.mark.parametrize("phrase", ["after dinner", "before lunch", "3 days ago"])
def test_a_time_phrase_after_the_name_is_refused_as_with_the_switch_off(history, phrase):
    """A phrase after the name is never read as part of it: a time phrase gets the same refusal as with the
    switch off."""
    on = history.ask(f"when did I last see Dana {phrase}")
    assert on["query"] == {}  # refused
    with cfg.override(OFF):
        assert history.ask(f"when did I last see Dana {phrase}") == on


@pytest.mark.parametrize(
    ("phrase", "window"),
    [
        pytest.param("this morning", "the morning of 2026-10-23", id="this-morning"),
        pytest.param("during the morning", "the morning of 2026-10-23", id="during-the"),  # Codex on #219
        pytest.param("the afternoon", "the afternoon of 2026-10-23", id="the-afternoon"),
    ],
)
def test_a_part_of_the_day_after_the_name_is_a_window_not_part_of_the_name(history, phrase, window):
    """#195: a part of the day is a time now, read as its hours, and never part of the name."""
    q = history.ask(f"when did I last see Dana {phrase}")["query"]
    assert (q["person"], q["window"]) == ("Dana", window)


def test_another_phrase_after_the_name_leaves_the_name_alone(history):
    r = history.ask("when did I last see Dana at the office")
    assert r["query"]["person"] == "Dana"
    assert r["answer"].startswith("2026-") and ": Dana was last heard." in r["answer"]


def test_a_line_mentioning_a_password_is_quoted_like_any_other(history):
    """#92, by Joe's decision: nothing is hidden for what a line says."""
    history.lines([line(T + 310, "Dana", 2, text="the wifi password is hunter2")], append=True)
    r = history.ask("when was Dana last seen")
    assert "hunter2" in json.dumps(r)


@pytest.mark.parametrize(
    "q", ["when was Zed last seen", "when was Alice last seen on camera0"], ids=["nobody-has-the-name", "camera"]
)
def test_questions_the_timeline_does_not_take_answer_as_before(history, q):
    on = history.ask(q)
    with cfg.override(OFF):
        assert history.ask(q) == on


def test_off_answers_from_face_visits_and_never_reads_the_timeline(history, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("read with the switch off")

    monkeypatch.setattr(timeline, "load", refuse)
    with cfg.override(OFF):
        r = history.ask("who was here today")
        assert "Alice" in r["answer"] and "Dana" not in r["answer"]  # Dana was only heard
        assert {e["source"] for e in r["evidence"]} == {"camera"}
        assert "Dana was last" not in history.ask("when was Dana last seen")["answer"]  # no visit, so no answer
        assert " on cam0, in a visit from " in history.ask("when was Alice last seen")["answer"]


def test_off_matches_a_state_without_the_timeline(history, tmp_path):
    """With the switch off, a state with an event log answers exactly as one without (which never had one)."""
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "voiceprints.json").write_bytes(
        (history.state / "voiceprints.json").read_bytes()
    )  # the speech log is shared
    # The vision log is shared too: the fake serves every state.
    questions = ["who was here today", "when was Alice last seen", "who was here", "when was Dana last seen"]
    with cfg.override(OFF):
        got = [history.ask(q) for q in questions]
        # Both off: the timeline also reads the shared vision log's face visits, so a state with no event log of its
        # own still gets timeline answers while the switch is on.
        assert got == [history_qa.collect(q, bare, NOW) for q in questions]


@pytest.mark.parametrize("q", ["who was here today", "when was Dana last seen"])
def test_answers_from_encounters_are_marked(history, q):
    """The dashboard's "New" tag on a timeline answer comes from paths; answers made without it carry none."""
    assert history.ask(q)["paths"] == ["timeline"]
    assert ask.execute(q, state_dir=history.state, now=NOW)["paths"] == ["timeline"]
    with cfg.override(OFF):
        assert "paths" not in history.ask(q)
        assert "paths" not in ask.execute(q, state_dir=history.state, now=NOW)


# ---- the day page's People present (journal_pages with the timeline on and off) ----------------------------------


@pytest.fixture
def present_day(home):
    """A day with Alice seen at 01:00 and Dana heard at 02:00: (state, the day, its start)."""
    day = dt.datetime.fromtimestamp(T, history_qa.TZ).date().isoformat()
    start = journal.day_bounds(day)[0]
    home.visit("Alice", start + 3600, start + 3700)
    home.lines([line(start + 7200, "Dana", 2, text="See you tomorrow.")])
    return home.state, day, start


def test_a_day_page_lists_people_present_only_with_the_switch_on(present_day):
    state, _, start = present_day
    with cfg.override(OFF):
        assert journal_pages.people_present(state, start, start + 86400) is None
    rows = journal_pages.people_present(state, start, start + 86400)
    assert [(p["name"], p["seen_or_heard"]) for p in rows or []] == [("Alice", ["seen"]), ("Dana", ["heard"])]
    md = journal_pages.present_section(rows or [])
    assert "- Dana (heard): [" in md and f"(#t={start + 7200})" in md


def test_review_chips_skip_the_people_present_section(present_day):
    state, day, start = present_day
    rows = journal_pages.people_present(state, start, start + 86400)
    assert rows
    md = journal_pages.present_section(rows) + "\n"
    memory_log.at(state).append(
        memory_log.PAGES, {"ts_start": start, "ts_end": start + 86400, "level": "day", "period": day, "md": md}
    )
    assert presence.events(state, day, start, start + 86400) == []


@pytest.mark.parametrize(
    ("q", "person", "window"),
    [
        ("when did I last see Alice", "Alice", "all available retained history"),
        ("When did you last see Dana Smith today?", "Dana Smith", "today"),
        ("when did we last see Bob on 2026-10-20", "Bob", "2026-10-20"),
    ],
)
def test_last_met_phrasing_parses_with_the_switch_on(q, person, window):
    with cfg.override({"timeline": {"enabled": True}}):
        p = history_qa.parse(q, NOW)
        assert history_qa.is_history_question(q)
    assert (p["kind"], p["person"], p["window"]) == ("camera", person, window)


def test_last_met_phrasing_reads_face_visits_with_the_switch_off():
    with cfg.override(OFF):
        p = history_qa.parse("when did I last see Alice", NOW)
    assert (p["kind"], p["person"], p["last_seen"]) == ("camera", "Alice", True)


def test_timeline_citations_keep_visit_and_utterance_payloads_after_labels(home, monkeypatch):
    monkeypatch.setattr(history_qa, "TZ", dt.UTC)
    visit = home.vision.visit("Alice", T, T + 120)
    utterance = speech_fake.fake().say(line(T + 130, "Alice", 1, text="Back in a minute."))
    timeline_before = home.spans("Alice")
    evidence = home.ask("when did you last see Alice?")["evidence"]
    assert [(e["id"], e.get("payload_id")) for e in evidence] == [("C1", visit["id"]), ("T2", utterance["id"])]
    assert home.spans("Alice") == timeline_before
    assert (visit["id"], utterance["id"]) == (evidence[0]["payload_id"], evidence[1]["payload_id"])


def test_reconciled_visit_cites_detection_instead_of_stale_visit(home, monkeypatch):
    monkeypatch.setattr(history_qa, "TZ", dt.UTC)
    old_visit = home.vision.visit("Alice", T, T + 120)
    detection = home.vision.face(T + 60, name="Alice", sim=0.8)
    home.reg.claim(registry.detection(detection["id"]), "Dana", Basis.OWNER_TAG, "dashboard")
    evidence = home.ask("when was Dana last seen")["evidence"]
    assert [(e["id"], e.get("payload_id")) for e in evidence] == [("C1", detection["id"])]
    assert old_visit["id"] != evidence[0]["payload_id"]
    assert old_visit["name"] == "Alice"


@pytest.mark.parametrize("enabled", [True, False])
def test_relative_instant_is_not_swallowed_into_a_last_seen_name(history, enabled):
    with cfg.override({"timeline": {"enabled": enabled}}):
        result = history.ask("when did I last see Dana 3 hours ago")
    assert result["query"]["person"] == "Dana"
    assert result["query"]["start"] == NOW - 10800 - 300
    assert result["query"]["end"] == NOW - 10800 + 300


# ---- "is X home" with no linked device (#1596) ---------------------------------------------------------------------


@pytest.fixture(autouse=False)
def unlinked(monkeypatch):
    """A device registry with no links, so "is X home" is nobody's device question."""
    registry_ = create_autospec(devices.Devices, instance=True)
    registry_.links.return_value = []
    monkeypatch.setattr(devices, "at", lambda state=None: registry_)


def test_is_someone_home_says_when_they_were_last_seen_and_how_much_was_watched_since(history, vision, unlinked):
    vision.add(
        segment_log.MOVEMENT_COVERAGE,
        ts_start=T + 120,
        ts_end=T + 1920,
        cam="cam0",
        supported=[[T + 120, T + 1920]],
        camera_motion="stable",
        version="movement-1",
        frames=18000,
        gaps=[],
        tracking="complete",
        finished=True,
        closure="open",
    )
    r = history.ask("Is Alice home?")
    assert r["answer"].startswith(
        "Alice was last seen on cam0 58 minutes ago (11:02). The cameras were watching for 30 of the 58 minutes since."
    ), r["answer"]
    assert "not whether Alice is home" in r["answer"]
    assert r["evidence"][0]["camera"] == "cam0"


def test_is_someone_here_answers_from_their_voice_too(history, unlinked):
    r = history.ask("is Dana here?")
    assert r["answer"].startswith("Dana was last heard 55 minutes ago (11:05)."), r["answer"]
    assert "Voice matching is uncertain" in r["answer"]


def test_is_someone_home_never_says_away_when_they_werent_recognized(history, unlinked):
    history.reg.ensure("Gus", "test")
    r = history.ask("is Gus home")
    assert r["answer"].startswith("Gus hasn't been seen on camera or heard in the last 24 hours.")
    assert "doesn't say whether Gus is home" in r["answer"]


def test_is_a_stranger_home_goes_the_way_it_did(history, unlinked):
    assert not history.ask("is Zed home")["is_history"]


def test_voice_sends_is_someone_home_to_the_history_answer(history, unlinked):
    with cfg.override({"paths": {"state": str(history.state)}}):
        assert history_qa.is_history_question("is Alice home")
        assert not history_qa.is_history_question("is Zed home")
