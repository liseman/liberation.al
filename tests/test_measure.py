"""tools/measure.py: the measurement set's label formats and its scoring, against a made-up state."""

import datetime as dt
import importlib.util
import json
from unittest.mock import create_autospec
from zoneinfo import ZoneInfo

import pytest

import speech_fake
from liberational import config as cfg
from liberational import segment_log
from liberational.memory import ask, history_qa, question, recall
from voices import write_voices

_spec = importlib.util.spec_from_file_location("measure", cfg.ROOT / "tools" / "measure.py")
assert _spec is not None and _spec.loader is not None
measure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure)

FIXTURES = cfg.ROOT / "tests" / "data" / "measurement"
LA = ZoneInfo("America/Los_Angeles")
COLLECT = history_qa.collect  # the real one, to spec the fakes by
FAKE_MODELS = {"models": {"ollama": {"llm": "fake-llm", "embedding": "fake-embed"}}}


def at(hour, minute, second=0.0):
    return dt.datetime(2026, 10, 20, hour, minute, tzinfo=LA).timestamp() + second


@pytest.fixture
def labels(tmp_path):
    (tmp_path / "labels").mkdir()
    return tmp_path / "labels"


@pytest.fixture
def state(tmp_path, monkeypatch, vision):
    """A state where Alice and Dana were seen and heard on 2026-10-20, and the fixtures label that day."""
    monkeypatch.setattr(history_qa, "TZ", LA)
    state = tmp_path / "state"
    state.mkdir()
    for ts, cam, name in (
        (at(14, 0, 5), "cam0", "Alice"),
        (at(14, 0, 10), "cam0", "Alice"),
        (at(14, 10, 5), "cam0", "Alice"),  # really Dana
        (at(14, 20, 0.3), "cam1", None),  # Dana, not recognized
        (at(14, 20, 0.5), "cam1", "Alice"),  # the labeler was unsure, so it counts either way
    ):
        vision.face(ts, cam, name, 0.6 if name else None)
    # Alice's visits, her faces on one camera with gaps up to 10 s
    vision.visit("Alice", at(14, 0, 5), at(14, 0, 10), "cam0", samples=2)
    vision.visit("Alice", at(14, 10, 5), at(14, 10, 5), "cam0")
    vision.visit("Alice", at(14, 20, 0.5), at(14, 20, 0.5), "cam1")
    lines = [
        dict(ts_start=at(15, 0, 5), ts_end=at(15, 0, 8), speaker="Speaker 3", speaker_id=3, device="kitchen",
             text="Let's plant tomatoes in the garden."),
        dict(ts_start=at(15, 30, 5), ts_end=at(15, 30, 7), speaker="Alice", speaker_id=1, device="kitchen",
             text="Turn it up."),
    ]  # fmt: skip
    speech_fake.fake().write(lines)
    write_voices(state, {1: "Alice", 3: "Dana"})
    return state


def write(labels, part, *rows):
    (labels / f"{part}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def files(state):
    """Every file's bytes, leaving out SQLite's shared-memory index (any reader of a WAL database makes one) and an
    empty write-ahead log."""
    return {p: p.read_bytes() for p in sorted(state.rglob("*")) if p.is_file() and p.suffix != ".sqlite3-shm"}


def identification(labels, state):
    return measure.score_identification(measure.load(labels, "identification"), state)


def retrieval(labels, state, **options):
    return measure.score_retrieval(measure.load(labels, "retrieval"), state, **options)


def fake_collect(monkeypatch, **kwargs):
    fake = create_autospec(COLLECT, **kwargs)
    monkeypatch.setattr(history_qa, "collect", fake)
    return fake


# Identification


def test_identification_fixtures_score_against_the_stores(state):
    r = identification(FIXTURES, state)
    assert (r["clips"], r["tp"], r["fp"], r["fn"]) == (5, 2, 2, 2)
    assert r["by_modality"] == {"face": 3, "voice": 2}
    assert (r["precision"], r["recall"]) == (0.5, 0.5)
    assert r["per_person"] == {"Alice": {"tp": 1, "fp": 2, "fn": 0}, "Dana": {"tp": 1, "fp": 0, "fn": 2}}
    assert r["confusions"] == [
        {"labeled": "Dana", "predicted": "Alice", "count": 1},
        {"labeled": "Dana", "predicted": "nobody", "count": 1},
        {"labeled": "nobody", "predicted": "Alice", "count": 1},
    ]
    clips = {c["id"]: c for c in r["per_clip"]}
    assert clips["kitchen-dana"]["predicted"] == ["Dana"]  # "Speaker 3" resolves to its cluster's name
    assert clips["hall-frame"]["predicted"] == []  # Alice is in unsure, so she is ignored
    assert clips["hall-frame"]["records"] == 2
    assert r["no_records"] == []


def test_names_compare_without_case(state, labels):
    write(labels, "identification", dict(id="a", modality="face", camera="cam0", start=at(14, 0), end=at(14, 0, 20),
                                         people=["alice"]))  # fmt: skip
    r = identification(labels, state)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)


@pytest.mark.parametrize(
    "clip",
    [
        pytest.param(dict(modality="face", camera="cam9", at=at(9, 0)), id="nothing-recorded"),
        pytest.param(dict(modality="voice", start=at(15, 0), end=at(15, 0, 30), device="porch"), id="other-device"),
    ],
)
def test_a_clip_with_nothing_recorded_is_listed(state, labels, clip):
    write(labels, "identification", dict(clip, id="far", people=["Dana"]))
    r = identification(labels, state)
    assert (r["tp"], r["fn"], r["no_records"], r["recall"], r["precision"]) == (0, 1, ["far"], 0.0, None)


def test_playback_is_a_line_and_never_a_name(state, labels):
    clip = dict(id="radio", modality="voice", start=at(16, 0), end=at(16, 0, 30), people=[])
    write(labels, "identification", clip, dict(clip, id="window", start=at(16, 5), end=at(16, 5, 30)))
    lines = [
        dict(ts_start=at(16, 0, 5), ts_end=at(16, 0, 8), speaker="Alice", speaker_id=1, cloned_playback=True,
             text="Hello from the speaker."),
        dict(ts_start=at(16, 5, 5), ts_end=at(16, 5, 8), speaker="Alice", speaker_id=1, text="Reading it aloud."),
    ]  # fmt: skip
    for x in lines:
        speech_fake.fake().say(x)
    (state / "playback_windows.json").write_text(json.dumps([dict(id="t", start=at(16, 5), end=at(16, 5, 10))]))
    r = identification(labels, state)
    assert (r["fp"], r["confusions"], r["no_records"]) == (0, [], [])
    assert [c["records"] for c in r["per_clip"]] == [1, 1]


def test_no_store_scores_as_missed(tmp_path):
    r = identification(FIXTURES, tmp_path)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 4)


# Retrieval through history_qa.collect


def test_retrieval_fixtures_ask_history_qa(state):
    r = retrieval(FIXTURES, state)
    q = {x["id"]: x for x in r["per_question"]}
    assert q["dana-garden"]["passed"], q["dana-garden"]
    assert q["who-yesterday"]["passed"], q["who-yesterday"]
    assert "Alice" in q["who-yesterday"]["answer"]
    assert not q["alice-garden"]["passed"]
    assert "no evidence within 30s" in q["alice-garden"]["failures"][0]
    assert (r["questions"], r["passed"], r["answers_checked"], r["answers_ok"]) == (3, 2, 1, 1)
    assert (r["evidence_expected"], r["evidence_found"]) == (2, 1)
    assert not r["recorded_frame_analysis"]


def test_uses_the_public_collect_at_the_question_time_with_frames_off(state, labels, monkeypatch):
    write(labels, "retrieval", dict(id="q", question="Who was here?", asked_at="2026-10-21T09:00:00-07:00",
                                    day="2026-10-20", answer_includes=["x"]))  # fmt: skip
    seen = []
    collect = fake_collect(monkeypatch, return_value={"answer": "x", "evidence": []})
    collect.side_effect = lambda *a, **k: seen.append(cfg.get("review.recorded_frame_analysis")) or collect.return_value
    monkeypatch.setenv("PACK_REVIEW_RECORDED_FRAME_ANALYSIS", "true")  # the environment can't turn frames on
    r = retrieval(labels, state)
    on = retrieval(labels, state, as_configured=True)
    asked = dt.datetime(2026, 10, 21, 9, tzinfo=LA).timestamp()
    collect.assert_any_call("Who was here?", state_dir=state, now=asked, day="2026-10-20")
    assert seen == [False, True]
    assert (r["recorded_frame_analysis"], on["recorded_frame_analysis"], r["passed"]) == (False, True, 1)


def test_answer_excludes_and_a_raising_question(state, labels, monkeypatch):
    write(
        labels,
        "retrieval",
        dict(id="a", question="Who was here yesterday?", asked_at=at(23, 0), answer_excludes=["alice"]),
        dict(id="b", question="Who was here yesterday?", asked_at=at(23, 0), answer_includes=["Dana"]),
    )
    fake_collect(monkeypatch, side_effect=[{"answer": "Alice was here"}, RuntimeError("boom")])
    r = retrieval(labels, state)
    a, b = r["per_question"]
    assert a["failures"] == ["answer has 'alice'"]
    assert "collect() raised RuntimeError: boom" in b["failures"]
    assert r["passed"] == 0


def test_answer_checks_say_why_collect_gave_no_text(state, labels):
    write(
        labels,
        "retrieval",
        dict(id="said", question="What did Dana say about the garden yesterday?",
             asked_at="2026-10-21T09:00:00-07:00", answer_includes=["tomatoes"]),
        dict(id="keys", question="Where are my keys?", asked_at=at(23, 0), answer_includes=["kitchen"]),
    )  # fmt: skip
    said, keys = retrieval(labels, state)["per_question"]
    assert said["is_history"]
    assert "collect() gave evidence but no answer text; check evidence instead" in said["failures"]
    assert not keys["is_history"]
    assert "collect() did not treat this as a history question" in keys["failures"]


def test_answer_excludes_fail_when_collect_gives_no_answer_text(state, labels, monkeypatch):
    write(
        labels,
        "retrieval",
        dict(id="said", question="What did Dana say?", asked_at=at(23, 0), answer_excludes=["Alice"]),
        dict(id="none", question="Who was here?", asked_at=at(23, 0), answer_excludes=["Alice"]),
    )
    found = {"answer": None, "evidence": [{"ts": 1, "text": "tomatoes"}]}
    fake_collect(monkeypatch, side_effect=[found, {"answer": None, "evidence": []}])
    r = retrieval(labels, state)
    said, none = r["per_question"]
    assert said["failures"] == ["collect() gave evidence but no answer text; check evidence instead"]
    assert none["failures"] == ["collect() gave no answer text"]
    assert (r["passed"], r["answers_checked"], r["answers_ok"]) == (0, 2, 0)


# Journal entries cover a stretch of time, so a labeled moment matches anywhere in the span, widened by within_s.
ENTRY = dict(kind="journal", source="journal", ts=1000.0, start=1000.0, end=1030.0, quote="Dana planted tomatoes.")
LINE = dict(ts=1015.0, text="tomatoes", source="transcript")
NO_SPAN = dict(ts=1000.0, start="1000", end=None, text="x")
# A segment-backed item says which stream it came from; its `source` stays camera.
WALK = dict(ts=1000.0, source="camera", stream="walkby.span", quote="Walked past cam0's counting line.")


@pytest.mark.parametrize(
    "item,expected,cites",
    [
        pytest.param(ENTRY, measure.Expected(1015.0, 0.0), True, id="inside-the-span"),
        pytest.param(ENTRY, measure.Expected(1035.0, 5.0), True, id="widened-after"),
        pytest.param(ENTRY, measure.Expected(995.0, 5.0), True, id="widened-before"),
        pytest.param(ENTRY, measure.Expected(1036.0, 5.0), False, id="just-after"),
        pytest.param(ENTRY, measure.Expected(994.0, 5.0), False, id="just-before"),
        pytest.param(ENTRY, measure.Expected(1015.0, 0.0, "tomatoes", "journal"), True, id="text-and-source"),
        pytest.param(ENTRY, measure.Expected(1015.0, 0.0, "shed", "journal"), False, id="other-text"),
        pytest.param(ENTRY, measure.Expected(1015.0, 0.0, None, "transcript"), False, id="other-source"),
        pytest.param(dict(ENTRY, source="memory"), measure.Expected(1015.0, 0.0, None, "journal"), True, id="by-kind"),
        pytest.param(LINE, measure.Expected(1015.0, 0.0, None, "journal"), False, id="a-line-is-not-journal"),
        pytest.param(NO_SPAN, measure.Expected(1020.0, 30.0), True, id="no-span-uses-ts"),
        pytest.param(NO_SPAN, measure.Expected(1020.0, 10.0), False, id="no-span-too-far"),
        pytest.param(WALK, measure.Expected(1000.0, 0.0, "counting line", "walkby.span"), True, id="its-stream"),
        pytest.param(WALK, measure.Expected(1000.0, 0.0, None, "zone.visit"), False, id="another-stream"),
        pytest.param(WALK, measure.Expected(1000.0, 0.0, None, "camera"), True, id="still-a-camera-item"),
        pytest.param(LINE, measure.Expected(1015.0, 0.0, None, "walkby.span"), False, id="an-untagged-item"),
    ],
)
def test_cites(item, expected, cites):
    assert measure._cites(item, expected) is cites


# Retrieval through ask.execute (--ask and --rounds)


class Execute:
    """ask.execute faked: records each call with the search rounds configured for it, and gives the queued replies."""

    def __init__(self):
        self.calls, self.replies = [], []

    def __call__(self, question, mode="local", state_dir=None, now=None, day=None, scope="history"):
        rounds = cfg.get("recall.max_rounds") if cfg.get("recall.enabled") else 0
        journal = bool(cfg.get("journal.enabled"))
        self.calls.append(
            dict(question=question, mode=mode, state_dir=state_dir, now=now, day=day, rounds=rounds, journal=journal)
        )
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def execute(monkeypatch):
    fake = Execute()
    monkeypatch.setattr(ask, "execute", fake)
    fake_collect(monkeypatch, side_effect=AssertionError("--ask goes through execute"))
    return fake


def test_scores_the_answer_and_evidence_execute_gives(state, labels, execute):
    write(
        labels,
        "retrieval",
        dict(id="said", question="What did Dana say about the garden yesterday?", asked_at=at(23, 0),
             day="2026-10-20", answer_includes=["tomatoes"], answer_excludes=["Alice"],
             evidence=[dict(at=at(15, 0, 20), within_s=0, includes="tomatoes", source="journal")]),
        dict(id="who", question="Who was here?", asked_at=at(23, 0), answer_includes=["Dana"]),
    )  # fmt: skip
    entry = dict(kind="journal", source="journal", ts=at(15, 0), start=at(15, 0), end=at(15, 0, 30),
                 quote="Dana planted tomatoes.")  # fmt: skip
    execute.replies = [
        dict(status="ok", answer="Dana said to plant tomatoes [J1].", evidence=[entry], is_history=True,
             model="ollama/fake", rounds=[{}, {}]),
        dict(status="partial", answer="Alice was here.", evidence=[], is_history=True, model="ollama/fake"),
    ]  # fmt: skip
    with cfg.override(FAKE_MODELS):
        r = retrieval(labels, state, use_ask=True)
    said, who = r["per_question"]
    assert said["passed"], said
    assert (said["status"], said["model"], said["rounds"]) == ("ok", "ollama/fake", 2)
    assert who["failures"] == ["answer lacks 'Dana'"]
    assert (who["rounds"], r["rounds"], r["through"]) == (0, 2, "ask.execute")
    assert said["latency_s"] >= 0
    assert r["latency_max_s"] == max(said["latency_s"], who["latency_s"])
    assert execute.calls[0] == dict(question="What did Dana say about the garden yesterday?", mode="local",
                                    state_dir=state, now=at(23, 0), day="2026-10-20", rounds=3,
                                    journal=True)  # fmt: skip
    assert r["settings"] == {
        "ask.model": "", "ask.num_ctx": 0, "ask.think": False, "recall.enabled": True, "recall.max_rounds": 3, "recall.model": "", "journal.enabled": True,
        "models.ollama.llm": "fake-llm", "semantic.enabled": True, "models.ollama.embedding": "fake-embed",
    }  # fmt: skip


def test_an_answer_past_asks_deadline_fails(state, labels, execute, monkeypatch):
    write(labels, "retrieval", dict(id="slow", question="Who was here?", asked_at=at(23, 0), answer_includes=["Dana"]))
    execute.replies = [dict(status="ok", answer="Dana", evidence=[], is_history=True)]
    clock = iter([0.0, 181.0])
    monkeypatch.setattr(measure.time, "perf_counter", lambda: next(clock))
    q = retrieval(labels, state, use_ask=True)["per_question"][0]
    assert q["failures"] == ["execute() gave its answer after 181 s, past Ask's 180 s deadline"]
    assert not q["answer_ok"]


def test_a_failed_answer_fails_the_answer_check(state, labels, execute):
    write(
        labels,
        "retrieval",
        dict(id="err", question="Who was here?", asked_at=at(23, 0), answer_excludes=["Alice"]),
        dict(id="raise", question="Who was here?", asked_at=at(23, 0), answer_includes=["x"]),
    )
    execute.replies = [
        dict(status="error", answer="The local model could not complete the answer.", error="URLError"),
        RuntimeError("boom"),
    ]
    r = retrieval(labels, state, use_ask=True)
    err, raised = r["per_question"]
    assert err["failures"] == ["execute() gave status error (URLError)"]
    assert not err["answer_ok"]
    assert "execute() raised RuntimeError: boom" in raised["failures"]
    assert "latency_s" in raised
    assert (r["passed"], r["answers_ok"]) == (0, 0)


def test_vectors_set_the_semantic_index_for_the_run(state, labels, execute, monkeypatch):
    write(labels, "retrieval", dict(id="q", question="Who was here?", asked_at=at(23, 0), answer_includes=["x"]))
    seen = []
    collect = fake_collect(monkeypatch, return_value=dict(answer="x", evidence=[], is_history=True))
    collect.side_effect = lambda *a, **k: seen.append(cfg.get("semantic.enabled")) or collect.return_value
    r = retrieval(labels, state, vectors=False)
    assert (seen, r["vectors"]) == ([False], False)  # through collect too
    execute.replies = [dict(status="ok", answer="x", evidence=[])]
    r = retrieval(labels, state, use_ask=True, vectors=True)
    assert (r["vectors"], r["settings"]["semantic.enabled"]) == (True, True)
    with monkeypatch.context() as env:
        env.setenv("PACK_SEMANTIC_ENABLED", "true")
        with pytest.raises(measure.RunError, match="PACK_SEMANTIC_ENABLED"):
            retrieval(labels, state, vectors=False)
    assert cfg.get("semantic.enabled")  # restored
    assert "semantic index off" in measure.report(dict(retrieval=dict(r, vectors=False)), "d", "s")


def test_journal_sets_the_journal_for_the_run(state, labels, execute, monkeypatch):
    write(labels, "retrieval", dict(id="q", question="Who was here?", asked_at=at(23, 0), answer_includes=["x"]))
    execute.replies = [dict(status="ok", answer="x", evidence=[])]
    r = retrieval(labels, state, use_ask=True, journal=False)
    assert ([c["journal"] for c in execute.calls], r["journal"], r["settings"]["journal.enabled"]) == (
        [False],
        False,
        False,
    )
    with monkeypatch.context() as env:
        env.setenv("PACK_JOURNAL_ENABLED", "true")
        with pytest.raises(measure.RunError, match="PACK_JOURNAL_ENABLED"):
            retrieval(labels, state, use_ask=True, journal=False)
    assert cfg.get("journal.enabled")  # restored
    assert "journal off" in measure.report(dict(retrieval=r), "d", "s")


def test_drafts_are_scored_and_counted_apart(state, labels, monkeypatch):
    write(
        labels,
        "retrieval",
        dict(id="sure", question="Who was here?", asked_at=at(23, 0), answer_includes=["x"]),
        dict(id="maybe", question="Who was here?", asked_at=at(23, 0), answer_includes=["y"], draft=True),
    )
    fake_collect(monkeypatch, return_value=dict(answer="x", evidence=[], is_history=True))
    r = retrieval(labels, state)
    assert (r["passed"], r["drafts"], r["drafts_passed"]) == (1, 1, 0)
    assert [q.get("draft") for q in r["per_question"]] == [None, True]
    text = measure.report(dict(retrieval=r), "d", "s")
    assert "1 are drafts, not yet confirmed; 0 of them passed" in text
    assert "FAIL maybe (draft)" in text


def test_a_draft_that_isnt_true_or_false_is_refused(labels):
    write(labels, "retrieval", dict(id="q", question="Who?", asked_at=at(23, 0), answer_includes=["x"], draft="yes"))
    with pytest.raises(measure.LabelError, match="draft must be true or false"):
        measure.load(labels, "retrieval")


def test_rounds_set_max_rounds_for_the_run_and_imply_ask(state, labels, execute):
    write(labels, "retrieval", dict(id="q", question="Who was here?", asked_at=at(23, 0), answer_includes=["x"]))
    execute.replies = [dict(status="ok", answer="x", evidence=[]), dict(status="ok", answer="x", evidence=[])]
    r = retrieval(labels, state, rounds=5)
    assert (r["settings"]["recall.max_rounds"], r["through"]) == (5, "ask.execute")
    r = retrieval(labels, state, rounds=0)  # no loop: recall.enabled goes off for the run
    assert [c["rounds"] for c in execute.calls] == [5, 0]
    assert (r["settings"]["recall.enabled"], r["settings"]["recall.max_rounds"]) == (False, 0)
    assert (cfg.get("recall.enabled"), cfg.get("recall.max_rounds")) == (True, 3)  # restored


def test_an_environment_variable_that_would_win_over_rounds_is_refused(state, labels, execute, monkeypatch, capsys):
    write(labels, "retrieval", dict(id="q", question="Who was here?", asked_at=at(23, 0), answer_includes=["x"]))
    monkeypatch.setenv("PACK_RECALL_MAX_ROUNDS", "1")
    with pytest.raises(measure.RunError, match="PACK_RECALL_MAX_ROUNDS"):
        retrieval(labels, state, rounds=3)
    assert measure.main(["--state-dir", str(state), "--dir", str(labels), "--rounds", "3"]) == 2
    out, err = capsys.readouterr()
    assert (out, execute.calls) == ("", [])
    assert "PACK_RECALL_MAX_ROUNDS" in err


def test_the_default_run_does_not_ask_and_keeps_its_fields(state, monkeypatch):
    monkeypatch.setattr(ask, "execute", create_autospec(ask.execute, side_effect=AssertionError("must not execute")))
    r = retrieval(FIXTURES, state)
    assert set(r) == {"questions", "passed", "answers_checked", "answers_ok", "evidence_expected", "evidence_found",
                      "recorded_frame_analysis", "timezone", "per_question"}  # fmt: skip
    assert not any("latency_s" in q or "rounds" in q for q in r["per_question"])


def test_a_real_ask_with_search_rounds_writes_nothing(state, labels, monkeypatch):
    """execute, collect, and the loop for real, with only the two model calls faked."""
    write(labels, "retrieval", dict(id="said", question="What did Dana say about the garden yesterday?",
                                    asked_at="2026-10-21T09:00:00-07:00", answer_includes=["tomatoes"],
                                    evidence=[dict(at=at(15, 0, 5), within_s=5, includes="tomatoes",
                                                   source="transcript")]))  # fmt: skip
    decisions = [{"decision": "search", "tool": "transcript", "args": {}}, {"decision": "answer"}]
    monkeypatch.setattr(recall, "_chat", create_autospec(recall._chat, side_effect=decisions))
    answer = create_autospec(ask.local_answer, return_value=("Dana wants tomatoes [T1].", "ollama/fake"))
    monkeypatch.setattr(ask, "local_answer", answer)
    before = files(state)
    q = retrieval(labels, state, rounds=2)["per_question"][0]
    assert files(state) == before
    assert q["passed"], q
    assert (q["rounds"], q["model"]) == (2, "ollama/fake")
    assert (q["evidence_given"], q["evidence_found"]) == (1, 1)  # ids in the model's real input, matched back


def test_evidence_found_but_cut_from_the_models_input_is_not_given(state, labels, monkeypatch):
    write(labels, "retrieval",
          dict(id="cut", question="What did Dana say?", asked_at=at(23, 0), answer_includes=["tomatoes"],
               evidence=[dict(at=at(15, 0), within_s=1, includes="tomatoes"),
                         dict(at=at(16, 0), within_s=1, includes="garden")]),
          dict(id="recorded", question="Who was here?", asked_at=at(23, 0), answer_includes=["Alice"],
               evidence=[dict(at=at(14, 0), within_s=1)]))  # fmt: skip
    lines = [dict(id="T1", ts=at(15, 0), quote="Plant tomatoes."), dict(id="T2", ts=at(16, 0), quote="The garden.")]
    seen = dict(id="C1", ts=at(14, 0), quote="Alice at the door.")

    def execute(question, mode="local", state_dir=None, now=None, day=None, scope="history"):
        if question == "Who was here?":  # a recorded answer: no model call
            return dict(status="ok", answer="Alice.", evidence=[seen], is_history=True, model="local-history-evidence")
        answer, model = ask.local_answer(question, json.dumps({"evidence": [{"id": "T2", "text": "The garden."}]}))
        return dict(status="ok", answer=answer, evidence=lines, is_history=True, model=model)

    monkeypatch.setattr(ask, "execute", execute)

    def answer(question, context, now=None, *, trace):
        trace.update(thinking="T2 is about the garden.", first_token_s=0.4, model_s=1.5)
        return "No tomatoes were mentioned.", "ollama/fake"

    monkeypatch.setattr(ask, "local_answer", answer)
    r = retrieval(labels, state, use_ask=True)
    cut, recorded = r["per_question"]
    assert (cut["evidence_found"], cut["evidence_given"], cut["model_input"]) == (2, 1, 1)
    assert (cut["evidence_ids"], cut["given_ids"]) == (["T1", "T2"], ["T2"])
    assert (cut["thinking"], cut["first_token_s"], cut["model_s"]) == ("T2 is about the garden.", 0.4, 1.5)
    assert (recorded["evidence_found"], recorded["evidence_given"], recorded["model_input"]) == (1, 1, None)
    assert "thinking" not in recorded and "first_token_s" not in recorded
    assert (r["evidence_found"], r["evidence_given"]) == (3, 2)
    assert "evidence found 3 of 3, 2 of them given to the model" in measure.report(dict(retrieval=r), "d", "s")


# Label formats

FACE = dict(id="a", modality="face", camera="cam0", people=[])
CLIP = dict(id="a", modality="face", camera="cam0", start=1, end=2, people=["Dana"])
QUESTION = dict(id="q", question="Who was here yesterday?", asked_at=1792600000, answer_includes=["Alice"])
EVENT = dict(id="s", camera="cam1", what="opened the fridge", start=1, end=2, count=5)


def without(row, key):
    return {k: v for k, v in row.items() if k != key}


@pytest.mark.parametrize(
    "part,row,message",
    [
        pytest.param("identification", dict(FACE, start="2026-10-20T14:00:00", end=1), "attach its zone", id="naive"),
        pytest.param("identification", dict(FACE, start="yesterday", end=1), "not an ISO 8601 time", id="not-iso"),
        pytest.param("identification", dict(FACE, start=10, end=5), "end is before start", id="backwards"),
        pytest.param("identification", dict(FACE, at=5, start=1, end=9), "not both", id="at-and-span"),
        pytest.param("identification", FACE, "needs start and end, or at", id="no-time"),
        pytest.param("identification", dict(CLIP, modality="smell"), "modality must be one of", id="modality"),
        pytest.param("identification", without(CLIP, "people"), "people is required", id="no-people"),
        pytest.param("identification", without(CLIP, "camera"), "camera must be", id="face-no-camera"),
        pytest.param("identification", dict(CLIP, people="Dana"), "people must be a list", id="people-str"),
        pytest.param("identification", dict(CLIP, unsure=["dana"]), "both people and unsure", id="people-unsure"),
        pytest.param("retrieval", without(QUESTION, "answer_includes"), "nothing to check", id="nothing-to-check"),
        pytest.param("retrieval", without(QUESTION, "asked_at"), "asked_at", id="no-asked-at"),
        pytest.param("retrieval", dict(QUESTION, day="20/10/2026"), "day must be YYYY-MM-DD", id="day"),
        pytest.param(
            "retrieval", dict(QUESTION, evidence=[{"at": 1, "source": "dream"}]), "evidence source", id="source"
        ),
        pytest.param("retrieval", dict(QUESTION, evidence=[{"at": 1, "within_s": -1}]), "within_s", id="within"),
        pytest.param("events", dict(EVENT, count=2.5), "count must be", id="count"),
    ],
)
def test_bad_labels_are_refused_with_their_line(tmp_path, part, row, message):
    write(tmp_path, part, row)
    with pytest.raises(measure.LabelError, match=rf"{part}\.jsonl:1: .*{message}"):
        measure.load(tmp_path, part)


def test_missing_file_is_no_labels(tmp_path):
    assert measure.load(tmp_path, "retrieval") == []


def test_accepted_labels(tmp_path):
    write(tmp_path, "identification", dict(CLIP, start=1792533600000, end=1792533601000),
          dict(without(CLIP, "camera"), id="v", modality="voice"), dict(FACE, id="f", at=100))  # fmt: skip
    ms, voice, frame = measure.load(tmp_path, "identification", frame_window=2.5)
    assert (ms.start, ms.end) == (1792533600.0, 1792533601.0)  # milliseconds, as the kernel reads them
    assert voice.camera is None
    assert (frame.start, frame.end, frame.frame) == (97.5, 102.5, True)
    write(tmp_path, "retrieval", dict(id="j", question="?", asked_at=1, evidence=[dict(at=1, source="journal")]))
    assert measure.load(tmp_path, "retrieval")[0].evidence[0].source == "journal"


def test_duplicate_ids_and_comments(tmp_path):
    row = json.dumps(EVENT)
    (tmp_path / "events.jsonl").write_text(f"# the kitchen, Tuesday\n\n{row}\n")
    assert len(measure.load(tmp_path, "events")) == 1
    (tmp_path / "events.jsonl").write_text(f"{row}\n{row}\n")
    with pytest.raises(measure.LabelError, match=r"events.jsonl:2: id 's' is used twice"):
        measure.load(tmp_path, "events")


def test_not_json(tmp_path):
    (tmp_path / "events.jsonl").write_text("{nope\n")
    with pytest.raises(measure.LabelError, match="events.jsonl:1: "):
        measure.load(tmp_path, "events")


# The command


@pytest.fixture
def run(state, capsys):
    def run(*args):
        code = measure.main(["--state-dir", str(state), *args])
        out, err = capsys.readouterr()
        return code, out, err

    return run


def test_report_reads_and_never_writes_state(state, run):
    before = files(state)
    code, out, _ = run("--dir", str(FIXTURES))
    assert code == 0
    assert files(state) == before
    for figures in ("5 labeled clips", "precision 0.50", "recall 0.50", "Dana -> Alice", "passed 2 of 3",
                    "FAIL alice-garden", "6 occurrences"):  # fmt: skip
        assert figures in out


def test_one_part_as_json(run):
    code, out, _ = run("--dir", str(FIXTURES), "--part", "events", "--json")
    assert (code, list(json.loads(out))) == (0, ["events"])


def test_no_labels_yet(labels, run):
    with cfg.override({"paths": {"measurement": str(labels)}}):
        code, out, _ = run()
    assert code == 0
    assert str(labels) in out


@pytest.mark.parametrize("option", ["--frame-window", "--rounds"])
def test_negative_numbers_are_refused(run, option):
    with pytest.raises(SystemExit) as e:
        run("--dir", str(FIXTURES), option, "-1")
    assert e.value.code == 2


def test_ask_report_and_json_name_the_settings(labels, run, monkeypatch):
    write(labels, "retrieval", dict(id="q", question="Who was here?", asked_at=at(23, 0), answer_includes=["x"]))
    reply = dict(status="ok", answer="x", evidence=[], model="ollama/fake", rounds=[{}])
    monkeypatch.setattr(ask, "execute", create_autospec(ask.execute, return_value=reply))
    with cfg.override(FAKE_MODELS):
        code, out, _ = run("--dir", str(labels), "--part", "retrieval", "--ask")
        _, data, _ = run("--dir", str(labels), "--part", "retrieval", "--rounds", "2", "--json")
    assert code == 0
    for setting in ("ask.execute", "3 rounds", "fake-llm", "fake-embed", "search rounds 1"):
        assert setting in out
    r = json.loads(data)["retrieval"]
    assert (r["settings"]["recall.max_rounds"], r["per_question"][0]["rounds"]) == (2, 1)


def test_compare_two_json_results(labels, run, monkeypatch):
    write(
        labels,
        "retrieval",
        dict(id="a", question="Who was here?", asked_at=at(23, 0), answer_includes=["Dana"]),
        dict(id="b", question="Who was here?", asked_at=at(23, 0), answer_includes=["Alice"]),
    )
    single, loop = labels / "single.json", labels / "loop.json"
    monkeypatch.setattr(
        ask, "execute", execute := create_autospec(ask.execute, return_value=dict(status="ok", answer="Alice"))
    )
    single.write_text(run("--dir", str(labels), "--part", "retrieval", "--rounds", "0", "--json")[1])
    execute.return_value = dict(status="ok", answer="Dana", rounds=[{}])
    loop.write_text(run("--dir", str(labels), "--part", "retrieval", "--rounds", "3", "--json")[1])
    code, out, _ = run("--compare", str(single), str(loop))
    assert code == 0
    assert "search loop off" in out and "search loop on, 3 rounds" in out
    assert "1 of 2  ->  1 of 2" in out
    assert "given to model  0 of 0  ->  0 of 0" in out
    assert "better (failed in A, passed in B): a" in out
    assert "worse (passed in A, failed in B): b" in out
    assert run("--compare", str(single), str(labels / "missing.json"))[0] == 2


def test_bad_label_exits_with_its_line(labels, run):
    write(labels, "retrieval", dict(id="q", question="?", asked_at=1))
    code, out, err = run("--dir", str(labels))
    assert (code, out) == (2, "")
    assert "retrieval.jsonl:1: nothing to check" in err


def test_answer_text_is_matched_without_its_surrounding_spaces(state, labels, monkeypatch):
    write(
        labels,
        "retrieval",
        dict(
            id="q", question="Who was here?", asked_at=at(23, 0), answer_includes=[" Dana "], answer_excludes=[" Al "]
        ),
    )
    fake_collect(monkeypatch, return_value={"answer": "Dana was here. Alice too."})
    assert retrieval(labels, state)["per_question"][0]["failures"] == ["answer has 'Al'"]


def test_an_answer_any_group_passes_on_any_one_of_its_wordings(state, labels, monkeypatch):
    write(
        labels,
        "retrieval",
        dict(id="digit", question="How many?", asked_at=at(23, 0), answer_any=[["three", "3"], ["cards"]]),
        dict(id="word", question="How many?", asked_at=at(23, 0), answer_any=[["three", "3"]]),
        dict(id="none", question="How many?", asked_at=at(23, 0), answer_any=[["three", "3"]]),
    )
    fake_collect(
        monkeypatch,
        side_effect=[{"answer": "3 of the Cards"}, {"answer": "Three of them"}, {"answer": "Two of them"}],
    )
    r = retrieval(labels, state)
    assert [q["failures"] for q in r["per_question"]] == [[], [], ["answer lacks any of ['three', '3']"]]
    assert r["answers_checked"] == 3


@pytest.mark.parametrize("value", [["three"], [[]], "three", [["three", ""]]])
def test_answer_any_must_be_groups_of_text(labels, run, value):
    write(labels, "retrieval", dict(id="q", question="How many?", asked_at=at(23, 0), answer_any=value))
    code, _, err = run("--dir", str(labels))
    assert code == 2
    assert "answer_any must be a list of" in err


# Identification: a label about one record (People's right and wrong marks)


def sighting_ids(vision, cam):
    return [r["id"] for r in vision.range(segment_log.FACE_MATCHES, 0, 2e9) if r["cam"] == cam]


def test_a_label_about_one_sighting_scores_only_it(state, labels, vision):
    # cam1 at 14:20 has an unrecognized face (really Dana) and, half a second later, one the pack called Alice.
    unknown, _alice = sighting_ids(vision, "cam1")
    write(
        labels,
        "identification",
        dict(id="just-unknown", modality="face", camera="cam1", at=at(14, 20, 0.3), sighting=unknown, people=["Dana"]),
        dict(id="frame", modality="face", camera="cam1", at=at(14, 20, 0.3), people=["Dana"]),
    )
    one, frame = identification(labels, state)["per_clip"]
    assert (one["predicted"], one["missed"], one["records"]) == ([], ["Dana"], 1)
    assert (frame["extra"], frame["records"]) == (["Alice"], 2)  # the whole frame also scores Alice's sighting


def test_a_label_about_one_line_scores_only_it(state, labels):
    first = at(15, 0, 5)
    window = dict(modality="voice", start=first, end=at(15, 30, 7), people=["Dana"])
    write(labels, "identification", dict(id="one", line=first, **window), dict(id="both", **window))
    one, both = identification(labels, state)["per_clip"]
    assert (one["predicted"], one["records"]) == (["Dana"], 1)
    assert (both["extra"], both["records"]) == (["Alice"], 2)


@pytest.mark.parametrize(
    ("row", "error"),
    [
        (dict(modality="voice", start=1, end=2, sighting=4), "sighting is for face labels"),
        (dict(modality="face", camera="cam0", at=1, line=1.5), "line is for voice labels"),
        (dict(modality="face", camera="cam0", at=1, sighting=2), "sighting must be a sighting id"),
        (dict(modality="face", camera="cam0", at=1, sighting=""), "sighting must be a sighting id"),
        (dict(modality="voice", start=1, end=2, line=-1), "line must be a line ts_start"),
    ],
)
def test_a_one_record_label_is_checked(labels, row, error):
    write(labels, "identification", dict(id="x", people=[], **row))
    with pytest.raises(measure.LabelError, match=error):
        measure.load(labels, "identification")


def test_a_label_in_a_renamed_persons_current_name_scores_as_right(state, labels, vision):
    # People shows a person's lines and shots under their name now, so a mark made there names them that way. The
    # pack's own names, heard or seen before the rename, must be read the same way, or a right mark scores as wrong.
    from liberational.people import registry

    registry.at(state).rename("Alice", "Alicia", "test")
    sighting = sighting_ids(vision, "cam0")[0]  # recognized as Alice at 14:00:05
    write(
        labels,
        "identification",
        dict(
            id="line", modality="voice", start=at(15, 30, 5), end=at(15, 30, 7), line=at(15, 30, 5), people=["Alicia"]
        ),
        dict(id="shot", modality="face", camera="cam0", at=at(14, 0, 5), sighting=sighting, people=["Alicia"]),
    )
    score = identification(labels, state)
    assert (score["tp"], score["fp"], score["fn"]) == (2, 0, 0)


# Identification by threshold


@pytest.fixture
def sweep_state(tmp_path, monkeypatch, vision):
    """Faces and voices with known similarities, and labels saying who was really there. Hand-derived counts below."""
    monkeypatch.setattr(history_qa, "TZ", LA)
    state = tmp_path / "sweep-state"
    state.mkdir()
    for ts, name, sim in (
        (at(9, 0), "Alice", 0.50),  # right
        (at(9, 1), "Alice", 0.80),  # right
        (at(9, 2), "Bob", 0.60),  # wrong: really Alice
        (at(9, 3), "Carol", None),  # right, named by a person
    ):
        vision.face(ts, "cam0", name, sim)
    lines = [
        dict(ts_start=at(10, 0), ts_end=at(10, 0, 2), speaker="Dana", speaker_id=4, speaker_sim=0.55, device="k"),
        dict(ts_start=at(10, 1), ts_end=at(10, 1, 2), speaker="Dana", speaker_id=4, speaker_sim=0.70, device="k"),
        dict(ts_start=at(10, 2), ts_end=at(10, 2, 2), speaker="Ed", speaker_id=5, speaker_sim=None, device="k"),
    ]  # the first right; the second really Alice; the third right, named by a person
    speech_fake.fake().write(lines)
    write_voices(state, {1: "Alice", 4: "Dana", 5: "Ed"})
    labels = tmp_path / "sweep-labels"
    labels.mkdir()

    def clip(cid, modality, hour, minute, who, **where):
        start = dt.datetime(2026, 10, 20, hour, minute, tzinfo=LA)
        return {"id": cid, "modality": modality, "start": start.isoformat(),
                "end": (start + dt.timedelta(seconds=10)).isoformat(), "people": [who], **where}  # fmt: skip

    write(labels, "identification",
          clip("f1", "face", 9, 0, "Alice", camera="cam0"), clip("f2", "face", 9, 1, "Alice", camera="cam0"),
          clip("f3", "face", 9, 2, "Alice", camera="cam0"), clip("f4", "face", 9, 3, "Carol", camera="cam0"),
          clip("v1", "voice", 10, 0, "Dana", device="k"), clip("v2", "voice", 10, 1, "Alice", device="k"),
          clip("v3", "voice", 10, 2, "Ed", device="k"))  # fmt: skip
    return state, labels


def rows(sweep, modality):
    return {r["threshold"]: (r["tp"], r["fp"], r["fn"], r["today"]) for r in sweep[modality]["rows"]}


def test_the_sweep_counts_each_threshold_from_the_stored_similarities(sweep_state):
    state, labels = sweep_state
    sweep = measure.sweep_identification(measure.load(labels, "identification"), state)
    face = rows(sweep, "face")
    # A face is named above the threshold (sim > t): Alice at 0.50 goes at 0.50, Bob's wrong 0.60 at 0.60, Alice's
    # 0.80 at 0.80. Carol, named by a person, stays at every threshold.
    assert [face[t] for t in (0.45, 0.5, 0.6, 0.8, 0.9)] == [
        (3, 1, 1, True), (2, 1, 2, False), (2, 0, 2, False), (1, 0, 3, False), (1, 0, 3, False),
    ]  # fmt: skip
    voice = rows(sweep, "voice")
    # A voice is named at the threshold or over (sim >= t): Dana's right 0.55 goes above 0.55, her wrong 0.70 above
    # 0.70; Ed, named by a person, stays.
    assert [voice[t] for t in (0.55, 0.6, 0.7, 0.75)] == [
        (2, 1, 1, True), (1, 1, 2, False), (1, 1, 2, False), (1, 0, 2, False),
    ]  # fmt: skip
    assert (min(face), max(face), min(voice), max(voice)) == (0.45, 0.9, 0.55, 0.9)


def test_the_sweep_at_todays_threshold_matches_todays_score(sweep_state):
    state, labels = sweep_state
    clips = measure.load(labels, "identification")
    today = measure.score_identification(clips, state)
    sweep = measure.sweep_identification(clips, state)
    first = [sweep[m]["rows"][0] for m in ("face", "voice")]
    assert sum(r["tp"] for r in first) == today["tp"] and sum(r["fp"] for r in first) == today["fp"]


def test_sweep_report_marks_today_and_reads_nothing_new(sweep_state, capsys):
    state, labels = sweep_state
    code = measure.main(["--state-dir", str(state), "--dir", str(labels), "--part", "identification", "--sweep"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Identification by threshold" in out
    assert "*0.45" in out and "*0.55" in out and " 0.60" in out
    assert "a record left unnamed keeps no candidate" in out


def test_sweep_needs_identification(sweep_state, capsys):
    state, labels = sweep_state
    with pytest.raises(SystemExit):
        measure.main(["--state-dir", str(state), "--dir", str(labels), "--part", "retrieval", "--sweep"])
    assert "--sweep scores identification" in capsys.readouterr().err


@pytest.mark.parametrize(
    "source", ["transcript", "journal", "walkby.span", "zone.visit", "action.span", "movement.track"]
)
def test_a_label_can_expect_a_source_or_a_segment_stream(source):
    assert measure._expected({"at": 1000.0, "source": source}).source == source


def test_a_label_naming_another_stream_is_refused():
    with pytest.raises(measure.LabelError, match="evidence source must be one of"):
        measure._expected({"at": 1000.0, "source": "body.detection"})


def read_labels(labels, *lines):
    (labels / "readings.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    return measure.load(labels, "readings")


WHO = {
    "id": "who",
    "question": "who was here at 3pm",
    "reading": {"source": "people", "measure": "who", "clock": "3pm"},
}
WHO_AGAIN = {"id": "who-again", "question": "who was around at 3pm", "same_as": "who"}
WHO_READ = question.Reading(source="people", measure="who", clock="3pm")


def test_a_reading_is_scored_field_by_field_against_its_label(labels, readings):
    readings["who was here at 3pm"] = question.Reading(source="people", measure="who", clock="3", people=("Dana",))
    r = measure.score_readings(read_labels(labels, WHO))
    assert (r["questions"], r["exact"], r["unreadable"]) == (1, 0, 0)
    assert r["per_question"][0]["wrong"] == {
        "people": {"expected": [], "got": ["dana"]},
        "clock": {"expected": "3pm", "got": "3"},
    }
    assert r["fields"]["clock"] == {"right": 0, "of": 1}
    assert r["fields"]["source"] == {"right": 1, "of": 1}


def test_a_reworded_copy_is_scored_on_its_originals_reading_and_whether_the_two_agree(labels, readings):
    readings["who was here at 3pm"] = WHO_READ
    readings["who was around at 3pm"] = question.Reading(source="people", measure="who", clock="3pm", now=True)
    r = measure.score_readings(read_labels(labels, WHO, WHO_AGAIN))
    assert (r["exact"], r["pairs"], r["pairs_agreed"]) == (1, 1, 0)
    assert r["per_question"][1]["wrong"] == {"now": {"expected": False, "got": True}}
    readings["who was around at 3pm"] = WHO_READ
    r = measure.score_readings(read_labels(labels, WHO, WHO_AGAIN))
    assert (r["exact"], r["pairs_agreed"]) == (2, 1)


def test_names_are_compared_without_case_or_order_and_a_refused_time_by_being_refused(labels, readings):
    label = {
        "id": "q",
        "question": "what did Dana and Sam say on Friday",
        "reading": {"people": ["Dana", "Sam"], "other_time": "on Friday"},
    }
    readings[label["question"]] = question.Reading(people=("sam", "dana"), other_time="Friday")
    assert measure.score_readings(read_labels(labels, label))["exact"] == 1


def test_a_question_that_isnt_history_is_scored_on_that_alone(labels, readings):
    label = {"id": "joke", "question": "tell me a joke", "reading": {"history": False}}
    readings["tell me a joke"] = question.Reading(history=False, people=("me",), source="both")
    r = measure.score_readings(read_labels(labels, label))
    assert r["exact"] == 1
    assert r["fields"]["history"] == {"right": 1, "of": 1}
    assert r["fields"]["source"] == {"right": 0, "of": 0}


def test_no_reading_from_the_model_counts_every_field_wrong(labels, monkeypatch):
    def down(asked):
        raise question.Unreadable("TimeoutError")

    monkeypatch.setattr(question, "read", down)
    r = measure.score_readings(read_labels(labels, WHO))
    assert (r["exact"], r["unreadable"]) == (0, 1)
    assert r["fields"]["history"] == {"right": 0, "of": 1}
    assert "MISREAD who: no reading" in measure.report({"readings": r}, labels, labels)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        pytest.param({"id": "a", "question": "q"}, "reading .* or same_as", id="neither"),
        pytest.param({"id": "a", "question": "q", "reading": {}, "same_as": "b"}, "not both", id="both"),
        pytest.param({"id": "a", "question": "q", "reading": {"mood": "curious"}}, "doesn't", id="unknown-field"),
        pytest.param(
            {"id": "a", "question": "q", "reading": {"source": "memories"}}, "reading.source", id="bad-choice"
        ),
    ],
)
def test_a_readings_label_that_cant_be_scored_is_refused(labels, line, message):
    with pytest.raises(measure.LabelError, match=message):
        read_labels(labels, line)


def test_a_copy_of_a_label_that_isnt_there_is_refused(labels):
    with pytest.raises(measure.LabelError, match="same_as 'who'"):
        measure.score_readings(read_labels(labels, WHO_AGAIN))


def test_the_readings_report_names_the_weakest_fields_and_the_misread_questions(labels, readings):
    readings["who was here at 3pm"] = question.Reading(source="people", measure="who")
    readings["who was around at 3pm"] = WHO_READ
    out = measure.report({"readings": measure.score_readings(read_labels(labels, WHO, WHO_AGAIN))}, labels, labels)
    assert "read exactly right 1 of 2; no reading for 0" in out
    assert "reworded copies read like their original 0 of 1" in out
    assert "  clock: 1 of 2" in out
    assert "MISREAD who: clock '3pm' read as ''" in out
    assert "MISREAD who-again" not in out


def test_readings_run_only_when_named(state, labels, monkeypatch):
    monkeypatch.setattr(
        measure, "score_readings", create_autospec(measure.score_readings, return_value={"questions": 0})
    )
    assert "readings" not in measure.run(FIXTURES, state)
    assert measure.run(FIXTURES, state, ["readings"]) == {"readings": {"questions": 0}}


def test_the_made_up_readings_load_and_every_copy_has_its_original():
    made_up = measure.load(FIXTURES, "readings")
    originals = {label.id for label in made_up if label.reading is not None}
    assert len(made_up) > 20
    assert {label.same_as for label in made_up if label.same_as} <= originals
    # A label that repeats one of the reader's worked examples would score the prompt against itself.
    examples = {" ".join(asked.lower().rstrip("?").split()) for asked, _ in question.SHOTS}
    assert not {label.id for label in made_up if " ".join(label.question.lower().rstrip("?").split()) in examples}


def test_a_retrieval_row_keeps_the_query_and_its_reading(state, labels):
    (labels / "retrieval.jsonl").write_text(
        json.dumps(
            {
                "id": "who",
                "question": "Who was here yesterday?",
                "asked_at": "2026-10-21T09:00:00-07:00",
                "answer_includes": ["Alice"],
            }
        )
        + "\n"
    )
    row = measure.score_retrieval(measure.load(labels, "retrieval"), state)["per_question"][0]
    assert row["query"]["reading"]["source"] == "people"
    assert row["query"]["window"] == "yesterday"
