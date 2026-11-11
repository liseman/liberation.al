"""Who "she", "he", or "they" is in a question to Ask or voice (memory/pronouns.py)."""

import pytest

from liberational.memory import ask_history, pronouns, subject_activity

NOW = 1_800_000_000.0
MAY = {"id": "p-may", "name": "May", "species": "cat", "sex": "female"}
TOM = {"id": "p-tom", "name": "Tom", "species": "cat", "sex": "male"}


@pytest.fixture
def said(monkeypatch):
    """Earlier questions on each surface, newest first as Ask's history lists them: (seconds ago, question)."""
    asked: dict[str, list[tuple[float, str]]] = {}

    class History:
        def list_entries(self, *, surface=None, limit=20, **_):
            rows = sorted(asked.get(surface or "", []))
            return [{"question": q, "created_at": NOW - ago, "surface": surface} for ago, q in rows][:limit]

    monkeypatch.setattr(ask_history, "at", lambda state=None: History())
    monkeypatch.setattr(subject_activity, "_pets", lambda state: [dict(MAY)])
    return asked


def resolve(tmp_path, question, surface="voice"):
    return pronouns.resolve(question, tmp_path, surface, NOW)


def test_the_earlier_question_read_is_bounded_to_the_recent_window(tmp_path, monkeypatch):
    seen: dict[str, object] = {}

    class History:
        def list_entries(self, *, surface=None, limit=20, **kwargs):
            seen.update(kwargs, limit=limit, surface=surface)
            return []

    monkeypatch.setattr(ask_history, "at", lambda state=None: History())
    monkeypatch.setattr(subject_activity, "_pets", lambda state: [])
    assert resolve(tmp_path, "where is she") is None
    assert seen["since"] == NOW - pronouns.RECENT_S


def test_she_is_the_animal_the_last_question_asked_about(tmp_path, said):
    said["voice"] = [(120, "what has the cat been doing lately")]
    r = resolve(tmp_path, "how many times did she use the litter box today")
    assert r is not None
    assert r.question == "how many times did the cat use the litter box today"
    assert r.note == 'Taking "she" as the cat.'


def test_a_person_named_earlier_is_who_he_is(tmp_path, said, monkeypatch):
    from liberational.memory import history_qa

    monkeypatch.setattr(history_qa, "people_named", lambda q: ["Dana"] if "Dana" in q else [])
    said["live"] = [(60, "when was Dana last seen"), (30, "what did he say then")]
    r = resolve(tmp_path, "where did he go after that", surface="live")
    assert r is not None and r.meant == "Dana", "followed through the earlier pronoun to Dana"
    assert r.question == "where did Dana go after that"


def test_they_is_the_people_named_together(tmp_path, said, monkeypatch):
    from liberational.memory import history_qa

    monkeypatch.setattr(history_qa, "people_named", lambda q: ["Dana", "Eve"] if "Dana" in q else [])
    said["live"] = [(90, "what did Dana and Eve talk about yesterday")]
    r = resolve(tmp_path, "what did they decide about their trip", surface="live")
    assert r is not None
    assert r.question == "what did Dana and Eve decide about Dana and Eve's trip"


def test_another_surface_or_an_old_question_doesn_t_set_the_subject(tmp_path, said, monkeypatch):
    monkeypatch.setattr(subject_activity, "_pets", lambda state: [])
    said["live"] = [(30, "what has the cat been doing")]
    said["voice"] = [(pronouns.RECENT_S + 5, "what has the dog been doing")]
    assert resolve(tmp_path, "where is she") is None


@pytest.mark.parametrize(
    ("pets", "question", "meant"),
    [
        pytest.param([MAY, TOM], "where is she", "May", id="the-female-pet"),
        pytest.param([MAY, TOM], "where is he", "Tom", id="the-male-pet"),
        pytest.param([{**MAY, "sex": None}], "where is she", "May", id="the-only-pet-without-a-sex"),
        pytest.param([{**TOM}], "where is she", None, id="the-only-pet-is-male"),
        pytest.param([{**MAY, "sex": None}, {**TOM, "sex": None}], "where is she", None, id="two-pets-no-sexes"),
        pytest.param([MAY], "where are they", None, id="they-is-never-a-guess"),
    ],
)
def test_with_no_recent_subject_she_or_he_is_a_pet_by_sex_or_the_only_pet(
    tmp_path, said, monkeypatch, pets, question, meant
):
    monkeypatch.setattr(subject_activity, "_pets", lambda state: [dict(p) for p in pets])
    r = resolve(tmp_path, question)
    assert (r.meant if r else None) == meant


def test_a_question_that_names_its_subject_keeps_its_pronouns(tmp_path, said):
    said["voice"] = [(30, "what has the dog been doing")]
    assert resolve(tmp_path, "did May eat before she slept") is None


def test_ask_answers_the_resolved_question_and_says_who_it_took_the_pronoun_as(tmp_path, said, monkeypatch):
    from liberational.memory import ask

    said["live"] = [(30, "what has May been doing lately")]
    asked = []
    monkeypatch.setattr(ask, "STATE", tmp_path)
    jobs = ask.JobManager(runner=lambda q, **o: asked.append(q) or {"status": "ok", "answer": "Twice.", "evidence": []})
    monkeypatch.setattr(ask, "_history_start", lambda *a, **k: None)
    monkeypatch.setattr(ask, "_history_finish", lambda ident, result, *a, **k: (result, False))
    job = jobs.submit("how many times did she use the litter box today", surface="live")
    job = _wait(jobs, job["id"])
    assert asked == ["how many times did May use the litter box today"]
    assert job["result"]["answer"] == 'Taking "she" as May. Twice.'


def _wait(jobs, jid):
    import time

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = jobs.get(jid)
        if job and job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.01)
    raise AssertionError("the job never finished")


def test_voice_sends_a_pronoun_question_about_the_cat_to_history(tmp_path, said, monkeypatch):
    from liberational import config as cfg
    from liberational.assistant import voice
    from liberational.memory import history_qa

    said["voice"] = [(30, "what has the cat been doing lately")]
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        assert not history_qa.is_history_question("what has she been doing")
        assert history_qa.is_history_question(voice._with_names("what has she been doing", owner=True))
        # Anyone else's question is classified as said: a pronoun never makes it a history question past the password.
        assert voice._with_names("what has she been doing", owner=False) == "what has she been doing"
