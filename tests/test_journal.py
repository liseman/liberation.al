"""The journal worker (memory/journal.py) on a real broker: entries, rollups, quiet markers, reconcile, forgetting,
and embeddings on the memory log."""

import hashlib
import json
import re
import types
from unittest.mock import create_autospec

import pytest
from hypothesis import given
from hypothesis import strategies as st

from liberational import config as cfg
from liberational import continuum_link, memory_log, segment_log, speech_log, vision_log
from liberational.domain.identity import Basis
from liberational.memory import journal
from liberational.people import registry
from voices import write_voices

DAY = "2026-03-02"
T0 = journal.day_bounds(DAY)[0] + 10 * 3600  # 10:00 local, on a day with no clock change
NOW = T0 + 2 * 3600  # long after every window here is final


def test_render_uses_current_names_and_merges(tmp_path):
    write_voices(tmp_path, {3: "Dana", 5: None}, forwards={7: 3})
    text = "[S3] asked about the venue. [S7] agreed. [S5] laughed. [S9] left."
    assert (
        journal.render(text, tmp_path) == "Dana asked about the venue. Dana agreed. Speaker 5 laughed. Speaker 9 left."
    )


def test_render_without_speakers_file(tmp_path):
    assert journal.render("[S2] waved", tmp_path) == "Speaker 2 waved"


def test_speaker_ids_resolves_a_name_through_merges(tmp_path):
    write_voices(tmp_path, {3: "Dana", 4: "Alice"}, forwards={7: 3, 8: 7})
    assert journal.speaker_ids("dana", tmp_path) == {"3", "7", "8"}
    assert journal.speaker_ids("Bob", tmp_path) == set()
    assert journal.speaker_ids("", tmp_path) == set()


def test_render_and_speaker_ids_follow_person_of(tmp_path):
    write_voices(tmp_path, {3: None, 4: "Eve"})
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(3), "Dana", Basis.OWNER_TAG, "test")  # named after the voices were written
    assert journal.render("[S3] met [S4].", tmp_path) == "Dana met Eve."
    assert journal.speaker_ids("dana", tmp_path) == {"3"}
    reg.rename("Dana", "Dana Cole", "test")
    assert journal.render("[S3] waved", tmp_path) == "Dana Cole waved"


# ---------------------------------------------------------------- the pass ----
def default_reply(system, user):
    """Stands in for the model: names every speaker token and plain speaker name it was given, so tests can see what
    reached it."""
    data = json.loads(user.split("\n\nYour last")[0])
    names = {i["speaker"] for i in data.get("items", []) if i.get("type") == "speech"} - {"Alice", "someone"}
    who = " and ".join(sorted(set(re.findall(r"\[S\d+\]", json.dumps(data))) | names)) or "nobody"
    return json.dumps(
        {
            "text": f"Alice talked with {who}." if "items" in data else f"Summary: Alice met {who}.",
            "source_ids": [r["id"] for r in data.get("sources", [])],
        }
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """The journal on the test's broker: real speech, vision, and memory logs on it (the suite's in-memory fakes
    would give entries parents the broker doesn't have), the model stubbed."""
    state = tmp_path / "state"
    state.mkdir()
    socket = continuum_link.socket_path()  # starts the test's broker
    speech = speech_log.SpeechLog(socket, client_name="test-speech", segments=state / "segments")
    vision = vision_log.VisionLog(socket, client_name="test-vision", state=state)
    memory = memory_log.MemoryLog(socket, state=state, client_name="test-memory")
    monkeypatch.setattr(speech_log, "shared", lambda: speech)
    monkeypatch.setattr(vision_log, "at", lambda state=None: vision)
    # Captions are read from the real segment store under the test's state, not the suite's fake.
    monkeypatch.setattr(segment_log, "at", lambda state=None: segment_log.Log(segment_log.root(state)))
    monkeypatch.setattr(memory_log, "at", lambda state=None: memory)
    write_voices(state, {1: "Alice", 3: "Dana", 5: None})
    e = types.SimpleNamespace(
        state=state, calls=[], replies=[], slept=[], before=None, up=True, speech=speech, vision=vision, memory=memory
    )

    e.chat_spec = journal.chat

    def chat(system, user, model, num_predict, *, structured=False, max_chars=None, ids=(), response_schema=None):
        e.calls.append(
            dict(
                system=system,
                user=user,
                data=json.loads(user.split("\n\nYour last")[0]),
                model=model,
                num_predict=num_predict,
                max_chars=max_chars,
                ids=list(ids),
            )
        )
        if e.before:
            e.before(e.calls[-1])  # a test's hook: raise, or do something while the call is in flight
        reply = e.replies.pop(0) if e.replies else None
        if reply is None:
            return default_reply(system, user)
        return reply

    monkeypatch.setattr(journal, "chat", create_autospec(journal.chat, side_effect=chat))
    monkeypatch.setattr(journal, "model_version", lambda model: "sha256:abc")
    monkeypatch.setattr(journal, "ollama_up", lambda: e.up)
    monkeypatch.setattr(journal, "_sleep", lambda s: e.slept.append(s))
    monkeypatch.setattr(journal, "_now", lambda: NOW)
    monkeypatch.setattr(journal, "known_faces", lambda: None)  # no face gallery: keep every face
    journal._TRIES.clear()
    journal._GAVE_UP.clear()
    with cfg.override(
        {
            "owner": {"name": "Alice", "speaker_label": ""},
            "assistant": {"name": "Bob"},
            "models": {"ollama": {"llm": "test-llm"}},
            "journal": {
                "enabled": True,
                "levels": [30, 180, 600, 3600],
                "lag_s": 120,
                "reconcile_days": 2,
                "model": "",
            },
            "semantic": {"enabled": False},
        }
    ):
        yield e
    for log in (speech, vision, memory):
        log.close()


def say(env, ts, text, sid=None, label=None, device="mic"):
    line = {"ts_start": round(ts, 3), "ts_end": round(ts + 2, 3), "text": text, "speaker_id": sid, "device": device}
    line["speaker"] = label if label is not None else {1: "Alice", 3: "Dana", 4: "Eve"}.get(sid, f"Speaker {sid}")
    return env.speech.append(speech_log.UTTERANCES, line)


def bob(env, ts, text):
    return env.speech.append(speech_log.REPLIES, {"ts_start": ts, "ts_end": ts + 2, "text": text, "speaker": "Bob"})


def seen(env, ts, cam, caption, objects=(), status="fresh", unchanged=None):
    """A caption in the test's segment store, as the caption worker writes one: no faces, which the face
    worker's records hold."""
    record = {
        "id": segment_log.new_id(),
        "ts_start": ts,
        "cam": cam,
        "caption": caption,
        "objects": [{"label": o, "bbox": [0, 0, 1, 1]} for o in objects],
        "observation": {"version": 1, "status": status},
        "unchanged": unchanged,
    }
    writers = segment_log.Writers(segment_log.root(env.state))
    writers.append(segment_log.CAPTIONS, cam, [record])
    writers.close()
    return record["id"]


def face(env, ts, cam, name, cluster=None):
    """A face see-actions matched on a tracked head, in the test's segment store (`face.match`)."""
    record = {
        "id": segment_log.new_id(),
        "ts_start": ts,
        "track_id": "7",
        "frame_id": segment_log.new_id(),
        "bbox": [0.1, 0.1, 0.2, 0.2],
        "det": 0.9,
        "name": name,
        "sim": 0.7,
        "cluster": cluster,
        "quality": 0.8,
    }
    writers = segment_log.Writers(segment_log.root(env.state))
    writers.append(segment_log.FACE_MATCHES, cam, [record])
    writers.close()
    return record["id"]


def records(env, level=30, quiet=False):
    """The current entries at `level`, in time order (quiet markers too when asked)."""
    return env.memory.entries(T0 - 86400, NOW + 86400, level=level, include_quiet=quiet)


def entry_calls(env):
    return [c for c in env.calls if "items" in c["data"]]


def purge(env, *ids):
    env.memory.link.call(lambda c: c.purge(ids=list(ids), reason="test"))


def test_builds_an_entry_with_parents_subjects_payload_and_producer(env):
    one = say(env, T0 + 1, "Let's book the venue for Friday.", sid=1)
    two = say(env, T0 + 4, "I can call them tomorrow.", sid=3)
    seen(env, T0 + 5, "cam0", "Two people talking at a table.", objects=["mug"])
    seen(env, T0 + 6, "cam1", "Two people talking at a table")
    seen(env, T0 + 7, "cam2", "Two people talking at a table.")
    bob(env, T0 + 20, "Noted.")

    result = journal.run_pass(env.state, now=NOW)

    assert result == {"built": 4, "calls": 4, "superseded": 0}  # the entry, then one rollup per longer level
    data = entry_calls(env)[0]["data"]
    speech = [(i["speaker"], i["text"]) for i in data["items"] if i["type"] == "speech"]
    assert speech == [("Alice", "Let's book the venue for Friday."), ("[S3]", "I can call them tomorrow.")]
    cams = [i for i in data["items"] if i["type"] == "camera"]
    assert len(cams) == 1  # the same caption from two cameras a second apart is one scene
    assert cams[0]["cameras"] == ["cam0", "cam1", "cam2"]
    assert cams[0]["objects"] == ["mug"]
    assert [i["text"] for i in data["items"] if i["type"] == "assistant"] == ["Noted."]
    assert env.calls[0]["model"] == "test-llm"

    [r] = records(env)
    assert (r["ts_start"], r["ts_end"], r["level"]) == (T0, T0 + 30, 30)
    assert r["text"] == "Alice talked with [S3]."
    assert r["search"] == "Alice talked with Dana."  # the text a search index files, with names at write time
    assert r["speakers"] == ["1", "3"]
    assert r["line_speakers"][two] == {"sid": "3", "me": False, "label_hash": journal.label_hash("Dana")}
    assert r["line_speakers"][one]["me"] is True
    assert r["parents"] == []  # what it was written from is in the segment store, which the broker can't cite
    assert set(r["line_speakers"]) == {one, two}  # its lines, by id
    reg = registry.at(env.state)
    assert set(r["subjects"]) == {reg.person_id("Dana"), reg.person_id("Alice")}
    assert r["producer"] == {
        "producer_id": "journal",
        "version_hash": "",
        "model": "test-llm",
        "version": "sha256:abc",
        "prompt_id": "journal.entry.30.v3",
        "prompt_hash": hashlib.sha256(env.calls[0]["system"].encode()).hexdigest()[:16],
    }
    # Every other window of the day so far is a quiet marker, so the catalog says the day is done up to the lag.
    quiet = [q for q in records(env, quiet=True) if q.get("status") == "quiet"]
    assert quiet and all(q.get("text") is None and q["parents"] == [] for q in quiet)
    assert quiet[0]["ts_start"] == journal.day_bounds(DAY)[0]


def test_each_call_may_cite_only_its_windows_own_sources_and_has_room_for_the_longest_reply(env):
    say(env, T0 + 1, "Let's book the venue for Friday.", sid=1)
    say(env, T0 + 4, "I can call them tomorrow.", sid=3)

    journal.run_pass(env.state, now=NOW)

    call = entry_calls(env)[0]
    assert call["ids"] == [s["id"] for s in call["data"]["sources"]] and len(call["ids"]) == 2
    longest = call["max_chars"] // 2 + len(call["ids"]) * journal.ID_TOKENS + journal.REPLY_OVERHEAD_TOKENS
    assert call["num_predict"] >= longest


@pytest.fixture
def sent(monkeypatch):
    """What chat asks Ollama for."""
    out = {}

    def llm(system, user, **kwargs):
        out.update(kwargs)
        return '{"text": "ok", "source_ids": []}', "m"

    monkeypatch.setattr(journal.summarize, "llm", llm)
    return out


def test_the_structured_schema_bounds_the_text_and_the_cited_ids(sent):
    journal.chat("s", "u", "m", 500, structured=True, max_chars=480, ids=["a", "b", "a"])
    properties = sent["response_format"]["properties"]
    assert properties["text"]["maxLength"] == 480
    assert properties["source_ids"]["items"]["enum"] == ["a", "b"]
    assert properties["source_ids"]["maxItems"] == 2


def test_a_window_with_more_sources_than_an_entry_may_cite_caps_them(sent):
    ids = [f"id{i}" for i in range(journal.MAX_SOURCE_IDS + 5)]
    journal.chat("s", "u", "m", 500, structured=True, max_chars=480, ids=ids)
    assert sent["response_format"]["properties"]["source_ids"]["maxItems"] == journal.MAX_SOURCE_IDS


def test_skips_quiet_windows_and_unchanged_scenes(env):
    for t in range(0, 90, 3):
        seen(env, T0 + t, "cam0", "An empty desk with a lamp.")
    seen(env, T0 + 95, "cam0", "A person sits down at the desk.")
    seen(env, T0 + 125, "cam0", "Someone stands up from the desk.")
    seen(env, T0 + 160, "cam0", "Nothing", status="not_observed")  # gate-skipped rows aren't observations
    seen(env, T0 + 190, "cam0", "A person sits down at the desk.", unchanged=True)  # carried forward, not seen

    journal.run_pass(env.state, now=NOW)

    assert [r["ts_start"] - T0 for r in records(env)] == [0, 90, 120]
    assert len(entry_calls(env)) == 3


def test_scenes_merges_a_near_copy_and_keeps_dissimilar_captions_apart():
    rows = [
        dict(ts=T0, cam="cam0", caption="A person sits at the desk.", faces=[], objects=[]),
        dict(ts=T0 + 3, cam="cam1", caption="A person sat at the desk.", faces=[], objects=[]),
        dict(ts=T0 + 6, cam="cam0", caption="An empty desk with a lamp.", faces=[], objects=[]),
        dict(ts=T0 + 9, cam="cam0", caption="The window shows a grey afternoon sky.", faces=[], objects=[]),
    ]
    kept = journal.scenes(rows)
    assert [k["caption"] for k in kept] == [
        "A person sits at the desk.",
        "An empty desk with a lamp.",
        "The window shows a grey afternoon sky.",
    ]
    assert kept[0]["cams"] == ["cam0", "cam1"]


@pytest.mark.parametrize(
    "a, b",
    [
        ("", "a person sits at the desk"),
        ("a person sits at the desk", "a person sits at the desk"),
        ("a person sits at the desk", "a person sat at the desk"),
        ("a person sits at the desk", "a person sits at the desk with a cup"),
        ("an empty desk with a lamp", "a person sits at the desk"),
        ("the window shows a grey afternoon sky", "a cat on the sofa"),
    ],
)
def test_the_caption_prefilter_keeps_exactly_the_pairs_difflib_would(a, b):
    """The multiset bound only skips pairs ratio() could not accept, so it changes no merge."""
    matcher = journal.difflib.SequenceMatcher(None, a, b, autojunk=False)
    exact = a == b or (matcher.quick_ratio() >= journal.SAME and matcher.ratio() >= journal.SAME)
    assert journal._alike(a, b) is exact


def test_owner_is_named_and_unnamed_speakers(env):
    say(env, T0 + 1, "Morning.", sid=1)
    say(env, T0 + 2, "Hi there.", label="Alice")  # no voice id, but the owner's label
    say(env, T0 + 3, "Who left this?", label="unknown")
    say(env, T0 + 4, "Me.", label="Eve")
    say(env, T0 + 5, "Hello.", sid=5)  # a voice with no name yet is still a token

    journal.run_pass(env.state, now=NOW)

    data = entry_calls(env)[0]["data"]
    assert [i["speaker"] for i in data["items"]] == ["Alice", "Alice", "someone", "Eve", "[S5]"]
    assert records(env)[0]["speakers"] == ["1", "5"]


@pytest.mark.parametrize("level", [30, 180, 600, 3600])
def test_generation_names_configured_owner_in_third_person(env, level):
    with cfg.override({"owner": {"name": "Sam"}}):
        p = journal.Pass(env.state, NOW)
        try:
            p.write(level, {"start": "10:00", "end": "10:30", "items": [], "sources": []}, set())
        finally:
            p.close()
    assert 'The owner is "Sam"' in env.calls[0]["system"]
    assert "third person, past tense" in env.calls[0]["system"]


@pytest.mark.parametrize(
    "text",
    [
        "I sat.",
        "Dana spoke to me.",
        "My cup fell.",
        "It was mine.",
        "We left.",
        "Our meal ended.",
        "I’m here.",
        "The wearer sat.",
    ],
)
def test_narration_rejects_first_person(text):
    assert not journal.narration_clean(text)


@pytest.mark.parametrize("text", ["Alice sat.", "Someone left.", "[S3] handed Sam a cup.", "Alice discussed the US."])
def test_narration_accepts_third_person(text):
    assert journal.narration_clean(text)


@pytest.mark.parametrize(
    "text",
    [
        'Alice said, "I will bring my book."',
        "Dana said 'I'm bringing my book.' Alice agreed.",
        "“We finished our work,” said Sam.",
        "Alice replied, ‘I’ll bring mine.’",
    ],
)
def test_narration_accepts_quoted_first_person(text):
    assert journal.narration_clean(text)


@pytest.mark.parametrize(
    "text",
    [
        '"I brought my book."',
        "'I'm bringing my book.'",
        'Alice said "I brought a book." Then I left.',
        "Alice said 'I brought a book.' My bag was empty.",
        'Alice said "I brought a book.',
    ],
)
def test_quotes_do_not_hide_first_person_narration(text):
    assert not journal.narration_clean(text)


def test_generation_keeps_quoted_speech_at_the_end_of_an_entry(env):
    env.replies[:] = [json.dumps({"text": 'Alice said "I will bring my book."', "source_ids": []})]
    p = journal.Pass(env.state, NOW)
    try:
        text, _ = p.write(30, {"start": "10:00", "end": "10:00:30", "items": [], "sources": []}, set())
    finally:
        p.close()
    assert text == 'Alice said "I will bring my book."'
    assert len(env.calls) == 1


def test_a_cut_quote_is_retried_instead_of_published_as_first_person(env):
    env.replies[:] = [
        json.dumps({"text": text, "source_ids": []})
        for text in (
            'Alice said "' + "I read my book. " * 80 + '"',
            "Alice discussed her book.",
        )
    ]
    p = journal.Pass(env.state, NOW)
    try:
        text, _ = p.write(30, {"start": "10:00", "end": "10:00:30", "items": [], "sources": []}, set())
    finally:
        p.close()
    assert text == "Alice discussed her book."
    assert len(env.calls) == 2


@pytest.mark.parametrize("second, succeeds", [("Alice sat.", True), ("We sat.", False)])
def test_first_person_gets_one_retry_and_is_never_stored(env, second, succeeds, caplog):
    env.replies[:] = [json.dumps({"text": t, "source_ids": []}) for t in ("I sat.", second)]
    p = journal.Pass(env.state, NOW)
    payload = {"start": "10:00", "end": "10:00:30", "items": [], "sources": []}
    try:
        if succeeds:
            assert p.write(30, payload, set())[0] == "Alice sat."
        else:
            with pytest.raises(journal.EmptyReply, match="first-person"):
                p.write(30, payload, set())
    finally:
        p.close()
    assert len(env.calls) == 2
    assert "entirely in third person" in env.calls[1]["user"]
    assert "journal narration rejected" in caplog.text


def test_rewrite_preserves_original_evidence_and_resumes_newest_first(env, monkeypatch):
    evidence = {
        "status": "verified",
        "sources": [{"id": "source-1", "camera": "cam0", "ts_start": T0, "ts_end": T0 + 5}],
    }
    old = env.memory.append(
        memory_log.JOURNAL, {"ts_start": T0, "ts_end": T0 + 30, "level": 30, "text": "I sat.", "evidence": evidence}
    )
    newer = env.memory.append(
        memory_log.JOURNAL,
        {"ts_start": T0 + 60, "ts_end": T0 + 90, "level": 30, "text": "I left.", "evidence": evidence},
    )
    corrected = create_autospec(env.chat_spec, side_effect=['{"text":"Alice left."}', '{"text":"Alice sat."}'])
    monkeypatch.setattr(journal, "chat", corrected)
    result = journal.rewrite_third_person(env.state, now=NOW, days=1)
    assert result["superseded"] == 2
    assert json.loads(corrected.call_args_list[0].args[1]) == {"text": "I left."}
    rows = env.memory.entries(T0, NOW)
    assert [r["supersedes"] for r in rows] == [old, newer]
    assert [r["text"] for r in rows] == ["Alice sat.", "Alice left."]
    assert [r["evidence"] for r in rows] == [evidence, evidence]
    assert len(env.memory.range(memory_log.JOURNAL, T0, NOW)) == 4
    assert journal.rewrite_third_person(env.state, now=NOW)["would_supersede"] == 0
    assert corrected.call_count == 2


def test_rewrite_failure_keeps_original_and_recording_links(env, monkeypatch):
    original = {"ts_start": T0, "ts_end": T0 + 30, "level": 30, "text": "I sat [clip](#t=123)."}
    old = env.memory.append(memory_log.JOURNAL, original)
    monkeypatch.setattr(
        journal, "chat", create_autospec(env.chat_spec, return_value='{"text":"Alice sat [clip](#t=999)."}')
    )
    assert journal.rewrite_third_person(env.state, now=NOW)["failed"] == 1
    assert [(r["id"], r["text"]) for r in env.memory.entries(T0, NOW)] == [(old, original["text"])]


def test_owner_only_rewrite_uses_schema_and_corrects_narrator_drift(env, monkeypatch):
    model = create_autospec(
        journal.summarize.llm,
        side_effect=[
            ('{"text":"I read my book."}', None),
            ('{"text":"Alice read her book."}', None),
        ],
    )
    monkeypatch.setattr(journal, "chat", env.chat_spec)
    monkeypatch.setattr(journal.summarize, "llm", model)
    assert journal.rewrite_prose({"text": "I read my book."}) == {"text": "Alice read her book."}
    assert model.call_count == 2
    first, second = model.call_args_list
    assert first.kwargs["response_format"]["required"] == ["text"]
    assert first.kwargs["response_format"]["additionalProperties"] is False
    assert first.kwargs["temperature"] == 0
    assert "I sat becomes Alice sat" in first.args[0]
    assert "naming the owner Alice" in second.args[0]


def test_long_rollup_waits_for_converted_children_and_keeps_the_original(env, monkeypatch):
    evidence = {
        "status": "verified",
        "sources": [
            {
                "id": "source-1",
                "stream": speech_log.UTTERANCES,
                "ts_start": T0,
                "ts_end": T0 + 5,
                "text": "A book was discussed.",
                "people": [],
            }
        ],
    }
    child = env.memory.append(
        memory_log.JOURNAL, {"ts_start": T0, "ts_end": T0 + 30, "level": 30, "text": "I read.", "evidence": evidence}
    )
    old = env.memory.append(
        memory_log.JOURNAL,
        {"ts_start": T0, "ts_end": T0 + 180, "level": 180, "text": "I read a book. " * 200, "evidence": evidence},
        parents=[child],
    )
    model = create_autospec(
        env.chat_spec,
        side_effect=[
            '{"text":"Alice read."}',
            '{"text":"Alice read a book.","source_ids":["source-1"]}',
        ],
    )
    monkeypatch.setattr(journal, "chat", model)
    result = journal.rewrite_third_person(env.state, now=NOW)
    assert (result["superseded"], result["regenerated"], result["failed"]) == (2, 1, 0)
    [rewritten] = env.memory.entries(T0, NOW, level=30)
    [rebuilt] = env.memory.entries(T0, NOW, level=180)
    assert rebuilt["supersedes"] == old
    assert rebuilt["parents"] == [rewritten["id"]]
    assert rebuilt["text"] == "Alice read a book."
    assert rebuilt["evidence"] == evidence
    assert len(env.memory.range(memory_log.JOURNAL, T0, NOW)) == 4
    assert json.loads(model.call_args_list[1].args[1])["entries"][0]["text"] == "Alice read."
    assert journal.rewrite_third_person(env.state, now=NOW)["would_supersede"] == 0


def test_failed_rewrite_regenerates_rollup_from_current_children(env, monkeypatch):
    old_child = env.memory.append(
        memory_log.JOURNAL, {"ts_start": T0, "ts_end": T0 + 30, "level": 30, "text": "I read."}
    )
    current_child = env.memory.append(
        memory_log.JOURNAL,
        {"ts_start": T0, "ts_end": T0 + 30, "level": 30, "text": "Alice read.", "supersedes": old_child},
    )
    old = env.memory.append(
        memory_log.JOURNAL, {"ts_start": T0, "ts_end": T0 + 180, "level": 180, "text": "I read."}, parents=[old_child]
    )
    monkeypatch.setattr(
        journal,
        "chat",
        create_autospec(
            env.chat_spec,
            side_effect=[
                "",
                "",
                '{"text":"Alice read.","source_ids":[]}',
            ],
        ),
    )
    assert journal.rewrite_third_person(env.state, now=NOW)["regenerated"] == 1
    [rebuilt] = env.memory.entries(T0, NOW, level=180)
    assert (rebuilt["supersedes"], rebuilt["parents"]) == (old, [current_child])


def test_regeneration_never_uses_uncorrected_children(env, monkeypatch):
    child = env.memory.append(memory_log.JOURNAL, {"ts_start": T0, "ts_end": T0 + 30, "level": 30, "text": "I read."})
    old = env.memory.append(
        memory_log.JOURNAL, {"ts_start": T0, "ts_end": T0 + 180, "level": 180, "text": "I read."}, parents=[child]
    )
    monkeypatch.setattr(journal, "chat", create_autospec(env.chat_spec, return_value='{"text":"I read."}'))
    result = journal.rewrite_third_person(env.state, now=NOW)
    assert (result["superseded"], result["regenerated"], result["failed"]) == (0, 0, 2)
    assert {r["id"] for r in env.memory.entries(T0, NOW)} == {child, old}


def test_deferred_rollups_regenerate_from_fine_to_coarse_in_one_run(env, monkeypatch):
    child = env.memory.append(
        memory_log.JOURNAL, {"ts_start": T0, "ts_end": T0 + 30, "level": 30, "text": "Alice read."}
    )
    body = {"ts_start": T0, "text": "I read a book. " * 200}
    three = env.memory.append(memory_log.JOURNAL, {**body, "ts_end": T0 + 180, "level": 180}, parents=[child])
    ten = env.memory.append(memory_log.JOURNAL, {**body, "ts_end": T0 + 600, "level": 600}, parents=[three])
    hour = env.memory.append(memory_log.JOURNAL, {**body, "ts_end": T0 + 3600, "level": 3600}, parents=[ten])
    monkeypatch.setattr(
        journal, "chat", create_autospec(env.chat_spec, return_value='{"text":"Alice read.","source_ids":[]}')
    )
    result = journal.rewrite_third_person(env.state, now=NOW)
    assert (result["regenerated"], result["failed"]) == (3, 0)
    [new_three] = env.memory.entries(T0, NOW, level=180)
    [new_ten] = env.memory.entries(T0, NOW, level=600)
    [new_hour] = env.memory.entries(T0, NOW, level=3600)
    assert (new_three["supersedes"], new_three["parents"]) == (three, [child])
    assert (new_ten["supersedes"], new_ten["parents"]) == (ten, [new_three["id"]])
    assert (new_hour["supersedes"], new_hour["parents"]) == (hour, [new_ten["id"]])


def test_rewrite_page_keeps_sections_links_and_parent_identifiers(env, monkeypatch):
    parent = env.memory.append(
        memory_log.JOURNAL, {"ts_start": T0, "ts_end": T0 + 30, "level": 30, "text": "Alice sat."}
    )
    start, end = journal.day_bounds(DAY)
    old = env.memory.append(
        memory_log.PAGES,
        {
            "ts_start": start,
            "ts_end": end,
            "level": "day",
            "period": DAY,
            "md": "I sat [h1](#t=123).",
            "sections": {"in_short": "I sat."},
        },
        parents=[parent],
    )
    monkeypatch.setattr(
        journal,
        "chat",
        create_autospec(
            env.chat_spec,
            return_value=json.dumps(
                {
                    "md": "Alice sat [h1](#t=123).",
                    "sections": {"in_short": "Alice sat."},
                }
            ),
        ),
    )
    assert journal.rewrite_third_person(env.state, now=NOW)["superseded"] == 1
    row = env.memory.page("day", DAY)
    assert row is not None
    assert (row["supersedes"], row["parents"], row["sections"], row["md"]) == (
        old,
        [parent],
        {"in_short": "Alice sat."},
        "Alice sat [h1](#t=123).",
    )
    assert journal.rewrite_third_person(env.state, now=NOW)["would_supersede"] == 0


def test_rewrite_dry_run_is_private_bounded_and_writes_nothing(env, monkeypatch):
    env.memory.append(
        memory_log.JOURNAL, {"ts_start": T0, "ts_end": T0 + 30, "level": 30, "text": "I discussed a private topic."}
    )
    env.memory.append(
        memory_log.JOURNAL, {"ts_start": T0 - 3 * 86400, "ts_end": T0 - 3 * 86400 + 30, "level": 30, "text": "I left."}
    )
    monkeypatch.setattr(
        journal,
        "chat",
        create_autospec(
            env.chat_spec,
            side_effect=[
                '{"text":"Alice discussed a private topic."}',
                '{"text":"Alice made tea."}',
                '{"text":"Dana handed Alice a book."}',
                '{"text":"Alice and Dana discussed their shopping list."}',
            ],
        ),
    )
    result = journal.rewrite_third_person(env.state, now=NOW, days=1, dry_run=True)
    assert result["would_supersede"] == 1
    assert result["by_level"] == {"journal.entry:30": 1}
    assert result["estimated_payload_growth_bytes"] > 0
    assert len(result["invented_examples"]) == 3
    assert "private topic" not in json.dumps(result)
    assert len(env.memory.range(memory_log.JOURNAL, 0, NOW)) == 2


def test_speaker_ids_sort_numbers_first_and_never_raise():
    assert journal._ids({"10", "3", "x", "2"}) == ["2", "3", "10", "x"]


@given(st.text(max_size=60))
def test_cleaning_a_reply_twice_changes_nothing(text):
    once = journal._clean(text)
    assert journal._clean(once) == once


@pytest.mark.parametrize(
    "reply, stored",
    [
        ('"I had coffee."', "I had coffee."),
        ("```\nI had coffee.\n```", "I had coffee."),
        ("**I had coffee.**", "I had coffee."),
        ("<think>hmm</think>I had coffee.", "I had coffee."),
    ],
)
def test_a_reply_is_stored_without_quotes_or_markdown(reply, stored):
    assert journal._clean(reply) == stored


def test_playback_and_bot_lines_never_reach_the_model(env):
    say(env, T0 + 4, "Sure thing.", label="Bob")
    say(env, T0 + 10, "Echo of the speaker.", sid=3)
    (env.state / "playback_windows.json").write_text(json.dumps([{"id": "x", "start": T0 + 9, "end": T0 + 13}]))

    journal.run_pass(env.state, now=NOW)

    assert env.calls == []
    assert records(env) == []


def test_lines_that_mention_passwords_or_codes_reach_the_model(env):
    """The journal hides nothing based on what a line says."""
    say(env, T0 + 1, "My password is hunter2.", sid=3)
    seen(env, T0 + 31, "cam0", "A screen shows the passcode 9911.")
    bob(env, T0 + 62, "Your api key is set.")

    journal.run_pass(env.state, now=NOW)

    sent = json.dumps(env.calls)
    assert "hunter2" in sent and "passcode 9911" in sent and "api key is set" in sent


def test_unknown_tokens_get_one_retry_then_become_someone(env):
    say(env, T0 + 1, "Hi.", sid=3)
    say(env, T0 + 31, "Hi again.", sid=3)
    env.replies[:] = [
        json.dumps({"text": t, "source_ids": []})
        for t in ["Alice greeted [S9].", "Alice greeted [S3].", "Alice saw [S9] and [S3].", "Alice saw [S9] and [S3]."]
    ]

    journal.run_pass(env.state, now=NOW)

    first, second = records(env)
    assert first["text"] == "Alice greeted [S3]."  # the retry fixed it
    assert second["text"] == "Alice saw someone and [S3]."  # still wrong after the retry
    assert len(entry_calls(env)) == 4
    assert "[S9]" in entry_calls(env)[1]["user"] and "[S3]" in entry_calls(env)[1]["user"].split("Your last")[1]


def test_rollups_cite_their_children_and_union_people(env):
    say(env, T0 + 1, "Coffee?", sid=3)
    seen(env, T0 + 2, "cam0", "Dana holds a mug.")
    say(env, T0 + 61, "Sure.", sid=5)
    seen(env, T0 + 62, "cam0", "Someone waves at the door.")

    journal.run_pass(env.state, now=NOW)

    kids = records(env, 30)
    [three] = records(env, 180)
    [ten] = records(env, 600)
    [hour] = records(env, 3600)
    assert (three["ts_start"], ten["ts_start"], hour["ts_start"]) == (
        T0 // 180 * 180,
        T0 // 600 * 600,
        T0 // 3600 * 3600,
    )
    assert three["parents"] == [k["id"] for k in kids]
    assert three["speakers"] == ["3", "5"]
    assert three["text"] == "Summary: Alice met [S3] and [S5]."
    assert ten["parents"] == [three["id"]]
    assert hour["speakers"] == ["3", "5"]
    assert set(hour["subjects"]) == set(kids[0]["subjects"]) | set(kids[1]["subjects"])
    rollup = next(c for c in env.calls if "entries" in c["data"])
    assert [e["text"] for e in rollup["data"]["entries"]] == [k["text"] for k in kids]
    assert hour["producer"]["prompt_id"] == "journal.entry.3600.v3"


def test_rollups_wait_for_final_children(env):
    say(env, T0 + 1, "Coffee?", sid=3)
    journal.run_pass(env.state, now=T0 + 150)  # the 30 s window is final, its 3 minute window isn't

    assert len(records(env, 30)) == 1 and records(env, 180) == []
    journal.run_pass(env.state, now=T0 + 180 + 121)
    assert len(records(env, 180)) == 1 and records(env, 600) == []


def test_a_second_pass_with_nothing_new_makes_no_calls_and_the_catalog_is_the_cursor(env):
    say(env, T0 + 1, "Coffee?", sid=3)
    journal.run_pass(env.state, now=NOW)
    n = len(env.calls)
    assert journal.run_pass(env.state, now=NOW + 60) == {"built": 0, "calls": 0, "superseded": 0}
    assert len(env.calls) == n
    done = max(q["ts_end"] for q in records(env, quiet=True))
    assert done == NOW + 60 - 120  # NOW + 60 less the 120 s lag: every window before it has an entry or a marker


def test_a_relabeled_line_gets_its_window_rebuilt_and_the_rollup_above_it(env):
    line = say(env, T0 + 1, "Coffee?", sid=3)
    say(env, T0 + 31, "Tea.", sid=1)
    journal.run_pass(env.state, now=NOW)
    [old] = [r for r in records(env) if r["ts_start"] == T0]
    [old_hour] = records(env, 3600)
    # The owner says Eve, not Dana's voice, said the first line: a label on the utterance in the event log.
    registry.at(env.state).claim(registry.utterance(T0 + 1, "mic"), "Eve", Basis.OWNER_TAG, "dashboard")
    env.calls.clear()

    result = journal.run_pass(env.state, now=NOW + 60)

    assert result["superseded"] == 4  # the entry, and the three rollups whose child it was
    [new] = [r for r in records(env) if r["ts_start"] == T0]
    assert new["supersedes"] == old["id"] and new["id"] != old["id"]
    assert [i["speaker"] for i in entry_calls(env)[0]["data"]["items"]] == ["Eve"]
    assert line in new["line_speakers"] and new["speakers"] == []  # written from the line, now nobody's voice
    [three] = records(env, 180)
    [hour] = records(env, 3600)
    assert new["id"] in three["parents"] and hour["supersedes"] == old_hour["id"]
    assert journal.run_pass(env.state, now=NOW + 120) == {"built": 0, "calls": 0, "superseded": 0}


def test_a_line_gone_from_the_speech_log_gets_its_window_written_again_without_it(env, monkeypatch):
    line = say(env, T0 + 1, "Coffee?", sid=3)
    say(env, T0 + 4, "Tea.", sid=1)
    journal.run_pass(env.state, now=NOW)
    [old] = records(env)
    assert len(records(env, 3600)) == 1

    # The speech log forgets the line (People: Delete on the speech side): it's no longer read back.
    kept = env.speech.range
    monkeypatch.setattr(env.speech, "range", lambda *a, **k: [x for x in kept(*a, **k) if x["id"] != line])
    env.calls.clear()
    journal.run_pass(env.state, now=NOW + 60)
    [r] = records(env)
    assert r["supersedes"] == old["id"]
    assert [i["text"] for i in entry_calls(env)[0]["data"]["items"]] == ["Tea."]
    assert line not in r["line_speakers"] and r["speakers"] == ["1"]
    assert len(records(env, 3600)) == 1


@pytest.fixture(params=[False, True], ids=["voice-list", "missing-voice-list"])
def reviewed_voice_list(env, monkeypatch, request):
    if request.param:
        table = journal.speakers(env.state)
        table.voices.clear()
        monkeypatch.setattr(journal, "speakers", lambda state=None: table)


def test_rejecting_every_line_hides_old_journal_and_normal_pass_rebuilds_quiet_then_undo(env, reviewed_voice_list):
    from liberational.listen import revisions

    line = say(env, T0 + 1, "Incorrect words.", sid=3)
    journal.run_pass(env.state, now=NOW)
    raw = env.speech.range(T0, T0 + 30)
    token = revisions.reject_lines(env.state, raw)
    assert env.memory.entries(T0, T0 + 3600) == []
    env.calls.clear()
    journal.run_pass(env.state, now=NOW + 60)
    assert env.memory.entries(T0, T0 + 3600) == []
    assert env.calls == []
    marker = next(r for r in env.memory.entries(T0, T0 + 30, level=30, include_quiet=True) if r["ts_start"] == T0)
    assert marker["status"] == "quiet"
    assert marker["transcript_revision"] == revisions.review_revision(env.state, T0, T0 + 30)
    revisions.undo_correction(env.state, token)
    journal.run_pass(env.state, now=NOW + 120)
    restored = env.memory.entries(T0, T0 + 30, level=30)
    assert len(restored) == 1 and line in restored[0]["line_speakers"]


def test_transcript_rejected_during_model_call_cannot_publish_current_journal_text(env):
    from liberational.listen import revisions

    say(env, T0 + 1, "Incorrect words.", sid=3)
    raw = env.speech.range(T0, T0 + 30)
    env.before = lambda call: revisions.reject_lines(env.state, raw)
    journal.run_pass(env.state, now=NOW)
    assert env.memory.entries(T0, T0 + 3600) == []
    env.before = None
    env.calls.clear()
    journal.run_pass(env.state, now=NOW + 60)
    assert env.memory.entries(T0, T0 + 3600) == []
    assert env.calls == []


def test_old_reviewed_window_rebuilds_outside_the_normal_lookback(env):
    from liberational.listen import revisions

    say(env, T0 + 1, "Incorrect words.", sid=3)
    journal.run_pass(env.state, now=NOW)
    token = revisions.reject_lines(env.state, env.speech.range(T0, T0 + 30))
    later = NOW + 5 * 86400
    journal.run_pass(env.state, now=later)
    marker = next(r for r in env.memory.entries(T0, T0 + 30, level=30, include_quiet=True) if r["ts_start"] == T0)
    assert marker["status"] == "quiet"
    revisions.undo_correction(env.state, token)
    journal.run_pass(env.state, now=later + 60)
    assert len(env.memory.entries(T0, T0 + 30, level=30)) == 1


def test_forget_purges_a_persons_entries_and_the_windows_are_written_again_without_them(env, monkeypatch):
    hers = say(env, T0 + 1, "Coffee?", sid=3)  # her line is all the first window has
    say(env, T0 + 31, "Tea, please.", sid=1)
    seen(env, T0 + 32, "cam1", "Someone at the door.")
    say(env, T0 + 61, "Lunch?", sid=1)
    bob(env, T0 + 62, "Dana said she is out today.")  # a mention of her, which stays
    say(env, T0 + 91, "Nice day.", sid=1)
    journal.run_pass(env.state, now=NOW)
    assert [r["ts_start"] - T0 for r in records(env)] == [0, 30, 60, 90]

    n = journal.forget("Dana", {"3"}, env.state)

    assert n >= 1 + 3  # her window, and the rollups above it
    assert [r["ts_start"] - T0 for r in records(env)] == [30, 60, 90]
    assert records(env, 3600) == []
    assert journal.forget("Nobody", {"42"}, env.state) == 0

    # The speech side forgets her line; the journal writes the windows without her.
    kept = env.speech.range
    monkeypatch.setattr(env.speech, "range", lambda *a, **k: [x for x in kept(*a, **k) if x["id"] != hers])
    write_voices(env.state, {1: "Alice", 5: None})
    env.calls.clear()
    journal.run_pass(env.state, now=NOW + 60)

    assert [r["ts_start"] - T0 for r in records(env)] == [30, 60, 90]  # nothing is left of the first window
    assert len(records(env, 3600)) == 1
    assert not any("[S3]" in json.dumps(r) for r in records(env, quiet=True))


def test_a_failing_window_is_retried_then_skipped(env):
    say(env, T0 + 1, "Coffee?", sid=3)
    say(env, T0 + 31, "Tea.", sid=1)

    def fail(call):
        if "Coffee?" in call["user"]:
            raise RuntimeError("the model fell over")

    env.before = fail
    env.up = False
    assert "unreachable" in journal.run_pass(env.state, now=NOW)["error"]
    assert journal._TRIES == {}  # Ollama being down costs the window nothing
    env.up = True
    for n in range(1, journal.MAX_RETRIES + 1):
        assert f"failure {n}" in journal.run_pass(env.state, now=NOW)["error"]
        assert journal._TRIES == {f"30:{int(T0)}": n} and records(env) == []

    result = journal.run_pass(env.state, now=NOW)  # one failure more: skipped, and the rest goes on

    assert "error" not in result
    assert [r["ts_start"] - T0 for r in records(env)] == [30]
    assert len(records(env, 3600)) == 1  # rollups go ahead with the children there are
    assert journal._TRIES == {}
    assert sum("Coffee?" in c["user"] for c in env.calls) == 2 + journal.MAX_RETRIES
    # Settled in the catalog: a later pass, in this process or after a restart, doesn't open the window again.
    [failed] = [r for r in records(env, quiet=True) if r.get("status") == "failed"]
    assert (failed["ts_start"], failed["level"]) == (T0, 30)
    journal._TRIES.clear()
    assert journal.run_pass(env.state, now=NOW + 60) == {"built": 0, "calls": 0, "superseded": 0}
    assert sum("Coffee?" in c["user"] for c in env.calls) == 2 + journal.MAX_RETRIES
    # `--day` gives it another go.
    env.before = None
    journal.rebuild_day(DAY, env.state, now=NOW + 60)
    assert [r["ts_start"] - T0 for r in records(env)] == [0, 30]


def test_a_rollup_rebuild_that_keeps_failing_is_left_until_its_inputs_change(env):
    # A failed rebuild leaves the old entry standing, so its window stayed stale and was opened again, and
    # failed again, every few passes, forever.
    say(env, T0 + 1, "Coffee?", sid=3)
    say(env, T0 + 31, "Tea.", sid=1)
    journal.run_pass(env.state, now=NOW)
    [old_hour] = records(env, 3600)
    # The owner relabels a line: its window is written again, so the rollups over it, the hour among them, are stale.
    registry.at(env.state).claim(registry.utterance(T0 + 1, "mic"), "Eve", Basis.OWNER_TAG, "dashboard")

    def fail(call):
        if "covering about 1 hour" in call["system"]:
            raise RuntimeError("journal chunk summaries did not fit after eight reductions")

    def hour_calls():
        return sum("covering about 1 hour" in c["system"] for c in env.calls)

    env.before = fail
    for n in range(1, journal.MAX_RETRIES + 1):
        assert f"failure {n}" in journal.run_pass(env.state, now=NOW + 60)["error"]
    assert "error" not in journal.run_pass(env.state, now=NOW + 60)  # given up on: the old hour stands
    tries = hour_calls()
    assert "error" not in journal.run_pass(env.state, now=NOW + 120)
    assert hour_calls() == tries and records(env, 3600) == [old_hour]  # and not opened again
    env.before = None
    registry.at(env.state).claim(registry.utterance(T0 + 31, "mic"), "Fay", Basis.OWNER_TAG, "dashboard")  # new inputs
    journal.run_pass(env.state, now=NOW + 180)
    [hour] = records(env, 3600)
    assert hour["supersedes"] == old_hour["id"]


def test_a_finest_window_left_after_failing_wakes_when_its_lines_are_relabeled_again(env):
    # A 30 s window's own relabel changes neither the entry it supersedes nor the review.
    say(env, T0 + 1, "Coffee?", sid=3)
    journal.run_pass(env.state, now=NOW)
    registry.at(env.state).claim(registry.utterance(T0 + 1, "mic"), "Eve", Basis.OWNER_TAG, "dashboard")

    def fail(call):
        if "Coffee?" in call["user"]:
            raise RuntimeError("the model fell over")

    def tries():
        return sum("Coffee?" in c["user"] for c in env.calls)

    env.before = fail
    for n in range(1, journal.MAX_RETRIES + 1):
        assert f"failure {n}" in journal.run_pass(env.state, now=NOW + 60)["error"]
    assert "error" not in journal.run_pass(env.state, now=NOW + 60)  # given up on
    rested = tries()
    journal.run_pass(env.state, now=NOW + 120)
    assert tries() == rested  # left alone while nothing changes
    registry.at(env.state).claim(registry.utterance(T0 + 1, "mic"), "Fay", Basis.OWNER_TAG, "dashboard")
    journal.run_pass(env.state, now=NOW + 180)
    assert tries() == rested + 1  # its line changed speaker: tried again


def test_an_empty_reply_is_a_failure_too(env):
    say(env, T0 + 1, "Coffee?", sid=3)
    env.replies[:] = [""]
    assert "empty reply" in journal.run_pass(env.state, now=NOW)["error"]
    assert journal._TRIES == {f"30:{int(T0)}": 1}
    assert "error" not in journal.run_pass(env.state, now=NOW)
    assert len(records(env)) == 1 and journal._TRIES == {}


@pytest.mark.parametrize(
    "reply,reason",
    [
        ("A prose answer instead of JSON.", "invalid JSON reply"),
        ('{"text": "An unfinished answer', "invalid JSON reply"),
        ('{"text": 42, "source_ids": []}', "invalid structured reply"),
        ('{"text": "", "source_ids": []}', "empty reply"),
    ],
)
def test_bad_model_replies_report_the_actual_failure_without_logging_their_content(env, reply, reason):
    env.replies[:] = [reply]
    with pytest.raises(journal.EmptyReply, match=reason) as error:
        journal.Pass(env.state, NOW).write(30, context_payload(), {"[S3]"})
    assert reply not in str(error.value)


def test_bad_speaker_correction_reply_keeps_and_sanitizes_the_first_answer(env):
    env.replies[:] = [json.dumps({"text": "[S9999] spoke.", "source_ids": ["source-0"]}), "not JSON"]
    text, evidence = journal.Pass(env.state, NOW).write(30, context_payload(), {"[S3]"})
    assert text == "someone spoke."
    assert evidence["status"] == "verified"
    assert len(env.calls) == 2


def test_yesterdays_last_windows_are_finished_after_midnight(env):
    """The fill started at today's start, so a day's last two minutes (inside the lag at midnight)
    and its final rollups were never written."""
    midnight = journal.day_bounds(DAY)[1]
    say(env, midnight - 25, "Good night.", sid=3)
    journal.run_pass(env.state, now=midnight + 30)  # just after midnight: the last window is still inside the lag
    assert records(env) == []
    journal.run_pass(env.state, now=midnight + 200)
    [entry] = records(env)
    assert entry["ts_start"] == midnight - 30
    assert [r["ts_start"] for r in records(env, 3600)] == [midnight - 3600], "yesterday's last hour is rolled up"
    assert journal.run_pass(env.state, now=midnight + 260) == {"built": 0, "calls": 0, "superseded": 0}


def test_a_line_written_late_lands_in_its_own_window(env):
    say(env, T0 + 1, "Coffee?", sid=1)
    say(env, T0 + 40, "Tea.", sid=1)
    say(env, T0 + 29.5, "Me too.", sid=3)  # appended after a later line, but it belongs to the first window
    journal.run_pass(env.state, now=NOW)
    first, second = records(env)
    assert (first["speakers"], second["speakers"]) == (["1", "3"], ["1"])


def test_stale_playback_window_cannot_starve_journal(env):
    say(env, T0 + 1, "Coffee?", sid=3)
    (env.state / "playback_windows.json").write_text(json.dumps([{"id": "x", "start": NOW - 1, "end": 1e10}]))
    journal.run_pass(env.state, now=NOW)
    assert len(env.calls) > 0
    assert records(env)


def test_rebuild_day_and_show(env, capsys):
    say(env, T0 + 1, "Coffee?", sid=3)
    say(env, T0 + 31, "Tea.", sid=1)
    journal.run_pass(env.state, now=NOW)
    before = {r["id"] for r in records(env)}
    env.calls.clear()

    journal.rebuild_day(DAY, env.state, now=NOW)

    assert {r["supersedes"] for r in records(env)} == before
    assert len(records(env)) == 2 and len(env.calls) == 5  # two entries, three rollups
    out = journal.show(DAY, env.state)
    assert out.splitlines() == [
        "10:00:00-10:00:30  Alice talked with Dana.",
        "10:00:30-10:01:00  Alice talked with nobody.",
    ]
    write_voices(env.state, {3: "Dana Cole"})
    journal.main(["--show", DAY, "--level", "180", "--state-dir", str(env.state)])
    assert capsys.readouterr().out == "10:00:00-10:03:00  Summary: Alice met Dana Cole.\n"
    assert journal.show("2026-03-01", env.state) == "No entries at the 30 second level for 2026-03-01."


@pytest.mark.parametrize(
    "prior,error",
    [(False, journal.EmptyReply("empty model text")), (True, RuntimeError("the model rejected this window"))],
    ids=["new-empty-window", "failed-rebuild-supersedes-old-entry"],
)
def test_day_rebuild_marks_a_bad_window_and_continues_later_windows_and_rollups(env, prior, error):
    say(env, T0 + 1, "Coffee?", sid=3)
    say(env, T0 + 31, "Tea.", sid=1)
    old = journal.Pass(env.state, NOW).build(30, int(T0)) if prior else None

    def fail_first_window(call):
        if "Coffee?" in call["user"]:
            raise error

    env.before = fail_first_window
    result = journal.rebuild_day(DAY, env.state, now=NOW)

    assert result["failed"] == 1
    assert "error" not in result
    assert [r["ts_start"] - T0 for r in records(env)] == [30]
    assert len(records(env, 3600)) == 1
    [failed] = [r for r in records(env, quiet=True) if r.get("status") == "failed"]
    assert (failed["ts_start"], failed["level"], failed.get("supersedes")) == (T0, 30, old)
    assert journal._TRIES == {}


@pytest.mark.parametrize(
    "up,error",
    [
        (False, RuntimeError("connection refused")),
        (True, journal.summarize.ollama_health.Unavailable("model unavailable")),
        (True, RuntimeError("unknown parent while appending")),
    ],
    ids=["ollama-down", "typed-model-unavailable", "input-purged"],
)
def test_day_rebuild_stops_on_shared_infrastructure_failure_without_marking_windows_failed(env, up, error):
    say(env, T0 + 1, "Coffee?", sid=3)
    say(env, T0 + 31, "Tea.", sid=1)
    env.up = up

    def fail(call):
        raise error

    env.before = fail
    result = journal.rebuild_day(DAY, env.state, now=NOW)

    assert "error" in result
    assert result.get("failed", 0) == 0
    assert not any(r.get("status") == "failed" for r in records(env, quiet=True))
    assert records(env) == []
    assert len(env.calls) == 1
    assert journal._TRIES == {}


def test_the_service_writes_nothing_while_switched_off(env, monkeypatch):
    def stop(seconds):
        raise SystemExit(seconds)

    monkeypatch.setattr(journal, "_sleep", stop)
    monkeypatch.setattr(journal, "run_pass", lambda *a, **k: pytest.fail("a pass ran while the journal was off"))
    with cfg.override({"journal": {"enabled": False}}), pytest.raises(SystemExit) as e:
        journal.daemon(env.state)
    assert e.value.code == 3600  # it idles instead of exiting, so the service manager doesn't restart it


def test_entries_are_embedded_once_as_payloads_for_the_vector_worker(env, monkeypatch):
    from liberational.memory import embedding

    say(env, T0 + 1, "Coffee?", sid=3)
    say(env, T0 + 31, "Tea.", sid=1)
    journal.run_pass(env.state, now=NOW)
    asked = []
    admission = []
    monkeypatch.setattr(journal.voice_turn, "wait_quiet", lambda: admission.append("quiet"))

    def embed(texts, model, dims, timeout, query=False):
        assert admission == ["quiet"], "background embedding must wait before submission"
        asked.append(list(texts))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(embedding, "embed", embed)
    monkeypatch.setattr(embedding, "digest", lambda model, timeout=5, fresh=False: ("sha256:feedfacefeedface", None))
    monkeypatch.setattr(embedding, "settings", lambda: ("test-embed", 4))
    with cfg.override({"semantic": {"enabled": True}}):
        first = journal.embed(env.state, now=NOW)
        again = journal.embed(env.state, now=NOW)
    assert (first, again) == ({"embedded": 5}, {"embedded": 0})  # two entries and three rollups, once
    assert sorted(asked[0])[:2] == [
        "Alice talked with Dana.",
        "Alice talked with nobody.",
    ]  # rendered text, names current
    stream = journal.embedding_stream("sha256:feedfacefeedface")
    assert stream == "journal.embedding.feedfacefeed"
    everything = {r["id"] for lv in (30, 180, 600, 3600) for r in records(env, lv)}
    assert env.memory.parents_in(stream, T0 - 1, NOW) == everything
    with cfg.override({"semantic": {"enabled": False}}):
        assert journal.embed(env.state, now=NOW) == {"skipped": "off"}


def test_a_model_repulled_during_an_embedding_pass_stores_nothing_under_its_old_digest(env, monkeypatch):
    # Vectors are kept only when the digest read after the call still matches. Because /api/tags
    # is cached, the read after a batch must be fresh, or new weights' vectors land in the old digest's stream.
    from liberational import ollama_health
    from liberational.memory import embedding

    say(env, T0 + 1, "Coffee?", sid=3)
    journal.run_pass(env.state, now=NOW)
    tags = {"digest": "sha256:aaaaaaaaaaaa1111"}

    class Response:
        def __init__(self, digest: str) -> None:
            self.digest = digest

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"models": [{"name": "test-embed", "model": "test-embed", "digest": self.digest, "size": 1}]}

    def repulled_while_embedding(texts, model, dims, timeout, query=False):
        tags["digest"] = "sha256:bbbbbbbbbbbb2222"
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(ollama_health, "_TAGS_CACHE", None)
    monkeypatch.setattr(ollama_health, "_get", lambda url, timeout: Response(tags["digest"]))
    monkeypatch.setattr(journal.voice_turn, "wait_quiet", lambda: None)
    monkeypatch.setattr(embedding, "embed", repulled_while_embedding)
    monkeypatch.setattr(embedding, "settings", lambda: ("test-embed", 4))
    monkeypatch.setattr(embedding, "_trouble", {"why": None})
    with cfg.override({"semantic": {"enabled": True}, "ollama": {"url": "http://127.0.0.1:11434"}}):
        result = journal.embed(env.state, now=NOW)
    assert result == {"embedded": 0, "skipped": "model changed"}
    old = journal.embedding_stream("sha256:aaaaaaaaaaaa1111")
    assert env.memory.parents_in(old, T0 - 1, NOW) == set()


def test_ollama_is_down_as_soon_as_it_stops_answering_whatever_the_tags_cache_holds(monkeypatch):
    # A failed call while Ollama is gone costs no retry (`ollama_up`), so it can't answer from the tags cached while
    # Ollama was still there (the tags cache keeps them 5 minutes).
    import requests

    from liberational import ollama_health

    class Listed:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"models": [{"name": "test-llm", "model": "test-llm", "digest": "sha256:aaaa", "size": 1}]}

    def gone(url: str, timeout: float) -> None:
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setattr(ollama_health, "_TAGS_CACHE", None)
    with cfg.override({"ollama": {"url": "http://127.0.0.1:11434"}}):
        monkeypatch.setattr(ollama_health, "_get", lambda url, timeout: Listed())
        assert journal._tags() is not None  # answered: the cache now holds its tags
        monkeypatch.setattr(ollama_health, "_get", gone)
        assert journal.ollama_up() is False


def test_who_was_seen_comes_from_face_matches_near_each_cameras_caption(env):
    seen(env, T0 + 2, "cam0", "Someone holds a mug.")
    face(env, T0 + 4, "cam0", "Dana")
    face(env, T0 + 5, "cam1", "Eve")  # another camera, with no caption in the window
    face(env, T0 + 6, "cam0", None, cluster="Face 3")  # nobody's name yet
    seen(env, T0 + 62, "cam0", "The room is empty.")  # the next window: no face near it

    journal.run_pass(env.state, now=NOW)

    first, second = records(env)
    assert first["faces"] == ["Dana"]
    assert [name for name, _ in first["faces_at"]] == ["Dana"]
    assert second["faces"] == []


@pytest.mark.parametrize("selected", ["speech", "face", "object"])
def test_evidence_contains_only_selected_atomic_sources_not_other_people_in_window(env, selected):
    speech = say(env, T0 + 1, "Let's reserve the venue.", sid=3)
    caption = seen(env, T0 + 2, "cam0", "A mug is on the table.")
    matched = face(env, T0 + 3, "cam0", "Eve")
    ids = {"speech": speech, "face": matched, "object": caption}
    env.replies[:] = [json.dumps({"text": "Alice noted the event.", "source_ids": [ids[selected]]})]
    built = journal.Pass(env.state, NOW).entry(T0, T0 + 30)
    assert built is not None
    evidence = built["payload"]["evidence"]
    assert evidence["status"] == "verified"
    assert [s["id"] for s in evidence["sources"]] == [ids[selected]]
    source = evidence["sources"][0]
    assert (
        source["stream"] == {"speech": "speech.utterance", "face": "face.match", "object": "camera.caption"}[selected]
    )
    assert source["camera"] == {"speech": None, "face": "cam0", "object": "cam0"}[selected]
    assert [p["name"] for p in source["people"]] == {"speech": ["Dana"], "face": ["Eve"], "object": []}[selected]


@pytest.mark.parametrize("selected", [None, [], ["made-up-source"], "not-a-list"])
def test_missing_or_unknown_attribution_is_unverified_without_window_fallback(env, selected):
    say(env, T0 + 1, "Let's reserve the venue.", sid=3)
    env.replies[:] = [json.dumps({"text": "Alice planned the event.", "source_ids": selected})]
    built = journal.Pass(env.state, NOW).entry(T0, T0 + 30)
    assert built is not None
    assert built["payload"]["evidence"] == {"status": "unverified", "sources": []}


def test_rollup_selects_atomic_child_sources_and_legacy_children_stay_unverified(env):
    first = say(env, T0 + 1, "Book the venue.", sid=3)
    second = seen(env, T0 + 31, "cam1", "A mug is on the table.")
    writer = journal.Pass(env.state, NOW)
    writer.build(30, int(T0))
    writer.build(30, int(T0 + 30))
    env.replies[:] = [json.dumps({"text": "Alice noticed a mug.", "source_ids": [second]})]
    rollup = writer.rollup(180, 30, T0, T0 + 180)
    assert rollup is not None
    assert rollup["payload"]["evidence"]["status"] == "verified"
    assert [s["id"] for s in rollup["payload"]["evidence"]["sources"]] == [second]
    assert first not in [s["id"] for s in rollup["payload"]["evidence"]["sources"]]
    del writer.present[30][int(T0)]["evidence"]
    env.replies[:] = [json.dumps({"text": "Alice noticed a mug.", "source_ids": [second]})]
    legacy_rollup = writer.rollup(180, 30, T0, T0 + 180)
    assert legacy_rollup is not None
    assert legacy_rollup["payload"]["evidence"]["status"] == "unverified"


def test_sources_removed_by_payload_fitting_cannot_be_selected(env, monkeypatch):
    caption = seen(env, T0 + 1, "cam0", "A mug is on the table.")
    say(env, T0 + 2, "A different subject.", sid=3)
    monkeypatch.setattr(journal, "MAX_INPUT", 500)
    env.replies[:] = [json.dumps({"text": "Alice noticed a mug.", "source_ids": [caption]})]
    built = journal.Pass(env.state, NOW).entry(T0, T0 + 30)
    sent = env.calls[-1]["data"]
    assert caption not in [s["id"] for s in sent["sources"]]
    assert built is not None
    assert built["payload"]["evidence"] == {"status": "unverified", "sources": []}


@pytest.mark.parametrize(
    "stream, category, raw_kind, person_name, names",
    [
        (segment_log.ACTION_SPANS, "action", "drink", "Dana", ["Dana"]),
        (segment_log.ACTION_SPANS, "action", "drink", None, ["Dana"]),
        (segment_log.WALKBY_SPANS, "walkby", "crossing", "Dana", []),
        (segment_log.ZONE_VISITS, "zone", None, "Dana", []),
    ],
)
def test_activity_sources_keep_actual_camera_kind_and_source_id(env, stream, category, raw_kind, person_name, names):
    record = {
        "id": segment_log.new_id(),
        "ts_start": T0 + 2,
        "ts_end": T0 + 5,
        "track_id": "7",
        "subject_type": "person",
        "kind": raw_kind or "zone",
        "person_name": person_name,
        "person": str(registry.at(env.state).person_id("Dana")),
        "zone": "work",
        "zone_name": "Work area",
        "definition": "def1",
        "subject_kind": "person",
        "label": None,
        "dwell_s": 3.0,
        "frames": 30,
        "entry_seen": True,
        "exit_seen": True,
        "ended": "left",
    }
    store = segment_log.Writers(segment_log.root(env.state))
    store.append(stream, "cam2", [record])
    store.close()
    env.replies[:] = [json.dumps({"text": "Alice noted an unverified event.", "source_ids": [record["id"]]})]
    built = journal.Pass(env.state, NOW).entry(T0, T0 + 30)
    assert built is not None
    source = built["payload"]["evidence"]["sources"][0]
    assert (source["id"], source["stream"], source["camera"], source["kind"], source.get("activity_kind")) == (
        record["id"],
        stream,
        "cam2",
        category,
        raw_kind,
    )
    assert (source["ts_start"], source["ts_end"]) == (T0 + 2, T0 + 5)
    assert [p["name"] for p in source["people"]] == names


def test_large_unicode_source_context_is_bounded_and_serializable(env):
    seen(env, T0 + 1, "cam0", "A mug is on the table.")
    say(env, T0 + 2, "界" * 9000, sid=3)
    writer = journal.Pass(env.state, NOW)
    rid = writer.build(30, int(T0))
    [stored] = [r for r in records(env) if r["id"] == rid]
    assert len(json.dumps(env.calls[-1]["data"], ensure_ascii=False).encode("utf-8")) <= journal.MAX_INPUT
    assert len(json.dumps(stored["evidence"], ensure_ascii=False).encode("utf-8")) < 8400
    assert stored["text"]


@pytest.mark.parametrize(
    "stream, bound_to, stored_name, removed",
    [
        (segment_log.FACE_MATCHES, "Dana", "Old name", True),
        (segment_log.ACTION_SPANS, None, "Dana", True),
        (segment_log.ACTION_SPANS, "Alice", "Dana", False),
        (segment_log.FACE_MATCHES, None, "Eve", False),
    ],
)
def test_forget_purges_source_only_identity_and_derived_records(env, stream, bound_to, stored_name, removed):
    reg = registry.at(env.state)
    person = {"id": str(reg.person_id(bound_to)) if bound_to else None, "name": stored_name}
    source = {
        "id": segment_log.new_id(),
        "stream": stream,
        "ts_start": T0 + 1,
        "ts_end": T0 + 2,
        "people": [person],
        "camera": "cam1",
        "kind": None,
        "text": "A selected input",
    }
    finest = env.memory.append(
        memory_log.JOURNAL,
        {"level": 30, "text": "Alice noticed an event.", "evidence": {"status": "verified", "sources": [source]}},
        span_ns=(int(T0) * memory_log.NS, int(T0 + 30) * memory_log.NS),
    )
    other = env.memory.append(
        memory_log.JOURNAL,
        {"level": 30, "text": "An unrelated entry."},
        span_ns=(int(T0 + 30) * memory_log.NS, int(T0 + 60) * memory_log.NS),
    )
    rollup = env.memory.append(
        memory_log.JOURNAL,
        {"level": 180, "text": "Alice noted the day."},
        parents=[finest, other],
        span_ns=(int(T0) * memory_log.NS, int(T0 + 180) * memory_log.NS),
    )
    review = env.memory.append(
        memory_log.REVIEW_SUMMARY,
        {"level": 180, "text": "Noted the day", "status": "ready", "source_hash": "test", "topics": []},
        parents=[rollup],
        span_ns=(int(T0) * memory_log.NS, int(T0 + 180) * memory_log.NS),
    )
    [entry] = [r for r in records(env) if r["id"] == finest]
    assert entry["subjects"] == [] and not entry.get("line_speakers")
    assert journal.forget("Dana", state=env.state) == 3 * removed
    assert (finest not in {r["id"] for r in records(env)}) is removed
    assert other in {r["id"] for r in records(env)}
    assert (rollup not in {r["id"] for r in records(env, 180)}) is removed
    assert (review not in {r["id"] for r in env.memory.range(memory_log.REVIEW_SUMMARY, T0, T0 + 180)}) is removed


@pytest.mark.parametrize(
    "bound, reuse_name, cached",
    [(True, False, False), (False, False, False), (True, True, False), (False, True, False), (True, True, True)],
)
def test_retained_action_source_cannot_restore_a_forgotten_person(env, bound, reuse_name, cached):
    reg = registry.at(env.state)
    old_person = reg.person_id("Dana")
    record = {
        "id": segment_log.new_id(),
        "ts_start": T0 + 2,
        "ts_end": T0 + 5,
        "track_id": "7",
        "subject_type": "person",
        "kind": "drink",
        "person_name": "Dana",
        "person": str(old_person) if bound else None,
    }
    store = segment_log.Writers(segment_log.root(env.state))
    store.append(segment_log.ACTION_SPANS, "cam2", [record])
    store.close()
    writer = journal.Pass(env.state, NOW)
    if cached:
        writer.event_log.between(T0, T0 + 30)
    reg.forget("Dana", "test")
    if reuse_name:
        reg.ensure("Dana", "test")
    assert reg.resolve(old_person) is None
    assert journal.speakers(env.state).forgotten_at("Dana", T0 + 2)
    env.replies[:] = [json.dumps({"text": "Alice noted an unverified drink event.", "source_ids": [record["id"]]})]
    built = writer.entry(T0, T0 + 30)
    assert built is not None
    [source] = built["payload"]["evidence"]["sources"]
    assert source["people"] == []
    assert built["subjects"] == []
    assert "Dana" not in json.dumps(env.calls[-1]["data"])
    assert str(old_person) not in json.dumps(env.calls[-1]["data"])
    assert source["stream"] == segment_log.ACTION_SPANS and source["camera"] == "cam2"


@pytest.mark.parametrize("caption", [False, True])
def test_a_face_only_window_is_not_quiet_even_without_a_changed_caption(env, caption):
    if caption:
        seen(env, T0 - 1, "cam0", "A mug is on a table.")
        seen(env, T0 + 1, "cam0", "A mug is on a table.", unchanged=True)
    matched = face(env, T0 + 3, "cam0", "Dana")
    env.replies[:] = [json.dumps({"text": "Alice saw Dana.", "source_ids": [matched]})]
    writer = journal.Pass(env.state, NOW)
    rid = writer.build(30, int(T0))
    [stored] = [r for r in records(env, quiet=True) if r["id"] == rid]
    assert stored.get("status") != "quiet"
    assert stored["evidence"]["status"] == "verified"
    [source] = stored["evidence"]["sources"]
    assert source["id"] == matched and source["stream"] == segment_log.FACE_MATCHES
    assert source["people"][0]["name"] == "Dana"


@pytest.mark.parametrize("method, current_name", [("rename", "Dee"), ("merge", "Alice")])
def test_retained_action_source_follows_a_living_person_through_rename_or_merge(env, method, current_name):
    reg = registry.at(env.state)
    original = reg.person_id("Dana")
    record = {
        "id": segment_log.new_id(),
        "ts_start": T0 + 2,
        "ts_end": T0 + 5,
        "track_id": "7",
        "subject_type": "person",
        "kind": "drink",
        "person_name": "Dana",
        "person": str(original),
    }
    store = segment_log.Writers(segment_log.root(env.state))
    store.append(segment_log.ACTION_SPANS, "cam2", [record])
    store.close()
    getattr(reg, method)("Dana", current_name, "test")
    env.replies[:] = [json.dumps({"text": "Alice noted an unverified event.", "source_ids": [record["id"]]})]
    built = journal.Pass(env.state, NOW).entry(T0, T0 + 30)
    assert built is not None
    [source] = built["payload"]["evidence"]["sources"]
    assert source["people"] == [{"id": str(reg.person_id(current_name)), "name": current_name}]
    assert built["subjects"] == [str(reg.person_id(current_name))]


@pytest.mark.parametrize("name, forgotten", [(None, False), ("Dana", True)])
def test_unknown_or_forgotten_face_alone_keeps_a_window_quiet(env, name, forgotten):
    face(env, T0 + 3, "cam0", name, cluster="Face 3")
    if forgotten:
        registry.at(env.state).forget("Dana", "test")
    rid = journal.Pass(env.state, NOW).build(30, int(T0))
    [stored] = [r for r in records(env, quiet=True) if r["id"] == rid]
    assert stored["status"] == "quiet"
    assert env.calls == []


def context_payload():
    return {
        "start": "10:00:00",
        "end": "10:00:30",
        "items": [
            {
                "type": "speech",
                "speaker": "[S3]",
                "text": "Keep this complete transcript line.",
                "source_ids": [f"source-{i}" for i in range(8)],
            }
        ],
        "sources": [{"id": f"source-{i}", "text": "evidence " * 100} for i in range(8)],
    }


def test_prompt_budget_trims_evidence_before_4190_token_overflow(env):
    def reject_oversized(call):
        if len(call["data"]["sources"]) == 8:
            raise RuntimeError("request (4190 tokens) exceeds available context size (4096 tokens)")

    env.before = reject_oversized
    text, evidence = journal.Pass(env.state, NOW).write(30, context_payload(), {"[S3]"})
    assert text
    assert len(env.calls) == 1
    assert env.calls[0]["data"]["items"][0]["text"] == "Keep this complete transcript line."
    assert len(env.calls[0]["data"]["sources"]) < 8
    assert evidence["status"] == "verified"
    assert env.calls[0]["data"]["items"][0]["source_ids"] == [r["id"] for r in env.calls[0]["data"]["sources"]]
    assert (
        journal.summarize.prompt_tokens(env.calls[0]["system"], env.calls[0]["user"], "test-llm")
        + env.calls[0]["num_predict"]
        <= 4096
    )


def test_oversized_transcript_is_split_without_discarding_text(env):
    original = "A complete transcript. " * 1000
    payload = context_payload()
    payload["items"][0]["text"] = original
    journal.Pass(env.state, NOW).write(30, payload, {"[S3]"})
    fragments = [item["text"] for call in env.calls for item in call["data"].get("items", [])]
    assert len(fragments) > 1
    assert "".join(fragments) == original
    assert all(
        journal.summarize.prompt_tokens(call["system"], call["user"], "test-llm") + call["num_predict"] <= 4096
        for call in env.calls
    )


def test_a_runaway_summary_is_cut_to_its_limit_so_a_split_window_still_finishes(env):
    # Live 11-06 13:00: Ollama let a summary run to 4,329 characters past its 2,400 maxLength, longer than
    # what it summed up, so splitting never shrank anything and gave up after eight levels.
    payload = context_payload()
    payload["items"][0]["text"] = "A complete transcript. " * 1000
    runaway = json.dumps({"text": "Alice talked at length. " * 400, "source_ids": []})
    env.before = lambda call: env.replies.append(runaway)
    text, _ = journal.Pass(env.state, NOW).write(30, payload, {"[S3]"})
    assert len(text) <= env.calls[-1]["max_chars"] and text.endswith("at length.")


@pytest.mark.parametrize("repeat", [False, True])
def test_context_400_retries_once_with_less_evidence(env, repeat):
    def overflow(call):
        if repeat or len(env.calls) == 1:
            raise journal.summarize.ContextLimit(4190, 4096)

    env.before = overflow
    writer = journal.Pass(env.state, NOW)
    if repeat:
        with pytest.raises(journal.summarize.ContextLimit):
            writer.write(30, context_payload(), {"[S3]"})
    else:
        assert writer.write(30, context_payload(), {"[S3]"})[0]
    assert len(env.calls) == 2
    assert len(env.calls[1]["data"]["sources"]) < len(env.calls[0]["data"]["sources"])
    assert env.calls[1]["data"]["items"][0]["text"] == "Keep this complete transcript line."


def test_oversized_speaker_correction_does_not_overflow_context(env):
    env.replies[:] = [json.dumps({"text": "[S9999] spoke.", "source_ids": ["source-0"]})]
    text, _ = journal.Pass(env.state, NOW).write(30, context_payload(), {f"[S{i}]" for i in range(1000)})
    assert text == "someone spoke."
    assert len(env.calls) == 1


def test_other_model_errors_are_not_retried_as_context_overflows(env):
    def fail(call):
        raise RuntimeError("invalid model option")

    env.before = fail
    with pytest.raises(RuntimeError, match="invalid model option"):
        journal.Pass(env.state, NOW).write(30, context_payload(), {"[S3]"})
    assert len(env.calls) == 1


def test_a_long_phase_says_where_it_is_about_once_a_minute(monkeypatch, capsys):
    # A pass logs its counts only at its end, so a reconcile over a day of corrected transcript looked hung.
    import datetime as dt

    from liberational.memory import history_qa

    monkeypatch.setattr(history_qa, "TZ", dt.UTC)
    t = [0.0]
    progress = journal.Progress("reconciling", 100, clock=lambda: t[0])
    day = dt.datetime(2026, 11, 6, tzinfo=dt.UTC).timestamp()
    for n in range(100):
        t[0] += 2.0  # two seconds a window
        progress.tick(day + 30 * n)
    lines = [ln for ln in capsys.readouterr().err.splitlines() if "reconciling:" in ln]
    # 200 s over 100 windows: once at 60 s, 120 s, and 180 s, never at the end (the pass's own count says that)
    assert len(lines) == 3
    assert "reconciling: 30 of 100 windows done, through 2026-11-06 00:14, about 2 min left" in lines[0]


@pytest.mark.parametrize(("total", "says"), [(100, False), (101, True)])
def test_a_big_phase_says_its_size_as_it_starts(capsys, total, says):
    journal.Progress("reconciling", total, clock=lambda: 0.0)
    err = capsys.readouterr().err
    assert (f"reconciling: {total:,} windows to write" in err) is says
