"""Facts about people (memory/facts.py) on the memory log: distilled from hour entries the facts pass follows under
its durable name, consolidated per person, one record per hour behind a fact, folded with their support when read."""

import json
import logging
import types

import pytest

from liberational import config as cfg
from liberational import memory_log, voice_turn
from liberational.domain import kernel
from liberational.memory import facts, journal, recall
from liberational.people import registry
from voices import write_voices

T0 = 1793000000.0 - 1793000000.0 % 3600  # an hour boundary
ASK = facts._ask  # the real model call, before `env` replaces it


@pytest.fixture
def env(tmp_path, monkeypatch, memory):
    state = tmp_path / "state"
    write_voices(state, {3: "Dana"})
    e = types.SimpleNamespace(state=state, calls=[], replies=[], memory=memory)

    def ask(system, user, schema, **options):
        e.calls.append(dict(system=system, data=json.loads(user)))
        return e.replies.pop(0)

    monkeypatch.setattr(facts, "_ask", ask)
    monkeypatch.setattr(journal, "model_version", lambda model: "sha256:abc")
    reg = registry.at(state)
    reg.ensure("Alice", "test")
    e.alice, e.dana = reg.known_as("Alice"), reg.known_as("Dana")
    with cfg.override(
        {
            "owner": {"name": "Alice", "speaker_label": ""},
            "models": {"ollama": {"llm": "test-llm"}},
            "journal": {"enabled": True, "facts": True, "levels": [30, 180, 600, 3600], "model": ""},
        }
    ):
        yield e


def hour(env, start, text, speakers=("3",), faces=(), supersedes=None, transcript_revision=0):
    """A 1 hour journal entry at start, as the journal writes one. Returns its id."""
    body = {
        "ts_start": start,
        "ts_end": start + 3600,
        "level": 3600,
        "text": text,
        "speakers": list(speakers),
        "faces": list(faces),
        "supersedes": supersedes,
        "transcript_revision": transcript_revision,
    }
    return env.memory.append(memory_log.JOURNAL, body, parents=["01a18be2-c1e9-740f-83c3-c4ebf8ea9999"])


def forget_heard_in(env, hour_id):
    """Someone heard in the hour is forgotten: the broker purges the hour and every fact citing it."""
    env.memory.purge([hour_id], "test")


def rebuild(env, hour_id, text):
    """The journal rewrites the hour: a new entry superseding the old one."""
    old = next(r for r in env.memory.range(memory_log.JOURNAL, 0, 1e12) if r["id"] == hour_id)
    return hour(env, old["ts_start"], text, supersedes=hour_id)


def held(env, pid):
    return facts.held(env.memory, pid, registry.at(env.state), state=env.state)


def held_facts(env):
    return [r["fact"] for r in held(env, env.dana)]


def every_fact(env):
    """Every fact record written about Dana and not purged, superseded ones too."""
    return env.memory.range(memory_log.FACTS, 0, 1e12, subjects=[env.dana])


def found(*pairs):
    """An extraction reply: (kind, fact) pairs about P1, the first person it was given."""
    return {"facts": [{"person": "P1", "kind": kind, "fact": fact} for kind, fact in pairs]}


def decided(*triples):
    """A consolidation reply: (new id, action, held id) triples."""
    return {"decisions": [{"new": new, "action": action, "existing": old} for new, action, old in triples]}


def acme(env):
    """Dana's hour at T0, and a pass that reads one fact from it. Returns the hour's id."""
    h = hour(env, T0, "[S3] manages Acme.")
    env.replies.append(found(("role", "Dana manages the Acme account.")))
    facts.facts_pass(env.state)
    return h


def test_people_in_an_hour_are_resolved_to_person_ids(env):
    reg = registry.at(env.state)
    got = facts.people_in({"ts_start": T0, "speakers": ["3", "99"], "faces": ["Dana", "Nobody"]}, reg)
    assert got == {env.dana: "Dana", env.alice: "Alice"}  # the owner is always there; unknowns are left out


def test_rejected_transcript_hides_existing_facts_and_rebuilt_hour_teaches_again(env):
    from liberational.listen import revisions

    old = acme(env)
    assert held_facts(env) == ["Dana manages the Acme account."]
    revisions.reject_lines(env.state, [{"device": "test", "ts_start": T0 + 1, "ts_end": T0 + 2}])
    assert held_facts(env) == []
    stamp = revisions.review_revision(env.state, T0, T0 + 3600)
    hour(env, T0, "[S3] manages a different account.", supersedes=old, transcript_revision=stamp)
    env.replies.append(found(("role", "Dana manages a different account.")))
    facts.facts_pass(env.state)
    assert held_facts(env) == ["Dana manages a different account."]
    assert every_fact(env)[0]["transcript_revision"] == stamp


def test_hour_rejected_while_fact_extraction_runs_does_not_publish_usable_fact(env, monkeypatch):
    from liberational.listen import revisions

    hour(env, T0, "[S3] manages Acme.")

    def rejected_during_extract(*args, **kwargs):
        revisions.reject_lines(env.state, [{"device": "test", "ts_start": T0 + 1, "ts_end": T0 + 2}])
        return found(("role", "Dana manages Acme."))

    monkeypatch.setattr(facts, "_ask", rejected_during_extract)
    facts.facts_pass(env.state)
    assert held_facts(env) == []


@pytest.mark.parametrize("junk", ["--3", "²", "3.0", ""], ids=["double-minus", "superscript", "decimal", "empty"])
def test_people_in_skips_a_speaker_id_that_is_not_a_whole_number(env, junk):
    reg = registry.at(env.state)
    got = facts.people_in({"ts_start": T0, "speakers": [junk, "3"]}, reg)
    assert env.dana in got  # the good id is still read, and the bad one doesn't stop the pass


def test_a_pass_extracts_facts_with_their_hours_as_parents(env):
    h = hour(env, T0, "[S3] said she manages the Acme account and never drinks coffee.")
    reply = found(
        ("role", "Dana manages the Acme account."),
        ("preference", "Dana doesn't drink coffee."),
        ("mood", "Dana seemed tired."),  # not a kind it may use
    )
    reply["facts"].append({"person": "P9", "kind": "role", "fact": "Someone not listed."})  # not a ref it was given
    env.replies.append(reply)
    assert facts.facts_pass(env.state) == {"hours": 1, "facts": 2}
    call = env.calls[0]["data"]
    assert call["hour"] == "Dana said she manages the Acme account and never drinks coffee."  # tokens rendered
    assert {p["name"] for p in call["people"]} == {"Dana", "Alice"}
    assert {p["name"] for p in call["people"] if "note" in p} == {"Alice"}  # only the owner is marked
    got = held(env, env.dana)
    assert [r["fact"] for r in got] == ["Dana manages the Acme account.", "Dana doesn't drink coffee."]
    assert got[0]["parents"] == [h] and got[0]["subjects"] == [env.dana]
    assert {k: got[0][k] for k in ("fact", "kind", "person", "support")} == {
        "fact": "Dana manages the Acme account.",
        "kind": "role",
        "person": env.dana,
        "support": 1,
    }
    assert (got[0]["ts_start"], got[0]["ts_end"]) == (T0, T0 + 3600)
    assert got[0]["producer"]["prompt_id"] == "person.fact.extract.v1"

    env.calls.clear()
    assert facts.facts_pass(env.state) == {"hours": 0, "facts": 0}  # an hour is read once: the broker keeps the place
    assert env.calls == []


def test_consolidation_adds_repeats_and_replaces(env):
    first = hour(env, T0, "[S3] manages Acme and walks the dog.")
    env.replies.append(
        found(("role", "Dana manages the Acme account."), ("habit", "Dana walks the dog every evening."))
    )
    facts.facts_pass(env.state)
    second = hour(env, T0 + 3600, "[S3] moved to Globex, still walks the dog, and plays chess.")
    env.replies += [
        found(
            ("role", "Dana works at Globex now."),
            ("habit", "Dana walks her dog in the evenings."),
            ("habit", "Dana plays chess."),
        ),
        decided(("N1", "replace", "F1"), ("N2", "same", "F2"), ("N3", "add", None)),
    ]
    assert facts.facts_pass(env.state) == {"hours": 1, "facts": 3}
    assert [f["fact"] for f in env.calls[-1]["data"]["held"]] == [
        "Dana manages the Acme account.",
        "Dana walks the dog every evening.",
    ]
    now = {r["fact"]: r for r in held(env, env.dana)}
    assert set(now) == {"Dana works at Globex now.", "Dana walks the dog every evening.", "Dana plays chess."}
    walk = now["Dana walks the dog every evening."]
    assert walk["support"] == 2  # the same fact, seen in a second hour
    assert set(walk["parents"]) == {first, second}
    assert now["Dana works at Globex now."]["parents"] == [second]
    superseded = {r["fact"] for r in every_fact(env)} - set(now)
    assert superseded == {"Dana manages the Acme account."}  # the dog fact's second record has the held words
    assert len(every_fact(env)) == 5  # two from the first hour, and three written by the second
    assert walk["records"] == [
        first_dog := walk["id"],
        *[r["id"] for r in every_fact(env) if r.get("same") == first_dog],
    ]
    assert next(r["fact"] for r in facts.about("dana", env.state)) == "Dana walks the dog every evening."


def test_bad_consolidation_decisions_fall_back_to_add(env):
    acme(env)
    hour(env, T0 + 3600, "[S3] plays chess and runs.")
    env.replies += [
        found(("habit", "Dana plays chess."), ("habit", "Dana runs.")),
        decided(("N1", "same", "F7")),  # no such held fact; N2 not decided
    ]
    facts.facts_pass(env.state)
    assert set(held_facts(env)) == {"Dana manages the Acme account.", "Dana plays chess.", "Dana runs."}


def test_a_rebuilt_hour_takes_its_facts_with_it_and_is_read_again(env):
    old = acme(env)
    rebuild(env, old, "[S3] manages Globex.")  # a relabel rewrote the hour: the new entry supersedes the old
    env.replies.append(found(("role", "Dana manages the Globex account.")))
    assert facts.facts_pass(env.state) == {"hours": 1, "facts": 1}
    assert held_facts(env) == ["Dana manages the Globex account."]  # the fact citing only the old hour was purged


def test_a_pass_reads_nothing_while_facts_are_off_or_without_an_hour_level(env):
    hour(env, T0, "[S3] manages Acme.")
    with cfg.override({"journal": {"facts": False}}):
        assert facts.facts_pass(env.state) == {"skipped": "off"}
        assert "facts" not in recall.tools()
    with cfg.override({"journal": {"levels": [30, 180, 600]}}):
        assert list(facts.facts_pass(env.state)) == ["skipped"]
    assert env.calls == []


def test_the_search_loop_reads_facts_about_a_person(env):
    acme(env)
    assert "facts" in recall.tools()
    with pytest.raises(ValueError, match="facts needs a person"):
        recall.arguments("facts", {}, (0.0, T0 + 7200))
    args, _ = recall.arguments("facts", {"person": "Dana"}, (0.0, T0 + 7200))
    items, note = recall.run("facts", args, "what does Dana do?", env.state, [])
    assert note is None
    [item] = items
    assert (item["source"], item["quote"]) == ("facts", "Dana manages the Acme account.")
    assert "drew from 1 hour of the journal" in item["method"]
    assert recall.run("facts", dict(args, person="Nobody"), "q", env.state, [])[0] == []


def test_the_cli_prints_what_is_held_about_someone(env, capsys):
    acme(env)
    journal.main(["--facts", "Dana", "--state-dir", str(env.state)])
    assert capsys.readouterr().out == "role         Dana manages the Acme account.  (from 1 hour)\n"


def by_content(monkeypatch, said, decide):
    """A fake model that answers from what it's shown: said maps a word in the hour to a fact about Dana (P1), and
    decide(held, new) gives the consolidation decisions. Returns the list of requests it gets."""
    calls = []

    def ask(system, user, schema, **options):
        data = json.loads(user)
        calls.append(data)
        if "hour" in data:
            return found(*(fact for word, fact in said.items() if word in data["hour"]))
        return {"decisions": decide(data["held"], data["new"])}

    monkeypatch.setattr(facts, "_ask", ask)
    return calls


def same_by_text(held_, new):
    """Consolidation decisions: a new fact with a held fact's exact text is the same, anything else is added."""
    names = {h["fact"]: h["id"] for h in held_}
    return [
        {"new": n["id"], "action": "same", "existing": names[n["fact"]]}
        if n["fact"] in names
        else {"new": n["id"], "action": "add", "existing": None}
        for n in new
    ]


def test_an_older_hour_never_replaces_a_newer_fact(env, monkeypatch):
    said = {"Acme": ("role", "Dana works at Acme."), "Globex": ("role", "Dana works at Globex.")}
    by_content(
        monkeypatch, said, lambda held, new: [{"new": n["id"], "action": "replace", "existing": "F1"} for n in new]
    )
    first = hour(env, T0, "[S3] works at Acme.")
    hour(env, T0 + 3600, "[S3] moved to Globex.")
    facts.facts_pass(env.state)  # a backlog: both hours unread at once
    assert held_facts(env) == ["Dana works at Globex."]

    rebuild(env, first, "[S3] still works at Acme.")  # an older hour, rebuilt and read after the newer one
    facts.facts_pass(env.state)
    assert held_facts(env) == ["Dana works at Globex."]
    assert facts.facts_pass(env.state) == {"hours": 0, "facts": 0}  # read once


def test_a_fact_keeps_the_hours_that_still_support_it_when_one_is_rebuilt(env, monkeypatch):
    said = {"dog": ("habit", "Dana walks the dog."), "chess": ("habit", "Dana plays chess.")}
    by_content(monkeypatch, said, same_by_text)
    a = hour(env, T0, "[S3] walked the dog.")
    b = hour(env, T0 + 3600, "[S3] walked the dog again.")
    facts.facts_pass(env.state)
    [dog] = held(env, env.dana)
    assert dog["support"] == 2 and set(dog["parents"]) == {a, b}

    rebuild(env, a, "[S3] played chess.")  # hour a is rebuilt and no longer mentions the dog
    facts.facts_pass(env.state)
    now = {r["fact"]: r for r in held(env, env.dana)}
    assert set(now) == {"Dana plays chess.", "Dana walks the dog."}  # b still supports the dog
    assert (now["Dana walks the dog."]["parents"], now["Dana walks the dog."]["support"]) == ([b], 1)

    facts.facts_pass(env.state)  # nothing changed: nothing re-read, nothing inflated
    assert {r["support"] for r in held(env, env.dana)} == {1}


def test_a_purged_hour_takes_only_its_own_records_and_the_rebuilt_hour_adds_to_what_stands(env, monkeypatch):
    """A fact from hours a and b is one record per hour, so purging a leaves it standing on b, and a's
    rewritten window restates it instead of starting it over."""
    by_content(
        monkeypatch, {"dog": ("habit", "Dana walks the dog."), "tea": ("preference", "Dana likes tea.")}, same_by_text
    )
    a = hour(env, T0, "[S3] walked the dog and had tea with someone.")
    b = hour(env, T0 + 3600, "[S3] walked the dog again.")
    facts.facts_pass(env.state)
    before = {r["fact"]: r for r in held(env, env.dana)}
    assert {k: r["support"] for k, r in before.items()} == {"Dana walks the dog.": 2, "Dana likes tea.": 1}
    name = before["Dana walks the dog."]["id"]

    forget_heard_in(env, a)  # someone in hour a is forgotten: the broker takes the hour and the records citing it
    assert {r["fact"]: (r["support"], r["parents"]) for r in held(env, env.dana)} == {"Dana walks the dog.": (1, [b])}
    assert held(env, env.dana)[0]["id"] == name  # still the same fact, though its first record went with hour a
    again = hour(env, T0, "[S3] walked the dog and had tea.")  # the journal writes the window again without that person
    facts.facts_pass(env.state)
    now = {r["fact"]: r for r in held(env, env.dana)}
    assert {k: (r["support"], r["parents"]) for k, r in now.items()} == {
        "Dana walks the dog.": (2, [again, b]),
        "Dana likes tea.": (1, [again]),
    }
    assert now["Dana walks the dog."]["id"] == name and (
        now["Dana walks the dog."]["ts_start"],
        now["Dana walks the dog."]["ts_end"],
    ) == (T0, T0 + 7200)


def test_a_pass_never_creates_the_event_log(tmp_path, monkeypatch, memory):
    # The event log's existence switches "who was here" to the timeline, so a facts pass must not make an empty one.
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(facts, "_ask", lambda *a, **options: pytest.fail("no model call without an event log"))
    monkeypatch.setattr(registry, "log_exists", lambda state=None: False)
    with cfg.override({"journal": {"enabled": True, "facts": True, "levels": [30, 180, 600, 3600]}}):
        memory.append(memory_log.JOURNAL, {"ts_start": T0, "ts_end": T0 + 3600, "level": 3600, "text": "[S3] x"})
        assert facts.facts_pass(state) == {"skipped": "no event log yet"}
        assert facts.about("Dana", state) == []
    assert memory.cursors == {}  # the hours wait for the log, so they are read once people resolve


def test_a_purge_during_consolidation_keeps_the_held_fact(env, monkeypatch):
    def purge_then_decide(held_, new):  # People: Delete forgets someone heard in the new hour while the model thinks
        forget_heard_in(env, second)
        return same_by_text(held_, new)

    by_content(monkeypatch, {"dog": ("habit", "Dana walks the dog.")}, purge_then_decide)
    a = hour(env, T0, "[S3] walked the dog.")
    facts.facts_pass(env.state)
    second = hour(env, T0 + 3600, "[S3] walked the dog again.")
    facts.facts_pass(env.state)
    facts.facts_pass(env.state)
    [dog] = held(env, env.dana)  # the fact from the first hour, which nobody forgot, still stands
    assert dog["parents"] == [a] or dog["support"] == 1


def test_facts_follow_a_merge(env):
    reg = registry.at(env.state)
    carl = reg.ensure("Carl", "test")
    hour(env, T0, "[S3] runs Acme.")
    env.replies.append(found(("role", "Dana runs Acme.")))
    facts.facts_pass(env.state)
    reg.merge("Dana", "Carl", "test")  # Dana was Carl all along
    assert [r["fact"] for r in facts.about("Carl", env.state)] == ["Dana runs Acme."]
    assert [r["fact"] for r in facts.about("Dana", env.state)] == ["Dana runs Acme."]

    hour(env, T0 + 3600, "[S3] still runs Acme.")  # voice 3 is Carl's now
    env.replies += [found(("role", "Carl runs Acme.")), decided(("N1", "same", "F1"))]
    facts.facts_pass(env.state)
    assert env.calls[-1]["data"]["held"] == [{"id": "F1", "fact": "Dana runs Acme."}]  # consolidation sees it
    [fact] = facts.about("Carl", env.state)
    assert (fact["subjects"], fact["support"]) == ([carl], 2)


def test_a_face_is_whoever_had_its_name_when_it_was_seen(env, monkeypatch):
    reg = registry.at(env.state)
    monkeypatch.setattr(kernel, "now", lambda: T0 - 600)
    bea = reg.ensure("Bea", "test")
    monkeypatch.setattr(kernel, "now", lambda: T0 + 1800)
    reg.rename("Bea", "Bee", "test")  # half an hour into T0's hour, Bea is renamed Bee
    new = reg.ensure("Bea", "test")  # and someone new is called Bea

    def people(start, *seen):  # seen: when each sighting was; none, as in an entry written before faces_at
        payload = {"ts_start": start, "speakers": [], "faces": ["Bea"]}
        if seen:
            payload["faces_at"] = [["Bea", at] for at in seen]
        return facts.people_in(payload, reg)

    assert people(T0) == {bea: "Bee", env.alice: "Alice"}  # read as of the hour's start
    assert people(T0 + 3600) == {new: "Bea", env.alice: "Alice"}
    assert people(T0, T0 + 2000) == {new: "Bea", env.alice: "Alice"}  # seen after the name changed hands
    assert people(T0, T0 + 900, T0 + 2000) == {bea: "Bee", new: "Bea", env.alice: "Alice"}  # each sighting alone
    assert reg.known_as("Dana", T0) == env.dana  # registered after that hour (a backfill): the name's first holder

    reg.forget("Bee", "test")  # facts from her hours never land on the new Bea
    assert people(T0) == {env.alice: "Alice"}


def test_a_pass_reads_a_few_hours_at_a_time_and_stops_for_voice(env, monkeypatch):
    calls = by_content(monkeypatch, {"dog": ("habit", "Dana walks the dog.")}, same_by_text)
    for i in range(4):
        hour(env, T0 + i * 3600, f"[S3] walked the dog, hour {i}.")
    assert facts.facts_pass(env.state, per_pass=2)["hours"] == 2
    busy = types.SimpleNamespace(now=True)
    assert facts.facts_pass(env.state, busy=lambda: busy.now)["hours"] == 0  # voice first: nothing read
    busy.now = False
    assert facts.facts_pass(env.state, busy=lambda: busy.now)["hours"] == 2
    assert [d["hour"] for d in calls if "hour" in d] == [f"Dana walked the dog, hour {i}." for i in range(4)]
    [dog] = held(env, env.dana)
    assert dog["support"] == 4


def test_retiring_a_facts_last_support_takes_the_fact(env, monkeypatch):
    """A fact must not come back in an earlier form once nothing supports it."""
    by_content(monkeypatch, {"dog": ("habit", "Dana walks the dog.")}, same_by_text)
    a = hour(env, T0, "[S3] walked the dog.")
    b = hour(env, T0 + 3600, "[S3] walked the dog again.")
    facts.facts_pass(env.state)
    assert [r["support"] for r in held(env, env.dana)] == [2]
    rebuild(env, a, "[S3] read.")  # a no longer supports it: trimmed to b
    facts.facts_pass(env.state)
    assert [(r["support"], r["parents"]) for r in held(env, env.dana)] == [(1, [b])]
    rebuild(env, b, "[S3] napped.")  # b neither: the fact and every version before it go
    facts.facts_pass(env.state)
    assert held(env, env.dana) == []
    assert every_fact(env) == []  # every record of it cited a rebuilt hour


def test_a_replaced_fact_stays_replaced_while_any_record_of_it_remains_and_comes_back_when_none_does(env, monkeypatch):
    def decide(held_, new):
        return [{"new": n["id"], "action": "replace", "existing": h["id"]} for n, h in zip(new, held_)]

    by_content(
        monkeypatch, {"Acme": ("role", "Dana works at Acme."), "Globex": ("role", "Dana works at Globex.")}, decide
    )
    hour(env, T0, "[S3] works at Acme.")
    facts.facts_pass(env.state)
    moved = hour(env, T0 + 3600, "[S3] moved to Globex.")
    facts.facts_pass(env.state)
    assert held_facts(env) == ["Dana works at Globex."]
    rebuild(env, moved, "[S3] napped.")  # the hour that said so is rebuilt and no longer says it
    facts.facts_pass(env.state)
    assert held_facts(env) == ["Dana works at Acme."]  # the fact it replaced stands again on its own hour


class Ollama:
    """Ollama's chat endpoint, faked: `reply(system, data, options)` gives the message content (a string, which may be
    broken JSON) and its done_reason."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.lines: list[tuple[int, str]] = []  # (level, line) for each facts log line

    def answer(self, sent):
        body = sent.payload
        system, user = (m["content"] for m in body["messages"])
        self.calls.append(dict(system=system, data=json.loads(user), options=body["options"]))
        content, reason = self.reply(system, json.loads(user), body["options"])
        return {"message": {"content": content}, "done_reason": reason}


@pytest.fixture
def ollama(env, monkeypatch, model_server):
    """The real `_ask` against a faked Ollama; `ollama.reply` decides each reply."""
    fake = Ollama(lambda system, data, options: pytest.fail("no reply set"))
    monkeypatch.setattr(facts, "_ask", ASK)
    model_server.reply = fake.answer
    monkeypatch.setattr(voice_turn, "wait_quiet", lambda: None)
    monkeypatch.setattr(cfg, "ollama_url", lambda: "http://127.0.0.1:11434")

    class Lines(logging.Handler):
        def emit(self, record):
            fake.lines.append((record.levelno, record.getMessage()))

    handler = Lines()
    facts.LOG.addHandler(handler)
    yield fake
    facts.LOG.removeHandler(handler)


def test_an_hour_whose_reply_stays_broken_is_skipped_and_the_next_hour_is_read(env, ollama):
    """A reply cut off mid-string (JSONDecodeError) left the hour unacknowledged, so every pass tried it again
    and the facts pass sat on it for hours. A bad reply is retried a few times, then the hour is skipped with a log
    line naming it, and the pass goes on."""
    broken = hour(env, T0, "[S3] talked for a long while.")
    good = hour(env, T0 + 3600, "[S3] manages Acme.")
    cut_off = '{"facts": [{"person": "P1", "kind": "role", "fact": "Dana'

    def reply(system, data, options):
        if data.get("hour") == "Dana talked for a long while.":
            return cut_off, "length"
        return json.dumps(found(("role", "Dana manages the Acme account."))), "stop"

    ollama.reply = reply
    out = facts.facts_pass(env.state)
    assert "error" not in out
    assert out["hours"] == 1 and out["facts"] == 1
    assert out["unreadable"] == [broken]
    assert held_facts(env) == ["Dana manages the Acme account."]
    tries = [c for c in ollama.calls if c["data"].get("hour") == "Dana talked for a long while."]
    assert len(tries) == facts.EXTRACT_TRIES > 1
    [line] = [ln for lv, ln in ollama.lines if lv == facts.logs.ERROR]
    assert broken in line and "Unterminated string" in line and "length" in line
    assert len([ln for lv, ln in ollama.lines if lv == facts.logs.WARN and broken in ln]) == facts.EXTRACT_TRIES - 1
    assert good not in " ".join(ln for _, ln in ollama.lines)
    ollama.calls.clear()
    assert facts.facts_pass(env.state) == {"hours": 0, "facts": 0}  # the skipped hour was acknowledged
    assert ollama.calls == []


def test_a_reply_cut_off_at_the_token_limit_is_asked_again_with_more_room(env, ollama):
    hour(env, T0, "[S3] manages Acme.")
    budgets = []

    def reply(system, data, options):
        budgets.append(options["num_predict"])
        if len(budgets) == 1:
            return '{"facts": [{"person": "P1", "kind": "ro', "length"
        return json.dumps(found(("role", "Dana manages the Acme account."))), "stop"

    ollama.reply = reply
    assert facts.facts_pass(env.state) == {"hours": 1, "facts": 1}
    assert budgets[1] > budgets[0]
    assert held_facts(env) == ["Dana manages the Acme account."]
