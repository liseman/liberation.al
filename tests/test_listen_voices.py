"""Voice matching on Continuum (listen/voices.py): matching each line to a voiceprint through the vector worker's
search, filing its embedding, numbering voiceprints, and the dashboard's voice tags against a real broker."""

import asyncio
import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from liberational import segment_log, speech_log
from liberational.domain.identity import Basis
from liberational.listen import revisions, voices
from liberational.people import registry, transcript_view

NS = 1_000_000_000
ALICE = "0199a000-0000-7000-8000-00000000a11c"


def unit(*xs: float) -> np.ndarray:
    v = np.zeros(192, np.float32)  # ReDimNet2 B3 checkpoint dimension, also exercised against the native schema
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


A, B = unit(1, 0, 0), unit(0, 1, 0)


class FakeSearch:
    """Answers with `hits`, or raises a bare TimeoutError, with no text, as Searcher does when the vector worker can't
    answer."""

    def __init__(self, hits: Sequence[Mapping[str, Any]] | None = ()) -> None:
        self.hits = hits

    async def nearest(
        self, model: str, version: str, vector: Sequence[float], k: int = 20, *, timeout: float = 0.5
    ) -> Sequence[Mapping[str, Any]]:
        if self.hits is None:
            raise TimeoutError
        return self.hits


class FakePeople:
    def __init__(self, persons: dict[int, str] | None = None) -> None:
        self.persons = persons or {}

    def person_of(self, n: int) -> str | None:
        return self.persons.get(n)

    def voice_of(self, person: str) -> int | None:
        return min((n for n, p in self.persons.items() if p == person), default=None)


def matcher(
    tmp_path: Path, search: FakeSearch, people: FakePeople | None = None, known: int = 0
) -> tuple[voices.VoiceMatcher, Path]:
    store = tmp_path / "segments"
    prints = voices.Voiceprints(tmp_path / "voiceprints.json", lambda: known)
    m = voices.VoiceMatcher(prints, people or FakePeople(), search, segments=store)
    return m, store


def filed(store: Path) -> list[dict[str, Any]]:
    """Every voice embedding in the store, oldest first."""
    return segment_log.Log(store).range(voices.EMBEDDINGS, 0.0, 2.0**32)


def hit(entity: str, score: float) -> dict[str, Any]:
    return {"payload_id": "p", "entity_id": entity, "parent_id": None, "start_ns": 0, "end_ns": 1, "score": score}


SECOND = (0, 2 * NS)  # a two-second clip
SHORT = (0, NS // 2)


def test_a_new_voice_opens_a_voiceprint_and_its_next_line_matches_it_before_the_search_has_it(tmp_path: Path) -> None:
    m, _ = matcher(tmp_path, FakeSearch([]), known=4)

    async def run() -> list[voices.Match | None]:
        return [await m.match(A, *SECOND), await m.match(A, *SECOND), await m.match(B, *SECOND)]

    first, again, other = asyncio.run(run())
    assert first is not None and again is not None and other is not None
    assert (first.speaker_id, first.speaker) == (5, "Speaker 5"), "above every number the registry knows"
    assert again.speaker_id == 5, "the search hasn't filed it yet, but the matcher remembers it"
    assert other.speaker_id == 6


def test_the_nearest_hit_names_the_voiceprint_it_was_filed_under(tmp_path: Path) -> None:
    near, far = voices.subject(3), voices.subject(2)
    m, _ = matcher(tmp_path, FakeSearch([hit(near, 0.2), hit(far, 0.3)]), known=3)
    got = asyncio.run(m.match(A, *SECOND))
    assert got is not None
    assert (got.speaker, got.speaker_id, got.sim) == ("Speaker 3", 3, pytest.approx(0.8))


@pytest.mark.parametrize(("similarity", "expected"), [(0.44, 4), (0.46, 3)])
def test_anonymous_matching_respects_the_calibrated_floor(tmp_path: Path, similarity: float, expected: int) -> None:
    m, _ = matcher(tmp_path, FakeSearch([hit(voices.subject(3), 1 - similarity)]), known=3)
    got = asyncio.run(m.match(A, *SECOND))
    assert got is not None and got.speaker_id == expected


def test_a_named_index_hit_without_confirmed_references_stays_unnamed(tmp_path: Path) -> None:
    m, _ = matcher(tmp_path, FakeSearch([hit(ALICE, 0.1)]), FakePeople({7: ALICE}), known=7)
    got = asyncio.run(m.match(A, *SECOND))
    assert got is None


def test_when_the_search_cant_answer_the_line_is_unmatched_and_no_voiceprint_opens(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    m, _ = matcher(tmp_path, FakeSearch(None), known=2)
    with caplog.at_level(logging.WARNING, logger=voices.log.name):
        assert asyncio.run(m.match(A, *SECOND)) is None
    assert m.prints.next == 3
    # #3073: the timeout has no text, so the line names its type rather than reading "unavailable ()".
    assert "voice search unavailable (TimeoutError); 1 lines unmatched so far" in caplog.messages


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        pytest.param(0.46, None, id="sim-0.54-isnt-enough-for-a-short-clip"),
        pytest.param(0.44, 1, id="sim-0.56-is"),
    ],
)
def test_a_short_clip_needs_a_closer_match(tmp_path: Path, score: float, expected: int | None) -> None:
    m, _ = matcher(tmp_path, FakeSearch([hit(voices.subject(1), score)]), known=1)
    got = asyncio.run(m.match(A, *SHORT))
    assert (got.speaker_id if got else None) == expected


def test_a_short_clip_never_opens_a_voiceprint(tmp_path: Path) -> None:
    m, _ = matcher(tmp_path, FakeSearch([]), known=1)
    assert asyncio.run(m.match(A, *SHORT)) is None
    assert m.prints.next == 2


def test_an_embedding_is_filed_with_its_voiceprint_its_person_and_its_line(tmp_path: Path) -> None:
    m, store = matcher(tmp_path, FakeSearch([]), FakePeople({3: ALICE}), known=4)
    m.learn(A, voices.Match("Speaker 3", 3, 0.9), "0199a000-0000-7000-8000-000000000001", *SECOND)
    m.learn(B, voices.Match("Speaker 4", 4, 0.9), "0199a000-0000-7000-8000-000000000002", *SECOND)
    m.learn(B, voices.Match("Speaker 4", 4, 0.9), "0199a000-0000-7000-8000-000000000003", *SHORT)  # too short
    m.learn(B, None, "0199a000-0000-7000-8000-000000000004", *SECOND)  # no voiceprint to file it under
    m.close()
    alice, nobody, unmatched = filed(store)
    assert unmatched["voiceprint"] == voices.subject(0), (
        "unmatched observations stay available for explicit line confirmation"
    )
    assert (alice["voiceprint"], alice["person"], alice["utterance"]) == (
        voices.subject(3), ALICE, "0199a000-0000-7000-8000-000000000001",
    )  # fmt: skip
    assert (nobody["voiceprint"], nobody["person"]) == (voices.subject(4), None), "an unnamed voice: no person"
    assert (alice["start_ns"], alice["end_ns"]) == SECOND
    assert np.array_equal(np.asarray(alice["vector"], np.float32), A)
    assert (store / voices.EMBEDDINGS / "listen").is_dir(), "the matcher's own slot"


def test_voiceprint_numbers_continue_after_a_restart_and_stay_above_the_registrys(tmp_path: Path) -> None:
    path = tmp_path / "voiceprints.json"
    first = voices.Voiceprints(path, lambda: 0)
    assert [first.new(), first.new()] == [1, 2]
    assert voices.Voiceprints(path, lambda: 0).new() == 3, "the next number is kept"
    assert voices.Voiceprints(path, lambda: 10).new() == 11, "never one the registry already has"
    assert voices.Voiceprints(path, lambda: 0).number(voices.subject(2)) == 2


def test_two_processes_never_hand_out_the_same_voiceprint(tmp_path: Path) -> None:
    # Listen and the dashboard's enroll (people/voice_edits.py) both hand numbers out, each from its own copy.
    path = tmp_path / "voiceprints.json"
    listen, enroll = voices.Voiceprints(path, lambda: 0), voices.Voiceprints(path, lambda: 0)
    assert [listen.new(), enroll.new(), listen.new()] == [1, 2, 3]
    assert listen.number(voices.subject(2)) == 2, "one the other handed out is known by its subject"


def test_an_enrolled_recording_is_filed_with_no_line_in_the_slot_it_names(tmp_path: Path) -> None:
    m, store = matcher(tmp_path, FakeSearch())
    m.file(A, 4, None, 5 * NS, 5 * NS, slot="enroll")
    m.close()
    ((row,),) = [filed(store)]
    assert (row["utterance"], row["voiceprint"], row["person"]) == (None, voices.subject(4), None)
    assert (store / voices.EMBEDDINGS / "enroll").is_dir()


def test_a_room_line_never_matches_a_voice_clone(tmp_path: Path) -> None:
    m, _ = matcher(tmp_path, FakeSearch([hit(voices.subject(2), 0.05), hit(voices.subject(1), 0.3)]), known=2)
    m.clones = lambda: frozenset({2})
    got = asyncio.run(m.match(A, *SECOND))
    assert got is not None and got.speaker_id == 1


def test_a_clones_embedding_names_no_person_even_when_its_voiceprint_does(tmp_path: Path) -> None:
    m, store = matcher(tmp_path, FakeSearch(), FakePeople({4: ALICE}))
    m.file(A, 4, None, 5 * NS, 5 * NS, own_subject=True)
    m.close()
    ((row,),) = [filed(store)]
    assert (row["voiceprint"], row["person"]) == (voices.subject(4), None)


def test_the_registry_lists_voiceprints_overall_and_a_persons(tmp_path: Path) -> None:
    reg = registry.at(tmp_path)
    reg.claim_many(
        [
            registry.Claim(registry.voiceprint(9), "Alice"),
            registry.Claim(registry.voiceprint(4), "Alice"),
            registry.Claim(registry.voiceprint(5), "Dana"),
        ],
        Basis.OWNER_TAG,
        "test",
    )
    alice = reg.person_id("Alice")
    assert reg.voiceprints() == [4, 5, 9]
    assert reg.voiceprints(alice) == [4, 9]
    assert reg.voiceprints(registry.PersonId("0199a000-0000-7000-8000-0000000000ff")) == [], "nobody's"


# ---- Voice tags, against a real broker and event log


def _queue(state: Path, *requests: dict[str, Any]) -> None:
    with (state / "voice_tags.jsonl").open("a") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")


def _setup(state: Path) -> tuple[voices.VoiceMatcher, dict[str, Any]]:
    """One line (wave mic, 100.0 to 102.0 s) heard as Speaker 1, its embedding filed."""
    prints = voices.Voiceprints(state / "voiceprints.json", lambda: 0)
    m = voices.VoiceMatcher(prints, voices.RegistryPeople(state), FakeSearch([]), segments=segment_log.root(state))
    n = prints.new()
    line = {"ts_start": 100.0, "ts_end": 102.0, "device": "wave", "text": "hello", "speaker": f"Speaker {n}"}
    writer = speech_log.Writer("wave", segment_log.root(state))
    writer.write(speech_log.UTTERANCES, {**line, "speaker_id": n}, span_ns=(100 * NS, 102 * NS))
    writer.close()
    (stored,) = speech_log.lines(99.0, 103.0, base=segment_log.root(state))
    m.file(A, n, stored["id"], 100 * NS, 102 * NS)
    return m, stored


def test_rejected_transcript_cannot_teach_a_voice_and_undo_refreshes_gallery(tmp_path):
    matcher_, line = _setup(tmp_path)
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    reg.claim(registry.utterance_of(line), "Alice", Basis.OWNER_TAG, "human")
    assert voices.confirm_lines(tmp_path, [line]) == 1
    gallery = voices.ReferenceGallery(tmp_path)
    assert len(gallery()) == 1
    token = revisions.reject_lines(tmp_path, [line])
    assert gallery() == []
    assert voices.confirm_lines(tmp_path, [line]) == 0
    revisions.undo_correction(tmp_path, token)
    assert len(gallery()) == 1
    winner = reg.person_of(registry.utterance_of(line)).winner
    assert winner is not None and winner.basis == Basis.OWNER_TAG
    matcher_.close()


def test_a_voice_tag_labels_the_line_and_teaches_the_persons_new_voice(tmp_path: Path) -> None:
    m, line = _setup(tmp_path)
    _queue(tmp_path, {"ts_start": 100.0, "line": registry.key_of(line), "name": "Alice", "requested_at": 1.0})
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    m.close()
    reg = registry.at(tmp_path)
    alice = reg.person_id("Alice")
    assert alice is not None
    assert reg.person_of(registry.utterance_of(line)).person == alice, "the line is hers in the event log"
    assert reg.voiceprints(alice) == [2], "she had no voice: the tag opened one for her"
    rows = filed(segment_log.root(tmp_path))
    taught = [r for r in rows if r["voiceprint"] == voices.subject(2)]
    assert taught[0]["person"] is None, "prepared new-person rows gain identity only through the committed registry"
    assert [(r["voiceprint"], r["utterance"], r["start_ns"], r["end_ns"]) for r in taught] == [
        (voices.subject(2), line["id"], 100 * NS, 102 * NS)
    ], "a copy of the line's own embedding (found by its line), over the line's span"
    assert np.array_equal(np.asarray(taught[0]["vector"], np.float32), A)
    assert len(rows) == 2
    assert (tmp_path / "voice_tags.offset").read_text() == str((tmp_path / "voice_tags.jsonl").stat().st_size)


def test_a_tag_asked_for_before_that_person_was_deleted_is_dropped(tmp_path: Path) -> None:
    m, line = _setup(tmp_path)
    m._remember(A, 1)
    _queue(
        tmp_path,
        {"ts_start": 100.0, "line": registry.key_of(line), "name": "Alice", "requested_at": 1.0},
        {"forget": [1], "name": "Alice"},
        {"ts_start": 5.0, "name": "Dana"},  # no line there: skipped, never retried
    )
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    m.close()
    assert not m.recent, "the forget drops the recent voices"
    assert registry.at(tmp_path).person_id("Alice") is None, "nobody registered under the deleted name"
    assert (tmp_path / "voice_tags.offset").read_text() == str((tmp_path / "voice_tags.jsonl").stat().st_size)


def test_prints_are_each_voiceprints_newest_embeddings_as_unit_vectors(tmp_path: Path) -> None:
    m, store = matcher(tmp_path, FakeSearch(), FakePeople({3: ALICE}))
    for k, v in enumerate([A, B, 2 * A]):
        m.file(v, 3, None, (10 + k) * NS, (11 + k) * NS)
    m.file(B, 4, None, 20 * NS, 21 * NS)
    m.close()
    got = voices.prints([3, 4, 5], limit=2, base=store)
    assert [[round(float(x), 6) for x in v[:2]] for v in got[3]] == [[0.0, 1.0], [1.0, 0.0]], "newest two, unit"
    assert len(got[4]) == 1 and got[5] == []


def test_forgetting_takes_the_embeddings_it_names(tmp_path: Path) -> None:
    m, store = matcher(tmp_path, FakeSearch(), FakePeople({3: ALICE}))
    m.file(A, 3, None, 10 * NS, 11 * NS)  # Alice's voiceprint, under her
    m.file(B, 4, None, 12 * NS, 13 * NS)  # an unnamed one
    m.close()
    voices.forget_voice(3, base=store)
    by_print = {voices.subject(n): n for n in (3, 4)}
    assert {by_print[r["voiceprint"]] for r in filed(store)} == {4}


def test_forgetting_one_embedding_by_id_keeps_the_rest(tmp_path: Path) -> None:
    m, store = matcher(tmp_path, FakeSearch())
    old = m.file(A, 3, None, 10 * NS, 11 * NS)
    new = m.file(B, 3, None, 12 * NS, 13 * NS)
    m.close()
    voices.forget_embedding(old, base=store)
    assert [r["id"] for r in filed(store)] == [new]


def _taught(
    state: Path,
    n: int,
    vector: np.ndarray,
    *,
    model: str = voices.MODEL,
    version: str = voices.VERSION,
    at: int = 5 * NS,
) -> str:
    """A taught voice vector in the knowledge store, as pack-knowledge imports one (a named voiceprint's)."""
    import uuid

    from liberational import knowledge

    rid = str(uuid.uuid7())
    value = {"voiceprint": n, "person": "", "vector": [float(x) for x in vector], "dims": len(vector),
             "model": model, "version": version, "source_id": rid, "source_kind": "voice.embedding"}  # fmt: skip
    record = knowledge.Record(rid, voices.REFERENCES, f"speaker:{n}", at, at, at, value, (voices.subject(n),))
    knowledge.Store(registry.knowledge_path(state)).put([record])
    return rid


def test_after_a_reset_the_taught_vectors_are_filed_under_their_voiceprint_and_person_once(tmp_path: Path) -> None:
    from voices import write_voices

    write_voices(tmp_path, {3: "Dana", 4: None})
    dana = registry.at(tmp_path).person_id("Dana")
    store = segment_log.root(tmp_path)
    kept = _taught(tmp_path, 3, A)
    _taught(tmp_path, 3, B, model="other")  # another model's vector: not this space
    assert voices.seed_taught(tmp_path, voices.RegistryPeople(tmp_path)) == 1
    ((row,),) = [filed(store)]
    assert (row["id"], row["voiceprint"], row["person"]) == (kept, voices.subject(3), str(dana))
    assert np.array_equal(np.asarray(row["vector"], np.float32), A)
    assert (store / voices.EMBEDDINGS / voices.TAUGHT).is_dir()
    # Seeded once: not again on the next start, and not after a forget took the rows either.
    assert voices.seed_taught(tmp_path, voices.RegistryPeople(tmp_path)) == 0
    voices.forget_voice(3, base=store)
    assert voices.seed_taught(tmp_path, voices.RegistryPeople(tmp_path)) == 0
    assert filed(store) == []


def test_a_taught_vector_of_the_wrong_size_is_skipped_and_counted_in_the_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from voices import write_voices

    write_voices(tmp_path, {3: "Dana"})
    kept = _taught(tmp_path, 3, A)
    _taught(tmp_path, 3, np.ones(512, np.float32))  # an old CAM++ vector is not a ReDimNet2 embedding
    with caplog.at_level(logging.WARNING, logger=voices.__name__):
        assert voices.seed_taught(tmp_path, voices.RegistryPeople(tmp_path)) == 1
    assert [r["id"] for r in filed(segment_log.root(tmp_path))] == [kept]
    assert "skipped 1 of 2 taught voice vectors" in caplog.text


class IndexOf:
    """The vector worker's index over the segment store's voice rows, keyed as it keys them (the person, else the
    voiceprint), answering nearest by cosine."""

    def __init__(self, store: Path) -> None:
        self.rows = filed(store)

    async def nearest(
        self, model: str, version: str, vector: Sequence[float], k: int = 20, *, timeout: float = 0.5
    ) -> Sequence[Mapping[str, Any]]:
        q = np.asarray(vector, np.float32)
        scored = sorted(
            ((1.0 - float(np.dot(q, np.asarray(r["vector"], np.float32))), r) for r in self.rows),
            key=lambda item: item[0],
        )
        return [
            {
                **hit(r["person"] or r["voiceprint"], score),
                "payload_id": r["id"],
                "start_ns": r["start_ns"],
                "end_ns": r["end_ns"],
            }
            for score, r in scored[:k]
        ]


def test_old_cluster_references_without_individual_confirmation_do_not_name_new_lines(tmp_path: Path) -> None:
    from voices import write_voices

    write_voices(tmp_path, {3: "Dana"})
    _taught(tmp_path, 3, A)
    # The reset: no segment store, no index. Listen starts and seeds it from the knowledge store.
    voices.seed_taught(tmp_path, voices.RegistryPeople(tmp_path))
    prints = voices.Voiceprints(tmp_path / "voiceprints.json", lambda: 3)
    m = voices.VoiceMatcher(prints, voices.RegistryPeople(tmp_path), IndexOf(segment_log.root(tmp_path)))
    got = asyncio.run(m.match(A, *SECOND))
    assert got is None, "old cluster vectors have no individual confirmation provenance"


def test_named_search_and_recent_auto_matches_cannot_teach_a_person_without_confirmed_references(tmp_path):
    m, _ = matcher(tmp_path, FakeSearch([hit(ALICE, 0.0)]), FakePeople({7: ALICE}), known=7)
    m._remember(A, 7)
    got = asyncio.run(m.match(A, *SECOND))
    assert got is None


def test_three_confirmed_references_accept_a_clear_voice_but_not_a_competing_cluster(tmp_path):
    m, _ = matcher(tmp_path, FakeSearch([]), FakePeople({7: ALICE}), known=7)
    m.references = lambda: [voices.Reference(ALICE, 7, str(i), A) for i in range(3)]
    got = asyncio.run(m.match(A, *SECOND))
    assert got is not None and got.speaker_id == 7
    m.search = FakeSearch([hit(voices.subject(6), 0.02)])
    ambiguous = asyncio.run(m.match(A, *SECOND))
    assert ambiguous is not None and ambiguous.speaker_id == 6


@pytest.mark.parametrize(("similarity", "accepted"), [(0.49, False), (0.51, True)])
def test_named_matching_requires_the_calibrated_cosine_floor(similarity, accepted):
    refs = [voices.Reference(ALICE, 7, str(i), A) for i in range(3)]
    query = unit(similarity, (1 - similarity**2) ** 0.5)
    assert (voices.match_person(query, refs) is not None) is accepted


@pytest.mark.parametrize(("competitor", "accepted"), [(0.76, False), (0.74, True)])
def test_named_matching_requires_the_unnamed_competitor_margin(competitor, accepted):
    refs = [voices.Reference(ALICE, 7, str(i), A) for i in range(3)]
    assert (voices.match_person(unit(0.8, 0.6), refs, competitor=competitor) is not None) is accepted


@pytest.mark.parametrize(("similarity", "accepted"), [(0.74, False), (0.76, True)])
def test_confirmed_negatives_raise_the_named_floor_above_the_global_floor(similarity, accepted):
    refs = [voices.Reference(ALICE, 7, str(i), A) for i in range(3)]
    refs.append(voices.Reference("dana", 8, "other-line", unit(0.6, 0.8)))
    # Dana's reference has cosine .6 to Alice, so the .15 margin raises Alice's floor to .75.
    query = unit(similarity, -((1 - similarity**2) ** 0.5))
    assert (voices.match_person(query, refs) is not None) is accepted


def test_repeating_one_confirmed_line_does_not_meet_the_reference_minimum(tmp_path):
    m, _ = matcher(tmp_path, FakeSearch([hit(ALICE, 0.0)]), FakePeople({7: ALICE}), known=7)
    m.references = lambda: [voices.Reference(ALICE, 7, "same-line", A)] * 20
    got = asyncio.run(m.match(A, *SECOND))
    assert got is None


def _spoken(seconds: float) -> tuple[int, int]:
    return 100 * NS, int((100 + seconds) * NS)


def test_a_stray_answer_is_left_out_of_the_core_instead_of_disabling_the_person():
    refs = [voices.Reference(ALICE, 7, str(i), v, span=_spoken(3)) for i, v in enumerate((A, A, B))]
    assert voices.calibrate(refs)[ALICE]["eligible"] is True
    assert voices.match_person(A, refs) is not None
    assert voices.match_person(B, refs) is None, "the stray answer can't name Alice"


def test_short_answers_count_as_answers_but_never_name_anyone():
    refs = [voices.Reference(ALICE, 7, str(i), A, span=_spoken(3)) for i in range(2)]
    refs += [voices.Reference(ALICE, 7, f"short-{i}", B, span=_spoken(1.5)) for i in range(2)]
    assert voices.calibrate(refs)[ALICE]["references"] == 4
    assert voices.match_person(B, refs) is None, "two short lines agreeing are still too short to trust"
    assert voices.calibrate(refs[2:] + refs[:1])[ALICE]["eligible"] is False, "one long answer is no core"


def test_a_line_is_named_by_its_nearest_answer_not_their_average():
    # Answers from three rooms that each agree with the first at .6; the line sounds like the second room's (.64)
    # though its cosine to their average is under .1.
    refs = [
        voices.Reference(ALICE, 7, "a", unit(1, 0, 0)),
        voices.Reference(ALICE, 7, "b", unit(0.6, 0.8, 0)),
        voices.Reference(ALICE, 7, "c", unit(0.6, 0, 0.8)),
    ]
    got = voices.match_person(unit(0, 0.8, -0.6), refs)
    assert got is not None and got.sim == pytest.approx(0.64)


def test_overlapping_people_disable_auto_assignment():
    overlapping = [voices.Reference(ALICE, 7, str(i), A) for i in range(3)] + [voices.Reference("dana", 8, "other", A)]
    assert voices.calibrate(overlapping)[ALICE]["eligible"] is False


@pytest.mark.parametrize(("negative", "accepted"), [(0.84, True), (0.86, False)])
def test_a_negative_near_the_persons_answers_raises_their_threshold(negative, accepted):
    refs = [voices.Reference(ALICE, 7, str(i), A) for i in range(3)]
    not_alice = voices.Negative(ALICE, "line", unit(negative, (1 - negative**2) ** 0.5))
    # Alice's answers are A itself, so a line can reach 1.0: a negative at .86 puts her threshold past it.
    assert (voices.match_person(A, refs, negatives=[not_alice]) is not None) is accepted


def test_a_line_that_sounds_like_the_persons_negative_is_not_named_them():
    refs = [voices.Reference(ALICE, 7, str(i), A) for i in range(3)]
    not_alice = voices.Negative(ALICE, "line", unit(0.3, 0.954))  # far from her answers: her threshold stays .5
    query = unit(0.6, 0.8)  # .6 to Alice's answers, .94 to the line the owner said isn't her
    assert voices.match_person(query, refs) is not None
    assert voices.match_person(query, refs, negatives=[not_alice]) is None


def test_only_individual_human_claims_promote_observed_vectors_and_retractions_remove_them(tmp_path):
    m, line = _setup(tmp_path)
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    assert voices.confirmed_references(tmp_path) == []
    reg.claim(registry.utterance_of(line), "Alice", Basis.OWNER_TAG, "test")
    voices.confirm_lines(tmp_path, [line])
    refs = voices.confirmed_references(tmp_path)
    assert len(refs) == 1
    assert np.array_equal(refs[0].vector, A)
    assert refs[0].key == registry.utterance_of(line).id
    reg.detach(registry.utterance_of(line), "unknown", "test", via=registry.voiceprint(1))
    assert voices.confirmed_references(tmp_path) == []
    m.close()


def test_a_stronger_named_cluster_with_no_references_blocks_another_person_and_does_not_churn(tmp_path):
    dana = "0199a000-0000-7000-8000-00000000daaa"
    m, _ = matcher(tmp_path, FakeSearch([hit(voices.subject(8), 0.07)]), FakePeople({7: ALICE, 8: dana}), known=8)
    m.references = lambda: [voices.Reference(ALICE, 7, str(i), A) for i in range(3)]
    query = unit(0.83, (1 - 0.83**2) ** 0.5)
    assert asyncio.run(m.match(query, *SECOND)) is None
    assert asyncio.run(m.match(query, *SECOND)) is None
    assert m.prints.next == 9


def test_an_ineligible_confirmed_person_still_competes_for_the_margin():
    refs = [voices.Reference(ALICE, 7, str(i), A) for i in range(3)]
    refs += [voices.Reference("dana", 8, "one-line", unit(0.5, 0.866))]
    assert voices.calibrate(refs)[ALICE]["eligible"] is True
    assert voices.match_person(unit(0.85, (1 - 0.85**2) ** 0.5), refs) is None


def test_recording_windows_count_as_one_confirmation_until_other_sources_are_confirmed():
    refs = [voices.Reference(ALICE, 7, str(i), A, "one-recording") for i in range(3)]
    assert voices.calibrate(refs)[ALICE]["references"] == 1
    assert voices.match_person(A, refs) is None
    refs.extend([voices.Reference(ALICE, 7, "line-a", A), voices.Reference(ALICE, 7, "line-b", A)])
    assert voices.match_person(A, refs) is not None


def test_reference_reads_follow_human_retags_and_confirmation_retries_are_idempotent(tmp_path):
    from liberational import knowledge

    m, line = _setup(tmp_path)
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    reg.claim(registry.voiceprint(2), "Dana", Basis.OWNER_TAG, "test")
    reg.claim(registry.utterance_of(line), "Alice", Basis.OWNER_TAG, "test")
    voices.confirm_lines(tmp_path, [line])
    voices.confirm_lines(tmp_path, [line])
    assert len(knowledge.Store(registry.knowledge_path(tmp_path)).query(voices.REFERENCES)) == 1
    gallery = voices.ReferenceGallery(tmp_path)
    assert gallery()[0].person == str(reg.person_id("Alice"))
    reg.claim(registry.utterance_of(line), "Dana", Basis.OWNER_TAG, "test")
    assert gallery()[0].person == str(reg.person_id("Dana"))
    assert gallery()[0].voiceprint == 2
    m.close()


def test_a_confirmed_turn_teaches_its_majority_but_not_an_unrelated_tv_voice(tmp_path):
    m, _first = _setup(tmp_path)
    writer = speech_log.Writer("wave", segment_log.root(tmp_path))
    for start, vector in [(103.0, A), (106.0, B)]:
        writer.write(
            speech_log.UTTERANCES,
            {
                "ts_start": start,
                "ts_end": start + 2,
                "device": "wave",
                "text": "test words",
                "speaker_id": 1,
                "speaker": "Speaker 1",
            },
            span_ns=(int(start * NS), int((start + 2) * NS)),
        )
        writer.sync()
        (line,) = speech_log.lines(start, start + 2, base=segment_log.root(tmp_path))
        m.file(vector, 1, line["id"], int(start * NS), int((start + 2) * NS))
    writer.close()
    lines = speech_log.lines(99, 109, base=segment_log.root(tmp_path))
    turn = [{"ts_start": e["ts_start"], "line": registry.key_of(e)} for e in lines]
    _queue(tmp_path, *[{**e, "group": "turn-a", "turn_lines": turn, "name": "Alice"} for e in turn])
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    viewed = registry.at(tmp_path).said([(registry.utterance_of(e), e.get("speaker_id")) for e in lines])
    assert [s.person == registry.at(tmp_path).person_id("Alice") for s in viewed] == [True, True, False]
    assert viewed[2].detached
    assert len(voices.confirmed_references(tmp_path)) == 2
    m.close()


def test_a_named_turn_whose_voices_agree_on_nothing_is_named_but_teaches_nothing(tmp_path):
    """The owner named a turn whose lines sound like nobody in particular (Joe 2026-11-06: a turn named Carly stayed
    Unmatched). With no majority voice there is no outlier to leave out: every line takes the name, and no voice is
    learned from audio that disagrees with itself."""
    m, _first = _setup(tmp_path)
    writer = speech_log.Writer("wave", segment_log.root(tmp_path))
    for start, vector in [(103.0, B), (106.0, unit(0, 0, 1))]:  # A, B, and C: no two alike
        writer.write(
            speech_log.UTTERANCES,
            {"ts_start": start, "ts_end": start + 2, "device": "wave", "text": "test words", "speaker": "unknown"},
            span_ns=(int(start * NS), int((start + 2) * NS)),
        )
        writer.sync()
        (line,) = speech_log.lines(start, start + 2, base=segment_log.root(tmp_path))
        m.file(vector, 1, line["id"], int(start * NS), int((start + 2) * NS))
    writer.close()
    lines = speech_log.lines(99, 109, base=segment_log.root(tmp_path))
    _queue(tmp_path, {"lines": [registry.key_of(e) for e in lines], "group": "turn-b", "name": "Alice"})
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    reg = registry.at(tmp_path)
    viewed = reg.said([(registry.utterance_of(e), e.get("speaker_id")) for e in lines])
    alice = reg.person_id("Alice")
    assert [s.person == alice for s in viewed] == [True, True, True]
    assert not any(s.detached for s in viewed)
    assert voices.confirmed_references(tmp_path) == []
    assert reg.voiceprints(alice) == []
    m.close()


def _turn_of(tmp_path: Path, m: voices.VoiceMatcher, *vectors: np.ndarray) -> list[dict[str, Any]]:
    """Lines heard 103 s on, 3 s apart, one per vector, by voice 1, and the turn naming them queued as Alice."""
    writer = speech_log.Writer("wave", segment_log.root(tmp_path))
    for k, vector in enumerate(vectors):
        start = 103.0 + 3 * k
        writer.write(
            speech_log.UTTERANCES,
            {"ts_start": start, "ts_end": start + 2, "device": "wave", "text": f"words {k}", "speaker_id": 1},
            span_ns=(int(start * NS), int((start + 2) * NS)),
        )
        writer.sync()
        (line,) = speech_log.lines(start, start + 2, base=segment_log.root(tmp_path))
        m.file(vector, 1, line["id"], int(start * NS), int((start + 2) * NS))
    writer.close()
    lines = speech_log.lines(102, 103 + 3 * len(vectors), base=segment_log.root(tmp_path))
    _queue(tmp_path, {"lines": [registry.key_of(e) for e in lines], "group": "turn-s", "name": "Alice"})
    return lines


C = unit(0, 0, 1)


def test_a_named_turn_keeps_its_majority_and_splits_the_rest_off_as_a_voice_to_name_later(tmp_path):
    """Joe 2026-11-06: "split the lines into multiple people and keep the ones that match"."""
    m, _first = _setup(tmp_path)
    lines = _turn_of(tmp_path, m, A, A, B)
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    reg = registry.at(tmp_path)
    alice = reg.person_id("Alice")
    said = reg.said([(registry.utterance_of(e), e.get("speaker_id")) for e in lines])
    assert [s.person == alice for s in said] == [True, True, False]
    viewed = transcript_view.View(tmp_path).lines(lines)
    split = viewed[2]["speaker_id"]
    assert split not in (None, 1) and viewed[2]["speaker"] == f"Speaker {split}" and not viewed[2].get("unlabeled")
    assert len(voices.confirmed_references(tmp_path)) == 2  # only the kept lines are Alice's references
    # Naming the new voice later names its line.
    assert reg.claim(registry.voiceprint(split), "Dana", Basis.OWNER_TAG, "test") is not None
    dana = registry.at(tmp_path).person_id("Dana")
    assert dana is not None
    assert registry.at(tmp_path).said([(registry.utterance_of(lines[2]), 1)])[0].person == dana
    assert transcript_view.View(tmp_path).lines(lines)[2]["speaker"] == "Dana"
    m.close()


def test_a_split_turn_keeps_the_side_that_sounds_like_the_person(tmp_path):
    m, first = _setup(tmp_path)  # Alice's voice, confirmed on its first line (A)
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    reg.claim(registry.utterance_of(first), "Alice", Basis.OWNER_TAG, "test")
    voices.confirm_lines(tmp_path, [first])
    lines = _turn_of(tmp_path, m, B, A, B, A)  # two and two: no majority
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    reg = registry.at(tmp_path)
    alice = reg.person_id("Alice")
    said = reg.said([(registry.utterance_of(e), e.get("speaker_id")) for e in lines])
    assert [s.person == alice for s in said] == [False, True, False, True]
    viewed = transcript_view.View(tmp_path).lines(lines)
    assert viewed[0]["speaker_id"] == viewed[2]["speaker_id"] not in (None, 1)  # the other side: one new voice
    m.close()


def test_a_split_turn_with_no_side_theirs_is_left_unknown(tmp_path):
    m, _first = _setup(tmp_path)
    lines = _turn_of(tmp_path, m, B, C, B, C)  # two and two, and Alice has no voice yet
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    reg = registry.at(tmp_path)
    said = reg.said([(registry.utterance_of(e), e.get("speaker_id")) for e in lines])
    assert [s.person for s in said] == [None] * 4 and all(s.detached and s.label == "unknown" for s in said)
    m.close()


@pytest.mark.parametrize("compact", [False, True], ids=["pending-legacy", "compact-batch"])
def test_nineteen_line_turn_reads_and_commits_once_off_the_audio_loop(tmp_path, monkeypatch, compact):
    import threading
    from collections import Counter

    from liberational import knowledge

    m, first = _setup(tmp_path)
    writer = speech_log.Writer("wave", segment_log.root(tmp_path))
    for i in range(1, 19):
        start = 100 + 3 * i
        writer.write(
            speech_log.UTTERANCES,
            {**first, "id": segment_log.new_id(), "ts_start": start, "ts_end": start + 2},
            span_ns=(start * NS, (start + 2) * NS),
        )
    writer.close()
    lines = speech_log.lines(99, 158, base=segment_log.root(tmp_path))
    for line in lines[1:]:
        m.file(A, 1, line["id"], int(line["ts_start"] * NS), int(line["ts_end"] * NS))
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(2), "Alice", Basis.OWNER_TAG, "test")
    turn = [{"ts_start": e["ts_start"], "line": registry.key_of(e)} for e in lines]
    requests = (
        [{"lines": [e["line"] for e in turn], "group": "nineteen", "name": "Alice"}]
        if compact
        else [{**e, "group": "nineteen", "turn_lines": turn, "name": "Alice"} for e in turn]
    )
    _queue(tmp_path, *requests)
    gallery = voices.ReferenceGallery(tmp_path)
    m.references = gallery
    gallery.refresh()
    counts = Counter()
    threads = set()
    entered, release = threading.Event(), threading.Event()
    read, sync, put, calibrate = segment_log.Log.range, segment_log.Writers.sync, knowledge.Store.put, voices.calibrate

    def counted_range(self, stream, *args, **kwargs):
        counts["embeddings" if stream == voices.EMBEDDINGS else "speech"] += 1
        return read(self, stream, *args, **kwargs)

    def counted_sync(self):
        threads.add(threading.get_ident())
        counts["sync"] += 1
        entered.set()
        assert release.wait(timeout=5)
        return sync(self)

    def counted_put(self, records):
        records = list(records)
        counts["references" if records and records[0].collection == voices.REFERENCES else "claims"] += 1
        return put(self, records)

    def counted_calibration(refs, negatives=()):
        counts["calibration"] += 1
        return calibrate(refs, negatives)

    monkeypatch.setattr(voices, "calibrate", counted_calibration)
    monkeypatch.setattr(segment_log.Log, "range", counted_range)
    monkeypatch.setattr(segment_log.Writers, "sync", counted_sync)
    monkeypatch.setattr(knowledge.Store, "put", counted_put)

    async def with_audio_progress():
        pending = asyncio.create_task(voices.VoiceTags(tmp_path, m).poll())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert reg.known_as("Alice") is not None, "shared registry reads must proceed during native sync"
            assert await asyncio.sleep(0, result="audio callback ran") == "audio callback ran"
        finally:
            release.set()
            await pending

    asyncio.run(with_audio_progress())
    assert dict(counts) == {"speech": 1, "embeddings": 1, "sync": 1, "claims": 1, "references": 1, "calibration": 1}
    assert threads and threading.get_ident() not in threads
    assert len(voices.confirmed_references(tmp_path)) == 19
    m.close()


def _three_lines(state: Path) -> tuple[voices.VoiceMatcher, list[dict[str, Any]]]:
    m, first = _setup(state)
    writer = speech_log.Writer("wave", segment_log.root(state))
    for start in (103, 106):
        writer.write(
            speech_log.UTTERANCES,
            {**first, "id": segment_log.new_id(), "ts_start": start, "ts_end": start + 2},
            span_ns=(start * NS, (start + 2) * NS),
        )
    writer.close()
    lines = speech_log.lines(99, 109, base=segment_log.root(state))
    for line in lines[1:]:
        m.file(A, 1, line["id"], int(line["ts_start"] * NS), int(line["ts_end"] * NS))
    return m, lines


def test_batch_publishes_new_person_once_and_matching_reuses_calibration(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from liberational import knowledge

    m, lines = _three_lines(tmp_path)
    gallery = voices.ReferenceGallery(tmp_path)
    m.references = gallery
    gallery.refresh()
    calls = []
    original = voices.calibrate

    def calibrated(refs, negatives=()):
        calls.append(len(refs))
        return original(refs, negatives)

    monkeypatch.setattr(voices, "calibrate", calibrated)
    _queue(tmp_path, {"lines": [registry.key_of(line) for line in lines], "group": "new-person", "name": "Alice"})
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    assert calls == [3]
    prepared = next(row for row in filed(segment_log.root(tmp_path)) if row["voiceprint"] == voices.subject(2))
    m.search = FakeSearch([hit(prepared["voiceprint"], 0.0)])
    matched = asyncio.run(m.match(A, *SECOND))
    assert matched is not None and matched.speaker_id == 2
    alice = registry.at(tmp_path).person_id("Alice")
    assert registry.at(tmp_path).person_of(registry.voiceprint(matched.speaker_id)).person == alice
    assert all(
        row["person"] is None for row in filed(segment_log.root(tmp_path)) if row["voiceprint"] == voices.subject(2)
    )
    for _ in range(19):
        assert gallery.match(A) is not None
    assert calls == [3], "stable matching must not recalibrate every utterance"
    # Heavy reads/calibration have no dependency on the shared registry's lock.
    gallery.mark = None
    with ThreadPoolExecutor(1) as pool, registry.at(tmp_path).lock:
        pool.submit(gallery.refresh).result(timeout=5)
    assert calls == [3, 3]

    # Cached store access is sufficient; a match must not construct another write-initializing Store.
    def unexpected_store(*args, **kwargs):
        raise AssertionError("a stable match must reuse its knowledge store")

    monkeypatch.setattr(knowledge.Store, "__init__", unexpected_store)
    assert gallery.match(A) is not None
    m.close()


def test_failed_owner_commit_keeps_queue_and_retry_reuses_prepared_rows_after_restart(tmp_path, monkeypatch):
    from liberational import knowledge

    m, lines = _three_lines(tmp_path)
    gallery = voices.ReferenceGallery(tmp_path)
    m.references = gallery
    gallery.refresh()
    _queue(tmp_path, {"lines": [registry.key_of(line) for line in lines], "group": "retry", "name": "Alice"})
    original = knowledge.Store.put
    failed = []
    clock = [1000.0]
    monkeypatch.setattr(voices.time, "monotonic", lambda: clock[0])

    def fail_references(self, records):
        records = list(records)
        if records and records[0].collection == voices.REFERENCES:
            failed.append(True)
            raise knowledge.StorageError("reference disk unavailable")
        return original(self, records)

    monkeypatch.setattr(knowledge.Store, "put", fail_references)
    tags = voices.VoiceTags(tmp_path, m)
    asyncio.run(tags.poll())
    assert registry.at(tmp_path).person_id("Alice") is None
    assert voices.confirmed_references(tmp_path) == []
    assert gallery.match(A) is None
    assert not (tmp_path / "voice_tags.offset").exists()
    assert len(filed(segment_log.root(tmp_path))) == 6, "only the original and prepared ordinary observations exist"
    asyncio.run(tags.poll())
    assert len(failed) == 1, "a transient failure must back off rather than retry in a tight loop"
    monkeypatch.setattr(knowledge.Store, "put", original)
    clock[0] += 2
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    assert len(filed(segment_log.root(tmp_path))) == 6, "a restarted consumer reuses the prepared voiceprint and rows"
    assert len(gallery()) == 3
    assert gallery.match(A) is not None
    assert registry.at(tmp_path).voiceprints(registry.at(tmp_path).person_id("Alice")) == [2]
    assert (tmp_path / "voice_tags.offset").read_text() == str((tmp_path / "voice_tags.jsonl").stat().st_size)
    m.close()


def test_dirty_gallery_coalesces_refresh_and_rejects_revoked_reference_membership(tmp_path, monkeypatch):
    import threading

    m, lines = _three_lines(tmp_path)
    gallery = voices.ReferenceGallery(tmp_path)
    m.references = gallery
    _queue(tmp_path, {"lines": [registry.key_of(line) for line in lines], "group": "dirty", "name": "Alice"})
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    assert gallery.match(A) is not None
    entered, release = threading.Event(), threading.Event()
    read = voices.confirmed_references
    calls = []

    def blocked(state):
        calls.append(True)
        entered.set()
        assert release.wait(timeout=5)
        return read(state)

    monkeypatch.setattr(voices, "confirmed_references", blocked)
    reg = registry.at(tmp_path)
    reg.detach(registry.utterance_of(lines[0]), "unknown", "test", via=registry.voiceprint(2))
    try:
        assert gallery.match(A) is None
        assert entered.wait(timeout=5)
        assert gallery.match(A) is None
        assert gallery.match(A) is None
        assert calls == [True], "one pending rebuild serves every dirty probe"
    finally:
        release.set()
        assert gallery.pending is not None
        gallery.pending.result(timeout=5)
    assert len(gallery()) == 2
    assert gallery.match(A) is None, "two references no longer satisfy the minimum"
    reg.forget("Alice", "test")
    assert gallery.match(A) is None
    assert gallery.pending is not None
    gallery.pending.result(timeout=5)
    assert gallery() == []
    m.close()


def test_inconsistent_batch_preserves_human_labels_and_reports_one_accurate_outcome(tmp_path, caplog):
    m, lines = _three_lines(tmp_path)
    # Replace the final line's observation with a genuinely different voice.
    m.file(B, 1, lines[2]["id"], int(lines[2]["ts_start"] * NS), int(lines[2]["ts_end"] * NS))
    reg = registry.at(tmp_path)
    reg.claim(registry.utterance_of(lines[2]), "Dana", Basis.OWNER_TAG, "human")
    _queue(tmp_path, {"lines": [registry.key_of(line) for line in lines], "group": "mixed", "name": "Alice"})
    caplog.clear()
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    assert reg.person_of(registry.utterance_of(lines[2])).person == reg.person_id("Dana")
    assert len(voices.confirmed_references(tmp_path)) == 2
    assert "2 labelled, 2 reference vectors, 0 left unknown, 1 human labels preserved" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert "confirm it individually" not in caplog.text
    m.close()


def test_compact_queue_uses_the_exact_device_key_when_two_microphones_share_a_timestamp(tmp_path):
    m, wanted = _setup(tmp_path)
    writer = speech_log.Writer("other", segment_log.root(tmp_path))
    writer.write(
        speech_log.UTTERANCES, {**wanted, "id": segment_log.new_id(), "device": "other"}, span_ns=(100 * NS, 102 * NS)
    )
    writer.close()
    lines = speech_log.lines(99, 103, base=segment_log.root(tmp_path))
    other = next(line for line in lines if line["device"] == "other")
    m.file(B, 1, other["id"], 100 * NS, 102 * NS)
    _queue(tmp_path, {"lines": [registry.key_of(wanted)], "group": "exact", "name": "Alice"})
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    reg = registry.at(tmp_path)
    assert reg.person_of(registry.utterance_of(wanted)).person == reg.person_id("Alice")
    assert reg.person_of(registry.utterance_of(other)).person is None
    assert [ref.key for ref in voices.confirmed_references(tmp_path)] == [registry.utterance_of(wanted).id]
    m.close()


def test_delete_between_preparation_and_transaction_wins_without_a_queue_marker(tmp_path, monkeypatch):
    import contextlib

    m, line = _setup(tmp_path)
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    _queue(tmp_path, {"lines": [registry.key_of(line)], "group": "deletion-race", "name": "Alice", "requested_at": 1.0})
    transaction = registry.transaction

    @contextlib.contextmanager
    def delete_before_begin(state):
        reg.forget("Alice", "test")
        with transaction(state) as local:
            yield local

    monkeypatch.setattr(registry, "transaction", delete_before_begin)
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    assert reg.person_id("Alice") is None, "an old queued tag must never recreate the deleted registration"
    assert voices.confirmed_references(tmp_path) == []
    assert (tmp_path / "voice_tags.offset").read_text() == str((tmp_path / "voice_tags.jsonl").stat().st_size)
    m.close()


def test_fresh_owner_teaching_after_forget_can_register_the_name_again(tmp_path):
    m, line = _setup(tmp_path)
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    old = reg.person_id("Alice")
    reg.forget("Alice", "test")
    _queue(
        tmp_path,
        {"lines": [registry.key_of(line)], "group": "new-teaching", "name": "Alice", "requested_at": time.time() + 1},
    )
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    assert reg.person_id("Alice") is not None and reg.person_id("Alice") != old
    assert len(voices.confirmed_references(tmp_path)) == 1
    m.close()


def test_cached_membership_resolves_an_index_entity_under_a_merged_person_id(tmp_path):
    m, _line = _setup(tmp_path)
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    old_person = reg.person_id("Alice")
    people = voices.RegistryPeople(tmp_path)
    assert people.snapshot().voice_of(str(old_person)) == 1
    reg.merge("Alice", "Dana", "test")
    current = reg.person_id("Dana")
    cached = people.snapshot()
    assert cached.voice_of(str(old_person)) == 1
    assert cached.person_of(1) == str(current)
    m.close()


def test_rename_between_preparation_and_transaction_uses_the_current_name(tmp_path, monkeypatch):
    import contextlib

    m, line = _setup(tmp_path)
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    person = reg.person_id("Alice")
    _queue(tmp_path, {"lines": [registry.key_of(line)], "group": "rename-race", "name": "Alice", "requested_at": 1.0})
    transaction = registry.transaction

    @contextlib.contextmanager
    def rename_before_begin(state):
        reg.rename("Alice", "Dee", "test")
        with transaction(state) as local:
            yield local

    monkeypatch.setattr(registry, "transaction", rename_before_begin)
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    assert [p.display_name for p in reg.people()] == ["Dee"]
    assert reg.person_of(registry.utterance_of(line)).person == person
    assert voices.confirmed_references(tmp_path)[0].person == str(person)
    m.close()


def test_old_request_already_pending_when_person_is_forgotten_cannot_recreate_them(tmp_path):
    m, line = _setup(tmp_path)
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    _queue(tmp_path, {"lines": [registry.key_of(line)], "group": "old-pending", "name": "Alice", "requested_at": 1.0})
    reg.forget("Alice", "test")
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    assert reg.person_id("Alice") is None
    assert voices.confirmed_references(tmp_path) == []
    assert (tmp_path / "voice_tags.offset").read_text() == str((tmp_path / "voice_tags.jsonl").stat().st_size)
    m.close()


@pytest.mark.parametrize("mutation", ["forget", "remember", "prune"])
def test_recent_scoring_survives_worker_thread_mutation_without_holding_a_lock_during_dot(
    tmp_path, monkeypatch, mutation
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    m, _store = matcher(tmp_path, FakeSearch())
    clock = [1000.0]
    monkeypatch.setattr(voices.time, "monotonic", lambda: clock[0])
    m._remember(A, 1)
    m._remember(B, 2)
    entered, release = threading.Event(), threading.Event()
    dot = voices.np.dot

    def paused_dot(a, b):
        entered.set()
        assert release.wait(timeout=5)
        return dot(a, b)

    def mutate_recent():
        if mutation == "forget":
            m.forget()
        else:
            if mutation == "prune":
                clock[0] += voices.RECENT_S + 1
            m._remember(A, 3)

    monkeypatch.setattr(voices.np, "dot", paused_dot)
    with ThreadPoolExecutor(2) as pool:
        reading = pool.submit(m._recent_best, A)
        try:
            assert entered.wait(timeout=5)
            # Mutation must finish while scoring is paused: no dot, registry lookup, or other slow work under lock.
            pool.submit(mutate_recent).result(timeout=5)
        finally:
            release.set()
        assert reading.result(timeout=5) == (1.0, 1), "the in-flight read keeps its original ordered snapshot"
    expected = {"forget": [], "remember": [1, 2, 3], "prune": [3]}[mutation]
    assert [n for _t, _v, n in m.recent] == expected
    m.close()


@pytest.mark.parametrize(
    ("model", "version", "dims", "eligible"),
    [
        ("redimnet2-b3", "vox2-lm-v1", 192, True),
        ("campp", "v1", 512, False),
        ("campp", "v1", 192, False),
        ("redimnet2-b3", "vox2-lm-v1", 512, False),
        ("redimnet2-b3", "other-checkpoint", 192, False),
    ],
)
def test_confirmed_reference_membership_never_mixes_embedding_model_spaces(tmp_path, model, version, dims, eligible):
    from liberational import knowledge

    reg = registry.at(tmp_path)
    line = registry.utterance(100.0, "wave")
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    reg.claim(line, "Alice", Basis.OWNER_TAG, "human")
    person = str(reg.person_id("Alice"))
    value = {
        "voiceprint": 1,
        "person": person,
        "vector": [1.0] + [0.0] * (dims - 1),
        "dims": dims,
        "model": model,
        "version": version,
        "source_id": "0199a000-0000-7000-8000-000000000123",
        "source_kind": "voice.embedding",
        "confirmation_subject": line.id,
    }
    record = knowledge.Record(
        "0199a000-0000-7000-8000-000000000124",
        voices.REFERENCES,
        "speaker:1",
        100 * NS,
        102 * NS,
        103 * NS,
        value,
        (person, voices.subject(1)),
    )
    knowledge.at(registry.knowledge_path(tmp_path)).put([record])
    refs = voices.confirmed_references(tmp_path)
    assert bool(refs) is eligible


def spanned(entity: str, score: float, start: int, end: int) -> dict[str, Any]:
    return {**hit(entity, score), "start_ns": start, "end_ns": end}


# Three lines first heard as unnamed Speaker 1, then taught as Alice: each reference is a copy of a line's vector.
TAUGHT_SPANS = [(10 * NS, 12 * NS), (20 * NS, 22 * NS), (30 * NS, 32 * NS)]


def taught_alice() -> list[voices.Reference]:
    return [voices.Reference(ALICE, 7, f"line-{i}", A, span=span) for i, span in enumerate(TAUGHT_SPANS)]


def test_the_unnamed_originals_of_taught_lines_do_not_compete_with_their_copies(tmp_path):
    originals = [spanned(voices.subject(1), 0.0, start, end) for start, end in TAUGHT_SPANS]
    m, _ = matcher(tmp_path, FakeSearch(originals), FakePeople({7: ALICE}), known=7)
    m.references = taught_alice
    got = asyncio.run(m.match(A, *SECOND))
    assert got is not None and (got.speaker_id, got.speaker) == (7, "Speaker 7")


def test_an_unrelated_line_in_the_same_unnamed_cluster_still_competes(tmp_path):
    originals = [spanned(voices.subject(1), 0.0, start, end) for start, end in TAUGHT_SPANS]
    other_line = spanned(voices.subject(1), 0.02, 40 * NS, 42 * NS)
    m, _ = matcher(tmp_path, FakeSearch([*originals, other_line]), FakePeople({7: ALICE}), known=7)
    m.references = taught_alice
    got = asyncio.run(m.match(A, *SECOND))
    assert got is not None and got.speaker_id == 1


def test_an_overlapping_but_different_line_still_competes():
    # Cross-talk: another voice's line overlaps a taught one in time but isn't the same utterance.
    crosstalk = [(0.98, 11 * NS, 13 * NS)]
    assert voices.match_person(A, taught_alice(), unnamed=crosstalk) is None


def test_a_reference_without_a_span_keeps_every_unnamed_competitor():
    refs = [voices.Reference(ALICE, 7, f"line-{i}", A) for i in range(3)]
    assert voices.match_person(A, refs, unnamed=[(1.0, 10 * NS, 12 * NS)]) is None


def test_a_voice_taught_from_an_unnamed_cluster_is_named_when_heard_again(tmp_path):
    prints = voices.Voiceprints(tmp_path / "voiceprints.json", lambda: 0)
    m = voices.VoiceMatcher(
        prints, voices.RegistryPeople(tmp_path), FakeSearch([]), segments=segment_log.root(tmp_path)
    )
    n = prints.new()
    writer = speech_log.Writer("wave", segment_log.root(tmp_path))
    for start in (100, 110, 120):
        row = {"ts_start": float(start), "ts_end": start + 2.0, "device": "wave", "text": "hello", "speaker_id": n}
        writer.write(speech_log.UTTERANCES, {**row, "speaker": f"Speaker {n}"}, span_ns=(start * NS, (start + 2) * NS))
        writer.sync()
        (line,) = speech_log.lines(start, start + 2, base=segment_log.root(tmp_path))
        m.file(A, n, line["id"], start * NS, (start + 2) * NS)
    writer.close()
    lines = speech_log.lines(99, 123, base=segment_log.root(tmp_path))
    turn = [{"ts_start": e["ts_start"], "line": registry.key_of(e)} for e in lines]
    _queue(tmp_path, *[{**e, "group": "turn-a", "turn_lines": turn, "name": "Alice"} for e in turn])
    asyncio.run(voices.VoiceTags(tmp_path, m).poll())
    gallery = voices.ReferenceGallery(tmp_path)
    gallery.refresh()
    m.search, m.references = IndexOf(segment_log.root(tmp_path)), gallery
    got = asyncio.run(m.match(A, 200 * NS, 202 * NS))
    alice = registry.at(tmp_path).person_id("Alice")
    assert got is not None and got.speaker_id != n
    assert registry.local(tmp_path).person_of(registry.voiceprint(got.speaker_id)).person == alice


def test_a_one_word_line_never_opens_an_unknown_voice(tmp_path):
    m, _ = matcher(tmp_path, FakeSearch([]), known=5)
    assert asyncio.run(m.match(A, *SECOND, words=1)) is None
    assert m.prints.next == 6, "no voiceprint opened"
    again = asyncio.run(m.match(A, *SECOND, words=3))
    assert again is not None and again.speaker_id == 6, "and nothing remembered it to match the next line"


def test_a_one_word_line_does_not_join_an_unknown_voice_or_feed_it(tmp_path):
    m, store = matcher(tmp_path, FakeSearch([hit(voices.subject(3), 0.05)]), known=3)
    got = asyncio.run(m.match(A, *SECOND, words=1))
    assert got is None
    m.learn(A, got, segment_log.new_id(), *SECOND)
    assert [r["voiceprint"] for r in filed(store)] == [voices.subject(0)], "filed unassigned, not under Speaker 3"
    assert m.recent == ()


def test_a_one_word_line_still_names_a_confirmed_person(tmp_path):
    m, _ = matcher(tmp_path, FakeSearch([]), FakePeople({7: ALICE}), known=7)
    m.references = lambda: [voices.Reference(ALICE, 7, str(i), A) for i in range(3)]
    got = asyncio.run(m.match(A, *SECOND, words=1))
    assert got is not None and got.speaker_id == 7


@pytest.mark.parametrize(
    ("text", "words"),
    [("Yeah.", 1), ("It's", 1), ("  ", 0), ("Hoochie. Hoochy.", 2), ("Okay, sure thing", 3), ("12 o'clock", 2)],
)
def test_words_are_runs_of_letters_digits_and_apostrophes(text, words):
    assert voices.word_count(text) == words


# ---- naming a voice teaches it (#3247)


def _heard(state: Path, vectors: Sequence[np.ndarray]) -> list[dict[str, Any]]:
    """One line a vector, each two seconds long and heard as Speaker 1 on the wave mic, its embedding filed."""
    prints = voices.Voiceprints(state / "voiceprints.json", lambda: 0)
    m = voices.VoiceMatcher(prints, voices.RegistryPeople(state), FakeSearch([]), segments=segment_log.root(state))
    n = prints.new()
    writer = speech_log.Writer("wave", segment_log.root(state))
    for k in range(len(vectors)):
        at = 100.0 + 10 * k
        line = {"ts_start": at, "ts_end": at + 2, "device": "wave", "text": "hello there", "speaker": f"Speaker {n}"}
        writer.write(speech_log.UTTERANCES, {**line, "speaker_id": n}, span_ns=(int(at * NS), int((at + 2) * NS)))
    writer.close()
    stored = speech_log.lines(99.0, 100.0 + 10 * len(vectors), base=segment_log.root(state))
    for line, v in zip(stored, vectors, strict=True):
        m.file(v, n, line["id"], int(line["ts_start"] * NS), int(line["ts_end"] * NS))
    m.close()
    return stored


def test_naming_a_voice_teaches_it_so_a_held_out_line_of_that_voice_is_named(tmp_path):
    taught = _heard(tmp_path, [unit(1, 0.10, 0), unit(1, 0, 0.10), unit(1, -0.10, 0)])
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "dashboard")  # what Who's this? naming writes
    held_out = unit(1, 0.05, -0.08)
    gallery = voices.ReferenceGallery(tmp_path)
    assert voices.match_person(held_out, gallery()) is None, "a name alone teaches nothing"
    assert voices.teach(tmp_path, taught, "Alice") == 3
    got = voices.match_person(held_out, gallery())
    assert got is not None and got.speaker_id == 1
    assert voices.match_person(B, gallery()) is None, "someone else's voice still isn't Alice"
    for line in taught:  # the naming's undo
        reg.release(registry.utterance_of(line), "dashboard")
    assert gallery() == []


def _named(gallery: voices.ReferenceGallery, query: np.ndarray) -> voices.Match | None:
    gallery.refresh()
    return gallery.match(query)


def test_owner_answers_on_room_audio_name_the_next_similar_line_and_a_not_stops_a_wrong_name(tmp_path):
    # Two answers agree, one is a stray from a noisy room (the old gate refused all three); the fourth line is
    # Dana's, heard unnamed, that sounds .7 like Alice.
    stray, dana = unit(0, 0, 0, 1), unit(0.7, 0, 0, 0, 0.714)
    *taught, theirs = _heard(tmp_path, [unit(1, 0.1, 0), unit(1, 0, 0.1), stray, dana])
    reg = registry.at(tmp_path)
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "dashboard")
    gallery = voices.ReferenceGallery(tmp_path)
    assert voices.teach(tmp_path, taught, "Alice") == 3
    held_out, like_dana = unit(1, 0.05, -0.08), unit(0.68, 0, 0, 0, 0.733)
    got = _named(gallery, held_out)
    assert got is not None and got.speaker_id == 1
    assert _named(gallery, like_dana) is not None, "before the owner's answer Dana's next line is named Alice"
    alice = str(reg.person_id("Alice"))
    assert voices.deny(tmp_path, [theirs], alice) == 1
    assert _named(gallery, like_dana) is None
    assert _named(gallery, held_out) is not None, "Alice's own voice is still named"
    voices.deny(tmp_path, [theirs], alice, active=False)  # the undo
    assert _named(gallery, like_dana) is not None


def test_a_negative_goes_once_the_owner_labels_that_line_as_the_person_after_all(tmp_path):
    (line,) = _heard(tmp_path, [A])
    reg = registry.at(tmp_path)
    alice = str(reg.ensure("Alice", "test"))
    voices.deny(tmp_path, [line], alice)
    assert [n.key for n in voices.confirmed_negatives(tmp_path)] == [registry.utterance_of(line).id]
    reg.claim(registry.utterance_of(line), "Alice", Basis.OWNER_TAG, "dashboard")
    assert voices.confirmed_negatives(tmp_path) == []


def test_teaching_a_line_again_adds_no_second_reference(tmp_path):
    (line,) = _heard(tmp_path, [A])
    registry.at(tmp_path).claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "dashboard")
    voices.teach(tmp_path, [line], "Alice")
    voices.teach(tmp_path, [line], "Alice")
    assert len(voices.confirmed_references(tmp_path)) == 1


def _line(key: str, sim: float, dur: float = 2.0, text: str = "hello there") -> dict[str, Any]:
    return {"device": "wave", "ts_start": float(key), "ts_end": float(key) + dur, "text": text, "speaker_sim": sim}


def test_teaching_takes_the_offered_clip_then_the_most_typical_usable_lines():
    lines = [
        _line("10", 0.50),
        _line("20", 0.90),
        _line("30", 0.95, dur=0.8),  # too short to have an embedding
        _line("40", 0.92, text="Yeah."),  # one word
        _line("50", 0.70, dur=6.0),
        _line("60", 0.70, dur=3.0),
    ]
    offered = registry.key_of(lines[0])
    picked = voices.teaching_lines(lines, offered)
    assert [e["ts_start"] for e in picked] == [10.0, 20.0, 50.0]
    assert [e["ts_start"] for e in voices.teaching_lines(lines, None, limit=2)] == [20.0, 50.0]
    assert voices.teaching_lines([], offered) == []
