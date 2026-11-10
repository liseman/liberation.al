"""Telling pets apart (people/pet_recognition.py): sightings, groups, owner labels, and the held-out check that
keeps recognition off until the labels show it works."""

from unittest.mock import create_autospec

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from liberational import knowledge, segment_log
from liberational.people import pet_recognition as R

T = 1_800_000_000.0
DIM = 384
MIA, POPPY = "0199a3c1-0000-7000-8000-000000000001", "0199a3c1-0000-7000-8000-000000000002"
BESS = "0199a3c1-0000-7000-8000-00000000000b"
PETS = [{"id": MIA, "name": "Mia", "species": "cat"}, {"id": POPPY, "name": "Poppy", "species": "cat"}]


def look(identity: int, k: int) -> list[float]:
    """An embedding of animal `identity`, varied by sighting `k`: 0.95 along the identity's own axis and 0.31 along
    an axis no other sighting uses, so two sightings of one animal are about 0.90 alike and of two animals 0."""
    v = np.zeros(DIM)
    v[identity] = 0.95
    v[100 + k] = 0.31
    return (v / np.linalg.norm(v)).tolist()


def row(track, identity, k, t, species="cat", box=(0.1, 0.1, 0.3, 0.3)):
    return {
        "id": f"emb-{track}-{k}",
        "ts_start": t,
        "cam": "cam0",
        "track_id": track,
        "species": species,
        "box": [*box, 0.9],
        "vector": look(identity, k),
        "model": R.embedding_model(),
    }


def seen(*tracks):
    """Sightings for (track, identity, k, t) tuples."""
    return R.sightings([row(*t) for t in tracks])


def test_a_tracks_embeddings_are_one_sighting_with_its_span_and_largest_box():
    rows = [
        row("a", 0, 1, T + 5, box=(0.1, 0.1, 0.2, 0.2)),
        row("a", 0, 2, T, box=(0.1, 0.1, 0.4, 0.4)),
        {"ts_start": T, "track_id": "a", "species": "cat", "vector": []},  # malformed: skipped
        row("b", 1, 3, T + 1),
    ]
    a, b = R.sightings(rows)
    assert (a.track_id, a.start, a.end, len(a.vectors), a.shot["ts"]) == ("a", T, T + 5, 2, T)
    assert b.track_id == "b"
    assert float(np.linalg.norm(a.mean)) == pytest.approx(1.0)


def test_sightings_of_one_animal_group_together_and_other_animals_and_species_do_not():
    found = seen(("a1", 0, 1, T), ("b1", 1, 2, T + 1), ("a2", 0, 3, T + 2), ("a3", 0, 4, T + 3))
    found += R.sightings([row("d1", 0, 5, T + 4, species="dog")])  # the same look, another species
    assert [[s.track_id for s in g] for g in R.groups(found)] == [["a1", "a2", "a3"], ["b1"], ["d1"]]


def test_recognition_names_a_clear_match_and_only_suggests_an_unclear_one():
    mia, poppy, query = seen(("m", 0, 1, T), ("p", 1, 2, T), ("q", 0, 3, T))
    assert R.recognize(query, [(mia, MIA), (poppy, POPPY)])[0] == MIA
    blend = R.sightings([{**row("x", 0, 4, T), "vector": list((np.array(look(0, 4)) + np.array(look(1, 4))) / 1.6)}])
    label, _, suggestion = R.recognize(blend[0], [(mia, MIA), (poppy, POPPY)])
    assert label is None and suggestion in (MIA, POPPY), "as like one as the other: unknown"
    far = R.sightings([row("f", 2, 5, T)])[0]
    assert R.recognize(far, [(mia, MIA), (poppy, POPPY)])[0] is None, "like neither"


def test_a_labeled_sighting_is_never_recognized_from_its_own_label():
    # Itself is the most alike sighting there is; it must be compared only with the other labeled ones.
    mia, poppy = seen(("m", 0, 1, T), ("p", 1, 2, T))
    assert R.recognize(mia, [(mia, MIA), (poppy, POPPY)])[0] is None
    assert R.recognize(mia, R.References([(mia, MIA), (poppy, POPPY)]))[0] is None


@pytest.mark.parametrize(
    ("with_dog", "exclude", "expected"),
    [
        pytest.param(True, (), None, id="as-like-the-dog"),
        pytest.param(False, (), MIA, id="no-dog-labeled"),
        pytest.param(True, (BESS,), MIA, id="owner-said-not-the-dog"),
    ],
)
def test_a_cat_box_as_like_the_labeled_dog_as_a_cat_is_named_after_neither(with_dog, exclude, expected):
    # The detector sometimes boxes the dog as a cat. Halfway between Mia's look and Bess's, the box is 0.67 like
    # each: enough for Mia, but no further from the dog than from her.
    mia, poppy = seen(("m", 0, 1, T), ("p", 1, 2, T))
    bess = R.sightings([row("b", 5, 3, T, species="dog")])[0]
    half = np.zeros(DIM)
    half[0] = half[5] = 1 / np.sqrt(2)
    query = R.sightings([{**row("q", 0, 4, T), "vector": half.tolist()}])[0]
    labeled = [(mia, MIA), (poppy, POPPY), *([(bess, BESS)] if with_dog else [])]
    label, sim, _ = R.recognize(query, labeled, min_sim=0.6, exclude=exclude)
    assert (label, sim) == (expected, pytest.approx(0.95 / np.sqrt(2), abs=0.001))


def visits(identity, label, start, n, gap=3600.0):
    """n labeled sightings of one animal an hour apart."""
    return [
        (s, label) for s in seen(*((f"{label}{k}", identity, 10 * identity + k, start + k * gap) for k in range(n)))
    ]


def test_held_out_check_turns_recognition_on_when_labels_tell_two_pets_apart():
    labeled = visits(0, MIA, T, 4) + visits(1, POPPY, T + 1800, 4)
    check = R.held_out(labeled, "cat")
    assert (check["on"], check["right"], check["wrong"], check["unknown"]) == (True, 8, 0, 0)


def test_one_kind_of_label_can_never_show_a_mistake_so_recognition_stays_off():
    check = R.held_out(visits(0, MIA, T, 8), "cat")
    assert check["on"] is False and "another animal" in check["reason"]


def test_sightings_from_one_visit_cannot_vouch_for_each_other():
    one_visit = visits(0, MIA, T, 4, gap=60.0) + visits(1, POPPY, T + 1800, 4, gap=60.0)
    check = R.held_out(one_visit, "cat")
    assert check["on"] is False and check["unknown"] == 8


def test_labels_that_disagree_with_appearance_keep_recognition_off():
    mixed = visits(0, MIA, T, 4) + visits(0, POPPY, T + 1800, 4)  # one look, labeled as two pets
    assert R.held_out(mixed, "cat")["on"] is False


def test_too_few_labels_keep_recognition_off():
    check = R.held_out(visits(0, MIA, T, 2) + visits(1, POPPY, T + 1800, 2), "cat")
    assert check["on"] is False and check["reason"].startswith("needs 2 more")


def store(state, rows, stream=segment_log.ANIMAL_EMBEDDINGS):
    """Embedding rows written to the segment store under `state`, as see-actions (or, with `stream`, see-captions)
    writes them."""
    writers = segment_log.Writers(segment_log.root(state))
    writers.append(stream, "cam0", rows)
    writers.close()
    return state


class Log:
    """A state directory with embeddings in its segment store, and the owner labels written to its knowledge store."""

    def __init__(self, state):
        self.state = state

    def labels(self):
        return [r.record for r in knowledge.Store(self.state / "knowledge.sqlite3").query(R.LABELS)]


def owner(state, *labels):
    """Owner labels written straight to the knowledge store: (track id, label or "" to clear, time)."""
    knowledge.Store(state / "knowledge.sqlite3").put(
        knowledge.Record(
            f"019c0000-0000-7000-8000-{k:012x}",
            R.LABELS,
            track,
            int(ts * 1e9),
            int(ts * 1e9),
            int(ts * 1e9),
            {"track_id": track, "label": label, "active": bool(label), "source": "owner"},
        )
        for k, (track, label, ts) in enumerate(labels)
    )


FIXTURE_ROWS = [
    # Mia and Poppy, four visits each, an hour apart; then one more of each, unlabeled
    *(r for k in range(4) for r in (row(f"m{k}", 0, k, T + 3600 * k), row(f"p{k}", 1, 10 + k, T + 3600 * k + 1800))),
    row("m-new", 0, 20, T + 20000),
    row("p-new", 1, 21, T + 20001),
    row("dog", 2, 22, T + 1, species="dog"),
]


@pytest.fixture
def log(monkeypatch, tmp_path):
    continuum_py = pytest.importorskip("continuum_py")
    if not hasattr(continuum_py, "SegmentWriter"):
        pytest.skip("this continuum_py predates the segment store")
    return Log(store(tmp_path, FIXTURE_ROWS))


def test_labeling_groups_turns_recognition_on_and_names_new_sightings(log):
    now = T + 30000
    before = R.view(log.state, PETS, now=now)
    assert all(not c["on"] for c in before["recognition"])
    assert R.label(log.state, [f"m{k}" for k in range(4)], MIA, PETS, now=now) == 4
    R.label(log.state, [f"p{k}" for k in range(4)], POPPY, PETS, now=now + 1)
    after = R.view(log.state, PETS, now=now + 2)
    (cat,) = [c for c in after["recognition"] if c["species"] == "cat"]
    assert cat["on"] is True
    rows = {s["track_id"]: s for g in after["groups"] for s in g["sightings"]}
    assert (rows["m-new"]["recognized"]["label"], rows["p-new"]["recognized"]["label"]) == (MIA, POPPY)
    assert "recognized" not in rows["dog"] and "suggestion" not in rows["dog"], "no dog labels to compare with"


def labeled_cats(state, now):
    R.label(state, [f"m{k}" for k in range(4)], MIA, PETS, now=now)
    R.label(state, [f"p{k}" for k in range(4)], POPPY, PETS, now=now + 1)


def test_a_read_with_no_new_label_runs_no_check_and_a_new_label_runs_them_again(log, monkeypatch):
    now = T + 30000
    labeled_cats(log.state, now)
    held_out = create_autospec(R.held_out, side_effect=R.held_out)
    monkeypatch.setattr(R, "held_out", held_out)
    first = R.view(log.state, PETS, now=now + 2)
    assert [c.args[1] for c in held_out.call_args_list] == ["cat", "dog"]
    again = R.view(log.state, PETS, now=now + 3)
    assert held_out.call_count == 2, "nothing was labeled since: the kept checks answer"
    assert again["recognition"] == first["recognition"]
    R.label(log.state, ["m-new"], MIA, PETS, now=now + 4)
    after = R.view(log.state, PETS, now=now + 5)
    assert [c.args[1] for c in held_out.call_args_list] == ["cat", "dog", "cat", "dog"]
    (cat,) = [c for c in after["recognition"] if c["species"] == "cat"]
    assert (first["recognition"][0]["labeled"], cat["labeled"]) == (8, 9)


def test_a_labeled_track_s_new_embedding_runs_its_species_check_again(log, monkeypatch):
    now = T + 30000
    labeled_cats(log.state, now)
    held_out = create_autospec(R.held_out, side_effect=R.held_out)
    monkeypatch.setattr(R, "held_out", held_out)
    R.view(log.state, PETS, now=now + 2)
    store(log.state, [row("m3", 0, 30, T + 3 * 3600 + 5)])  # Mia's last labeled visit, seen a little longer
    R.view(log.state, PETS, now=now + 3)
    assert [c.args[1] for c in held_out.call_args_list] == ["cat", "dog", "cat"], "the dog's inputs did not change"


def polled(state):
    return {s["track_id"] for g in R.view(state, PETS)["groups"] for s in g["sightings"]}


def test_a_poll_gets_the_kept_answer_until_a_label_or_a_new_sighting_changes_it(log, monkeypatch):
    now = T + 30000
    monkeypatch.setattr(R.time, "time", lambda: now)
    first = R.view(log.state, PETS)
    assert R.view(log.state, PETS) is first
    assert polled(log.state) >= {"m-new", "p-new"}
    R.label(log.state, ["m-new"], MIA, PETS, now=now)
    assert "m-new" not in polled(log.state)
    store(log.state, [row("m-later", 0, 31, T + 25000)])
    assert "m-later" in polled(log.state)
    renamed = R.view(log.state, [{**PETS[0], "name": "Mittens"}, PETS[1]])
    assert renamed["names"][MIA] == "Mittens"


def test_a_kept_answer_is_given_again_only_for_a_while(log, monkeypatch):
    monkeypatch.setattr(R.time, "time", lambda: T + 30000)
    monkeypatch.setattr(R, "VIEW_TTL_S", 0.0)  # its label_days cutoff moves with time, so it's worked out again
    first = R.view(log.state, PETS)
    again = R.view(log.state, PETS)
    assert again is not first and again == first


def test_until_the_check_passes_a_match_is_only_a_suggestion(log):
    now = T + 30000
    R.label(log.state, ["m0", "m1"], MIA, PETS, now=now)
    rows = {s["track_id"]: s for g in R.view(log.state, PETS, now=now + 1)["groups"] for s in g["sightings"]}
    assert "recognized" not in rows["m-new"]
    assert rows["m-new"]["suggestion"]["label"] == MIA
    assert rows["m-new"]["suggestion"]["clear"] is True, "a clear match the check doesn't trust yet"


def test_an_unclear_nearest_pet_is_a_suggestion_that_is_not_clear():
    mia, poppy = seen(("m", 0, 1, T), ("p", 1, 2, T))
    blend = R.sightings([{**row("x", 0, 4, T), "vector": list((np.array(look(0, 4)) + np.array(look(1, 4))) / 1.6)}])
    judged = R._judge(blend[0], [(mia, MIA), (poppy, POPPY)], {"on": False})
    assert judged["suggestion"]["clear"] is False


def listed(state, now):
    return {s["track_id"]: s for g in R.view(state, PETS, now=now)["groups"] for s in g["sightings"]}


@pytest.mark.parametrize("value", [MIA, R.OTHER], ids=["a-pet", "another-animal"])
def test_a_labeled_sighting_leaves_the_list_and_clearing_its_label_brings_it_back(log, value):
    now = T + 30000
    assert "m0" in listed(log.state, now)
    R.label(log.state, ["m0", "m1"], value, PETS, now=now)
    after = listed(log.state, now + 1)
    assert "m0" not in after and "m1" not in after
    assert "m2" in after  # the rest of its group still waits
    R.label(log.state, ["m0"], None, PETS, now=now + 2)
    assert listed(log.state, now + 3)["m0"]["label"] is None


def test_each_named_pets_newest_sightings_are_listed_labeled_or_recognized(log):
    now = T + 30000
    R.label(log.state, [f"m{k}" for k in range(4)], MIA, PETS, now=now)
    R.label(log.state, [f"p{k}" for k in range(4)], POPPY, PETS, now=now + 1)
    seen = R.view(log.state, PETS, now=now + 2)["seen"]
    mia = [(r["track_id"], r["how"]) for r in seen[MIA]]
    assert set(mia) == {*((f"m{k}", "labeled") for k in range(4)), ("m-new", "recognized")}
    assert [r["end"] for r in seen[MIA]] == sorted((r["end"] for r in seen[MIA]), reverse=True)  # newest first
    assert "dog" not in {r["track_id"] for rows in seen.values() for r in rows}


def test_a_pet_never_seen_has_no_sightings(log):
    assert R.view(log.state, PETS, now=T + 30000)["seen"] == {MIA: [], POPPY: []}


def test_naming_a_sighting_as_a_pet_of_another_species_corrects_its_species(log):
    now = T + 30000
    assert R.label(log.state, ["dog"], MIA, PETS, now=now) == 1  # the detector said dog; the owner says it's Mia
    assert R.labels(log.state)["dog"]["species"] == "cat"
    view = R.view(log.state, PETS, now=now + 1)
    assert "dog" in {r["track_id"] for r in view["seen"][MIA]}
    assert "dog" not in {c["species"] for c in view["recognition"]}  # no dog sightings are left


def test_a_sighting_of_a_species_no_pet_here_is_listed_as_some_animal(log):
    state = store(log.state / "zoo", [row("e", 5, k, T + k, species="elephant") for k in range(3)])
    view = R.view(state, PETS, now=T + 100)
    assert [(g["species"], [s["species"] for s in g["sightings"]]) for g in view["groups"]] == [("animal", ["animal"])]
    assert view["recognition"] == []  # no check for "some animal"
    R.label(state, ["e"], MIA, PETS, now=T + 101)  # it was Mia all along
    assert R.labels(state)["e"]["species"] == "cat"
    assert [r["track_id"] for r in R.view(state, PETS, now=T + 102)["seen"][MIA]] == ["e"]


def test_an_ignored_sighting_leaves_the_list_and_never_teaches_recognition(log):
    now = T + 30000
    R.label(log.state, [f"m{k}" for k in range(4)], MIA, PETS, now=now)
    R.label(log.state, ["p0", "p1", "m-new"], R.IGNORED, PETS, now=now + 1)
    view = R.view(log.state, PETS, now=now + 2)
    listed = {s["track_id"] for g in view["groups"] for s in g["sightings"]}
    assert not listed & {"p0", "p1", "m-new"}
    assert "m-new" not in {r["track_id"] for rows in view["seen"].values() for r in rows}
    (cat,) = [c for c in view["recognition"] if c["species"] == "cat"]
    assert cat["labeled"] == 4  # Mia's four; the ignored ones aren't examples
    R.label(log.state, ["p0"], None, PETS, now=now + 3)  # Undo
    assert "p0" in listed_now(log.state, now + 4)


def listed_now(state, now):
    return {s["track_id"] for g in R.view(state, PETS, now=now)["groups"] for s in g["sightings"]}


def test_a_blocked_sighting_and_the_ones_like_it_leave_the_list_until_it_is_unblocked(log):
    now = T + 30000
    assert {"m0", "m-new", "p0"} <= listed_now(log.state, now)
    assert R.block(log.state, ["m0"], PETS, now=now) == 1
    after = listed_now(log.state, now + 1)
    assert not after & {"m0", "m1", "m2", "m3", "m-new"}  # the same animal, about 0.90 alike
    assert {"p0", "p-new", "dog"} <= after  # another cat and a dog: nothing like it
    R.block(log.state, ["m0"], PETS, active=False, now=now + 2)
    assert "m-new" in listed_now(log.state, now + 3)


def test_a_block_hides_only_what_waits_for_a_name(log):
    now = T + 30000
    R.label(log.state, [f"m{k}" for k in range(4)], MIA, PETS, now=now)
    R.label(log.state, [f"p{k}" for k in range(4)], POPPY, PETS, now=now + 1)
    R.block(log.state, ["m-new"], PETS, now=now + 2)
    seen = R.view(log.state, PETS, now=now + 3)["seen"]
    assert "m-new" in {r["track_id"] for r in seen[MIA]}  # a named pet is still recognized


@pytest.mark.parametrize(
    ("alike", "hidden"),
    [
        pytest.param(0.90, True, id="the-same-animal"),
        pytest.param(0.76, True, id="just-over"),
        pytest.param(0.74, False, id="just-under"),
        pytest.param(0.66, False, id="alike-enough-to-group-but-not-to-block"),
    ],
)
def test_a_sighting_is_blocked_only_at_high_confidence(alike, hidden):
    wall = np.zeros(DIM, np.float32)
    wall[0] = 1.0
    v = np.zeros(DIM)
    v[[0, 50]] = alike, np.sqrt(1 - alike**2)
    (s,) = R.sightings([{**row("t", 0, 1, T), "vector": v.tolist()}])
    walls = {"w": {"species": "cat", "vector": wall}}
    assert R.blocked([s], walls) == ({"t"} if hidden else set())
    assert R.blocked([s], {"w": {"species": "dog", "vector": wall}}) == set()  # another species' block


def test_the_block_threshold_sits_above_grouping_and_recognition():
    assert R.BLOCK_SIM > R.GROUP_SIM > R.MATCH_SIM


def test_ignored_and_blocked_sightings_are_listed_newest_first_for_taking_back(log):
    now = T + 30000
    R.label(log.state, ["p0"], R.IGNORED, PETS, now=now)
    R.label(log.state, ["p1"], R.IGNORED, PETS, now=now + 1)
    R.block(log.state, ["m0"], PETS, now=now + 2)
    hidden = R.hidden(log.state)
    assert [r["track_id"] for r in hidden["ignored"]] == ["p1", "p0"]
    assert [(r["track_id"], r["species"], r["cam"]) for r in hidden["blocked"]] == [("m0", "cat", "cam0")]
    assert (hidden["ignored_count"], hidden["blocked_count"], hidden["block_similarity"]) == (2, 1, R.BLOCK_SIM)
    R.label(log.state, ["p0"], None, PETS, now=now + 3)  # un-ignore
    R.block(log.state, ["m0"], PETS, active=False, now=now + 4)  # unblock
    hidden = R.hidden(log.state)
    assert [r["track_id"] for r in hidden["ignored"]] == ["p1"] and hidden["blocked"] == []


def test_a_block_is_taught_state_in_the_knowledge_store(log):
    from liberational import knowledge_cli

    R.block(log.state, ["m0"], PETS, now=T + 30000)
    (record,) = [r.record for r in knowledge.Store(log.state / "knowledge.sqlite3").query(R.BLOCKS)]
    assert record.key == "m0" and record.value["model"] == R.embedding_model()
    assert R.BLOCKS in knowledge_cli._DOCUMENTS


@pytest.mark.parametrize(
    ("tracks", "error"),
    [
        pytest.param([], ValueError, id="none"),
        pytest.param(["gone"], LookupError, id="not-recorded"),
        pytest.param([7], ValueError, id="not-an-id"),
    ],
)
def test_a_block_that_cannot_be_made_writes_nothing(log, tracks, error):
    with pytest.raises(error):
        R.block(log.state, tracks, PETS, now=T + 30000)
    assert R.blocks(log.state) == {}


def test_the_species_offered_are_the_household_ones_and_any_a_pet_already_is():
    from liberational.people import pets

    assert pets.offered([]) == ["dog", "cat", "bird"]
    assert pets.offered([{"species": "horse"}, {"species": "cat"}]) == ["dog", "cat", "bird", "horse"]


def test_the_newest_label_wins(log):
    now = T + 30000
    R.label(log.state, ["m0"], POPPY, PETS, now=now)
    R.label(log.state, ["m0"], MIA, PETS, now=now + 1)
    assert R.labels(log.state)["m0"]["label"] == MIA


@pytest.mark.parametrize(
    ("tracks", "value", "error"),
    [
        pytest.param(["m0"], "0199a3c1-0000-7000-8000-00000000dead", LookupError, id="no-such-pet"),
        pytest.param(["gone"], MIA, LookupError, id="no-longer-recorded"),
        pytest.param([], MIA, ValueError, id="nothing-to-label"),
    ],
)
def test_a_label_that_cannot_be_true_is_refused_and_nothing_is_written(log, tracks, value, error):
    with pytest.raises(error):
        R.label(log.state, tracks, value, PETS, now=T + 30000)
    assert log.labels() == []


def test_unlabeled_sightings_expire_at_the_configured_detail_window(log):
    from liberational import config

    now = T + 20 * 86400
    with config.override({"aging": {"days": 14}}):
        result = R.view(log.state, PETS, now=now)
        assert result["groups"] == []
        assert result["label_days"] == 14
        with pytest.raises(LookupError, match="no longer recorded"):
            R.label(log.state, ["m0"], MIA, PETS, now=now)
    assert log.labels() == []


@pytest.mark.parametrize("label", [MIA, R.OTHER])
def test_owner_reference_keeps_every_old_embedding_and_can_be_relabelled(log, label):
    now = T + 60 * 86400
    state = store(log.state / "reference", [row("reference", 0, k, T + k) for k in range(70)])
    owner(state, ("reference", label, T + 100))
    (reference,) = R.recorded(state, now)
    assert (reference.track_id, len(reference.vectors), R.labels(state)["reference"]["label"]) == (
        "reference",
        70,
        label,
    )
    assert R.view(state, PETS, now=now)["groups"] == []  # labeled: not waiting for a label
    assert R.label(state, ["reference"], MIA, PETS, now=now) == 1


def test_clearing_an_old_reference_does_not_resurrect_its_earlier_label(log):
    now = T + 60 * 86400
    owner(log.state, ("m0", MIA, T + 100), ("m0", "", T + 101))
    assert R.view(log.state, PETS, now=now)["groups"] == []
    with pytest.raises(LookupError, match="no longer recorded"):
        R.label(log.state, ["m0"], MIA, PETS, now=now)


def test_aging_unlabeled_detail_preserves_old_reference_recognition_and_photo_selection(log):
    R.label(log.state, [f"m{k}" for k in range(4)], MIA, PETS, now=T + 30000)
    R.label(log.state, [f"p{k}" for k in range(4)], R.OTHER, PETS, now=T + 30001)
    now = T + 60 * 86400
    store(log.state, [row("recent", 0, 30, now - 1)])
    before = R.view(log.state, PETS, now=now)
    assert before["recognition"][0]["on"] is True
    recent = next(s for g in before["groups"] for s in g["sightings"] if s["track_id"] == "recent")
    assert recent["recognized"]["label"] == MIA
    # Without the old unlabeled detail, as the store's history window drops it, owner reference tracks still answer.
    kept = [r for r in FIXTURE_ROWS if r["track_id"] not in ("m-new", "p-new", "dog")]
    aged = store(log.state / "aged", [*kept, row("recent", 0, 30, now - 1)])
    knowledge.Store(aged / "knowledge.sqlite3").put(log.labels())  # the owner's labels, kept with the state
    assert R.view(aged, PETS, now=now) == before
    reference = next(s for s in R.recorded(aged, now=now) if s.track_id == "m0")
    assert reference.shot["ts"] == T
    assert reference.shot["box"] == pytest.approx([0.1, 0.1, 0.3, 0.3])


def test_unlabeled_window_follows_the_pets_label_days_setting(log):
    from liberational import config

    with config.override({"pets": {"label_days": 7}}):
        result = R.view(log.state, PETS, now=T + 10 * 86400)
        assert result["groups"] == []
        assert result["label_days"] == 7


def test_unavailable_reference_read_never_returns_a_partial_recognition_answer(log, monkeypatch):
    def unavailable(self, stream, start, end, cameras=None):
        raise TimeoutError("segment read incomplete")

    monkeypatch.setattr(segment_log.Log, "range", unavailable)
    with pytest.raises(TimeoutError, match="segment read incomplete"):
        R.view(log.state, PETS, now=T + 30000)


def test_label_reads_before_the_evidence_lock_and_writes_under_it(log, monkeypatch):
    # The read under the lock took about half a second, so quick labels queued behind each other.
    import fcntl

    from liberational import locking

    lockfile = locking.lock_path(log.state / R.EVIDENCE_LOCK)
    lockfile.parent.mkdir(parents=True)
    original_segments, original_put = segment_log.Log.range, knowledge.Store.put

    def locked():
        with lockfile.open("a") as other:
            try:
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(other.fileno(), fcntl.LOCK_UN)
            return False

    reads, writes = [], []

    def segments(*args, **kwargs):
        reads.append(locked())
        return original_segments(*args, **kwargs)

    def put(*args, **kwargs):
        writes.append(locked())
        return original_put(*args, **kwargs)

    monkeypatch.setattr(segment_log.Log, "range", segments)
    monkeypatch.setattr(knowledge.Store, "put", put)
    assert R.label(log.state, ["m0", "m1"], MIA, PETS, now=T + 30000) == 2
    assert reads and not any(reads) and writes == [True]
    assert [r.key for r in log.labels()] == ["m0", "m1"]
    assert not locked()


def test_embeddings_are_read_again_only_for_the_hours_whose_files_changed(log, monkeypatch):
    # Every label and every pet box read every embedding ever recorded.
    original = segment_log.Log.range
    read: list[float] = []

    def counted(self, stream, start, end, cameras=None):
        read.append(start)
        return original(self, stream, start, end, cameras)

    monkeypatch.setattr(segment_log.Log, "range", counted)
    first = R.embedding_rows(log.state)
    assert {r["track_id"] for r in first} == {r["track_id"] for r in FIXTURE_ROWS}
    read.clear()
    assert len(R.embedding_rows(log.state)) == len(first) and read == []  # nothing changed: nothing read
    hour = 3600 * (T // 3600)
    store(log.state, [row("late", 0, 30, T + 5)])  # a row written late into an hour already read
    store(log.state, [row("next", 1, 31, T + 90000)])  # and a new hour
    rows = R.embedding_rows(log.state)
    assert {r["track_id"] for r in rows} == {r["track_id"] for r in FIXTURE_ROWS} | {"late", "next"}
    assert sorted(set(read)) == [hour, 3600 * ((T + 90000) // 3600)]  # those two hours only, each stream once


def test_a_label_from_before_the_reset_keeps_its_imported_embeddings(log):
    """After the reset the segment store holds none of a labeled track's embeddings: the imported references
    still make it a sighting with its label, so recognition keeps its examples."""
    state = log.state / "after-reset"
    state.mkdir()
    vectors = [row("old-mia", 0, k, T + k) for k in range(3)]
    knowledge.Store(state / "knowledge.sqlite3").put(
        knowledge.Record(
            f"019c0000-0000-7000-8000-{k:012x}",
            R.REFERENCES,
            "old-mia",
            1,
            1,
            1,
            {**r, "vector": [float(v) for v in r["vector"]], "model": R.embedding_model()},
        )
        for k, r in enumerate(vectors, start=100)
    )
    owner(state, ("old-mia", MIA, T + 10))
    (sighting,) = R.recorded(state, T + 60 * 86400)
    assert (sighting.track_id, R.labels(state)["old-mia"]["label"], len(sighting.vectors)) == ("old-mia", MIA, 3)


def after_reset(log, rows):
    """A state directory as a reset leaves it: the old one's knowledge store (labels, references), and a segment
    store holding only `rows`."""
    state = log.state / "after-reset"
    state.mkdir()
    old = knowledge.Store(log.state / "knowledge.sqlite3")
    kept = [r.record for c in (R.LABELS, R.REFERENCES) for r in old.query(c)]
    knowledge.Store(state / "knowledge.sqlite3").put(kept)
    return store(state, rows)


def references(state):
    return [r.record for r in knowledge.Store(state / "knowledge.sqlite3").query(R.REFERENCES)]


def test_labels_keep_recognition_across_a_reset_that_drops_their_embeddings(log):
    now = T + 30000
    R.label(log.state, [f"m{k}" for k in range(4)], MIA, PETS, now=now)
    R.label(log.state, [f"p{k}" for k in range(4)], POPPY, PETS, now=now + 1)
    state = after_reset(log, [row("m-new", 0, 20, T + 20000), row("p-new", 1, 21, T + 20001)])
    rows = listed(state, now + 2)
    assert (rows["m-new"]["recognized"]["label"], rows["p-new"]["recognized"]["label"]) == (MIA, POPPY)


def test_an_ignored_label_keeps_no_example_and_relabeling_keeps_one(log):
    now = T + 30000
    R.label(log.state, ["m0"], R.IGNORED, PETS, now=now)
    assert references(log.state) == []
    R.label(log.state, ["m1"], MIA, PETS, now=now + 1)
    R.label(log.state, ["m1"], R.OTHER, PETS, now=now + 2)
    assert [r.key for r in references(log.state)] == ["m1"]


def test_a_reference_of_another_embedding_space_is_no_example_and_leaves_its_label_stranded(log):
    """An earlier square-crop reference doesn't compare with tight-crop embeddings."""
    state = after_reset(log, [])
    label = {"track_id": "old-mia", "species": "cat", "cam": "cam0", "label": MIA, "sighting_start": T}
    square = R.reference(np.array(look(0, 1)), label, 1)
    square.value["model"] = "facebook/dinov3:square"
    body = {**label, "active": True, "source": "owner"}
    knowledge.Store(state / "knowledge.sqlite3").put(
        [square, knowledge.Record("019c0000-0000-7000-8000-0000000000aa", R.LABELS, "old-mia", 2, 2, 2, body)]
    )
    assert R.recorded(state, T + 60) == []
    assert [r["track_id"] for r in R.stranded(state)] == ["old-mia"]
    assert R.keep_references(state, [R.reference(np.array(look(0, 2)), label, 3)]) == 1
    assert [s.track_id for s in R.recorded(state, T + 60)] == ["old-mia"]


def test_a_sighting_embedded_in_another_space_is_not_listed_labeled_or_named(log):
    # Earlier square-crop rows were listed beside tight ones, and labeling one kept its square mean
    # as a tight reference.
    taught(log)
    store(log.state, [{**row("m-square", 0, 30, T + 20005), "model": "facebook/dinov3:square"}])  # Mia's look
    assert "m-square" not in listed(log.state, T + 30010)
    with pytest.raises(LookupError, match="no longer recorded"):
        R.label(log.state, ["m-square"], MIA, PETS, now=T + 30020)
    named = R.resolve(log.state, "cam0", T + 19990, T + 20010, ["m-new", "m-square"], PETS)
    assert set(named) == {"m-new"}


def test_a_reference_is_what_pack_knowledge_accepts():
    from liberational import knowledge_cli

    owner_label = {"track_id": "run:cat:1", "species": "cat", "cam": "cam0", "label": MIA, "sighting_start": T}
    record = R.reference(np.array(look(0, 1)), owner_label, 1)
    knowledge_cli._reference(record)  # raises on a record an export or import would refuse
    assert np.isclose(np.linalg.norm(record.value["vector"]), 1.0)


def test_stranded_labels_are_those_without_embeddings_or_a_reference_and_restoring_settles_them(log):
    state = after_reset(log, [{**row("m-new", 0, 20, T + 20000), "model": R.embedding_model()}])
    labeled = {"species": "cat", "cam": "cam0", "sighting_start": T, "active": True, "source": "owner"}
    knowledge.Store(state / "knowledge.sqlite3").put(
        knowledge.Record(f"019c0000-0000-7000-8000-{k:012x}", R.LABELS, track, k, k, k, {**labeled, **body})
        for k, (track, body) in enumerate(
            [
                ("gone-mia", {"track_id": "gone-mia", "label": MIA}),
                ("gone-other", {"track_id": "gone-other", "label": R.OTHER}),
                ("gone-ignored", {"track_id": "gone-ignored", "label": R.IGNORED}),
                ("m-new", {"track_id": "m-new", "label": MIA}),  # its embeddings are still kept
            ],
            start=1,
        )
    )
    assert sorted(r["track_id"] for r in R.stranded(state)) == ["gone-mia", "gone-other"]
    restored = [R.reference(np.array(look(0, 30)), r, 9) for r in R.stranded(state)]
    assert R.keep_references(state, restored) == 2
    assert R.stranded(state) == []
    assert R.keep_references(state, restored) == 0, "a second run writes nothing"


def test_a_one_pet_species_is_recognized_from_its_own_labels_at_a_threshold_they_set():
    check = R.gallery(visits(0, BESS, T, 6), "cat", BESS)
    # Two sightings of one animal are 0.95^2 / (0.95^2 + 0.31^2) = 0.9038 alike (`look`): every held-out nearest one.
    assert (check["on"], check["right"], check["unknown"], check["wrong"]) == (True, 6, 0, 0)
    assert check["threshold"] == pytest.approx(0.9038, abs=1e-3)


def test_a_one_pet_gallery_never_accepts_less_alike_than_its_floor():
    unalike = [(R.sightings([row(f"b{k}", k, k, T + 3600 * k, species="dog")])[0], BESS) for k in range(6)]
    check = R.gallery(unalike, "dog", BESS)
    assert check["threshold"] == R.MIN_GALLERY_SIM, "labeled sightings 0.1 alike can't lower it"
    assert (check["right"], check["unknown"]) == (0, 6)


def test_a_one_pet_gallery_needs_enough_labels_from_separate_visits():
    assert R.gallery(visits(0, BESS, T, 5), "cat", BESS)["reason"] == "needs 1 more labeled sightings"
    one_visit = R.gallery(visits(0, BESS, T, 6, gap=60.0), "cat", BESS)
    assert (one_visit["on"], one_visit["reason"]) == (False, "needs labeled sightings from more separate visits")


def test_the_only_dog_is_recognized_and_a_strange_dog_is_not(tmp_path):
    pytest.importorskip("continuum_py")
    rows = [row(f"bess{k}", 5, 30 + k, T + 3600 * k, species="dog") for k in range(6)]
    rows += [row("bess-new", 5, 40, T + 30000, species="dog"), row("stranger", 6, 41, T + 30001, species="dog")]
    state = store(tmp_path, rows)
    pets = [*PETS, {"id": BESS, "name": "Bess", "species": "dog"}]
    now = T + 40000
    R.label(state, [f"bess{k}" for k in range(6)], BESS, pets, now=now)
    view = R.view(state, pets, now=now + 1)
    (dog,) = [c for c in view["recognition"] if c["species"] == "dog"]
    assert dog["on"] is True
    sightings = {s["track_id"]: s for g in view["groups"] for s in g["sightings"]}
    assert sightings["bess-new"]["recognized"]["label"] == BESS
    assert "recognized" not in sightings["stranger"]


def test_sightings_come_from_see_actions_and_see_captions_streams_alike(tmp_path):
    pytest.importorskip("continuum_py")
    store(tmp_path, [row("actions", 0, 1, T)])
    store(tmp_path, [row("captions", 0, 2, T + 60)], segment_log.ANIMAL_GROUNDING_EMBEDDINGS)
    assert [s.track_id for s in R.recorded(tmp_path, now=T + 120)] == ["actions", "captions"]


# Review's boxes: `resolve` names a window's tracks from owner-label spans, never from all of history.


def taught(log, now=T + 30000):
    """Mia's and Poppy's four visits each labeled through the Pets page, which records each sighting's span."""
    R.label(log.state, [f"m{k}" for k in range(4)], MIA, PETS, now=now)
    R.label(log.state, [f"p{k}" for k in range(4)], POPPY, PETS, now=now + 1)


def reads(monkeypatch):
    """Every segment-store read as (stream, start, end, cameras)."""
    seen: list[tuple[str, float, float, list[str] | None]] = []
    original = segment_log.Log.range

    def spy(self, stream, start, end, cameras=None):
        seen.append((stream, start, end, cameras))
        return original(self, stream, start, end, cameras)

    monkeypatch.setattr(segment_log.Log, "range", spy)
    return seen


def test_review_names_a_window_s_tracks_by_appearance_from_the_labels_own_spans(log, monkeypatch):
    taught(log)
    got = reads(monkeypatch)
    window = (T + 19990, T + 20010)
    named = R.resolve(log.state, "cam0", *window, ["m-new", "p-new"], PETS)
    assert {t: m["label"] for t, m in named.items()} == {"m-new": MIA, "p-new": POPPY}
    assert all(start >= T and end - start <= 2.0 for _, start, end, _ in got[:-2]), "each label's own sighting span"
    assert [(start, end, cams) for _, start, end, cams in got[-2:]] == [(*window, ["cam0"])] * 2, "then the window"
    got.clear()
    R.resolve(log.state, "cam0", *window, ["m-new"], PETS)
    assert [(start, end) for _, start, end, _ in got] == [window] * 2, "the gallery is cached"


def test_review_never_names_a_track_before_the_check_is_on(log):
    R.label(log.state, ["m0", "m1"], MIA, PETS, now=T + 30000)  # the Pets page would only suggest Mia
    assert R.resolve(log.state, "cam0", T + 19990, T + 20010, ["m-new"], PETS) == {}


def test_review_s_gallery_follows_new_labels(log):
    R.label(log.state, [f"m{k}" for k in range(4)], MIA, PETS, now=T + 30000)
    assert R.resolve(log.state, "cam0", T + 19990, T + 20010, ["m-new"], PETS) == {}, "one class: off"
    R.label(log.state, [f"p{k}" for k in range(4)], POPPY, PETS, now=T + 30001)
    assert R.resolve(log.state, "cam0", T + 19990, T + 20010, ["m-new"], PETS)["m-new"]["label"] == MIA


def test_review_leaves_out_owner_labeled_tracks_and_tracks_outside_the_window(log):
    taught(log)
    R.label(log.state, ["m-new"], MIA, PETS, now=T + 30002)  # the owner's own label shows, not a recognized one
    assert R.resolve(log.state, "cam0", T + 19990, T + 20010, ["m-new", "p-new"], PETS).keys() == {"p-new"}
    assert R.resolve(log.state, "cam1", T + 19990, T + 20010, ["p-new"], PETS) == {}, "another camera's window"
    assert R.resolve(log.state, "cam0", T + 30000, T + 30060, ["p-new"], PETS) == {}, "no embeddings in the window"


def test_a_label_that_doesnt_say_where_its_sighting_was_is_no_reference_for_review(log):
    owner(
        log.state,
        *((f"m{k}", MIA, T + 30000 + k) for k in range(4)),
        *((f"p{k}", POPPY, T + 30010 + k) for k in range(4)),
    )
    assert R.view(log.state, PETS, now=T + 30100)["recognition"][0]["on"] is True, "the Pets page reads all history"
    assert R.resolve(log.state, "cam0", T + 19990, T + 20010, ["m-new"], PETS) == {}, "unknown, not guessed"


def test_review_names_the_only_dog_and_not_a_strange_one(tmp_path):
    pytest.importorskip("continuum_py")
    rows = [row(f"bess{k}", 5, 30 + k, T + 3600 * k, species="dog") for k in range(6)]
    rows += [row("bess-new", 5, 40, T + 30000, species="dog"), row("stranger", 6, 41, T + 30001, species="dog")]
    state = store(tmp_path, rows)
    pets = [*PETS, {"id": BESS, "name": "Bess", "species": "dog"}]
    R.label(state, [f"bess{k}" for k in range(6)], BESS, pets, now=T + 40000)
    named = R.resolve(state, "cam0", T + 29990, T + 30010, ["bess-new", "stranger"], pets)
    assert named.keys() == {"bess-new"} and named["bess-new"]["label"] == BESS


def test_a_pet_the_owner_said_it_isnt_is_left_out_and_the_next_class_answers():
    mia, poppy, query = seen(("m", 0, 1, T), ("p", 1, 2, T), ("q", 0, 3, T))
    assert R.recognize(query, [(mia, MIA), (poppy, POPPY)])[0] == MIA
    label, _, suggestion = R.recognize(query, [(mia, MIA), (poppy, POPPY)], exclude={MIA})
    assert (label, suggestion) == (None, POPPY), "Poppy is far from it: only a suggestion, never Mia"


def test_not_this_pet_moves_a_sightings_suggestion_on_and_taking_it_back_restores_it(log):
    now = T + 30000
    R.label(log.state, [f"m{k}" for k in range(4)], MIA, PETS, now=now)
    R.label(log.state, [f"p{k}" for k in range(4)], POPPY, PETS, now=now + 1)
    assert listed(log.state, now + 2)["m-new"]["recognized"]["label"] == MIA
    assert R.mark_not(log.state, ["m-new"], MIA, PETS, now=now + 3) == 1
    denied = listed(log.state, now + 4)["m-new"]
    assert (denied.get("recognized"), denied["suggestion"]["label"], denied["not"]) == (None, POPPY, [MIA])
    seen_mia = {r["track_id"] for r in R.view(log.state, PETS, now=now + 4)["seen"][MIA]}
    assert "m-new" not in seen_mia, "not on Mia's card either"
    R.mark_not(log.state, ["m-new"], MIA, PETS, active=False, now=now + 5)
    back = listed(log.state, now + 6)["m-new"]
    assert (back["recognized"]["label"], back["not"]) == (MIA, [])


def test_not_this_pet_on_a_sighting_labeled_as_that_pet_clears_the_label(log):
    now = T + 30000
    R.label(log.state, ["m0"], MIA, PETS, now=now)
    R.mark_not(log.state, ["m0"], MIA, PETS, now=now + 1)
    back = listed(log.state, now + 2)["m0"]
    assert (back["label"], back["not"]) == (None, [MIA]), "back in Sightings to name, and never Mia again"
    R.label(log.state, ["m1"], POPPY, PETS, now=now + 3)
    R.mark_not(log.state, ["m1"], MIA, PETS, now=now + 4)
    assert "m1" not in listed(log.state, now + 5), "a label as another pet stays"


@pytest.mark.parametrize(
    ("tracks", "pet", "error"),
    [
        pytest.param([], MIA, ValueError, id="no-sightings"),
        pytest.param(["m0"], "0199a3c1-0000-7000-8000-00000000dead", LookupError, id="no-such-pet"),
        pytest.param(["m0"], R.OTHER, LookupError, id="not-a-named-pet"),
        pytest.param(["gone"], MIA, LookupError, id="no-longer-recorded"),
    ],
)
def test_a_not_label_that_cannot_be_true_is_refused_and_nothing_is_written(log, tracks, pet, error):
    with pytest.raises(error):
        R.mark_not(log.state, tracks, pet, PETS, now=T + 30000)
    assert R.nots(log.state) == {}


def test_review_never_names_a_track_as_a_pet_the_owner_said_it_isnt(log):
    now = T + 30000
    R.label(log.state, [f"m{k}" for k in range(4)], MIA, PETS, now=now)
    R.label(log.state, [f"p{k}" for k in range(4)], POPPY, PETS, now=now + 1)
    window = (T + 19990, T + 20010)
    assert R.resolve(log.state, "cam0", *window, ["m-new"], PETS)["m-new"]["label"] == MIA
    R.mark_not(log.state, ["m-new"], MIA, PETS, now=now + 2)
    assert R.resolve(log.state, "cam0", *window, ["m-new"], PETS) == {}, "the cached gallery is read again"


def test_each_waiting_sighting_and_group_carries_the_guess_shape_faces_and_voices_share(log):
    now = T + 30000
    R.label(log.state, [f"m{k}" for k in range(4)], MIA, PETS, now=now)
    R.label(log.state, [f"p{k}" for k in range(4)], POPPY, PETS, now=now + 1)
    got = R.view(log.state, PETS, now=now + 2)
    assert got["count"] == len(got["groups"])
    row = {s["track_id"]: s for g in got["groups"] for s in g["sightings"]}["m-new"]
    assert row["guess"]["id"] == MIA and row["guess"]["name"] == "Mia"
    assert (row["guess"]["band"], row["guess"]["close"], row["not"]) == ("probable", None, [])
    assert row["guess"]["score"] == round(100 * row["recognized"]["similarity"])
    (group,) = [g for g in got["groups"] if any(s["track_id"] == "m-new" for s in g["sightings"])]
    assert group["guess"]["id"] == MIA and group["not"] == []


def _row(label, score, band, nots=()):
    guess = {"id": label, "name": label, "score": score, "band": band, "close": None} if label else None
    return {"guess": guess, "not": list(nots)}


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(
            [_row("mia", 90, "maybe"), _row("mia", 70, "maybe"), _row("poppy", 95, "probable")],
            ("poppy", 95, "probable"),
            id="one-probable-beats-more-maybes",
        ),
        pytest.param(
            [_row("mia", 90, "probable"), _row("poppy", 80, "probable"), _row("mia", 70, "probable")],
            ("mia", 80, "probable"),
            id="most-named-with-the-mean-of-their-scores",
        ),
        pytest.param(
            [_row("poppy", 60, "maybe"), _row("mia", 50, "maybe")],
            ("poppy", 60, "maybe"),
            id="a-tie-goes-to-the-first-named",
        ),
    ],
)
def test_a_groups_guess_is_what_its_sightings_name_most(rows, expected):
    guess = R.group_guess(rows, {})
    assert guess is not None
    assert (guess["id"], guess["score"], guess["band"]) == expected


def test_a_group_with_no_guessed_sighting_has_no_guess():
    assert R.group_guess([_row(None, 0, "none")], {}) is None


def test_a_groups_not_is_what_every_sighting_in_it_was_said_not_to_be(log):
    now = T + 30000
    R.mark_not(log.state, ["m0", "m1", "m2", "m3", "m-new"], MIA, PETS, now=now)  # Mia's whole group
    R.mark_not(log.state, ["m0"], POPPY, PETS, now=now + 1)
    (group,) = [
        g for g in R.view(log.state, PETS, now=now + 2)["groups"] if g["sightings"][0]["track_id"].startswith("m")
    ]
    assert group["not"] == [MIA]


def _groups_one_at_a_time(found):
    """Reference groups() implementation: each centroid compared alone, which the vectorized version must match."""
    out = []
    for s in found:
        best, sim = None, R.GROUP_SIM
        for k, (species, _, centroid) in enumerate(out):
            if species != s.species:
                continue
            value = float(s.mean @ (centroid / max(float(np.linalg.norm(centroid)), 1e-9)))
            if value >= sim:
                best, sim = k, value
        if best is None:
            out.append((s.species, [s], s.mean.copy()))
        else:
            species, members, centroid = out[best]
            out[best] = (species, [*members, s], centroid + s.mean)
    return sorted((m for _, m, _ in out), key=lambda m: (-len(m), m[0].start))


_unit = st.lists(st.floats(-1, 1, allow_nan=False, width=32), min_size=4, max_size=4).filter(
    lambda v: sum(x * x for x in v) > 0.01
)


@settings(max_examples=150, deadline=None)
@given(st.lists(st.tuples(st.sampled_from(["cat", "dog"]), _unit, st.integers(0, 3)), min_size=1, max_size=40))
def test_groups_match_comparing_each_centroid_alone(drawn):
    found = []
    for i, (species, base, jitter) in enumerate(drawn):
        s = R.Sighting(f"t{i}", species, "cam0", float(i), float(i))
        v = np.asarray(base, dtype=np.float32) + np.float32(0.05 * jitter)
        s.vectors.append(v / np.linalg.norm(v))
        found.append(s)
    assert [[s.track_id for s in g] for g in R.groups(found)] == [
        [s.track_id for s in g] for g in _groups_one_at_a_time(found)
    ]


def test_a_sighting_exactly_as_like_two_groups_joins_the_later_one():
    half = np.radians(25)  # the two are 50 degrees apart (0.64, two groups); the third is 25 from each (0.91)
    found = []
    for i, v in enumerate([(np.cos(half), -np.sin(half)), (np.cos(half), np.sin(half)), (1.0, 0.0)]):
        s = R.Sighting(f"t{i}", "cat", "cam0", float(i), float(i))
        s.vectors.append(np.asarray(v, dtype=np.float32))
        found.append(s)
    assert [[s.track_id for s in g] for g in R.groups(found)] == [["t1", "t2"], ["t0"]]


def test_a_label_keeps_its_sightings_saved_photo_with_the_taught_state(log):
    from liberational.people import face_shots, pet_shots

    face_shots.write(log.state, "m0", "3", b"photo of m0", pet_shots.KIND)
    R.label(log.state, ["m0", "m1"], MIA, PETS, now=T + 30000)
    assert R.photos(log.state) == {"m0": b"photo of m0"}  # m1 had no saved photo: a label without one
    R.label(log.state, ["m0"], POPPY, PETS, now=T + 30001)  # relabeled: its example and photo are already kept
    assert len(knowledge.Store(log.state / "knowledge.sqlite3").query(R.PHOTOS)) == 1
