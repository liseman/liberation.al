"""The owner's activity list (memory/activity_kinds.py): add and delete, the starting kinds carrying over, the
classes a new one is read from, what it learns from answers, and deleted kinds leaving every read."""

import pytest

import activity_annotations
from liberational import segment_log
from liberational.memory import activity_kinds, events
from liberational.people import registry
from liberational.see import action_clips

T = 1_800_000_000.0
NTU_MODEL = "stgcn++-ntu120-j"


def live(state) -> list[str]:
    return [k.kind for k in activity_kinds.kinds(state)]


def test_the_starting_kinds_keep_their_slugs_and_classes_without_creating_the_store(tmp_path):
    assert {k.kind: (k.ntu, k.kinetics) for k in activity_kinds.kinds(tmp_path)} == {
        "crossing": ((), ()),
        "sit_down": (("sitting down",), ()),
        "stand_up": (("standing up",), ()),
        "pick_up": (("pick up",), ()),
        "put_down": (("drop",), ()),
        "fall": (("falling",), ()),
        "drink": (("drink water",), ()),
        "eat": (("eat meal or snack",), ()),
        "phone_call": (("make a phone call",), ()),
        "phone_use": (("playing with phone or tablet",), ("looking at phone", "texting")),
        "read": ((), ("reading book", "reading newspaper")),
        "watch_tv": ((), ("watching tv",)),
        "sleep": ((), ("sleeping",)),
    }
    assert not registry.knowledge_path(tmp_path).exists()


def test_an_added_activity_joins_the_list_with_the_classes_its_name_matches(tmp_path):
    made = activity_kinds.add(tmp_path, "  Squats ", "deep knee bends")
    assert (made.kind, made.name, made.description) == ("squats", "Squats", "deep knee bends")
    assert (made.ntu, made.kinetics) == (("squat down",), ("squat",))
    # The starting kinds are written with it, so they stay in the list.
    assert live(tmp_path)[:2] == ["crossing", "sit_down"]
    assert live(tmp_path)[-1] == "squats"


@pytest.mark.parametrize(
    ("name", "description", "ntu", "kinetics"),
    [
        pytest.param("Vacuuming", "", (), ("vacuuming floor",), id="ing-form"),
        pytest.param("Watch TV", "", (), ("watching tv",), id="no-loose-match-beside-a-full-one"),
        pytest.param("Dishes", "washing dishes at the sink", (), ("washing dishes",), id="description"),
        pytest.param("Stretching", "", ("stretch oneself",), ("stretching arm", "stretching leg"), id="loose-match"),
        pytest.param("Zorbing", "", (), (), id="nothing-matches"),
    ],
)
def test_a_new_activitys_classes_come_from_its_words(name, description, ntu, kinetics):
    assert activity_kinds.match(name, description) == {"ntu": ntu, "kinetics": kinetics}


@pytest.mark.parametrize(
    ("name", "error"),
    [
        pytest.param("", "1 to 60", id="blank"),
        pytest.param("x" * 61, "1 to 60", id="too-long"),
        pytest.param("!!", "letter or a digit", id="no-letters"),
        pytest.param("sat DOWN", "already an activity called Sat down", id="same-name"),
        pytest.param("Sit down", "already an activity called Sat down", id="same-kind"),
    ],
)
def test_a_bad_or_taken_name_is_refused_and_nothing_is_written(tmp_path, name, error):
    with pytest.raises(ValueError, match=error):
        activity_kinds.add(tmp_path, name)
    assert not registry.knowledge_path(tmp_path).exists()


def test_delete_takes_an_activity_out_and_restore_brings_it_back(tmp_path):
    activity_kinds.delete(tmp_path, "sit_down")
    assert "sit_down" not in live(tmp_path)
    assert activity_kinds.deleted(tmp_path) == {"sit_down"}
    assert "sit_down" not in activity_kinds.model_kinds(tmp_path)
    activity_kinds.restore(tmp_path, "sit_down")
    assert "sit_down" in live(tmp_path)
    assert activity_kinds.deleted(tmp_path) == frozenset()


def test_deleting_an_unknown_activity_is_a_lookup_error(tmp_path):
    with pytest.raises(LookupError):
        activity_kinds.delete(tmp_path, "juggling")


def test_adding_a_deleted_activitys_name_again_brings_it_back(tmp_path):
    activity_kinds.add(tmp_path, "Squats")
    activity_kinds.delete(tmp_path, "squats")
    activity_kinds.add(tmp_path, "squats", "again")
    assert [(k.kind, k.description) for k in activity_kinds.kinds(tmp_path) if k.kind == "squats"] == [
        ("squats", "again")
    ]


def span(vision, kind: str, at: float, track: str = "7") -> dict:
    return vision.add(
        segment_log.ACTION_SPANS, ts_start=at, ts_end=at + 10, cam="cam0", kind=kind, track_id=track, candidate=True
    )


def window(vision, at: float, model: str, classes: list[int], scores: list[float], track: str = "7") -> None:
    vision.add(
        segment_log.ACTION_WINDOWS,
        ts_start=at,
        ts_end=at + 3,
        cam="cam0",
        track_id=track,
        classes=classes,
        scores=scores,
        model=model,
    )


def test_a_deleted_activitys_clips_and_answers_leave_the_reads_and_come_back_on_restore(tmp_path, vision):
    sat, stood = span(vision, "sit_down", T), span(vision, "stand_up", T + 20)
    activity_annotations.rate(tmp_path, sat, "right", stream=segment_log.ACTION_SPANS)
    activity_kinds.delete(tmp_path, "sit_down")
    listing = events.detected(tmp_path, T, T + 60)
    assert [i["id"] for i in listing["items"]] == [stood["id"]]
    assert "sit_down" not in {row["kind"] for row in listing["promotion"]}
    assert events.kind_answers(tmp_path)[0] == {"sit_down": 1}  # kept, for a restore
    activity_kinds.restore(tmp_path, "sit_down")
    listing = events.detected(tmp_path, T, T + 60)
    assert [i["id"] for i in listing["items"]] == [sat["id"], stood["id"]]
    assert [i["feedback"] for i in listing["items"]] == ["right", None]


def test_an_answer_keeps_what_the_models_saw_in_the_clip(tmp_path, vision):
    sat = span(vision, "sit_down", T)
    squat, sit = 79, 7  # NTU-120's "squat down" and "sitting down"
    window(vision, T, NTU_MODEL, [squat, sit], [0.6, 0.2])
    window(vision, T + 5, NTU_MODEL, [sit, squat], [0.5, 0.4])
    window(vision, T + 5, action_clips.MODEL, [492], [0.8])  # Kinetics-600's "squat"
    window(vision, T + 5, NTU_MODEL, [0], [0.9], track="8")  # another person's
    events.feedback_activity(tmp_path, events.FeedbackActivity(sat["id"], T, "right"))
    (value,) = activity_annotations.values(tmp_path, events.FEEDBACK)
    assert value["classes"] == {
        "ntu": [["squat down", 0.5], ["sitting down", 0.35]],
        "kinetics": [["squat", 0.8]],
    }


def answered(tmp_path, vision, kind: str, verdicts: list[str], top: int = 79) -> None:
    """Clips of `kind` whose model saw `top` (NTU-120's "squat down" by default), each answered."""
    for i, verdict in enumerate(verdicts):
        at = T + 100 * i
        clip = span(vision, kind, at, track=str(i))
        window(vision, at, NTU_MODEL, [top], [0.7], track=str(i))
        events.feedback_activity(tmp_path, events.FeedbackActivity(clip["id"], at, verdict))


@pytest.mark.parametrize(
    ("verdicts", "learned"),
    [
        pytest.param(["right"] * 3, {"sit_down": {"ntu": ("squat down",), "kinetics": ()}}, id="three-yes"),
        pytest.param(["right"] * 2, {}, id="two-yes"),
        pytest.param(["right"] * 3 + ["wrong"] * 2, {}, id="too-many-no"),
    ],
)
def test_an_activity_learns_a_class_that_tops_enough_of_its_yes_clips(tmp_path, vision, verdicts, learned):
    answered(tmp_path, vision, "sit_down", verdicts)
    assert activity_kinds.learned(tmp_path) == learned


def test_clips_called_by_a_new_activitys_name_teach_it_its_first_classes(tmp_path, vision):
    activity_kinds.add(tmp_path, "Zorbing")  # no class name matches it
    for i in range(3):
        clip = span(vision, "stand_up", T + 100 * i, track=str(i))
        window(vision, T + 100 * i, NTU_MODEL, [101], [0.7], track=str(i))  # NTU-120's "side kick"
        events.name_activity(tmp_path, events.NameActivity(clip["id"], T + 100 * i, "zorbing"))
    zorbing = next(k for k in activity_kinds.kinds(tmp_path) if k.kind == "zorbing")
    assert (zorbing.ntu, zorbing.kinetics) == ((), ())
    taught = activity_kinds.learned(tmp_path)
    assert taught == {"zorbing": {"ntu": ("side kick",), "kinetics": ()}}
    assert activity_kinds.indices([zorbing], taught, "ntu") == {"zorbing": (101,)}


def test_a_deleted_activity_learns_nothing(tmp_path, vision):
    answered(tmp_path, vision, "sit_down", ["right"] * 3)
    activity_kinds.delete(tmp_path, "sit_down")
    assert activity_kinds.learned(tmp_path) == {}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_see_actions_rereads_the_list_each_reload_and_keeps_the_last_when_a_read_fails(tmp_path, monkeypatch):
    clock = Clock()
    view = activity_kinds.Live(tmp_path, clock)
    assert "sit_down" in view.ntu()
    activity_kinds.delete(tmp_path, "sit_down")
    activity_kinds.add(tmp_path, "Squats")
    clock.now = activity_kinds.RELOAD_S - 1
    assert "sit_down" in view.ntu()  # not reread yet
    clock.now = activity_kinds.RELOAD_S
    assert ("sit_down" in view.ntu(), view.ntu()["squats"]) == (False, (79,))
    assert view.kinetics()["squats"] == (492,)

    def broken(state):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(activity_kinds, "kinds", broken)
    clock.now = 2 * activity_kinds.RELOAD_S
    assert view.ntu()["squats"] == (79,)
