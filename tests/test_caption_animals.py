from typing import Any, cast

import numpy as np
import pytest

from liberational import segment_log
from liberational.people import pet_recognition
from liberational.see import caption_animals
from liberational.see.caption_animals import CaptionAnimals
from liberational.see.caption_worker import CaptionWorker
from liberational.see.captions import CaptionPipeline

FRAME = np.zeros((540, 960, 3), np.uint8)
DOG = '[{"label": "dog", "bbox_2d": [400, 500, 500, 700]}]'  # 96 x 108 px of FRAME
TWO_CATS = '[{"label": "cat", "bbox_2d": [100, 100, 200, 300]}, {"label": "cat", "bbox_2d": [600, 100, 700, 300]}]'


class Find:
    """The VLM's animal grounding: the given answers in turn, then none."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, frame: np.ndarray) -> str:
        self.calls += 1
        return self.answers.pop(0) if self.answers else "[]"


class Embed:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.calls = fail, 0

    def __call__(self, frame: np.ndarray, box: Any) -> np.ndarray:
        self.calls += 1
        if self.fail:
            raise RuntimeError("no GPU")
        v = np.zeros(384, np.float32)
        v[0] = 1.0
        return v


@pytest.mark.parametrize(
    ("label", "species"),
    [
        ("dog", "dog"),
        ("small white dog", "dog"),
        ("kittens", "cat"),
        ("Puppy", "dog"),
        ("dog bed", None),
        ("cat tree", None),
        ("guitar", None),
        ("", None),
        ("zebra", None),  # not a pet here (#2035): no sighting
        ("a brown horse", None),
    ],
)
def test_a_grounded_label_names_a_species_by_its_last_word(label: str, species: str | None) -> None:
    assert caption_animals.species_of(label) == species


@pytest.mark.parametrize(
    ("caption", "named"),
    [
        ("One person on the couch. Dog on floor near TV.", True),
        ("Two cats asleep on the rug.", True),
        ("A pet bowl by the door.", True),
        ("Bess lies by the sofa.", True),
        ("Moe Meerkat on the windowsill.", True),
        ("Mae sits at the desk.", False),  # part of a name is a person as often as a pet
        ("Guitars hang on the wall; TV off.", False),
        ("A catalog on the table.", False),
        ("A poster of a giraffe on the wall.", False),  # not a household animal: no look
    ],
)
def test_a_caption_names_an_animal_by_a_species_word_or_a_whole_pet_name(caption: str, named: bool) -> None:
    assert caption_animals.names_animal(caption, ["Bess", "Moe Meerkat"]) is named


def test_a_caption_naming_an_animal_asks_for_its_box_at_most_every_interval() -> None:
    find = Find(DOG, DOG, DOG)
    a = CaptionAnimals("cam0", find, None, every_s=30.0)
    first = a.look(FRAME, 100.0, "A dog on the rug.", None)
    assert [(f.species, f.box) for f in first] == [("dog", [0.4, 0.5, 0.5, 0.7])]
    assert a.look(FRAME, 129.0, "A dog on the rug.", None) == []
    assert find.calls == 1
    assert len(a.look(FRAME, 130.0, "A dog on the rug.", None)) == 1
    assert find.calls == 2


def test_a_caption_without_an_animal_never_asks() -> None:
    find = Find(DOG)
    a = CaptionAnimals("cam0", find, None)
    assert a.look(FRAME, 100.0, "Guitars on the wall.", None) == []
    assert a.look(FRAME, 200.0, "", None) == [], "a frame with no fresh caption"
    assert find.calls == 0


def test_a_scheduled_groundings_animals_are_used_without_asking_again() -> None:
    find = Find(DOG)
    a = CaptionAnimals("cam0", find, None)
    grounded = [{"label": "guitar", "box": [0, 0, 100, 400]}, {"label": "cat", "box": [100, 100, 300, 400]}]
    found = a.look(FRAME, 100.0, "A cat by the guitars.", cast(Any, grounded))
    assert [(f.species, f.box) for f in found] == [("cat", [0.1, 0.1, 0.3, 0.4])]
    assert find.calls == 0


def test_a_taught_pets_name_in_a_caption_asks() -> None:
    find = Find(DOG)
    a = CaptionAnimals("cam0", find, None, lambda: ["Bess"])
    assert len(a.look(FRAME, 100.0, "Bess on the couch.", None)) == 1


def test_an_answer_with_no_animal_or_a_failed_one_gives_nothing_and_a_later_caption_asks_again(caplog) -> None:
    find = Find('[{"label": "dog bed", "bbox_2d": [100, 100, 300, 300]}]', "not json")
    a = CaptionAnimals("cam0", find, None, every_s=30.0)
    assert a.look(FRAME, 100.0, "A dog bed by the TV.", None) == []
    assert a.look(FRAME, 130.0, "A dog bed by the TV.", None) == []
    assert "animal grounding failed" in caplog.text
    assert find.calls == 2


def test_a_lone_animal_keeps_its_track_through_a_visit_and_starts_another_after_it() -> None:
    a = CaptionAnimals("cam0", Find(DOG, DOG, DOG), None, every_s=0.0)
    first = a.look(FRAME, 100.0, "dog", None)[0].track_id
    assert a.look(FRAME, 220.0, "dog", None)[0].track_id == first, "within SAME_VISIT_S of the last box"
    assert a.look(FRAME, 341.0, "dog", None)[0].track_id != first


def test_two_animals_of_a_species_at_once_never_share_a_track() -> None:
    a = CaptionAnimals(
        "cam0", Find(DOG, TWO_CATS, '[{"label": "cat", "bbox_2d": [100, 100, 200, 300]}]'), None, every_s=0
    )
    dog = a.look(FRAME, 100.0, "a dog", None)[0].track_id
    cats = [f.track_id for f in a.look(FRAME, 101.0, "two cats", None)]
    alone = a.look(FRAME, 102.0, "a cat", None)[0].track_id
    assert len({dog, *cats, alone}) == 4, "a lone cat after two can't tell which it is"


def test_only_boxes_big_enough_are_embedded_and_a_failing_embedder_stops_embeddings_only(caplog) -> None:
    small = '[{"label": "cat", "bbox_2d": [100, 100, 140, 150]}]'  # 38 x 27 px of FRAME
    embed = Embed()
    a = CaptionAnimals("cam0", Find(small, DOG), embed, every_s=0.0)
    assert a.look(FRAME, 100.0, "a cat", None)[0].vector is None
    assert embed.calls == 0, "no animal box big enough: the embedder is never loaded"
    assert a.look(FRAME, 101.0, "a dog", None)[0].vector is not None

    broken = Embed(fail=True)
    b = CaptionAnimals("cam1", Find(DOG, DOG), broken, every_s=0.0)
    assert [f.vector for f in b.look(FRAME, 100.0, "a dog", None)] == [None]
    assert len(b.look(FRAME, 101.0, "a dog", None)) == 1, "the sighting stays"
    assert broken.calls == 1
    assert "pet embeddings failed" in caplog.text


class VLM:
    name = "fake"

    def caption(self, img: np.ndarray, memory: Any) -> str:
        return "One person on the couch. Dog on the floor near the TV."


def test_the_pipeline_gives_a_frames_animals_with_its_caption() -> None:
    animals = CaptionAnimals("cam0", Find(DOG), None)
    out = CaptionPipeline("cam0", VLM(), animals=animals).feed(FRAME, 100.0, 1)
    assert out.caption is not None and out.caption["caption"].startswith("One person")
    assert [f.species for f in out.animals] == ["dog"]


class Segments:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, Any]]] = {}

    def append(self, stream: str, cam: str, records: list[dict[str, Any]]) -> None:
        segment_log.columns(stream, records)  # every row fits its stream's schema
        self.rows.setdefault(stream, []).extend({**r, "cam": cam} for r in records)

    def sync(self) -> None:
        pass


def test_the_worker_writes_sightings_that_pet_recognition_reads() -> None:
    segments = Segments()
    worker = CaptionWorker(
        cast(Any, None), [], cast(Any, None), model="ollama", segments=cast(Any, segments), embedding_model="dino"
    )
    out = CaptionPipeline("cam0", VLM(), animals=CaptionAnimals("cam0", Find(DOG), Embed())).feed(FRAME, 100.0, 1)
    worker._handle("01a1a4cc-0000-7000-8000-000000000001", 100_000_000_000, out, None)
    assert {segment_log.ANIMAL_GROUNDINGS, segment_log.ANIMAL_GROUNDING_EMBEDDINGS} <= set(segments.rows)
    assert not {segment_log.ANIMAL_DETECTIONS, segment_log.ANIMAL_EMBEDDINGS} & set(segments.rows), (
        "never see-actions' streams: one writer per stream and camera (#2062)"
    )
    [detection] = segments.rows[segment_log.ANIMAL_GROUNDINGS]
    assert (detection["species"], detection["verified"], detection["model"]) == ("dog", True, "grounding:ollama")
    assert detection["box"] == [0.4, 0.5, 0.5, 0.7, 1.0]
    [embedding] = segments.rows[segment_log.ANIMAL_GROUNDING_EMBEDDINGS]
    records = segment_log.records(
        segment_log.ANIMAL_GROUNDING_EMBEDDINGS,
        {
            **segment_log.columns(segment_log.ANIMAL_GROUNDING_EMBEDDINGS, [embedding]),
            "camera": ["cam0"],
            "row_seq": [0],
        },
    )
    [sighting] = pet_recognition.sightings(records)
    assert (sighting.species, sighting.cam, sighting.start) == ("dog", "cam0", 100.0)
    assert sighting.track_id == detection["track_id"]
    assert sighting.shot["box"] == pytest.approx([0.4, 0.5, 0.5, 0.7])
