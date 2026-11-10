"""The contact card's clip labels (memory/clip_labels.py): subject-less, present tense, no first person, no name,
from the camera caption through a cached rewrite. Made-up captions and a fake model; never the real data/."""

import pytest

from liberational.memory import clip_labels as cl


@pytest.mark.parametrize(
    ("reply", "want"),
    [
        pytest.param(
            "sitting on a chair, hand over mouth, glasses on",
            "Sitting on a chair, hand over mouth, glasses on.",
            id="kept-capitalized-full-stop",
        ),
        pytest.param('Label: "Pouring coffee at the counter."', "Pouring coffee at the counter.", id="lead-and-quotes"),
        pytest.param("I sat on a chair, hand over my mouth, glasses on.", None, id="first-person-journal-style"),
        pytest.param("Sitting with my glasses on.", None, id="my"),
        pytest.param("We're watching TV.", None, id="we"),
        pytest.param("Sitting, hand over his mouth.", None, id="pronoun"),
        pytest.param("Sitting, hand over her mouth.", None, id="her"),
        pytest.param("Man seated, hand over mouth, glasses on.", None, id="bare-man"),
        pytest.param("A woman pouring coffee.", None, id="a-woman"),
        pytest.param("Someone sitting on the couch.", None, id="someone"),
        pytest.param("Foreground: man, seated.", None, id="caption-lead-then-man"),
        pytest.param("Foreground: seated, hand over mouth.", "Seated, hand over mouth.", id="caption-lead-goes"),
        pytest.param("Two people talking, TV on.", "Two people talking, TV on.", id="the-scene-stays"),
        pytest.param("\u201cSitting on a chair.\u201d", "Sitting on a chair.", id="curly-quotes"),
        pytest.param("**Sitting on a chair.**", "Sitting on a chair.", id="markdown"),
        pytest.param("Sitting on a chair.\n\nNote: no person visible.", "Sitting on a chair.", id="first-line-only"),
        pytest.param("The person is sitting on the couch.", None, id="the-person"),
        pytest.param("Dana sitting on the couch.", None, id="their-name"),
        pytest.param("Sitting next to dana's guitar.", None, id="their-name-any-case"),
        pytest.param("Sitting on the couch with a guitar and a sewing machine nearby.", None, id="over-60-chars"),
        pytest.param("", None, id="empty"),
    ],
)
def test_clean_keeps_only_subject_less_labels(reply, want):
    assert cl.clean(reply, "Dana Smith") == want


def history(vision, rows):
    for ts, cam, text, *fields in rows:
        vision.caption(ts, cam, text, **(fields[0] if fields else {}))


def test_the_caption_is_the_real_one_nearest_the_middle_of_the_clip(tmp_path, vision):
    history(
        vision,
        [
            (100, "cam0", "Foreground: man, seated, reading."),
            # nearest, but a repeat for an unchanged view: says nothing new
            (150, "cam0", "Foreground: man, seated, reading.", dict(unchanged=True, since=100)),
            (148, "cam0", None),  # warming up
            (130, "cam0", "Foreground: man, standing, pouring coffee."),
            (150, "cam1", "Another camera."),
            (300, "cam0", "Long after the clip."),
        ],
    )
    assert cl.caption("cam0", 140, 160, tmp_path) == "Foreground: man, standing, pouring coffee."
    assert cl.caption("cam0", 500, 510, tmp_path) is None


def test_a_broker_that_cannot_be_reached_gives_no_caption(tmp_path, monkeypatch):
    class Down:
        def captions(self, start, end, cams=()):
            raise ConnectionRefusedError

    monkeypatch.setattr(cl.vision_log, "at", lambda state=None: Down())
    assert cl.caption("cam0", 140, 160, tmp_path) is None


@pytest.fixture(autouse=True)
def nothing_kept():
    cl._kept.clear()
    yield
    cl._kept.clear()


def labels(tmp_path, reply, clock=None):
    calls = []

    def rewrite(text):
        calls.append(text)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return cl.Labels(tmp_path, rewrite, background=False, clock=clock or (lambda: 1000.0)), calls


def test_the_card_never_waits_and_gets_the_label_once_written(tmp_path, vision):
    history(vision, [(150, "cam0", "Foreground: man, seated, hand over mouth, wearing glasses.")])
    lab, calls = labels(tmp_path, "Sitting, hand over mouth, glasses on.")
    assert lab.get("cam0", 140, 160, "Dana") is None  # not written yet: queued, no model call made here
    assert calls == []
    lab.drain()
    assert calls == ["Foreground: man, seated, hand over mouth, wearing glasses."]
    assert lab.get("cam0", 140, 160, "Dana") == "Sitting, hand over mouth, glasses on."
    lab.drain()
    assert len(calls) == 1  # cached per clip


def test_a_first_person_rewrite_is_never_shown_or_retried(tmp_path, vision):
    history(vision, [(150, "cam0", "Foreground: man, seated.")])
    lab, calls = labels(tmp_path, "I sat on a chair, hand over my mouth, glasses on.")
    lab.get("cam0", 140, 160, "Dana")
    lab.drain()
    assert lab.get("cam0", 140, 160, "Dana") is None
    lab.drain()
    assert len(calls) == 1


def test_a_new_caption_writes_a_new_label(tmp_path, vision):
    history(vision, [(150, "cam0", "Foreground: man, seated.")])
    lab, calls = labels(tmp_path, "Sitting.")
    lab.get("cam0", 140, 160)
    lab.drain()
    vision.records[cl.vision_log.CAPTIONS][0]["caption"] = "Foreground: man, standing."
    cl._kept.clear()  # the settled clip's kept lookup has expired (KEEP_S)
    assert lab.get("cam0", 140, 160) is None
    lab.drain()
    assert calls == ["Foreground: man, seated.", "Foreground: man, standing."]


def test_a_model_that_is_down_is_tried_again_later(tmp_path, vision):
    now = [1000.0]
    history(vision, [(150, "cam0", "Foreground: man, seated.")])
    lab, calls = labels(tmp_path, ConnectionError("ollama down"), clock=lambda: now[0])
    lab.get("cam0", 140, 160)
    lab.drain()
    assert lab.get("cam0", 140, 160) is None
    lab.drain()
    assert len(calls) == 1  # not on every card load
    now[0] += cl.RETRY_S + 1
    lab.get("cam0", 140, 160)
    lab.drain()
    assert len(calls) == 2


def test_the_background_writer_labels_queued_clips_and_stops_when_idle(tmp_path, vision):
    history(vision, [(150, "cam0", "Foreground: man, seated."), (450, "cam0", "Foreground: man, standing.")])
    lab = cl.Labels(tmp_path, lambda text: "Sitting." if "seated" in text else "Standing.", idle_s=0.05)
    assert lab.get("cam0", 140, 160) is None
    assert lab.get("cam0", 440, 460) is None
    worker = lab.worker
    assert worker is not None
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert lab.worker is None  # gone, so the next clip queued starts another
    assert (lab.get("cam0", 140, 160), lab.get("cam0", 440, 460)) == ("Sitting.", "Standing.")
    assert lab.queued == set()


def test_a_name_given_since_hides_a_label_that_has_it(tmp_path, vision):
    history(vision, [(150, "cam0", "Foreground: man, seated.")])
    lab, _ = labels(tmp_path, "Sitting by Sam's desk.")
    lab.get("cam0", 140, 160, "Dana")
    lab.drain()
    assert lab.get("cam0", 140, 160, "Dana") == "Sitting by Sam's desk."
    assert lab.get("cam0", 140, 160, "Sam") is None  # checked against the name each time it's shown


def test_the_cache_lets_go_of_the_oldest_label_past_max_cached(tmp_path, vision, monkeypatch):
    monkeypatch.setattr(cl, "MAX_CACHED", 2)
    history(vision, [(150, "cam0", "Seated."), (450, "cam0", "Standing."), (750, "cam0", "Pouring.")])
    lab, calls = labels(tmp_path, "Sitting.")
    for start in (140, 440, 740):
        lab.get("cam0", start, start + 20)
        lab.drain()
    assert lab.get("cam0", 440, 460) == "Sitting."
    assert lab.get("cam0", 740, 760) == "Sitting."
    assert lab.get("cam0", 140, 160) is None  # the oldest went: queued to be written again
    lab.drain()
    assert calls == ["Seated.", "Standing.", "Pouring.", "Seated."]


def test_a_settled_clips_caption_is_read_once_for_keep_s_and_a_recent_clip_every_time(tmp_path, monkeypatch):
    reads = []

    class Counting:
        def captions(self, start, end, cams=()):
            reads.append(start)
            return [{"id": "c1", "cam": "cam0", "ts": (start + end) / 2, "caption": "Foreground: man, seated."}]

    now = [10_000.0]
    monkeypatch.setattr(cl.vision_log, "at", lambda state=None: Counting())
    monkeypatch.setattr(cl.vision_log, "is_real", lambda r: True)
    monkeypatch.setattr(cl.time, "time", lambda: now[0])
    settled = now[0] - cl.WINDOW_S - cl.SETTLED_S - 10
    assert cl.caption("cam0", settled - 5, settled, tmp_path) == "Foreground: man, seated."
    assert cl.caption("cam0", settled - 5, settled, tmp_path) == "Foreground: man, seated."
    assert len(reads) == 1, "kept"
    now[0] += cl.KEEP_S
    cl.caption("cam0", settled - 5, settled, tmp_path)
    assert len(reads) == 2, "read again once KEEP_S has passed"
    recent = now[0] - 5
    cl.caption("cam0", recent - 5, recent, tmp_path)
    cl.caption("cam0", recent - 5, recent, tmp_path)
    assert len(reads) == 4, "a clip whose captions may still come is read every time"
