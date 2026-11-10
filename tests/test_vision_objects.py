"""Notable objects from the caption model's grounding (vision/objects.py), and storing them in camera history."""

import json
import time

import numpy as np

from liberational.vision import objects

GOOD = """[
  {"label": "tv", "bbox_2d": [540, 28, 745, 240]},
  {"label": "Guitar", "bbox_2d": [908, 0, 970, 130]},
  {"label": "couch", "bbox_2d": [268, 207, 480, 427]}
]"""
IMG = np.zeros((4, 4, 3), np.uint8)
KEY = ("cam0", 0.0)


def test_parse_reads_the_models_json():
    items = objects.parse(GOOD)
    assert [it["label"] for it in items] == ["tv", "guitar", "couch"]  # labels lowercased
    assert items[0]["box"] == [540, 28, 745, 240]


def test_parse_tolerates_fences_chatter_and_a_cut_off_tail():
    text = 'Here you go:\n```json\n[{"label": "lamp", "bbox": [10, 20, 110, 220]},\n {"label": "chair", "bbox_2d": [5'
    assert objects.parse(text) == [{"label": "lamp", "box": [10, 20, 110, 220]}]


def test_parse_drops_malformed_off_scale_tiny_whole_image_and_repeated_boxes():
    text = json.dumps(
        [
            {"label": "a", "bbox_2d": [1, 2, 3]},  # three numbers
            {"label": "", "bbox_2d": [0, 0, 100, 100]},  # no label
            {"label": "b", "bbox_2d": ["x", 0, 100, 100]},  # not numbers
            {"label": "c", "bbox_2d": [100, 100, 1400, 900]},  # pixels, not the 0 to 1000 scale
            {"label": "d", "bbox_2d": [100, 100, 105, 300]},  # a sliver
            {"label": "e", "bbox_2d": [0, 0, 1000, 1000]},  # the whole scene
            {"label": "f", "bbox_2d": [300, 400, 200, 100]},  # corners swapped: kept, put in order
            {"label": "f", "bbox_2d": [200, 100, 301, 400]},  # the same box again
        ]
    )
    assert objects.parse(text) == [{"label": "f", "box": [200, 100, 300, 400]}]
    assert objects.parse("I can't see anything.") == []
    assert objects.parse(None) == []


def test_parse_keeps_variety_when_the_model_repeats_one_kind():
    wall = [{"label": "guitar", "bbox_2d": [100 * i, 0, 100 * i + 60, 200]} for i in range(6)]
    text = json.dumps(
        [{"label": "tv", "bbox_2d": [0, 300, 400, 600]}, *wall, {"label": "lamp", "bbox_2d": [700, 300, 760, 900]}]
    )
    assert [it["label"] for it in objects.parse(text)] == ["tv", "guitar", "guitar", "lamp"]


def test_converts_the_0_to_1000_scale_to_pixels():
    items = [{"label": "tv", "box": [500, 250, 750, 500]}]
    assert objects.to_pixels(items, (720, 1280), ts=5.0) == [{"label": "tv", "bbox": [640, 180, 960, 360], "ts": 5.0}]
    assert objects.to_pixels(items, (252, 448))[0]["bbox"] == [224, 63, 336, 126]  # the tile
    assert objects.to_pixels(items, (252, 448), norm=True)[0]["bbox_norm"] == [0.5, 0.25, 0.75, 0.5]
    assert objects.to_pixels(items, (252, 448), norm=False)[0]["bbox_norm"] is None  # a warped tile


def wait_until(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            return False
        time.sleep(0.005)
    return True


def test_draws_labelled_boxes_and_caps_them():
    out = objects.draw(np.zeros((720, 1280, 3), np.uint8), [{"label": "tv", "bbox": [100, 100, 400, 300]}])
    assert (out[300, 150:350] > 100).all(axis=1).mean() > 0.9  # the bottom edge, drawn across
    assert out[80:100, 100:140].any()  # the label sits above the box
    assert not out[500:700, 600:1200].any()  # nothing elsewhere
    many = [{"label": str(i), "bbox": [10 + 80 * i, 400, 60 + 80 * i, 450]} for i in range(15)]
    out = objects.draw(np.zeros((720, 1280, 3), np.uint8), many)
    drawn = [i for i in range(15) if out[450, 10 + 80 * i : 60 + 80 * i].any()]
    assert len(drawn) == objects.MAX_OBJECTS
