"""The detection measurement harness (tools/measure_detection.py): its metric code, on made-up boxes and joints.

These use nothing from a model or a dataset, so they pin the arithmetic a report's numbers rest on: IoU, COCO's
average precision including the ignored-box rule, recall at an operating floor, PCK, OKS, mask matching, the
top-k decode both detector exports share, and the pose crop that must invert what pose_decode applies.

# An editor's pyright here resolves the system interpreter, which has neither of these; ty, the repository's
# checker, resolves the project's environment and passes. The ignores keep that environment mismatch from reading as
# a missing dependency in this file.
"""

import importlib.util
import pathlib
import sys

import numpy as np  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]

PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "measure_detection.py"
spec = importlib.util.spec_from_file_location("measure_detection", PATH)
assert spec and spec.loader
harness = importlib.util.module_from_spec(spec)
sys.modules["measure_detection"] = harness
spec.loader.exec_module(harness)


def scored(scores, similarity, ignored=None, crowd=None, areas=None):
    """One picture's row: scores, an N x M similarity matrix, which ground truth is ignored, which is a crowd
    region, and the detections' own areas (100 px square unless a test says otherwise)."""
    scores = np.asarray(scores, np.float64)
    similarity = np.asarray(similarity, np.float64).reshape(len(scores), -1) if len(scores) else np.zeros((0, 0))
    width = similarity.shape[1] if similarity.size else 0
    ignored = np.zeros(width, bool) if ignored is None else np.asarray(ignored, bool)
    crowd = np.zeros(width, bool) if crowd is None else np.asarray(crowd, bool)
    areas = np.full(len(scores), 100.0) if areas is None else np.asarray(areas, np.float64)
    return harness.Scored(scores, areas, similarity, ignored, crowd)


def test_iou_is_one_for_the_same_box_and_zero_for_one_that_does_not_touch():
    overlap = harness.iou_xyxy(
        np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    )
    assert overlap.tolist() == [[1.0, 0.0]]


def test_iou_halves_a_box_that_shares_half_its_area():
    # 10 x 10 against 10 x 10 shifted by 5 across: intersection 50, union 150.
    assert harness.iou_xyxy(np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([[5.0, 0.0, 15.0, 10.0]]))[
        0, 0
    ] == pytest.approx(1 / 3)


def test_a_perfect_detection_of_every_ground_truth_box_scores_ap_one():
    row = scored([0.9, 0.8], [[1.0, 0.0], [0.0, 1.0]])
    assert harness.average_precision([row], 0.5) == pytest.approx(1.0)


def test_ap_drops_to_the_share_of_ground_truth_found_not_to_the_share_of_hits():
    # Two people, one found and then a false positive: recall tops out at 0.5, so 51 of 101 recall points carry
    # precision 1 and the rest cannot be reached.
    row = scored([0.9, 0.8], [[1.0, 0.0], [0.0, 0.0]])
    assert harness.average_precision([row], 0.5) == pytest.approx(51 / 101)


def test_a_detection_matching_ignored_ground_truth_is_dropped_rather_than_counted_wrong():
    # The small box is ignored by the large band, so a hit on it is neither a true nor a false positive: the
    # remaining, large box is found, and AP stays 1.
    row = scored([0.9, 0.8], [[0.0, 1.0], [1.0, 0.0]], ignored=[False, True])
    assert harness.average_precision([row], 0.5) == pytest.approx(1.0)
    everything_ignored = scored([0.9], [[1.0]], ignored=[True])
    assert np.isnan(harness.average_precision([everything_ignored], 0.5))


def test_ap_needs_some_ground_truth_and_some_detections():
    assert np.isnan(harness.average_precision([scored([], np.zeros((0, 2)))], 0.5))
    assert np.isnan(harness.average_precision([scored([0.9], [[0.0]], ignored=[True])], 0.5))


def test_mean_ap_reads_the_ends_of_the_threshold_grid():
    # A box that overlaps its truth by exactly 0.6 is found at 0.50 and 0.55 and 0.60, and missed above.
    row = scored([0.9], [[0.6]])
    values = harness.mean_ap([row])
    assert values["ap50"] == pytest.approx(1.0)
    assert values["ap75"] == pytest.approx(0.0)


def test_recall_at_a_floor_counts_only_detections_above_it():
    row = scored([0.7, 0.4], [[1.0, 0.0], [0.0, 1.0]])
    assert harness.recall_at([row], 0.6) == pytest.approx(0.5)
    assert harness.recall_at([row], 0.3) == pytest.approx(1.0)
    assert harness.recall_at([row], 0.8) == pytest.approx(0.0)


def test_pck_counts_joints_within_a_tenth_of_the_box_side():
    box = np.array([[0.0, 0.0, 100.0, 200.0]])
    truth = np.array([[[10.0, 10.0], [50.0, 50.0]]])
    visible = np.array([[True, True]])
    near = np.array([[[10.0, 10.0], [55.0, 55.0]]])  # 7.1 px off, inside 10% of 200
    far = np.array([[[10.0, 10.0], [150.0, 150.0]]])
    assert harness.pck(near, truth, visible, box) == pytest.approx(1.0)
    assert harness.pck(far, truth, visible, box) == pytest.approx(0.5)


def test_oks_is_one_for_the_same_joints_and_near_zero_for_ones_across_the_picture():
    box = np.array([[0.0, 0.0, 100.0, 100.0]])
    truth = np.zeros((1, 17, 2))
    visible = np.ones((1, 17), bool)
    assert harness.oks(truth, truth, visible, box)[0] == pytest.approx(1.0)
    assert harness.oks(truth + 500.0, truth, visible, box)[0] < 0.01


def test_oks_ignores_joints_nobody_labelled():
    box = np.array([[0.0, 0.0, 100.0, 100.0]])
    truth = np.zeros((1, 17, 2))
    visible = np.zeros((1, 17), bool)
    visible[0, 0] = True
    perfect = harness.oks(truth, truth, visible, box)[0]
    assert perfect == pytest.approx(1.0)
    assert harness.oks(np.full((1, 17, 2), 500.0), truth, visible, box)[0] < 0.01


def test_joint_error_reports_a_share_of_the_box_and_nan_for_unseen_joints():
    box = np.array([[0.0, 0.0, 100.0, 100.0]])
    truth = np.zeros((1, 17, 2))
    visible = np.zeros((1, 17), bool)
    visible[0, 0] = True
    errors = harness.joint_error(truth + np.array([10.0, 0.0]), truth, visible, box)
    assert errors[0] == pytest.approx(0.1)
    assert np.isnan(errors[1])


def test_mask_matching_reports_recall_and_precision_against_overlapping_masks():
    truth = np.zeros((2, 8, 8), bool)
    truth[0, :4] = True
    truth[1, 4:] = True
    good = truth[0].copy()
    partial = np.zeros((8, 8), bool)
    partial[:3] = True
    perfect = harness.match_masks(np.stack([good]), truth)
    assert perfect["recall"] == pytest.approx(0.5)
    assert perfect["precision"] == pytest.approx(1.0)
    # 3 rows against a 4-row truth: IoU 0.75 clears a 0.5 threshold, a 0.8 one does not.
    assert harness.match_masks(np.stack([partial]), truth[0:1])["recall"] == pytest.approx(1.0)
    assert harness.match_masks(np.stack([partial]), truth[0:1], threshold=0.8)["recall"] == pytest.approx(0.0)
    assert np.isnan(harness.match_masks(np.zeros((0, 8, 8), bool), np.zeros((0, 8, 8), bool))["recall"])


def test_top_k_decode_is_sigmoid_globally_ranked_and_keeps_two_classes_of_one_query():
    # One query carries a confident person and a less confident cat: both survive the global top-k, which is what
    # lets a screen and a person on one query coexist and what the absence of NMS keeps honest.
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]]], np.float32)
    logits = np.zeros((1, 2, 91), np.float32)
    person, cat = harness.PERSON, harness.ANIMALS[1]
    logits[0, 0, person], logits[0, 0, cat], logits[0, 1, person] = 3.0, 1.0, -3.0
    detection = harness.top_k_decode(boxes, logits)
    assert detection.labels.tolist()[:2] == [person, cat]
    assert detection.scores[0] > 0.9 > detection.scores[1] > 0.5
    assert detection.boxes[0] == pytest.approx([0.4, 0.4, 0.6, 0.6])


def test_top_k_decode_clips_to_the_picture_and_drops_boxes_with_no_extent():
    boxes = np.array([[[0.02, 0.5, 0.2, 0.2], [0.5, 0.5, 0.0, 0.2]]], np.float32)
    logits = np.zeros((1, 2, 91), np.float32)
    logits[0, :, harness.PERSON] = 2.0
    detection = harness.top_k_decode(boxes, logits)
    # The decode ranks every query/class pair, as the running path does, and leaves the score floor to the caller.
    kept = detection.scores >= 0.7
    assert detection.boxes[kept].shape == (1, 4)
    assert detection.boxes[kept][0].tolist() == pytest.approx([0.0, 0.4, 0.12, 0.6], abs=1e-5)


def test_a_dense_label_layout_maps_to_official_category_ids():
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], np.float32)
    logits = np.zeros((1, 1, 80), np.float32)
    logits[0, 0, 14] = 4.0  # bird in the dense layout
    detection = harness.top_k_decode(boxes, logits, labels=harness.DENSE80)
    assert detection.labels[detection.scores >= 0.9].tolist() == [harness.ANIMALS[0]]


def test_the_pose_crop_puts_the_box_centre_at_the_patch_centre_and_decodes_back_to_it():
    picture = np.zeros((480, 640, 3), np.uint8)
    picture[200:280, 300:340] = 255  # a bright box the crop transform must land on
    patch, transform = harness.pose_crop(picture, [300.0, 200.0, 340.0, 280.0], (192, 256))
    assert patch.shape == (3, 256, 192)
    # A SimCC peak where the box centre falls in the patch decodes to the box centre in the picture.
    # A SimCC peak where the box centre falls decodes back to the box centre.
    # SimCC keeps two bins a pixel, so the patch centre's bin is the input width and height.
    peak_x, peak_y = 192, 256
    simcc_x = np.zeros((1, 1, 192 * 2), np.float32)
    simcc_y = np.zeros((1, 1, 256 * 2), np.float32)
    simcc_x[0, 0, peak_x] = 0.9
    simcc_y[0, 0, peak_y] = 0.9
    joints = harness.pose_decode(simcc_x, simcc_y, transform, (192, 256))
    assert joints[0, 0, 0] == pytest.approx(320.0, abs=1.0)
    assert joints[0, 0, 1] == pytest.approx(240.0, abs=1.0)
    assert joints[0, 0, 2] == pytest.approx(0.9)


def test_square_input_is_rgb_chw_and_normalised():
    picture = np.full((100, 200, 3), 255, np.uint8)  # BGR white
    blob = harness.square_input(picture, 64)
    assert blob.shape == (1, 3, 64, 64)
    expected = (1.0 - harness.IMAGE_MEAN) / harness.IMAGE_STD
    assert blob[0, :, 0, 0] == pytest.approx(expected, abs=1e-5)
    plain = harness.square_input(picture, 64, normalize=False)
    assert plain[0, :, 0, 0] == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)


def test_nms_keeps_one_of_two_boxes_on_the_same_object_and_both_when_they_are_apart():
    detection = harness.Detection(
        boxes=np.array([[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5], [50.0, 50.0, 60.0, 60.0]]),
        scores=np.array([0.9, 0.8, 0.7]),
        labels=np.array([1, 1, 1]),
    )
    kept = harness.non_max_suppression(detection, 0.5)
    assert kept.scores.tolist() == [0.9, 0.7]
    assert harness.non_max_suppression(detection, 0.0).scores.tolist() == [0.9, 0.8, 0.7]


def test_nms_compares_boxes_within_a_class_only():
    # A person's box and an animal's box overlap here; multiclass NMS must keep both, as the running path does
    # with no NMS at all across classes.
    detection = harness.Detection(
        boxes=np.array([[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5]]),
        scores=np.array([0.9, 0.8]),
        labels=np.array([harness.PERSON, harness.ANIMALS[1]]),
    )
    assert len(harness.non_max_suppression(detection, 0.5).scores) == 2


def test_oks_scales_a_pixel_error_by_the_person_area_and_the_joint_constant():
    # One visible nose (COCO's constant 0.26) off by 20 px on a 100 x 100 box: exp(-400 / ((0.26 * 2) ** 2 * 10000 * 2)).
    truth = np.zeros((1, 17, 2))
    visible = np.zeros((1, 17), bool)
    visible[0, 0] = True
    pred = truth + np.array([20.0, 0.0])
    expected = float(np.exp(-400.0 / ((0.26 * 2.0) ** 2 * 10000.0 * 2.0)))
    assert expected > 0.9
    assert harness.oks(pred, truth, visible, np.array([[0.0, 0.0, 100.0, 100.0]]))[0] == pytest.approx(expected)
    # The same error on a person a tenth the area counts for more.
    smaller = harness.oks(pred, truth, visible, np.array([[0.0, 0.0, 31.6, 31.6]]))[0]
    assert smaller < expected


def test_crowd_ground_truth_ignores_what_it_overlaps_instead_of_counting_it_wrong():
    # A COCO crowd region is not a person to find, and a detection inside one is neither right nor wrong. Without
    # its crowd flag the same detection reads as a false positive.
    boxes = np.array([[0.0, 0.0, 100.0, 100.0], [200.0, 200.0, 240.0, 260.0]])
    with_crowd = harness.scored(
        harness.Detection(
            np.array([[10.0, 10.0, 90.0, 90.0], [200.0, 200.0, 240.0, 260.0]]), np.array([0.9, 0.8]), np.array([1, 1])
        ),
        (boxes, np.array([True, False]), np.array([10000.0, 2400.0])),
        1,
        "all",
    )
    assert harness.average_precision([with_crowd], 0.5) == pytest.approx(1.0)
    assert harness.recall_at([with_crowd], 0.0) == pytest.approx(1.0)  # one real person, found
    # The same predictions with the crowd region not labelled at all: the box inside it is a plain false positive.
    as_person = harness.scored(
        harness.Detection(
            np.array([[10.0, 10.0, 90.0, 90.0], [200.0, 200.0, 240.0, 260.0]]), np.array([0.9, 0.8]), np.array([1, 1])
        ),
        (np.array([[200.0, 200.0, 240.0, 260.0]]), np.array([False]), np.array([2400.0])),
        1,
        "all",
    )
    # One person, one false positive ranked above the one hit: every recall point carries precision 0.5.
    assert harness.average_precision([as_person], 0.5) == pytest.approx(0.5)


def test_a_picture_with_nobody_in_it_costs_the_detector_that_finds_someone_there():
    # COCO keeps such a picture in the category's evaluation and its detections count as false positives, which is
    # what makes a detector that invents people look worse than one that does not.
    found = scored([0.8], [[1.0]])
    assert harness.average_precision([found], 0.5) == pytest.approx(1.0)
    # The invented box scores higher than the real find, so every recall point carries precision 0.5.
    invented = scored([0.9], [[], []])
    assert harness.average_precision([found, invented], 0.5) == pytest.approx(0.5)


def test_a_duplicate_box_still_finds_the_other_person_it_overlaps():
    # Two people, two detections: the better-scoring one takes the closer person, and the second must go to the
    # one still unclaimed rather than count as a false positive.
    row = harness.Scored(
        np.array([0.9, 0.8]),
        np.full(2, 100.0),
        np.array([[1.0, 0.6], [0.6, 1.0]]),
        np.zeros(2, bool),
        np.zeros(2, bool),
    )
    assert harness.average_precision([row], 0.5) == pytest.approx(1.0)


def test_a_crowd_region_measures_the_share_of_the_detection_inside_it():
    # COCO's crowd convention: a small box wholly inside a crowd matches it completely, so it is dropped as ignored
    # rather than counted as a false positive. Ordinary IoU would call that a miss.
    crowd_box = np.array([[0.0, 0.0, 100.0, 100.0]])
    small = np.array([[10.0, 10.0, 20.0, 20.0]])
    assert harness.iou_xyxy(small, crowd_box)[0, 0] < 0.05
    assert harness.iou_xyxy(small, crowd_box, np.array([True]))[0, 0] == pytest.approx(1.0)
    # A box half outside the crowd is half a match, not none.
    hangover = np.array([[90.0, 10.0, 110.0, 20.0]])
    assert harness.iou_xyxy(hangover, crowd_box, np.array([True]))[0, 0] == pytest.approx(0.5)
