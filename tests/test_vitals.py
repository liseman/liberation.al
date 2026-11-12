"""The maths of heart rate from skin colour (vitals.py): POS, CHROM, the band, and the confidence that gates it."""

import math

import numpy as np
import pytest

from liberational import vitals

FPS = 30.0
SECONDS = 12.0
SAMPLES = int(FPS * SECONDS)


def pulse(rate_bpm: float = 72.0, amplitude: float = 0.4, fps: float = FPS, samples: int = SAMPLES) -> np.ndarray:
    """A known heart rate's colour modulation as a (samples,) signal: 1.2 Hz at 72 bpm."""
    times = np.arange(samples) / fps
    return amplitude * np.sin(2 * np.pi * (rate_bpm / 60.0) * times)


def colour(pulse_signal: np.ndarray, noise: float, rng: np.random.Generator, drift: float = 0.0) -> np.ndarray:
    """A T x 3 R, G, B trace carrying `pulse_signal` over a mid skin tone, with a slow light drift and noise added.
    Green carries most of the pulse and blue almost none, as blood's absorption gives it."""
    samples = len(pulse_signal)
    times = np.arange(samples) / FPS
    light = drift * np.sin(2 * np.pi * 0.05 * times)  # the room's light changing over the window
    base = np.array([150.0, 110.0, 85.0])  # R, G, B of a mid skin tone, in the order region_means gives them
    weight = np.array([0.3, 1.0, 0.15])
    return base + light[:, None] + weight * pulse_signal[:, None] + rng.normal(0.0, noise, (samples, 3))


def trace_of(rgb: np.ndarray, *, side: int = 160, motion: float = 0.02, fps: float = FPS) -> vitals.Trace:
    trace = vitals.Trace()
    for i, sample in enumerate(rgb):
        trace.add(int(i * 1e9 / fps), sample, side, motion)
    return trace


def test_a_seated_pulse_is_recovered_within_a_beat_or_two():
    rng = np.random.default_rng(11)
    trace = trace_of(colour(pulse(72.0), noise=0.25, rng=rng))
    reading = trace.reading()
    assert reading is not None
    assert reading.bpm == pytest.approx(72.0, abs=2.0)
    assert reading.trusted


def test_chrom_agrees_with_pos_on_a_clean_pulse():
    rng = np.random.default_rng(3)
    rgb = colour(pulse(96.0), noise=0.1, rng=rng)
    reading = trace_of(rgb).reading()
    assert reading is not None
    assert reading.pos_bpm == pytest.approx(96.0, abs=2.0)
    assert reading.chrom_bpm == pytest.approx(reading.pos_bpm, abs=2.0)


def test_a_picture_with_no_colour_is_no_reading():
    """An infrared night camera gives R = G = B: there is no blood signal in the chrominance to find."""
    rng = np.random.default_rng(5)
    grey = colour(pulse(72.0), noise=0.2, rng=rng)[:, 1:2]
    reading = trace_of(np.repeat(grey, 3, axis=1)).reading()
    assert reading is not None
    assert reading.mono
    assert math.isnan(reading.bpm)
    assert not reading.trusted


def test_a_face_too_small_for_its_pixels_is_no_reading():
    rng = np.random.default_rng(6)
    reading = trace_of(colour(pulse(72.0), noise=0.2, rng=rng), side=90).reading()
    assert reading is not None
    assert reading.face_px == 90
    assert math.isnan(reading.bpm)


def test_a_moving_face_is_no_reading():
    rng = np.random.default_rng(8)
    reading = trace_of(colour(pulse(72.0), noise=0.2, rng=rng), motion=1.2).reading()
    assert reading is not None
    assert math.isnan(reading.bpm)


def test_noise_with_no_pulse_at_all_is_no_reading():
    """A still face in a still room: whatever the noise's spectrum peaks at is not a heartbeat."""
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        base = np.array([150.0, 110.0, 85.0])
        reading = trace_of(base + rng.normal(0.0, 1.5, (SAMPLES, 3))).reading()
        assert reading is not None
        assert math.isnan(reading.bpm), f"seed {seed} reported {reading.bpm}"


def test_a_window_with_too_little_of_a_trace_is_not_a_reading_yet():
    rng = np.random.default_rng(2)
    assert trace_of(colour(pulse(72.0), noise=0.2, rng=rng)[: int(3 * FPS)]).reading() is None


def test_dropping_a_third_of_the_frames_lowers_coverage_and_the_reading():
    """A camera that laps frames has holes in its window: the coverage says so, and a third of the frames missing
    leaves nothing trustworthy."""
    rng = np.random.default_rng(9)
    rgb = colour(pulse(72.0), noise=0.2, rng=rng)
    trace = vitals.Trace()
    for i, sample in enumerate(rgb):
        if i % 3 != 2:  # the third frame of every three never arrives
            trace.add(int(i * 1e9 / FPS), sample, 160, 0.02)
    reading = trace.reading()
    assert reading is not None
    assert reading.coverage == pytest.approx(2 / 3, abs=0.05)
    assert not reading.trusted


def test_a_camera_below_twenty_frames_a_second_is_no_reading():
    """A room camera at 10 fps: the heart band sits too close to the frame rate to measure anything."""
    rng = np.random.default_rng(4)
    trace = trace_of(colour(pulse(72.0), noise=0.2, rng=rng), fps=10.0)
    reading = trace.reading()
    assert reading is not None
    assert reading.fps == pytest.approx(10.0, abs=0.1)
    assert math.isnan(reading.bpm)


def test_the_rate_is_the_strongest_line_in_the_band():
    times = np.arange(SAMPLES) / FPS
    for bpm in (48.0, 72.0, 120.0, 165.0):
        signal = np.sin(2 * np.pi * (bpm / 60.0) * times)
        assert vitals.rate(signal, FPS) == pytest.approx(bpm, abs=1.5)


def test_a_rate_outside_the_band_is_never_reported():
    """A camera's flicker at 4 Hz is not a heart rate: the band's edges hold."""
    times = np.arange(SAMPLES) / FPS
    signal = np.sin(2 * np.pi * 5.0 * times)
    bpm = vitals.rate(signal, FPS)
    assert math.isnan(bpm) or vitals.BPM_MIN <= bpm <= vitals.BPM_MAX


def test_region_means_averages_the_patches_and_returns_rgb():
    # BGR as the backends hand it over: two patches, one of 100/50/10 and one of 200/60/20 (blue, green, red).
    pixels = np.zeros((4, 4, 3), np.uint8)
    pixels[0:2, 0:2] = (10, 50, 100)  # B, G, R
    pixels[2:4, 2:4] = (20, 60, 200)
    means = vitals.region_means(pixels, [(0, 0, 2, 2), (2, 2, 4, 4)])
    assert means.tolist() == [150.0, 55.0, 15.0]


def test_face_regions_puts_skin_on_the_forehead_and_both_cheeks():
    joints = np.zeros((26, 3), np.float64)
    # A face 200 px wide: pupils 87 px apart (2.3 interocular distances to the face's side), eye line at y=100.
    joints[0] = (100.0, 130.0, 0.9)  # nose
    joints[1] = (56.5, 100.0, 0.9)  # left eye
    joints[2] = (143.5, 100.0, 0.9)  # right eye
    joints[3] = (40.0, 110.0, 0.9)  # ears
    joints[4] = (160.0, 110.0, 0.9)
    regions = vitals.face_regions(joints, 400, 300)
    assert regions is not None
    assert regions.side == pytest.approx(200.0, abs=1.0)
    x1, y1, _x2, _y2 = regions.box
    centres = [((p[0] + p[2]) / 2 + x1, (p[1] + p[3]) / 2 + y1) for p in regions.patches]
    assert len(centres) == 3
    # The forehead above the eye line's middle, the cheeks beside the face's midline and below it.
    assert centres[0][1] < 100.0 - 30.0
    assert centres[0][0] == pytest.approx(100.0, abs=8.0)
    assert centres[1][0] < 100.0 - 40.0
    assert centres[2][0] > 100.0 + 40.0
    assert all(y > 110.0 for _, y in centres[1:])


def test_face_regions_refuses_a_head_turned_edge_on_or_too_far_away():
    joints = np.zeros((26, 3), np.float64)
    joints[0] = (100.0, 130.0, 0.9)
    joints[1] = (100.0, 100.0, 0.9)  # both eyes on the midline: the head is edge-on, so the face's size is unknown
    joints[2] = (100.0, 100.0, 0.9)
    assert vitals.face_regions(joints, 400, 300) is None
    far = np.zeros((26, 3), np.float64)
    far[0] = (100.0, 130.0, 0.9)
    far[1] = (98.0, 100.0, 0.9)  # pupils 2 px apart: a face a long way off
    far[2] = (100.0, 100.0, 0.9)
    assert vitals.face_regions(far, 400, 300) is None
    unplaced = np.zeros((26, 3), np.float64)
    assert vitals.face_regions(unplaced, 400, 300) is None
