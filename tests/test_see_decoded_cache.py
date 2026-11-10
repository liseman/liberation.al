"""Transient frame transport must never hand out overwritten pixels or invent source identity."""

import copy
import fcntl
import json
import os
from contextlib import ExitStack

import numpy as np
import pytest

from liberational.see.decoded_cache import DecodedCache, DecodedCacheReader


@pytest.fixture
def source():
    return {
        "id": "01900000-0000-7000-8000-000000000001",
        "stream_id": 7,
        "start_ns": 1_800_000_000_000_000_001,
        "end_ns": 1_800_000_000_000_000_001,
        "shape_name": "H264AnnexB",
        "seq": 19,
        "parents": [],
        "capture": {"width": 4, "height": 4, "device_seq": 19, "flip": True},
    }


@pytest.mark.parametrize("format,shape", [("bgr", (4, 4, 3)), ("nv12", (6, 4))])
def test_roundtrip_keeps_exact_native_header_and_owned_pixels(tmp_path, source, format, shape):
    pixels = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    original = pixels.copy()
    with (
        DecodedCache(tmp_path, format=format, max_pixels=64) as owner,
        DecodedCacheReader(tmp_path, max_pixels=64) as reader,
    ):
        descriptor = owner.publish(pixels, source, break_count=3, color_range="limited", color_matrix="bt709")
        assert descriptor is not None
        descriptor = json.loads(json.dumps(descriptor))
        assert descriptor["color_range"] == "limited"
        assert descriptor["color_matrix"] == "bt709"
        source["parents"].append("changed-after-publication")
        pixels[:] = 255
        result = reader.read(descriptor)
        assert result is not None
        image, header = result
        np.testing.assert_array_equal(image, original)
        assert header == descriptor["source_header"]
        assert header["parents"] == []
        assert header["start_ns"] == 1_800_000_000_000_000_001
        assert header["capture"]["flip"] is True
        assert descriptor["break_count"] == 3
        image[:] = 0
        again = reader.read(descriptor)
        assert again is not None
        np.testing.assert_array_equal(again[0], original)


def test_overwrite_rejects_stale_descriptor_without_changing_previous_copy(tmp_path, source):
    with DecodedCache(tmp_path, max_pixels=64) as owner, DecodedCacheReader(tmp_path, max_pixels=64) as reader:
        first = owner.publish(np.full((4, 4, 3), 1, np.uint8), source)
        assert first is not None
        copied = reader.read(first)
        assert copied is not None
        for value in [2, 3, 4, 5]:
            last = owner.publish(np.full((4, 4, 3), value, np.uint8), source)
        assert last is not None and last["slot"] == first["slot"]
        assert last["generation"] > first["generation"]
        assert reader.read(first) is None
        np.testing.assert_array_equal(copied[0], np.full((4, 4, 3), 1, np.uint8))
        current = reader.read(last)
        assert current is not None
        np.testing.assert_array_equal(current[0], np.full((4, 4, 3), 5, np.uint8))


def test_pinned_slots_are_skipped_and_all_busy_never_overwrites(tmp_path, source):
    with (
        DecodedCache(tmp_path, max_pixels=64) as owner,
        DecodedCacheReader(tmp_path, max_pixels=64) as reader,
        ExitStack() as pins,
    ):
        descriptors = [owner.publish(np.full((4, 4, 3), value, np.uint8), source) for value in range(4)]
        assert all(row is not None for row in descriptors)
        for descriptor in descriptors:
            assert descriptor is not None
            handle = pins.enter_context(open(descriptor["path"], "rb"))
            fcntl.flock(handle, fcntl.LOCK_SH)
        assert owner.publish(np.zeros((4, 4, 3), np.uint8), source) is None
        for value, descriptor in enumerate(descriptors):
            assert descriptor is not None
            result = reader.read(descriptor)
            assert result is not None
            assert np.all(result[0] == value)
        pins.close()
        next_frame = owner.publish(np.full((4, 4, 3), 9, np.uint8), source, break_count=1)
        assert next_frame is not None and next_frame["serial"] == 6 and next_frame["break_count"] == 1


def test_one_pinned_slot_does_not_stall_other_slots(tmp_path, source):
    with DecodedCache(tmp_path, max_pixels=64) as owner:
        first = owner.publish(np.zeros((4, 4, 3), np.uint8), source)
        assert first is not None
        with open(first["path"], "rb") as pin:
            fcntl.flock(pin, fcntl.LOCK_SH)
            for _ in range(6):
                descriptor = owner.publish(np.ones((4, 4, 3), np.uint8), source)
                assert descriptor is not None and descriptor["slot"] != first["slot"]


def test_reader_close_never_unlinks_owner_and_owner_close_invalidates_mapped_descriptor(tmp_path, source):
    owner = DecodedCache(tmp_path, max_pixels=64)
    reader = DecodedCacheReader(tmp_path, max_pixels=64)
    descriptor = owner.publish(np.zeros((4, 4, 3), np.uint8), source)
    assert descriptor is not None and reader.read(descriptor) is not None
    reader.close()
    assert os.path.exists(descriptor["path"])
    reader = DecodedCacheReader(tmp_path, max_pixels=64)
    assert reader.read(descriptor) is not None
    owner.close()
    assert reader.read(descriptor) is None
    assert not list(tmp_path.iterdir())
    reader.close()
    owner.close()


@pytest.mark.parametrize("field,value", [("generation", 999), ("break_count", 999), ("serial", 999)])
def test_descriptor_cannot_substitute_locked_slot_metadata(tmp_path, source, field, value):
    with DecodedCache(tmp_path, max_pixels=64) as owner, DecodedCacheReader(tmp_path, max_pixels=64) as reader:
        descriptor = owner.publish(np.zeros((4, 4, 3), np.uint8), source)
        assert descriptor is not None
        descriptor[field] = value
        assert reader.read(descriptor) is None


def test_descriptor_cannot_swap_native_header(tmp_path, source):
    with DecodedCache(tmp_path, max_pixels=64) as owner, DecodedCacheReader(tmp_path, max_pixels=64) as reader:
        descriptor = owner.publish(np.zeros((4, 4, 3), np.uint8), source)
        assert descriptor is not None
        descriptor["source_header"]["id"] = "01900000-0000-7000-8000-000000000002"
        assert reader.read(descriptor) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", float("nan")),
        ("height", True),
        ("width", 16_000_001),
        ("shape", [4, 4, 4]),
        ("length", 47),
        ("dtype", "float32"),
        ("slot", 4),
        ("owner", "bad"),
        ("serial", -1),
        ("format", "rgb"),
    ],
)
def test_malformed_descriptors_are_rejected_before_mapping(tmp_path, source, field, value):
    with DecodedCache(tmp_path, max_pixels=64) as owner, DecodedCacheReader(tmp_path, max_pixels=64) as reader:
        descriptor = owner.publish(np.zeros((4, 4, 3), np.uint8), source)
        assert descriptor is not None
        descriptor[field] = value
        with pytest.raises(ValueError):
            reader.read(descriptor)
        assert not reader._maps


@pytest.mark.parametrize(
    "format,pixels",
    [
        ("bgr", np.zeros((4, 4, 4), np.uint8)),
        ("bgr", np.zeros((4, 4, 3), np.float32)),
        ("bgr", np.zeros((10, 10, 3), np.uint8)),
        ("bgr", np.zeros((4, 4, 3), np.uint8)[:, ::-1]),
        ("nv12", np.zeros((5, 4), np.uint8)),
        ("nv12", np.zeros((6, 3), np.uint8)),
    ],
)
def test_owner_refuses_invalid_layout_dtype_size_and_strides(tmp_path, source, format, pixels):
    with DecodedCache(tmp_path, format=format, max_pixels=64) as owner, pytest.raises(ValueError):
        owner.publish(pixels, source)


def test_path_traversal_outside_root_and_symlinks_cannot_be_mapped(tmp_path, source):
    root = tmp_path / "root"
    root.mkdir()
    with DecodedCache(root, max_pixels=64) as owner, DecodedCacheReader(root, max_pixels=64) as reader:
        descriptor = owner.publish(np.zeros((4, 4, 3), np.uint8), source)
        assert descriptor is not None
        escaped = copy.deepcopy(descriptor)
        escaped["path"] = str(root / ".." / "slot-0.pixels")
        with pytest.raises(ValueError):
            reader.read(escaped)
        escaped["path"] = str(tmp_path / "slot-0.pixels")
        with pytest.raises(ValueError):
            reader.read(escaped)
        original = descriptor["path"]
        os.rename(original, original + ".old")
        os.symlink(original + ".old", original)
        with pytest.raises(OSError):
            reader.read(descriptor)
        assert not reader._maps


def test_mapping_lru_and_owner_slot_count_remain_bounded(tmp_path, source):
    with ExitStack() as stack:
        reader = stack.enter_context(DecodedCacheReader(tmp_path, max_maps=2, max_pixels=64))
        for _ in range(5):
            owner = stack.enter_context(DecodedCache(tmp_path, max_pixels=64))
            assert len(list(owner.path.iterdir())) == 4
            descriptor = owner.publish(np.zeros((4, 4, 3), np.uint8), source)
            assert descriptor is not None and reader.read(descriptor) is not None
            assert len(reader._maps) <= 2


@pytest.mark.parametrize(
    "change", [{"start_ns": True}, {"end_ns": -1}, {"stream_id": 0}, {"id": "not-a-native-id"}, {"extra": float("nan")}]
)
def test_invalid_native_headers_never_publish(tmp_path, source, change):
    source.update(change)
    with DecodedCache(tmp_path, max_pixels=64) as owner, pytest.raises(ValueError):
        owner.publish(np.zeros((4, 4, 3), np.uint8), source)


def test_layout_changes_keep_old_descriptor_exact_and_gap_counter_never_regresses(tmp_path, source):
    with DecodedCache(tmp_path, max_pixels=64) as owner, DecodedCacheReader(tmp_path, max_pixels=64) as reader:
        old = owner.publish(np.zeros((4, 4, 3), np.uint8), source, break_count=2)
        new = owner.publish(np.ones((2, 4, 3), np.uint8), source, break_count=3)
        assert old is not None and new is not None
        first = reader.read(old)
        second = reader.read(new)
        assert first is not None and first[0].shape == (4, 4, 3)
        assert second is not None and second[0].shape == (2, 4, 3)
        with pytest.raises(ValueError, match="backwards"):
            owner.publish(np.zeros((4, 4, 3), np.uint8), source, break_count=2)


@pytest.mark.parametrize("field,value", [("color_range", "full"), ("color_matrix", "bt601")])
def test_descriptor_cannot_forge_locked_color_signaling(tmp_path, source, field, value):
    with (
        DecodedCache(tmp_path, format="nv12", max_pixels=64) as owner,
        DecodedCacheReader(tmp_path, max_pixels=64) as reader,
    ):
        descriptor = owner.publish(np.zeros((6, 4), np.uint8), source, color_range="limited", color_matrix="bt709")
        assert descriptor is not None and reader.read(descriptor) is not None
        descriptor[field] = value
        assert reader.read(descriptor) is None


@pytest.mark.parametrize("field,value", [("color_range", "jpeg"), ("color_matrix", "guessed"), ("color_range", None)])
def test_unknown_color_signaling_is_explicit_and_invalid_values_are_rejected(tmp_path, source, field, value):
    with DecodedCache(tmp_path, max_pixels=64) as owner, DecodedCacheReader(tmp_path, max_pixels=64) as reader:
        descriptor = owner.publish(np.zeros((4, 4, 3), np.uint8), source)
        assert descriptor is not None
        assert descriptor["color_range"] == descriptor["color_matrix"] == "unspecified"
        descriptor[field] = value
        with pytest.raises(ValueError, match="color metadata"):
            reader.read(descriptor)
