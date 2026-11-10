"""Bounded, transient decoded-frame handoff. This module opens no cameras or broker connections.

One owner creates four private slot files beneath an explicit root. Nonblocking exclusive file locks keep
slow readers from stalling capture. Readers validate the descriptor against locked slot metadata, copy the
pixels, and release the shared lock before returning. A descriptor is not a new source identity or proof.
"""

import fcntl
import json
import mmap
import os
import stat
import struct
import tempfile
import threading
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self, cast

import numpy as np

PixelFormat = Literal["bgr", "nv12"]
SLOTS = 4
MAX_PIXELS = 16_000_000
_MAX_HEADER = 65_536
_U64 = (1 << 64) - 1
_META = struct.Struct("<8s16sIQIIQIIQQQII")
_MAGIC = b"DCACHE02"
_OFFSET = _META.size + _MAX_HEADER
_CODES = {"bgr": 1, "nv12": 2}
_RANGES = {"unspecified": 0, "full": 1, "limited": 2}
_MATRICES = {"unspecified": 0, "bt601": 1, "bt709": 2, "bt2020nc": 3, "fcc": 4, "smpte240m": 5}


def _uint(value: object, maximum: int = _U64, *, positive: bool = False) -> int:
    if type(value) is not int or not int(positive) <= value <= maximum:
        raise ValueError("invalid decoded cache integer")
    return value


def _color(value: object, codes: Mapping[str, int]) -> str:
    if not isinstance(value, str) or value not in codes:
        raise ValueError("invalid decoded cache color metadata")
    return value


def _owner(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid decoded cache owner")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise ValueError("invalid decoded cache owner") from error
    if not parsed.int or str(parsed) != value:
        raise ValueError("invalid decoded cache owner")
    return value


def _header(value: object) -> tuple[dict[str, Any], bytes]:
    if not isinstance(value, Mapping):
        raise ValueError("decoded cache needs a native source header")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > _MAX_HEADER:
            raise ValueError("source header exceeds decoded cache bound")
        result = json.loads(encoded)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("invalid decoded cache source header") from error
    _owner(result.get("id"))
    _uint(result.get("stream_id"), (1 << 32) - 1, positive=True)
    start, end = _uint(result.get("start_ns")), _uint(result.get("end_ns"))
    if end < start:
        raise ValueError("reversed native source span")
    return result, encoded


def _layout(
    format: object, width: object, height: object, shape: object, length: object, limit: int
) -> tuple[PixelFormat, int, int, tuple[int, ...], int]:
    if format not in ("bgr", "nv12"):
        raise ValueError("invalid decoded cache pixel format")
    w, h = _uint(width, MAX_PIXELS, positive=True), _uint(height, MAX_PIXELS, positive=True)
    if w * h > limit or (format == "nv12" and (w % 2 or h % 2)):
        raise ValueError("decoded frame exceeds bounds or has invalid NV12 dimensions")
    expected = (h, w, 3) if format == "bgr" else (h * 3 // 2, w)
    size = w * h * 3 if format == "bgr" else w * h * 3 // 2
    if (
        not isinstance(shape, (list, tuple))
        or any(type(v) is not int for v in shape)
        or tuple(shape) != expected
        or _uint(length) != size
    ):
        raise ValueError("decoded cache shape/length mismatch")
    return cast(PixelFormat, format), w, h, expected, size


class _Slot:
    def __init__(self, fd: int, size: int, *, readonly: bool = False) -> None:
        self.fd = fd
        try:
            self.mapping = mmap.mmap(fd, size, access=mmap.ACCESS_READ if readonly else mmap.ACCESS_WRITE)
        except BaseException:
            os.close(fd)
            raise
        self.inode = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)

    def close(self) -> None:
        self.mapping.close()
        os.close(self.fd)


class DecodedCache:
    """One camera/format owner. Root must already exist; close removes only this owner's temporary subtree."""

    def __init__(self, root: Path, *, format: PixelFormat = "bgr", max_pixels: int = MAX_PIXELS) -> None:
        _uint(max_pixels, MAX_PIXELS, positive=True)
        if format not in _CODES:
            raise ValueError("invalid decoded cache format")
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("decoded cache root is not a directory")
        self.format = format
        self.max_pixels = max_pixels
        self.owner = str(uuid.uuid4())
        self.capacity = max_pixels * 3 if format == "bgr" else max_pixels * 3 // 2
        self._directory = tempfile.TemporaryDirectory(prefix=f"decoded-{self.owner}-", dir=self.root)
        self.path = Path(self._directory.name)
        self._slots: list[_Slot] = []
        self._lock = threading.Lock()
        self._serial = self._generation = self._next = self._break_count = 0
        self._closed = False
        try:
            for number in range(SLOTS):
                fd = os.open(self.path / f"slot-{number}.pixels", os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    os.ftruncate(fd, _OFFSET + self.capacity)
                except BaseException:
                    os.close(fd)
                    raise
                self._slots.append(_Slot(fd, _OFFSET + self.capacity))
        except BaseException:
            self.close()
            raise

    def publish(
        self,
        pixels: np.ndarray,
        source_header: Mapping[str, Any],
        *,
        break_count: int = 0,
        format: PixelFormat | None = None,
        color_range: str = "unspecified",
        color_matrix: str = "unspecified",
    ) -> dict[str, Any] | None:
        """Publish contiguous uint8 BGR/NV12, or None when all four slots are pinned. No pixel conversion occurs.

        Serial advances for every valid attempt, including a busy refusal; caller must separately advance its
        break counter on loss. Generation advances only on successful publication. Source headers are copied. Color signaling is
        explicit and remains unspecified unless supplied; this transport performs no color conversion.
        """
        if format is not None and format != self.format:
            raise ValueError("one decoded cache instance has one pixel format")
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or not pixels.flags.c_contiguous:
            raise ValueError("decoded cache needs contiguous uint8 pixels")
        if self.format == "bgr":
            if pixels.ndim != 3:
                raise ValueError("decoded cache needs BGR height,width,3")
            height, width = pixels.shape[:2]
        else:
            if pixels.ndim != 2 or pixels.shape[0] % 3:
                raise ValueError("decoded cache needs NV12 height*3/2,width")
            height, width = pixels.shape[0] * 2 // 3, pixels.shape[1]
        _, width, height, shape, length = _layout(
            self.format, width, height, pixels.shape, pixels.nbytes, self.max_pixels
        )
        color_range = _color(color_range, _RANGES)
        color_matrix = _color(color_matrix, _MATRICES)
        native, encoded_header = _header(source_header)
        breaks = _uint(break_count)
        with self._lock:
            if self._closed:
                raise ValueError("decoded cache owner is closed")
            if breaks < self._break_count:
                raise ValueError("decoded cache break counter cannot go backwards")
            self._break_count = breaks
            serial = _uint(self._serial + 1)
            self._serial = serial
            for step in range(SLOTS):
                number = (self._next + step) % SLOTS
                slot = self._slots[number]
                try:
                    fcntl.flock(slot.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    generation = _uint(self._generation + 1)
                    # Invalidate the previous generation before touching its pixels. A failed copy can never
                    # leave the old descriptor authorizing partially replaced pixels.
                    slot.mapping[:8] = b"\0" * 8
                    slot.mapping[_OFFSET : _OFFSET + length] = memoryview(pixels).cast("B")
                    slot.mapping[_META.size : _META.size + len(encoded_header)] = encoded_header
                    slot.mapping[: _META.size] = _META.pack(
                        _MAGIC,
                        uuid.UUID(self.owner).bytes,
                        number,
                        generation,
                        width,
                        height,
                        length,
                        _CODES[self.format],
                        len(encoded_header),
                        breaks,
                        serial,
                        self.capacity,
                        _RANGES[color_range],
                        _MATRICES[color_matrix],
                    )
                    self._generation = generation
                    self._next = (number + 1) % SLOTS
                    return {
                        "path": str(self.path / f"slot-{number}.pixels"),
                        "owner": self.owner,
                        "slot": number,
                        "generation": generation,
                        "width": width,
                        "height": height,
                        "shape": list(shape),
                        "length": length,
                        "format": self.format,
                        "dtype": "uint8",
                        "color_range": color_range,
                        "color_matrix": color_matrix,
                        "source_header": native,
                        "break_count": breaks,
                        "serial": serial,
                    }
                finally:
                    fcntl.flock(slot.fd, fcntl.LOCK_UN)
        return None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for slot in self._slots:
                slot.close()
            self._slots.clear()
            self._directory.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class DecodedCacheReader:
    """A bounded per-process mapping LRU. Reads return owned pixel/header copies; readers never unlink files."""

    def __init__(self, root: Path, *, max_maps: int = 16, max_pixels: int = MAX_PIXELS) -> None:
        self.root = root.resolve(strict=True)
        self.max_maps = _uint(max_maps, 64, positive=True)
        self.max_pixels = _uint(max_pixels, MAX_PIXELS, positive=True)
        self._maps: OrderedDict[str, _Slot] = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False

    def _open(self, path: str, owner: str, number: int) -> int:
        candidate = Path(path)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("decoded descriptor path must be beneath its configured root")
        try:
            parts = candidate.relative_to(self.root).parts
        except ValueError as error:
            raise ValueError("decoded descriptor escaped its configured root") from error
        if len(parts) != 2 or not parts[0].startswith(f"decoded-{owner}-") or parts[1] != f"slot-{number}.pixels":
            raise ValueError("invalid decoded descriptor slot path")
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directory = None
        try:
            directory = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
            return os.open(parts[1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
        finally:
            if directory is not None:
                os.close(directory)
            os.close(root_fd)

    def read(self, descriptor: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Validate and copy one exact generation. Missing/overwritten owners return None; malformed input raises."""
        if not isinstance(descriptor, Mapping):
            raise ValueError("decoded cache descriptor must be an object")
        if descriptor.get("dtype") != "uint8":
            raise ValueError("decoded cache descriptor must use uint8")
        color_range = _color(descriptor.get("color_range", "unspecified"), _RANGES)
        color_matrix = _color(descriptor.get("color_matrix", "unspecified"), _MATRICES)
        owner = _owner(descriptor.get("owner"))
        number = _uint(descriptor.get("slot"), SLOTS - 1)
        generation = _uint(descriptor.get("generation"), positive=True)
        serial = _uint(descriptor.get("serial"), positive=True)
        breaks = _uint(descriptor.get("break_count"))
        format, width, height, shape, length = _layout(
            descriptor.get("format"),
            descriptor.get("width"),
            descriptor.get("height"),
            descriptor.get("shape"),
            descriptor.get("length"),
            self.max_pixels,
        )
        native, encoded_header = _header(descriptor.get("source_header"))
        path = descriptor.get("path")
        if not isinstance(path, str) or len(path) > 4096:
            raise ValueError("invalid decoded descriptor path")
        with self._lock:
            if self._closed:
                raise ValueError("decoded cache reader is closed")
            try:
                fd = self._open(path, owner, number)
            except FileNotFoundError:
                return None
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                os.close(fd)
                raise ValueError("decoded cache slot must be a regular file")
            size = info.st_size
            inode = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
            capacity_limit = self.max_pixels * 3 if format == "bgr" else self.max_pixels * 3 // 2
            if not _OFFSET < size <= _OFFSET + capacity_limit or size < _OFFSET + length:
                os.close(fd)
                raise ValueError("decoded cache file size exceeds bounds")
            slot = self._maps.pop(path, None)
            if slot is not None and slot.inode != inode:
                slot.close()
                slot = None
            if slot is None:
                while len(self._maps) >= self.max_maps:
                    self._maps.popitem(last=False)[1].close()
                slot = _Slot(fd, size, readonly=True)
            else:
                os.close(fd)
            self._maps[path] = slot
            fcntl.flock(slot.fd, fcntl.LOCK_SH)
            try:
                stored = _META.unpack(slot.mapping[: _META.size])
                expected = (
                    _MAGIC,
                    uuid.UUID(owner).bytes,
                    number,
                    generation,
                    width,
                    height,
                    length,
                    _CODES[format],
                    len(encoded_header),
                    breaks,
                    serial,
                    size - _OFFSET,
                    _RANGES[color_range],
                    _MATRICES[color_matrix],
                )
                if stored != expected or slot.mapping[_META.size : _META.size + len(encoded_header)] != encoded_header:
                    return None
                pixels = np.frombuffer(slot.mapping, np.uint8, count=length, offset=_OFFSET).reshape(shape).copy()
                return pixels, native
            finally:
                fcntl.flock(slot.fd, fcntl.LOCK_UN)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for slot in self._maps.values():
                slot.close()
            self._maps.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
