"""Read the vertex positions of a PLY point cloud (ASCII or binary, either byte order) as an Nx3 float64 array, and
write a coloured cloud as binary PLY (`dumps`).

Only the `vertex` element's `x`, `y`, and `z` are read; other vertex properties (colours, normals) are skipped by
their declared types. Elements declared before `vertex` must have no list properties in a binary file, since their
size can't be known without parsing them; reconstruction tools write `vertex` first, so this isn't a limit in
practice. No dependency beyond numpy.
"""

import pathlib
from collections.abc import Mapping

import numpy as np

_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


class PlyError(ValueError):
    """The file isn't a PLY point cloud this reader understands."""


def _element_dtype(props: list[tuple[str, str]], order: str) -> np.dtype:
    try:
        return np.dtype([(name, order + _TYPES[kind]) for name, kind in props])
    except KeyError as missing:
        raise PlyError(f"unknown PLY property type {missing}") from None


def load(path: pathlib.Path | str) -> np.ndarray:
    """The vertex positions in `path` as an (N, 3) float64 array."""
    data = pathlib.Path(path).read_bytes()
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise PlyError(f"{path} is not a PLY file")
    body = data[data.index(b"\n", end) + 1 :]
    fmt = ""
    elements: list[tuple[str, int, list[tuple[str, str]], bool]] = []  # name, count, scalar props, has a list prop
    for line in data[:end].decode("ascii", "replace").splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "format":
            fmt = words[1]
        elif words[0] == "element":
            elements.append((words[1], int(words[2]), [], False))
        elif words[0] == "property" and elements:
            name, count, props, has_list = elements[-1]
            if words[1] == "list":
                elements[-1] = (name, count, props, True)
            else:
                props.append((words[2], words[1]))
    names = [e[0] for e in elements]
    if "vertex" not in names:
        raise PlyError(f"{path} has no vertex element")
    _, count, props, has_list = elements[names.index("vertex")]
    if not {"x", "y", "z"} <= {p[0] for p in props} or has_list:
        raise PlyError(f"{path}'s vertices have no plain x, y, and z")
    if fmt == "ascii":
        lines = body.decode("ascii").splitlines()
        skip = sum(e[1] for e in elements[: names.index("vertex")])
        rows = np.array([line.split() for line in lines[skip : skip + count]], dtype=np.float64)
        columns = [p[0] for p in props]
        return rows[:, [columns.index(axis) for axis in ("x", "y", "z")]]
    order = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(fmt)
    if order is None:
        raise PlyError(f"unknown PLY format {fmt!r}")
    offset = 0
    for name, n, before, listed in elements[: names.index("vertex")]:
        if listed:
            raise PlyError(f"{path}: element {name!r} before the vertices has a list property")
        offset += n * _element_dtype(before, order).itemsize
    vertices = np.frombuffer(body, dtype=_element_dtype(props, order), count=count, offset=offset)
    return np.stack([vertices[axis].astype(np.float64) for axis in ("x", "y", "z")], axis=1)


def dumps(
    points: np.ndarray, colors: np.ndarray | None = None, counts: Mapping[str, np.ndarray] | None = None
) -> bytes:
    """An (N, 3) point cloud, with optional (N, 3) 0 to 255 RGB colours and optional per-point counts (a ushort
    property each, by name: tier 1's `seen` and `missed`), as a binary little-endian PLY: float32 x, y, z, then uchar
    red, green, blue when coloured, then the counts. `load` reads the points back, and `read` all of it."""
    xyz = np.asarray(points, dtype="<f4").reshape(-1, 3)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    props = ["property float x", "property float y", "property float z"]
    if colors is not None:
        rgb = np.asarray(colors).reshape(-1, 3)
        if len(rgb) != len(xyz):
            raise ValueError(f"{len(rgb)} colours for {len(xyz)} points")
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        props += ["property uchar red", "property uchar green", "property uchar blue"]
    for name, values in (counts or {}).items():
        if len(values) != len(xyz):
            raise ValueError(f"{len(values)} {name} counts for {len(xyz)} points")
        fields.append((name, "<u2"))
        props.append(f"property ushort {name}")
    rows = np.empty(len(xyz), dtype=fields)
    for i, axis in enumerate("xyz"):
        rows[axis] = xyz[:, i]
    if colors is not None:
        for i, channel in enumerate(("red", "green", "blue")):
            rows[channel] = np.clip(rgb[:, i], 0, 255)
    for name, values in (counts or {}).items():
        rows[name] = np.clip(np.asarray(values), 0, 65535)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}", *props, "end_header", ""]
    return "\n".join(header).encode("ascii") + rows.tobytes()


Cloud = tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]


def vertices(data: bytes) -> np.ndarray | None:
    """Every vertex of a binary little-endian PLY whose only element is `vertex` and whose properties are all scalar
    (what `dumps` writes, and a 3D Gaussian splat's PLY), as a structured array with a field per property; None for
    any other PLY. Raises PlyError when `data` isn't a PLY at all."""
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise PlyError("not a PLY file")
    lines = data[:end].decode("ascii", "replace").splitlines()
    fmt = next((ln.split()[1] for ln in lines if ln.startswith("format ")), "")
    elements = [ln.split() for ln in lines if ln.startswith("element ")]
    if fmt != "binary_little_endian" or len(elements) != 1 or elements[0][1] != "vertex":
        return None
    props = [ln.split() for ln in lines if ln.startswith("property ")]
    if any(words[1] == "list" or words[1] not in _TYPES for words in props):
        return None
    dtype = np.dtype([(words[2], "<" + _TYPES[words[1]]) for words in props])
    body = data[data.index(b"\n", end) + 1 :]
    return np.frombuffer(body, dtype=dtype, count=int(elements[0][2]))


def read(path: pathlib.Path | str) -> Cloud:
    """The points (N, 3, float64), the colours (N, 3, uint8) when the file has `red`, `green`, and `blue` vertex
    properties, and every other ushort vertex property by name (the counts `dumps` writes) of a PLY cloud. A file
    `vertices` reads (what `dumps` writes) is read directly; any other file gives its points through `load`, with no
    colours or counts."""
    try:
        rows = vertices(pathlib.Path(path).read_bytes())
    except PlyError as e:
        raise PlyError(f"{path} is not a PLY file") from e
    if rows is None:
        return load(path), None, {}
    names = rows.dtype.names or ()
    points = np.c_[rows["x"], rows["y"], rows["z"]].astype(np.float64)
    colors = None
    if {"red", "green", "blue"} <= set(names):
        colors = np.c_[rows["red"], rows["green"], rows["blue"]].astype(np.uint8)
    counts = {n: rows[n].astype(np.int64) for n in names if rows.dtype[n] == np.dtype("<u2")}
    return points, colors, counts
