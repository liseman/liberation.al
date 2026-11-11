"""The SOG copy of a splat (spatial/sog.py), checked by a decoder written from PlayCanvas's reader
(splat-transform 3.10.3, `readSogSourceV2`)."""

import io
import json
import math
import zipfile

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from liberational.spatial import sog

SH_COEFFS = {0: 0, 1: 3, 2: 8, 3: 15}


def decode(data: bytes) -> dict:
    """A `.sog` file's Gaussians, in the file's order: centres, opacities, DC colours, log scales, rotations (w, x, y,
    z), and the higher SH coefficients (N, 3 channels, coefficients)."""
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        meta = json.loads(bundle.read("meta.json"))

        def image(name):
            bgr = cv2.imdecode(np.frombuffer(bundle.read(name), np.uint8), cv2.IMREAD_UNCHANGED)
            assert bgr is not None
            if bgr.shape[2] == 3:  # an image whose alpha is 255 throughout is stored without it
                bgr = np.dstack([bgr, np.full(bgr.shape[:2], 255, np.uint8)])
            return bgr[..., [2, 1, 0, 3]].reshape(-1, 4).astype(np.int64)

        n = meta["count"]
        assert meta["version"] == 2
        low, high = (image(f)[:n] for f in meta["means"]["files"])
        mins, maxs = np.array(meta["means"]["mins"]), np.array(meta["means"]["maxs"])
        logged = mins + (maxs - mins) * (low[:, :3] | (high[:, :3] << 8)) / 65535
        xyz = np.sign(logged) * (np.exp(np.abs(logged)) - 1)
        quats = np.zeros((n, 4))
        for g, (r, gr, b, tag) in enumerate(image(meta["quats"]["files"][0])[:n]):
            largest = tag - 252
            assert 0 <= largest <= 3
            others = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]][largest]
            quats[g, others] = (np.array([r, gr, b]) / 255 * 2 - 1) / math.sqrt(2)
            quats[g, largest] = math.sqrt(max(0.0, 1 - float((quats[g] ** 2).sum())))
        scales = np.array(meta["scales"]["codebook"])[image(meta["scales"]["files"][0])[:n, :3]]
        colour = image(meta["sh0"]["files"][0])[:n]
        dc = np.array(meta["sh0"]["codebook"])[colour[:, :3]]
        rest = np.zeros((n, 3, 0))
        if "shN" in meta:
            coeffs = SH_COEFFS[meta["shN"]["bands"]]
            centroids = cv2.imdecode(
                np.frombuffer(bundle.read(meta["shN"]["files"][0]), np.uint8), cv2.IMREAD_UNCHANGED
            )
            assert centroids is not None and centroids.shape[1] == 64 * coeffs
            rgb = centroids[..., [2, 1, 0]]
            labels = image(meta["shN"]["files"][1])[:n]
            label = labels[:, 0] | (labels[:, 1] << 8)
            assert label.max() < meta["shN"]["count"]
            book = np.array(meta["shN"]["codebook"])
            # entry `label` is `coeffs` pixels from column (label % 64) * coeffs of row label // 64
            cols = (label % 64)[:, None] * coeffs + np.arange(coeffs)
            rest = book[rgb[(label // 64)[:, None], cols]].transpose(0, 2, 1)  # (N, 3 channels, coefficients)
        return {"xyz": xyz, "alpha": colour[:, 3] / 255, "dc": dc, "scale": scales, "wxyz": quats, "rest": rest}


def gaussians(xyz, dc, opacity, scales, wxyz, rest):
    """A 3DGS PLY's vertices: `rest` is (N, 3 channels, coefficients), as the PLY lays it out channel by channel."""
    n, coeffs = len(xyz), rest.shape[2]
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", *(f"f_rest_{i}" for i in range(3 * coeffs))]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    cols = np.concatenate([xyz, dc, rest.reshape(n, -1), opacity[:, None], scales, wxyz], axis=1)
    return np.rec.fromarrays(cols.T.astype("<f4"), names=names)


def test_one_gaussian_encodes_to_the_values_worked_out_by_hand():
    rows = gaussians(
        xyz=np.array([[1.0, 2.0, -0.5]]),
        dc=np.full((1, 3), 0.25),
        opacity=np.array([0.0]),
        scales=np.full((1, 3), -4.0),
        wxyz=np.array([[0.0, -1.0, 0.0, 0.0]]),  # a half turn about x, stored with its largest component negative
        rest=np.zeros((1, 3, 0)),
    )
    got = decode(sog.encode(rows))
    assert got["xyz"][0] == pytest.approx([1.0, 2.0, -0.5], abs=1e-6)  # the map's frame, nothing flipped
    assert got["alpha"][0] == pytest.approx(128 / 255)  # sigmoid(0) = 0.5
    assert got["dc"][0] == pytest.approx([0.25] * 3) and got["scale"][0] == pytest.approx([-4.0] * 3)
    assert got["wxyz"][0] == pytest.approx([0.0, 1.0, 0.0, 0.0], abs=1 / 255)  # q and -q are one rotation


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@st.composite
def clouds(draw):
    n = draw(st.integers(1, 12))
    degree = draw(st.sampled_from([0, 1, 2, 3]))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    return gaussians(
        xyz=rng.uniform(-50, 50, (n, 3)),
        dc=rng.uniform(-3, 3, (n, 3)),
        opacity=rng.uniform(-5, 5, n),
        scales=rng.uniform(-9, 5, (n, 3)),
        wxyz=unit(rng.normal(size=(n, 4)) + 1e-3),
        rest=rng.uniform(-0.9, 0.9, (n, 3, SH_COEFFS[degree])),
    )


@settings(max_examples=60, deadline=None)
@given(clouds())
def test_every_gaussian_decodes_back_within_its_quantization(rows):
    got = decode(sog.encode(rows))
    xyz = np.c_[rows["x"], rows["y"], rows["z"]].astype(float)
    # The file is in Morton order: each decoded Gaussian is matched to the one it sits on.
    at = np.argmin(np.linalg.norm(got["xyz"][:, None] - xyz[None], axis=2), axis=1)
    assert sorted(at.tolist()) == list(range(len(rows))), "every Gaussian once"
    rows = rows[at]
    xyz = xyz[at]
    logged = np.sign(xyz) * np.log1p(np.abs(xyz))
    step = (logged.max(0) - logged.min(0)) / 65535  # 16 bits over the cloud's box, in log space
    assert (np.abs(got["xyz"] - xyz) <= step * (np.abs(xyz) + 1) * 0.51 + 1e-5).all()
    assert np.abs(got["alpha"] - 1 / (1 + np.exp(-rows["opacity"].astype(float)))).max() <= 0.5 / 255 + 1e-6
    # A dozen Gaussians have fewer values than a codebook has entries, so each comes back as it went in.
    assert got["dc"] == pytest.approx(np.c_[rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"]], abs=1e-5)
    assert got["scale"] == pytest.approx(np.c_[rows["scale_0"], rows["scale_1"], rows["scale_2"]], abs=1e-5)
    wxyz = np.c_[rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"]].astype(float)
    assert np.abs(np.abs((got["wxyz"] * wxyz).sum(axis=1)) - 1).max() < 1e-3
    coeffs = got["rest"].shape[2]
    rest = np.array([[rows[f"f_rest_{i}"][g] for i in range(3 * coeffs)] for g in range(len(rows))])
    # ... unless the SH coefficients alone outnumber them (bands 2 and 3), when each lands on a near entry.
    close = 1e-5 if rest.size <= 256 else 0.02
    assert got["rest"].reshape(len(rows), -1) == pytest.approx(rest.reshape(len(rows), -1), abs=close)


def test_many_values_share_a_codebook_that_crowds_where_they_do():
    rng = np.random.default_rng(0)
    values = np.r_[rng.normal(-4.0, 0.05, 50_000), rng.normal(1.0, 0.5, 500)]  # most scales alike, a few large
    book, at = sog.codebook(values)
    assert len(book) == 256 and (np.diff(book) >= 0).all()
    err = np.abs(book[at] - values)
    assert np.median(err[:50_000]) < 0.001 and err.max() < 0.5


def test_morton_order_walks_a_grid_cell_by_cell():
    corners = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)  # x fastest
    shuffled = corners[[5, 2, 7, 0, 3, 6, 1, 4]]
    assert shuffled[sog.morton(shuffled)].tolist() == corners.tolist()


def test_an_sh_count_that_isnt_a_degree_is_refused():
    rows = gaussians(
        np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(1), np.zeros((1, 3)), np.eye(1, 4), np.zeros((1, 3, 2))
    )
    with pytest.raises(ValueError, match="SH degree"):
        sog.encode(rows)


def with_views(rows: np.ndarray, views: np.ndarray) -> np.ndarray:
    """`rows` with a `view` property, as the trainer writes it (a float)."""
    names = [*(rows.dtype.names or ()), "view"]
    return np.rec.fromarrays([*(rows[k] for k in rows.dtype.names or ()), views.astype("<f4")], names=names)


@settings(max_examples=40, deadline=None)
@given(clouds())
def test_each_gaussians_view_is_given_in_the_files_own_order(rows):
    # Each Gaussian's view is its own index, so the view at pixel i names the Gaussian the file put there.
    tagged = with_views(rows, np.arange(len(rows)))
    got = decode(sog.encode(tagged))
    views = np.frombuffer(sog.views(tagged) or b"", "<u2")
    assert len(views) == len(rows)
    xyz = np.c_[rows["x"], rows["y"], rows["z"]].astype(float)
    nearest = np.argmin(np.linalg.norm(got["xyz"][:, None] - xyz[None], axis=2), axis=1)
    assert views.tolist() == nearest.tolist()


def test_a_splat_without_views_has_none_and_a_view_that_isnt_one_is_untagged():
    rows = gaussians(
        xyz=np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        dc=np.zeros((3, 3)),
        opacity=np.zeros(3),
        scales=np.zeros((3, 3)),
        wxyz=np.tile([1.0, 0, 0, 0], (3, 1)),
        rest=np.zeros((3, 3, 0)),
    )
    assert sog.views(rows) is None
    # Along x the Morton order is the rows' own.
    got = sog.views(with_views(rows, np.array([4.0, np.nan, 2.5])), untagged=99)
    assert np.frombuffer(got or b"", "<u2").tolist() == [4, 99, 99]
