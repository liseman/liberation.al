"""MoViNet-A2 padding, pooling, kinds, and scoring with temporary synthetic checkpoints.

These tests exercise strict loading and batch independence, not trained-weight accuracy or upstream compatibility.
"""

import numpy as np
import pytest
import torch

from liberational.see import movinet
from liberational.see.movinet_classes import CLASSES


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def checkpoint(tmp_path):
    # Synthetic weights exercise the real architecture and loader without installation files or downloads.
    # Keep the random state local, including on machines with accelerators.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        state = movinet.MoViNetA2().state_dict()
    path = tmp_path / "synthetic.pth"
    torch.save(state, path)
    return path


@pytest.mark.parametrize("fault", ["missing", "unexpected"])
def test_checkpoint_keys_must_match_strictly(checkpoint, fault):
    state = torch.load(checkpoint, weights_only=True)
    if fault == "missing":
        del state["conv1.conv_1.conv3d.weight"]
    else:
        state["unexpected.weight"] = torch.ones(1)
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key" if fault == "missing" else "Unexpected key"):
        movinet.ClipModel(checkpoint, torch.device("cpu"))


@pytest.mark.parametrize(
    ("width", "before", "after"),
    [pytest.param(7, 1, 1, id="odd-splits-evenly"), pytest.param(8, 0, 1, id="even-pads-after-only")],
)
def test_same_padding_puts_the_odd_pixel_after_as_tensorflow_does(width, before, after):
    x = torch.ones(1, 1, 1, width, width)
    padded = movinet.same_pad(x, kernel=3, stride=2)
    assert padded.shape[-1] == width + before + after
    assert padded[0, 0, 0, before:-after, before:-after].all()
    assert not padded[0, 0, 0, :, -1].any()


def test_the_average_pool_leaves_the_padding_out_of_each_mean():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4)
    # A 4 x 4 frame pools to 2 x 2 from windows at rows and columns 0 to 2 and 2 to 3; the second runs into padding.
    assert movinet.tf_avg_pool(x)[0, 0, 0].tolist() == [[5.0, 6.5], [11.0, 12.5]]


def test_each_clip_scores_independently_with_an_isolated_checkpoint(checkpoint):
    model = movinet.ClipModel(checkpoint, torch.device("cpu"))
    rng = np.random.default_rng(0)
    clips = rng.integers(0, 256, (2, 2, movinet.SIZE, movinet.SIZE, 3), dtype=np.uint8)
    together = model.score(clips)
    assert together.shape == (2, len(CLASSES))
    np.testing.assert_allclose(together.sum(1), 1.0, rtol=1e-5)
    alone = np.concatenate([model.score(clips[i : i + 1]) for i in range(2)])
    np.testing.assert_allclose(together, alone, rtol=1e-5, atol=1e-7)  # a clip's scores don't depend on its batch
