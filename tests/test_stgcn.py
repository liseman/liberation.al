"""ST-GCN++ graph, input, and scoring with temporary synthetic checkpoints.

These tests exercise strict loading and batch independence, not trained-weight accuracy or upstream compatibility.
Device comparison is opt-in: set LIBERATIONAL_TEST_DEVICE to the accelerator to test (cuda:N or mps).
The comparison uses only synthetic weights, and never searches the installation's model directory.
"""

import os

import numpy as np
import pytest
import torch

from liberational.see import stgcn


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
        state = stgcn.STGCNPlusPlus().state_dict()
    path = tmp_path / "synthetic.pth"
    torch.save(state, path)
    return path


@pytest.mark.parametrize("fault", ["missing", "unexpected"])
def test_checkpoint_keys_must_match_strictly(checkpoint, fault):
    state = torch.load(checkpoint, weights_only=True)
    if fault == "missing":
        del state["cls_head.fc_cls.weight"]
    else:
        state["unexpected.weight"] = torch.ones(1)
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key" if fault == "missing" else "Unexpected key"):
        stgcn.ActionModel(checkpoint, torch.device("cpu"))


def test_the_graph_is_pyskls_coco_spatial_adjacency():
    A = stgcn.adjacency().numpy()
    assert A.shape == (3, 17, 17)
    np.testing.assert_array_equal(A[0], np.eye(17))
    assert A[1][13, 15] == 1.0  # the left ankle's one inward link, to the left knee
    assert A[1][:, 0].sum() == 0  # the nose is the center: nothing inward from it
    assert [A[2][j, 0] for j in (1, 2, 5, 6)] == [0.25] * 4  # four links out of the nose, normalized


def test_windows_become_pyskls_normalized_two_person_input():
    w = np.zeros((1, stgcn.CLIP, 26, 3), np.float32)
    w[0, :, 0] = (0.5, 0.25, 0.9)
    w[0, :, 1] = (0.8, 0.8, 0.005)  # under pyskl's 0.01 threshold
    w[0, :, 20] = (0.9, 0.9, 0.9)  # a halpe26 foot joint, past COCO's 17
    x = stgcn.model_input(w)
    assert x.shape == (1, 2, stgcn.CLIP, 17, 3)
    assert x[0, 0, 0, 0].tolist() == pytest.approx([0.0, -0.5, 0.9])
    assert x[0, 0, 0, 1].tolist() == pytest.approx([0.0, 0.0, 0.005])
    assert not x[0, 1].any()  # the second person slot stays empty, as training padded a lone person


def test_each_window_scores_independently_with_an_isolated_checkpoint(checkpoint):
    model = stgcn.ActionModel(checkpoint, torch.device("cpu"))
    rng = np.random.default_rng(0)
    windows = rng.uniform(0.2, 0.8, (3, stgcn.CLIP, 26, 3)).astype(np.float32)
    together = model.score(windows)
    assert together.shape == (3, len(stgcn.CLASSES))
    np.testing.assert_allclose(together.sum(1), 1.0, rtol=1e-5)
    alone = np.concatenate([model.score(windows[i : i + 1]) for i in range(3)])
    np.testing.assert_allclose(together, alone, rtol=1e-5, atol=1e-7)  # a window's scores don't depend on its batch
    assert model.score(windows[:0]).shape == (0, len(stgcn.CLASSES))


@pytest.mark.skipif(not os.environ.get("LIBERATIONAL_TEST_DEVICE"), reason="set LIBERATIONAL_TEST_DEVICE to opt in")
def test_an_accelerated_device_scores_as_the_cpu_does(checkpoint):
    device = torch.device(os.environ["LIBERATIONAL_TEST_DEVICE"])
    windows = np.random.default_rng(1).uniform(0.2, 0.8, (4, stgcn.CLIP, 26, 3)).astype(np.float32)
    want = stgcn.ActionModel(checkpoint, torch.device("cpu")).score(windows)
    got = stgcn.ActionModel(checkpoint, device).score(windows)
    np.testing.assert_allclose(got, want, atol=2e-3)  # probabilities: CUDA's TF32 and MPS's kernels round a little
    assert (got.argmax(1) == want.argmax(1)).all()
