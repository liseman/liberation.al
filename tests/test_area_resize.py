"""area_resize matches F.interpolate(mode="area") and works where MPS can't (non-divisible sizes)."""

import pytest

torch = pytest.importorskip("torch")
gpu_pose = pytest.importorskip("liberational.see.gpu_pose")
F = torch.nn.functional


def test_matches_area_on_the_cpu() -> None:
    x = torch.rand(1, 3, 173, 211)
    assert torch.allclose(gpu_pose.area_resize(x, (64, 64)), F.interpolate(x, size=(64, 64), mode="area"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_non_divisible_sizes_work_on_mps() -> None:
    x = torch.rand(1, 3, 173, 211)
    out = gpu_pose.area_resize(x.to("mps"), (64, 64))
    assert out.device.type == "mps"
    assert torch.allclose(out.cpu(), F.interpolate(x, size=(64, 64), mode="area"), atol=1e-5)
