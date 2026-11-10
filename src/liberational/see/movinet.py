"""MoViNet-A2 (Kondratyuk et al., 2021) trained on Kinetics-600, for actions an RGB clip shows better than a
skeleton: eating, drinking, a phone, cooking, washing up, reading, TV, and cleaning.

A port of the base (not streaming) model from Atze00/MoViNet-pytorch (MIT), itself a port of the authors' TensorFlow
weights (Apache-2.0): 3D convolutions with TensorFlow's "same" padding. Only what those weights use is here, so the
checkpoint loads strictly. It reads clips of RGB in 0 to 1, N x 3 x T x 224 x 224; the weights were trained on
50-frame clips at 5 fps.
"""

import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

from liberational.see.movinet_classes import CLASSES

SIZE = 224  # a clip's frames are this many pixels square
FRAMES = 50  # and it holds this many, 10 s at 5 fps

# Per block: input, output, and expanded channels, kernel (t, h, w), and spatial stride, from the A2 config.
BLOCKS: tuple[tuple[tuple[int, int, int, tuple[int, int, int], int], ...], ...] = (
    ((16, 16, 40, (1, 5, 5), 2), (16, 16, 40, (3, 3, 3), 1), (16, 16, 64, (3, 3, 3), 1)),
    (
        (16, 40, 96, (3, 3, 3), 2),
        (40, 40, 120, (3, 3, 3), 1),
        (40, 40, 96, (3, 3, 3), 1),
        (40, 40, 96, (3, 3, 3), 1),
        (40, 40, 120, (3, 3, 3), 1),
    ),
    (
        (40, 72, 240, (5, 3, 3), 2),
        (72, 72, 160, (3, 3, 3), 1),
        (72, 72, 240, (3, 3, 3), 1),
        (72, 72, 192, (3, 3, 3), 1),
        (72, 72, 240, (3, 3, 3), 1),
    ),
    (
        (72, 72, 240, (5, 3, 3), 1),
        (72, 72, 240, (3, 3, 3), 1),
        (72, 72, 240, (3, 3, 3), 1),
        (72, 72, 240, (3, 3, 3), 1),
        (72, 72, 144, (1, 5, 5), 1),
        (72, 72, 240, (3, 3, 3), 1),
    ),
    (
        (72, 144, 480, (5, 3, 3), 2),
        (144, 144, 384, (1, 5, 5), 1),
        (144, 144, 384, (1, 5, 5), 1),
        (144, 144, 480, (1, 5, 5), 1),
        (144, 144, 480, (1, 5, 5), 1),
        (144, 144, 480, (3, 3, 3), 1),
        (144, 144, 576, (1, 3, 3), 1),
    ),
)


def same_pad(x: Tensor, kernel: int, stride: int) -> Tensor:
    """Pad height and width as TensorFlow's "same" does: the extra pixel, if any, goes after."""
    pads = []
    for size in (x.shape[-1], x.shape[-2]):
        total = max(kernel - (stride if size % stride == 0 else size % stride), 0)
        pads += [total // 2, total - total // 2]
    return F.pad(x, pads)


def make_divisible(v: float, divisor: int = 8) -> int:
    rounded = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * v else rounded


class Conv(nn.Module):
    """A 3D convolution with "same" padding in space and symmetric padding in time, then batch norm and swish if
    asked. Its weights sit at conv_1.conv3d and conv_1.norm, as in the checkpoint."""

    def __init__(
        self,
        cin: int,
        cout: int,
        kernel: tuple[int, int, int] = (1, 1, 1),
        stride: int = 1,
        *,
        groups: int = 1,
        norm: bool = True,
        act: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel, self.stride = kernel, stride
        self.conv_1 = nn.Sequential()
        self.conv_1.add_module(
            "conv3d",
            nn.Conv3d(cin, cout, kernel, (1, stride, stride), ((kernel[0] - 1) // 2, 0, 0), groups=groups, bias=bias),
        )
        if norm:
            self.conv_1.add_module("norm", nn.BatchNorm3d(cout, eps=0.001))
        if act:
            self.conv_1.add_module("act", nn.SiLU())

    def forward(self, x: Tensor) -> Tensor:
        if self.kernel[1] > 1 or self.stride > 1:
            x = same_pad(x, self.kernel[1], self.stride)
        return self.conv_1(x)


def tf_avg_pool(x: Tensor) -> Tensor:
    """TensorFlow's 3 x 3, stride 2, "same" average pool over height and width, which leaves the padding out of the
    average. Square frames only."""
    if x.shape[-1] % 2:
        return F.avg_pool3d(x, (1, 3, 3), (1, 2, 2), (0, 1, 1), count_include_pad=False)
    x = F.avg_pool3d(F.pad(x, (0, 1, 0, 1)), (1, 3, 3), (1, 2, 2))
    x[..., -1] = x[..., -1] * 9 / 6
    x[..., -1, :] = x[..., -1, :] * 9 / 6
    return x


class Squeeze(nn.Module):
    """Squeeze and excitation over the whole clip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        squeezed = make_divisible(channels // 4)
        self.fc1 = Conv(channels, squeezed, norm=False, act=False, bias=True)
        self.fc2 = Conv(squeezed, channels, norm=False, act=False, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        scale = self.fc2(F.silu(self.fc1(F.adaptive_avg_pool3d(x, 1))))
        return torch.sigmoid(scale) * x


class Bneck(nn.Module):
    def __init__(self, cin: int, cout: int, expanded: int, kernel: tuple[int, int, int], stride: int) -> None:
        super().__init__()
        self.pool = stride > 1
        self.res: nn.Sequential | None = None
        if stride > 1 or cin != cout:
            self.res = nn.Sequential()
            self.res.add_module("1" if self.pool else "0", Conv(cin, cout, act=False))
        self.expand = Conv(cin, expanded)
        self.deep = Conv(expanded, expanded, kernel, stride, groups=expanded)
        self.se = Squeeze(expanded)
        self.project = Conv(expanded, cout, act=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        if self.res is not None:
            residual = self.res(tf_avg_pool(x) if self.pool else x)
        return residual + self.alpha * self.project(self.se(self.deep(self.expand(x))))


class MoViNetA2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv(3, 16, (1, 3, 3), 2)
        self.blocks = nn.Sequential()
        for i, block in enumerate(BLOCKS):
            for j, (cin, cout, expanded, kernel, stride) in enumerate(block):
                self.blocks.add_module(f"b{i}_l{j}", Bneck(cin, cout, expanded, kernel, stride))
        self.conv7 = Conv(144, 640)
        self.classifier = nn.Sequential()
        self.classifier.add_module("0", Conv(640, 2048, norm=False, act=False, bias=True))
        self.classifier.add_module("1", nn.SiLU())
        self.classifier.add_module("3", Conv(2048, len(CLASSES), norm=False, act=False, bias=True))

    def forward(self, x: Tensor) -> Tensor:
        """Logits, N x 600, for clips N x 3 x T x H x W of RGB in 0 to 1."""
        x = self.conv7(self.blocks(self.conv1(x)))
        return self.classifier(F.adaptive_avg_pool3d(x, 1)).flatten(1)


class ClipModel:
    """MoViNet-A2 with its weights, ready to score clips on `device`."""

    def __init__(self, weights: pathlib.Path, device: torch.device) -> None:
        self.model = MoViNetA2()
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.model.eval().to(device)
        self.device = device

    def score(self, clips: NDArray[np.uint8] | Tensor) -> NDArray[np.float32]:
        """Class probabilities, N x 600, for N clips of RGB frames, N x T x SIZE x SIZE x 3 in 0 to 255."""
        x = torch.as_tensor(clips, device=self.device).permute(0, 4, 1, 2, 3).float() / 255
        with torch.inference_mode():
            return torch.softmax(self.model(x), dim=1).cpu().numpy()
