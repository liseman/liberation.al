"""The action model: ST-GCN++ on 2D skeletons, with pyskl's stock NTU RGB+D 120 weights.

ST-GCN++ (PYSKL, Duan et al. 2022) in plain PyTorch, the joint stream trained on HRNet 2D keypoints, NTU-120
cross-subject: 84.4% top-1 there, and well below that in a house. Fast first, accuracy
tuning later, so this is the stock model as pyskl tests it (`configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py`),
with no fine-tuning. The module names match pyskl's, so its checkpoint loads strictly: every key and shape must
match, which is what checks the port.

Input is what pyskl's test pipeline makes: COCO's 17 joints (the first 17 of halpe26, in the same order), x and y
in -1 to 1 across the frame (PreNormalize2D's fixed mode), the joint's confidence as the third channel, a joint at
or under 0.01 confidence zeroed, 100 frames, and two person slots with the second left empty as training padded a
lone person. Output is each class's softmax probability, as pyskl averages clips.

The checkpoint is read with `torch.load(weights_only=True)`, which rebuilds tensors and never runs code from the
file. (torch types `nn.Module.forward` as an assignable attribute, so ty flags each `forward` here as overriding
it; that's how every torch module is written.)
"""

import pathlib

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from liberational.see.action_classes import CLASSES, CLIP

__all__ = ["CLASSES", "CLIP", "ActionModel", "STGCNPlusPlus", "adjacency", "model_input"]

COCO_JOINTS = 17
EPS = 1e-4

_COCO_EDGES = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (9, 7), (7, 5),
    (10, 8), (8, 6), (5, 0), (6, 0), (1, 0), (3, 1), (2, 0), (4, 2),
)  # fmt: skip


def _normalized(edges: NDArray[np.float64]) -> NDArray[np.float64]:
    """pyskl's normalize_digraph: each column divided by its sum."""
    sums = edges.sum(0)
    return edges / np.where(sums > 0, sums, 1)


def adjacency() -> torch.Tensor:
    """pyskl's `Graph(layout='coco', mode='spatial').A`: self links, inward, and outward, column-normalized."""
    inward = np.zeros((COCO_JOINTS, COCO_JOINTS))
    for i, j in _COCO_EDGES:
        inward[j, i] = 1
    return torch.tensor(
        np.stack((np.eye(COCO_JOINTS), _normalized(inward), _normalized(inward.T))), dtype=torch.float32
    )


class UnitTCN(nn.Module):
    def __init__(
        self, cin: int, cout: int, kernel: int = 9, stride: int = 1, dilation: int = 1, norm: bool = True
    ) -> None:
        super().__init__()
        pad = (kernel + (kernel - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(cin, cout, (kernel, 1), padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(cout) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class MSTCN(nn.Module):
    """pyskl's mstcn: four dilated 3-tap branches, a max-pool branch, and a 1x1 branch, fused."""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        mid = cout // 6
        first = cout - mid * 5
        branches: list[nn.Module] = []
        for i, dilation in enumerate((1, 2, 3, 4)):
            c = first if i == 0 else mid
            branches.append(
                nn.Sequential(
                    nn.Conv2d(cin, c, 1), nn.BatchNorm2d(c), nn.ReLU(), UnitTCN(c, c, 3, stride, dilation, norm=False)
                )
            )
        branches.append(
            nn.Sequential(
                nn.Conv2d(cin, mid, 1),
                nn.BatchNorm2d(mid),
                nn.ReLU(),
                nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
            )
        )
        branches.append(nn.Conv2d(cin, mid, 1, stride=(stride, 1)))
        self.branches = nn.ModuleList(branches)
        total = first + mid * 5
        self.transform = nn.Sequential(nn.BatchNorm2d(total), nn.ReLU(), nn.Conv2d(total, cout, 1))
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.transform(torch.cat([branch(x) for branch in self.branches], dim=1)))


class UnitGCN(nn.Module):
    """pyskl's unit_gcn with adaptive='init' (a learned adjacency) and a residual."""

    def __init__(self, cin: int, cout: int, A: torch.Tensor) -> None:
        super().__init__()
        self.subsets = A.size(0)
        self.A = nn.Parameter(A.clone())
        self.conv = nn.Conv2d(cin, cout * self.subsets, 1)
        self.down = nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout)) if cin != cout else nn.Identity()
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        res = self.down(x)
        x = self.conv(x).view(n, self.subsets, -1, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, self.A).contiguous()
        return self.act(self.bn(x) + res)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, A: torch.Tensor, stride: int = 1, residual: bool = True) -> None:
        super().__init__()
        self.gcn = UnitGCN(cin, cout, A)
        self.tcn = MSTCN(cout, cout, stride)
        self.relu = nn.ReLU()
        self.skip = residual
        self.residual: nn.Module = (
            nn.Identity()
            if not residual or (cin == cout and stride == 1)
            else UnitTCN(cin, cout, kernel=1, stride=stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x) if self.skip else 0
        return self.relu(self.tcn(self.gcn(x)) + res)


class Backbone(nn.Module):
    """pyskl's STGCN: ten blocks, 64 channels doubling at blocks 5 and 8, where time is also halved."""

    def __init__(self) -> None:
        super().__init__()
        A = adjacency()
        self.data_bn = nn.BatchNorm1d(3 * COCO_JOINTS)
        blocks = [Block(3, 64, A, residual=False)]
        channels = 64
        for stage in range(2, 11):
            stride = 2 if stage in (5, 8) else 1
            out = channels * 2 if stage in (5, 8) else channels
            blocks.append(Block(channels, out, A, stride))
            channels = out
        self.gcn = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, m, t, v, c = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(n * m, v * c, t)
        x = self.data_bn(x).view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
        for block in self.gcn:
            x = block(x)
        return x.reshape((n, m) + x.shape[1:])


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_cls = nn.Linear(256, len(CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, m, c = x.shape[:3]
        return self.fc_cls(x.mean(dim=(3, 4)).reshape(n, m, c).mean(dim=1))


class STGCNPlusPlus(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.cls_head = Head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """N x 2 persons x CLIP frames x 17 joints x 3 (x, y, confidence) -> N x 120 logits."""
        return self.cls_head(self.backbone(x))


def model_input(windows: NDArray[np.floating]) -> NDArray[np.float32]:
    """Windows of one person's joints, N x CLIP x K x 3 (x and y normalized 0 to 1 across the frame, confidence;
    K is 17 or more, halpe26's first 17 being COCO's), as the model's input: N x 2 x CLIP x 17 x 3."""
    joints = np.asarray(windows, np.float32)[:, :, :COCO_JOINTS].copy()
    joints[..., :2] = joints[..., :2] * 2 - 1
    joints[joints[..., 2] <= 0.01, :2] = 0
    joints[..., 2] = joints[..., 2].clip(0, 1)
    out = np.zeros((len(joints), 2, *joints.shape[1:]), np.float32)
    out[:, 0] = joints
    return out


class ActionModel:
    """ST-GCN++ with its weights, ready to score windows on `device`."""

    def __init__(self, weights: pathlib.Path, device: torch.device) -> None:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.model = STGCNPlusPlus()
        self.model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        self.model.eval().to(device)
        self.device = device

    def score(self, windows: NDArray[np.floating]) -> NDArray[np.float32]:
        """Class probabilities, N x 120, for N windows of one person's joints (see `model_input`)."""
        if not len(windows):
            return np.zeros((0, len(CLASSES)), np.float32)
        x = torch.from_numpy(model_input(windows)).to(self.device)
        with torch.inference_mode():
            return torch.softmax(self.model(x), dim=1).cpu().numpy()
