"""
Reusable building blocks for the lightweight face segmentation model.

- DepthwiseSeparableConv: factored convolution (depthwise + pointwise)
- SqueezeExcitation: channel attention (recalibrate feature importance)
- DSResBlock: depthwise separable residual block + SE attention
- AttentionGate: additive attention on skip connections
- ASPP_Lite: lightweight atrous spatial pyramid pooling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: spatial depthwise + 1×1 pointwise."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Learns per-channel importance weights via global pooling → FC → ReLU → FC → Sigmoid.
    Very parameter-efficient: only 2×C×(C//reduction) params, but provides
    significant accuracy gains by letting the network recalibrate which
    feature channels are most discriminative.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class DSResBlock(nn.Module):
    """
    Depthwise Separable Residual Block with SE channel attention.

    Two DS-Convs with a residual skip + Squeeze-and-Excitation.
    If channel dims differ, a 1×1 projection aligns them.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, se_reduction: int = 4):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(
            in_channels, out_channels, stride=stride, padding=1
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Squeeze-and-Excitation channel attention
        self.se = SqueezeExcitation(out_channels, reduction=se_reduction)
        # Residual projection if needed
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.act(out + identity)
        return out


class AttentionGate(nn.Module):
    """
    Additive Attention Gate for skip connections.
    Filters irrelevant spatial regions before concat with decoder.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, gate: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        # gate comes from decoder (lower res), skip from encoder
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        # Upsample gate to skip resolution if sizes mismatch
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=False)
        att = self.relu(g + s)
        att = self.psi(att)
        return skip * att


class ASPP_Lite(nn.Module):
    """
    Lightweight Atrous Spatial Pyramid Pooling.

    Uses depthwise separable dilated convolutions + global average pooling
    to capture multi-scale context with minimal parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rates: tuple = (6, 12, 18),
    ):
        super().__init__()
        # 1×1 conv branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        # Dilated DS-Conv branches
        self.branches = nn.ModuleList()
        for rate in rates:
            self.branches.append(
                DepthwiseSeparableConv(
                    in_channels, out_channels, kernel_size=3,
                    padding=rate, dilation=rate,
                )
            )
        # Global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        # Fusion: 1×1 + len(rates) dilated + 1 GAP = len(rates)+2 branches
        num_branches = 2 + len(rates)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        feats = [self.conv1x1(x)]
        for branch in self.branches:
            feats.append(branch(x))
        gap = self.gap(x)
        gap = F.interpolate(gap, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(gap)
        return self.fuse(torch.cat(feats, dim=1))
