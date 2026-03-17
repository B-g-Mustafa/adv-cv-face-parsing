"""
AttentionLiteUNet — Lightweight face segmentation model (<1.7M parameters).

Architecture:
    4-stage encoder (DSResBlocks) → ASPP_Lite bottleneck → 4-stage decoder
    with Attention Gates on skip connections.

Design rationale (from research):
    - Depthwise separable convolutions reduce params by ~8-9× vs standard convs
    - Attention gates filter irrelevant background on skip connections
    - ASPP provides multi-scale receptive field without deepening the network
    - Sobel edge channel (4th input) delegates edge detection to classical CV
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DSResBlock, AttentionGate, ASPP_Lite, DepthwiseSeparableConv


class AttentionLiteUNet(nn.Module):
    """
    Lightweight U-Net with attention gates and depthwise separable convolutions.

    Args:
        in_channels: Number of input channels (3 for RGB, 4 with Sobel edge).
        num_classes: Number of segmentation classes.
        enc_channels: Channel sizes for each encoder stage.
        dec_channels: Channel sizes for each decoder stage.
        bottleneck_channels: Channels in ASPP bottleneck.
        aspp_rates: Dilation rates for ASPP.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 19,
        enc_channels: Optional[List[int]] = None,
        dec_channels: Optional[List[int]] = None,
        bottleneck_channels: int = 256,
        aspp_rates: tuple = (6, 12, 18),
        dropout: float = 0.1,
    ):
        super().__init__()
        if enc_channels is None:
            enc_channels = [32, 64, 128, 256]
        if dec_channels is None:
            dec_channels = [128, 64, 32]

        # ---- Initial conv ----
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(in_channels, enc_channels[0], 3, padding=1),
        )

        # ---- Encoder stages (each downsamples 2x via stride-2) ----
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = enc_channels[0]
        for ch_out in enc_channels[1:]:
            self.encoders.append(DSResBlock(ch_in, ch_out))
            self.pools.append(nn.MaxPool2d(2))
            ch_in = ch_out

        # ---- Bottleneck with ASPP ----
        self.bottleneck = ASPP_Lite(
            enc_channels[-1], bottleneck_channels, rates=aspp_rates
        )

        # ---- Decoder stages ----
        self.upconvs = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Reverse encoder channels for skip connections
        skip_channels = list(reversed(enc_channels))  # [256, 128, 64, 32]

        dec_in = bottleneck_channels
        for i, ch_out in enumerate(dec_channels):
            skip_ch = skip_channels[i]  # matches encoder stage
            # Upsample + reduce channels
            self.upconvs.append(
                nn.ConvTranspose2d(dec_in, ch_out, kernel_size=2, stride=2)
            )
            # Attention gate: gate from decoder, skip from encoder
            inter_ch = max(ch_out // 2, 16)
            self.att_gates.append(AttentionGate(ch_out, skip_ch, inter_ch))
            # Fuse: concat(skip, up) → DSResBlock
            self.decoders.append(DSResBlock(ch_out + skip_ch, ch_out))
            dec_in = ch_out

        # ---- Final upconv to match first encoder skip (enc_channels[0]) ----
        self.final_up = nn.ConvTranspose2d(
            dec_channels[-1], dec_channels[-1], kernel_size=2, stride=2
        )
        inter_ch_final = max(dec_channels[-1] // 2, 16)
        self.final_att = AttentionGate(dec_channels[-1], enc_channels[0], inter_ch_final)
        self.final_dec = DSResBlock(dec_channels[-1] + enc_channels[0], dec_channels[-1])

        # ---- Classification head ----
        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(dec_channels[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x0 = self.stem(x)  # (B, 32, H, W)

        # Encoder
        skips = [x0]
        out = x0
        for enc, pool in zip(self.encoders, self.pools):
            out = pool(out)
            out = enc(out)
            skips.append(out)
        # skips: [x0(32), e1(64), e2(128), e3(256)]

        # Bottleneck
        out = self.bottleneck(out)  # (B, 256, H/8, W/8)

        # Decoder (process skips in reverse, skipping the last one which is bottleneck input)
        skip_list = list(reversed(skips))  # [e3, e2, e1, x0]
        for i, (upconv, att, dec) in enumerate(
            zip(self.upconvs, self.att_gates, self.decoders)
        ):
            out = upconv(out)
            skip = skip_list[i]
            # Ensure spatial sizes match
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(
                    out, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            skip = att(gate=out, skip=skip)
            out = dec(torch.cat([out, skip], dim=1))

        # Final stage to full resolution
        out = self.final_up(out)
        skip = skip_list[-1]  # x0
        if out.shape[2:] != skip.shape[2:]:
            out = F.interpolate(
                out, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        skip = self.final_att(gate=out, skip=skip)
        out = self.final_dec(torch.cat([out, skip], dim=1))

        # Head
        out = self.dropout(out)
        out = self.head(out)
        return out


def build_model(cfg: dict) -> AttentionLiteUNet:
    """Instantiate model from config dictionary."""
    mcfg = cfg.get("model", {})
    return AttentionLiteUNet(
        in_channels=mcfg.get("in_channels", 4),
        num_classes=mcfg.get("num_classes", 19),
        enc_channels=mcfg.get("encoder_channels", [32, 64, 128, 256]),
        dec_channels=mcfg.get("decoder_channels", [128, 64, 32]),
        bottleneck_channels=mcfg.get("bottleneck_channels", 256),
        aspp_rates=tuple(mcfg.get("aspp_rates", [6, 12, 18])),
        dropout=mcfg.get("dropout", 0.1),
    )
