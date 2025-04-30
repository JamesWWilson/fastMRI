# fastmri/models/unet_lora.py

import torch
from torch import nn
import torch.nn.functional as F
from fastmri.models.unet import Unet as BaseUnet

class LoRAConv2d(nn.Module):
    """A Conv2d + low-rank adapters (A, B) à la LoRA."""
    def __init__(self, in_ch, out_ch, kernel_size, padding, bias, r, alpha, dropout):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              padding=padding, bias=bias)
        # LoRA adapters
        self.lora_A = nn.Conv2d(in_ch, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r, out_ch, kernel_size=1, bias=False)
        self.scaling = alpha / r
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # base = self.conv(x)
        # delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        # return base + delta

        # standard convolution path
        base = self.conv(x)
        # low-rank adapter path
        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        # ----- new: make sure spatial sizes agree -----
        if delta.shape[-2:] != base.shape[-2:]:
            # e.g. due to pooling/upsampling mis–alignments
            delta = F.interpolate(delta,
                                  size=base.shape[-2:],
                                  mode="nearest",
                                #   align_corners=None
                                  )
        return base + delta


class LoRAConvTranspose2d(nn.Module):
    """A ConvTranspose2d + LoRA adapters in the “feature” space."""
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias, r, alpha, dropout):
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_ch, out_ch,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=bias)
        # we inject adapters as 1×1 convolutions in feature‐space
        # **ADJUSTED**: adapters must accept out_ch (the channels of `base`), not in_ch
        self.lora_A = nn.Conv2d(out_ch, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r,   out_ch, kernel_size=1, bias=False)
        self.scaling = alpha / r
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # base = self.convT(x)
        # delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        # return base + delta
        # 1) up-sample
        base = self.convT(x)
        # 2) build LoRA delta on the **upsampled** features
        delta = self.lora_B(self.lora_A(self.dropout(base))) * self.scaling
        # 3) guard: if anything still mis-shaped, interpolate
        if delta.shape[-2:] != base.shape[-2:]:
            delta = F.interpolate(delta, size=base.shape[-2:], mode="nearest")
        # 4) residual add
        return base + delta



class LoRATransposeConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=2, padding=0,
                 bias=False, r=4, alpha=16.0, dropout=0.1):
        super().__init__()
        # frozen base transpose‐conv
        self.tconv = nn.ConvTranspose2d(in_ch, out_ch,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=bias)
        # LoRA adapters operate on the output of the upsample:
        self.lora_A = nn.Conv2d(out_ch, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r, out_ch, kernel_size=1, bias=False)
        self.scaling = alpha / r
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        base = self.tconv(x)
        # apply LoRA on the upsampled feature‐map
        d = self.lora_B(self.lora_A(self.dropout(base))) * self.scaling
        # sizes should match, but guard anyway
        if d.shape[-2:] != base.shape[-2:]:
            d = F.interpolate(d, size=base.shape[-2:], mode="nearest")
        return base + d



class UnetLoRA(BaseUnet):
    """
    Extends the FastMRI U-Net with LoRA adapters. Modes:
      - none:               no LoRA
      - bottleneck_only:    LoRA only in the middle ConvBlock
      - bottleneck_final:   LoRA in the middle + final 1×1 projection
      - decoder_only:       LoRA throughout the decoder (transpose + up convs)
      - all:                LoRA in every Conv2d (down, middle, up, final)
    """
    def __init__(
        self,
        in_chans, out_chans, chans, num_pool_layers, drop_prob,
        r=4, alpha=16.0, dropout=0.1,
        mode="bottleneck_final",
    ):
        super().__init__(in_chans, out_chans, chans, num_pool_layers, drop_prob)
        self.lora_mode = mode

        def wrap_conv(layer):
            # layer: nn.Conv2d
            return LoRAConv2d(
                layer.in_channels, layer.out_channels,
                kernel_size=layer.kernel_size[0],
                padding=layer.padding[0],
                bias=(layer.bias is not None),
                r=r, alpha=alpha, dropout=dropout
            )

        # 1) DOWN‐SAMPLE: only in “all”
        if mode == "all":
            for block in self.down_sample_layers:
                block.layers[0] = wrap_conv(block.layers[0])
                block.layers[4] = wrap_conv(block.layers[4])

        # 2) BOTTLENECK: in all except “none”
        if mode in ("bottleneck_only", "bottleneck_final", "decoder_only", "all"):
            # self.conv is a ConvBlock
            self.conv.layers[0] = wrap_conv(self.conv.layers[0])
            if mode == "all":
                # also wrap the second conv in that block
                self.conv.layers[4] = wrap_conv(self.conv.layers[4])

        # 3) FINAL 1×1 PROJECTION: in bottleneck_final, all, decoder_only
        if mode in ("bottleneck_final", "all", "decoder_only"):
            last = self.up_conv[-1]
            # last may be ConvBlock or nn.Sequential(ConvBlock, Conv2d)
            if isinstance(last, nn.Sequential):
                convblock, final_conv = last[0], last[1]
                last[1] = LoRAConv2d(
                    final_conv.in_channels, final_conv.out_channels,
                    kernel_size=1, padding=0, bias=(final_conv.bias is not None),
                    r=r, alpha=alpha, dropout=dropout
                )

        # 4) DECODER PATH: transpose + up convs, in decoder_only & all
        if mode in ("decoder_only", "all"):
            # 4a) transpose‐convs
            for tconv in self.up_transpose_conv:
                orig = tconv.layers[0]  # the ConvTranspose2d
                tconv.layers[0] = LoRAConvTranspose2d(
                    orig.in_channels, orig.out_channels,
                    kernel_size=orig.kernel_size[0],
                    stride=orig.stride[0],
                    padding=orig.padding[0],
                    bias=(orig.bias is not None),
                    r=r, alpha=alpha, dropout=dropout
                )
            # 4b) up‐sampling ConvBlocks
            for up in self.up_conv:
                # up is either ConvBlock or Sequential(ConvBlock, Conv2d)
                convblock = up[0] if isinstance(up, nn.Sequential) else up
                convblock.layers[0] = wrap_conv(convblock.layers[0])
                convblock.layers[4] = wrap_conv(convblock.layers[4])

        # freeze everything except the LoRA adapters
        self.freeze_base()

    def freeze_base(self):
        """Freeze all original parameters, leave only lora_A / lora_B trainable."""
        for name, p in self.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False

    def freeze_encoder(self):
        """Freeze just the encoder (down-sampling) path."""
        # down-sampling ConvBlocks
        for block in self.down_sample_layers:
            for p in block.parameters():
                p.requires_grad = False

    def freeze_decoder(self):
        """Freeze the decoder path (bottleneck + up-sampling)."""
        # bottleneck ConvBlock
        for p in self.conv.parameters():
            p.requires_grad = False
        # all transpose-conv adapters or base
        for tconv in self.up_transpose_conv:
            for p in tconv.parameters():
                p.requires_grad = False
        # all up-sampling ConvBlocks (+ final projection)
        for up in self.up_conv:
            # if it’s a Sequential([ConvBlock, Conv2d]), block is up[0]
            block = up[0] if isinstance(up, nn.Sequential) else up
            for p in block.parameters():
                p.requires_grad = False
            # if sequential, also freeze the final 1×1 conv
            if isinstance(up, nn.Sequential):
                for p in up[1].parameters():
                    p.requires_grad = False