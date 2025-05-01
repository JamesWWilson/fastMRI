import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from fastmri.models import Unet
import torch

class ReconNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
         # pull the patch size from the ViT for padding:
        try:
            self.patch_h, self.patch_w = net.patch_size
        except:
            # for Unet, we won’t pad
            self.patch_h = self.patch_w = None

    def forward(self, x):
        # 1) If [B,H,W], add channel dim → [B,1,H,W]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 2) RSS-collapse across coil dimension if C>1
        if x.size(1) > 1:
            x = torch.sqrt((x**2).sum(dim=1, keepdim=True))

        # 3) Pad so H,W are multiples of patch size (ViT only)
        wpad = hpad = (0,0)
        if self.patch_h and self.patch_w:
            B, C, H, W = x.shape
            pad_w = (self.patch_w - W % self.patch_w) % self.patch_w
            pad_h = (self.patch_h - H % self.patch_h) % self.patch_h
            # pad equally on both sides
            wpad = (pad_w//2, pad_w - pad_w//2)
            hpad = (pad_h//2, pad_h - pad_h//2)
            x = F.pad(x, (wpad[0], wpad[1], hpad[0], hpad[1]))

        # 4) Normalize to zero mean, unit std
        mean = x.view(x.size(0), 1, 1, -1).mean(-1, keepdim=True)
        std  = x.view(x.size(0), 1, 1, -1).std(-1, keepdim=True)
        x_norm = (x - mean) / (std + 1e-6)

        # 5) Run through the underlying net
        out = self.net(x_norm)  # VisionTransformer.forward *or* Unet.forward

        # 6) If out is 3-D, restore channel dim
        if out.dim() == 3:
            out = out.unsqueeze(1)

        # 7) Un-normalize
        out = out * (std + 1e-6) + mean

        # 8) Un-pad
        if self.patch_h and self.patch_w:
            out = out[..., 
                      hpad[0]: out.shape[-2] - hpad[1],
                      wpad[0]: out.shape[-1] - wpad[1]
                    ]

        return out