import torch
import torch.nn as nn

class UncertaintyEvaluator(nn.Module):
    """
    Simple conv‐regression net: takes in a reconstructed slice
    (1×H×W) and outputs a per‐pixel uncertainty map (1×H×W).
    """
    def __init__(self, chans=32, num_layers=4):
        super().__init__()
        layers = []
        in_ch = 1
        for i in range(num_layers):
            out_ch = chans * (2**i) if i < num_layers - 1 else chans
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        layers += [
            nn.Conv2d(in_ch, 1, kernel_size=3, padding=1),
            nn.Softplus(),  # ensure positivity
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, H, W), returns (B, 1, H, W)
        return self.net(x)