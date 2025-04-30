def always_true(x):
    return True

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import torch
from tqdm import tqdm

import fastmri
import fastmri.data.transforms as T
from fastmri.data import mri_data
from fastmri.models import Unet

UNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
MODEL_FNAMES = {
    "unet_knee_sc": "knee_sc_leaderboard_state_dict.pt",
    "unet_knee_mc": "knee_mc_leaderboard_state_dict.pt",
    "unet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def run_unet_model(batch, model, device):
    image, _, mean, std, fname, slice_num, _ = batch

    output = model(image.to(device).unsqueeze(1)).squeeze(1).cpu()

    mean = mean.unsqueeze(1).unsqueeze(2)
    std = std.unsqueeze(1).unsqueeze(2)
    output = (output * std + mean).cpu()

    return output, int(slice_num[0]), fname[0]

def run_inference(challenge, state_dict_file, data_path, output_path, device):
    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4)
    #model.load_state_dict(torch.load("epoch=9-step=5210.ckpt", map_location=device))
    checkpoint = torch.load("epoch=9-step=3300.ckpt", map_location=device)
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval().to(device)

    data_transform = T.UnetDataTransform(which_challenge="singlecoil")
    dataset = mri_data.SliceDataset(
        root=data_path,
        transform=data_transform,
        challenge="singlecoil",
        raw_sample_filter=always_true,
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=12)
    outputs = defaultdict(list)

    for batch in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            output, slice_num, fname = run_unet_model(batch, model, device)
        outputs[fname].append((slice_num, output))

    for fname in outputs:
        outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

    fastmri.save_reconstructions(outputs, output_path / "reconstructions-upweighted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--challenge",
        default="unet_knee_sc",
        choices=(
            "unet_knee_sc",
            "unet_knee_mc",
            "unet_brain_mc",
        ),
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to subsampled data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        
        required=True,
        help="Path for saving reconstructions",
    )

    args = parser.parse_args()

    run_inference(
        args.challenge,
        args.state_dict_file,
        args.data_path,
        args.output_path,
        torch.device(args.device),
    )
