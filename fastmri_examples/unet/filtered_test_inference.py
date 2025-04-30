import argparse
from collections import defaultdict
from pathlib import Path

import time 
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import fastmri
import fastmri.data.transforms as T
from fastmri.data import mri_data
from fastmri.models import Unet

# ---------- GLOBAL FILTER SET ----------
filtered_set = set()

# ---------- RAW SAMPLE FILTER ----------
from pathlib import Path

def filter_fn(sample_info):
    fname = Path(sample_info[0]).stem  # ensures we get "file100022" from "file100022.h5"
    slice_id = int(sample_info[1])
    return (fname, slice_id) in filtered_set



# ---------- INFERENCE UTILITY ----------
def run_unet_model(batch, model, device):
    image, _, mean, std, fname, slice_num, _ = batch
    output = model(image.to(device).unsqueeze(1)).squeeze(1).cpu()
    mean = mean.unsqueeze(1).unsqueeze(2)
    std = std.unsqueeze(1).unsqueeze(2)
    output = (output * std + mean).cpu()
    return output, int(slice_num[0]), fname[0]

# ---------- MAIN INFERENCE ----------
def run_inference(challenge, state_dict_file, data_path, output_path, device, csv_path):
    global filtered_set

    start_time = time.time()

    # 1. Load CSV filter set
    df = pd.read_csv(csv_path)
    df["filename"] = df["filename"].str.replace(".h5", "", regex=False)  # <--- add this
    filtered_set = set((row["filename"], int(row["slice"])) for _, row in df.iterrows())

    print(f"Total filtered slice targets: {len(filtered_set)}")

    # 2. Load model
    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4)
    #model.load_state_dict(torch.load("knee_sc_leaderboard_state_dict.pt"))
    checkpoint = torch.load("epoch=9-step=3300.ckpt", map_location=device)
    state_dict = {k.replace("unet.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval().to(device)

    # 3. Load data
    data_transform = T.UnetDataTransform(which_challenge="singlecoil")
    dataset = mri_data.SliceDataset(
    root=data_path,
    transform=data_transform,
    challenge="singlecoil",
    raw_sample_filter=filter_fn,
    )

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=12)

    # 4. Run inference
    outputs = defaultdict(list)
    for batch in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            output, slice_num, fname = run_unet_model(batch, model, device)
        outputs[fname].append((slice_num, output))

    # 5. Save reconstructions
    for fname in outputs:
        outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
    fastmri.save_reconstructions(outputs, output_path / "finetuned_upweighted_reconstructions_test")

    total_time = time.time() - start_time
    print(f"\n✅ Inference completed in {total_time:.2f} seconds")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", default="unet_knee_sc", choices=["unet_knee_sc", "unet_knee_mc", "unet_brain_mc"])
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument("--state_dict_file", type=Path, required=True, help="Path to .pth model file")
    parser.add_argument("--data_path", type=Path, required=True, help="Path to singlecoil_test folder")
    parser.add_argument("--output_path", type=Path, required=True, help="Where to save reconstructions")
    parser.add_argument("--csv_path", type=Path, required=True, help="Path to filtered_test_boxes.csv")

    args = parser.parse_args()

    run_inference(
        args.challenge,
        args.state_dict_file,
        args.data_path,
        args.output_path,
        torch.device(args.device),
        args.csv_path,
    )


