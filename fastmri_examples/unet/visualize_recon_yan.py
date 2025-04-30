import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import imageio.v2 as imageio

# Input and output paths
recon_path = Path("G:/Deep Learning/Group Project/fastMRI/unet/outputs/finetuned_upweighted_reconstructions_test")
save_path = Path("G:/Deep Learning/Group Project/fastMRI/unet/outputs/finetuned_upweighted__recon_images")
save_path.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist

# Iterate through .h5 files and save first slice as PNG
for file in recon_path.glob("*.h5"):
    with h5py.File(file, "r") as hf:
        image = hf["reconstruction"][()]  # shape: (num_slices, H, W)

    for slice_idx in range(image.shape[0]):
        slice_img = image[slice_idx]

        # Normalize to [0, 1]
        normalized_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        out_img = (normalized_img * 255).astype(np.uint8)
        out_img = np.squeeze(out_img)

        # Save slice with filename including index
        out_file = save_path / f"{file.stem}_slice{slice_idx:03d}.png"
        imageio.imwrite(out_file, out_img)

    print(f"Saved {image.shape[0]} slices from {file.name}")



