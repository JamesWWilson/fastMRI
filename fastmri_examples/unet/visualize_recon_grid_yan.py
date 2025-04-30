import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -------- CONFIG --------
recon_dir = Path("G:/Deep Learning/Group Project/fastMRI/unet/outputs/recon_filtered_leaderboard/pretrained_filtered_reconstructions")
save_dir = Path("G:/Deep Learning/Group Project/fastMRI/unet/outputs/recon_filtered_leaderboard/pretrained_filtered_grid_images")
save_dir.mkdir(parents=True, exist_ok=True)

# -------- VISUALIZATION LOOP --------
for file in sorted(recon_dir.glob("*.h5")):
    with h5py.File(file, "r") as hf:
        recon = hf["reconstruction"][:]  # shape: (N, 1, H, W)

    if recon.shape[1] == 1:
        recon = recon[:, 0, :, :]  # → (N, H, W)

    num_slices = recon.shape[0]
    cols = 6
    rows = int(np.ceil(num_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten()

    for i in range(rows * cols):
        ax = axes[i]
        if i < num_slices:
            ax.imshow(recon[i], cmap="gray")
            ax.set_title(f"Slice {i}")
        ax.axis("off")

    plt.tight_layout()
    grid_path = save_dir / f"{file.stem}_grid.png"
    plt.savefig(grid_path, dpi=150)
    plt.close()
    # print(f"✅ Saved grid: {grid_path}")
