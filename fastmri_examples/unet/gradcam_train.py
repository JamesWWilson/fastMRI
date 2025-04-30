import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

from fastmri.models import Unet
from fastmri.data.transforms import UnetDataTransform
from pathlib import Path
import h5py


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target is None:
            target = output.mean()
        target.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def extract_bbox_from_cam(cam_map, threshold=0.6):
    binary_map = cam_map > threshold
    coords = np.argwhere(binary_map)
    if coords.shape[0] == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return x0, y0, x1, y1


# Load pretrained U-Net
model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4)
model.load_state_dict(torch.load("knee_sc_leaderboard_state_dict.pt"))
model.eval().cuda()

# Target layer for Grad-CAM
cam = GradCAM(model, model.down_sample_layers[-1])

# Set paths
train_path = Path("G:/Deep Learning/Group Project/fastMRI/data/singlecoil_train")
save_dir = Path("G:/Deep Learning/Group Project/fastMRI/data/Filtered_Training")
save_dir.mkdir(parents=True, exist_ok=True)
crop_dir = save_dir / "cropped"
crop_dir.mkdir(exist_ok=True)
bbox_csv = save_dir / "training_bounding_boxes.csv"

# Write CSV header
with open(bbox_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "slice", "x0", "y0", "x1", "y1"])

# Process all training slices
for file in sorted(train_path.glob("*.h5")):
    with h5py.File(file, 'r') as hf:
        num_slices = hf["kspace"].shape[0]
        for i in range(num_slices):
            kspace = hf["kspace"][i]
            target = hf["reconstruction_esc"][i] if "reconstruction_esc" in hf else None
            attrs = dict(hf.attrs)

            transform = UnetDataTransform(which_challenge="singlecoil")
            sample = transform(kspace, None, target, attrs, str(file.name), i)
            image, _, mean, std, _, _, target = sample

            image = image.unsqueeze(0).unsqueeze(0).cuda()
            image.requires_grad = True

            cam_map = cam(image)
            input_img = image.detach().cpu().squeeze().numpy()
            bbox = extract_bbox_from_cam(cam_map, threshold=0.6)

            if bbox:
                x0, y0, x1, y1 = bbox
                area = (x1 - x0) * (y1 - y0)
                if area > 200:
                    # Save ROI crop
                    roi = input_img[y0:y1, x0:x1]
                    roi_path = crop_dir / f"{file.stem}_slice{i:02d}_roi.png"
                    plt.imsave(roi_path, roi, cmap='gray')

                    # Save bbox info
                    with open(bbox_csv, mode='a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([file.name, i, x0, y0, x1, y1])

                    print(f"Saved: {file.stem}_slice{i:02d}")
