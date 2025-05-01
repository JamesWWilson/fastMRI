import os
import random
import shutil
from pathlib import Path

# Input folder with images
src_dir = Path("G:/Deep Learning/Group Project/fastMRI/data/Filtered_Training/cropped")

# Output folder to store 1000 randomly selected images
dst_dir = Path("G:/Deep Learning/Group Project/fastMRI/data/Filtered_Training/random_1000")
dst_dir.mkdir(parents=True, exist_ok=True)

# Get all image file paths (you can filter by extension if needed)
image_files = [f for f in src_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

# Randomly select 1000 images
selected_images = random.sample(image_files, 1000)

# Copy them to the new directory
for image in selected_images:
    shutil.copy(image, dst_dir / image.name)

print(f"✅ Copied {len(selected_images)} images to {dst_dir}")
