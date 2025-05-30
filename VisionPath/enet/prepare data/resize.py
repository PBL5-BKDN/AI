import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Paths to your converted dataset
dataset_dir = "selected/"  
images_dir = os.path.join(dataset_dir, "images/val")
masks_dir = os.path.join(dataset_dir, "labels/val")
output_dir = "resized_dataset/" 
output_images_dir = os.path.join(output_dir, "images/val")
output_masks_dir = os.path.join(output_dir, "labels/val")

# Create output directories
Path(output_images_dir).mkdir(parents=True, exist_ok=True)
Path(output_masks_dir).mkdir(parents=True, exist_ok=True)

# Target size for ENet
target_size = (512, 256)

# Get list of image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

# Process each image and mask
for image_file in tqdm(image_files, desc="Resizing images and masks"):
    # Load image and mask
    image_path = os.path.join(images_dir, image_file)
    mask_file = image_file.replace('.jpg', '.png')  # Adjust extension if needed
    mask_path = os.path.join(masks_dir, mask_file)
    
    if not os.path.exists(mask_path):
        print(f"Mask not found for {image_file}, skipping...")
        continue
    
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as single-channel
    
    if image is None or mask is None:
        print(f"Failed to load {image_file} or its mask, skipping...")
        continue
    
    # Resize image with linear interpolation
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Resize mask with nearest-neighbor interpolation to preserve class IDs
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Save the resized image and mask
    output_image_path = os.path.join(output_images_dir, image_file)
    output_mask_path = os.path.join(output_masks_dir, mask_file)
    
    cv2.imwrite(output_image_path, image_resized)
    cv2.imwrite(output_mask_path, mask_resized)

print(f"Resized dataset saved to {output_dir}")