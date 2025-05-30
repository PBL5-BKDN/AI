import os
import numpy as np
from PIL import Image
from collections import defaultdict

pixel_counts = defaultdict(int)
mask_dir = 'data_traffic_supervision/labels/val'
for mask_file in os.listdir(mask_dir):
    if mask_file.endswith('.png'):
        mask = np.array(Image.open(os.path.join(mask_dir, mask_file)))
        unique, counts = np.unique(mask, return_counts=True)
        for id_val, count in zip(unique, counts):
            pixel_counts[id_val] += count

for id_val, count in pixel_counts.items():
    print(f"Class ID {id_val}: {count} pixels")