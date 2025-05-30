from PIL import Image
import numpy as np

mask = np.array(Image.open('segmentation_data/masks_integer/_0ZLZEpBrN8d2YGqPYQSlA.png'))
print("Shape:", mask.shape)  # Phải là (H, W)
print("Unique values:", np.unique(mask))  # Phải là các giá trị từ 0-24
print("Dtype:", mask.dtype)  # Phải là uint8