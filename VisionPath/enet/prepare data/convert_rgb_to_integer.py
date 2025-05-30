# import os
# import json
# import numpy as np
# from PIL import Image

# # Load mapping.json
# with open('segmentation_data/mapping.json', 'r') as f:
#     mapping = json.load(f)['labels']

# # Tạo dictionary ánh xạ màu -> ID
# color_to_id = {}
# for label in mapping:
#     color = tuple(label['color'])
#     color_to_id[color] = label['id']

# # Hàm chuyển mask RGB sang integer
# def convert_mask(mask_path, output_path):
#     # Đọc mask và đảm bảo là ảnh RGB
#     img = Image.open(mask_path)
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     mask = np.array(img)  # Shape: (H, W, 3)
#     height, width = mask.shape[:2]
#     converted_mask = np.zeros((height, width), dtype=np.uint8)

#     for color, id_val in color_to_id.items():
#         # Chuyển color thành numpy array và mở rộng shape để so sánh
#         color_array = np.array(color, dtype=np.uint8)  # Shape: (3,)
#         color_array = np.expand_dims(color_array, axis=(0, 1))  # Shape: (1, 1, 3)
#         # So sánh từng pixel với color
#         color_mask = np.all(mask == color_array, axis=2)  # Shape: (H, W)
#         converted_mask[color_mask] = id_val

#     # Lưu mask đã chuyển đổi
#     Image.fromarray(converted_mask).save(output_path)

# # Xử lý tất cả mask trong thư mục
# input_dir = 'training/labels/'
# output_dir = 'segmentation_data/masks_integer/'
# os.makedirs(output_dir, exist_ok=True)

# for mask_file in os.listdir(input_dir):
#     if mask_file.endswith('.png'):
#         input_path = os.path.join(input_dir, mask_file)
#         output_path = os.path.join(output_dir, mask_file)
#         convert_mask(input_path, output_path)

import os
import cv2
import numpy as np
from tqdm import tqdm

# Class mapping
class_mapping = {
    12: 0,  # road
    16: 1,  # lane marking - general
    6: 2,   # bike-lane
    13: 3,  # service-lane
    7: 4,   # crosswalk - plain
    15: 4,  # lane marking - crosswalk
    1: 5,   # curb
    8: 5,   # curb cut
    2: 6,   # fence
    3: 7,   # guard-rail
    4: 7,   # barrier
    5: 7,   # wall
    14: 8,  # sidewalk
    0: 9,   # background
    9: 9,   # parking
    10: 9,  # pedestrian-area
    11: 9,  # rail track
    17: 9,  # terrain
    18: 9,  # traffic light
    19: 9,  # traffic sign (front)
    20: 9,  # bicycle
    21: 9,  # bus
    22: 9,  # car
    23: 9,  # motorcycle
    24: 9   # truck
}

def remap_labels(label, mapping):
    new_label = np.full_like(label, 9, dtype=np.uint8)  # background
    for old_class, new_class in mapping.items():
        new_label[label == old_class] = new_class
    return new_label

def convert_labels(input_dir, output_dir, size=(512, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    label_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    for label_file in tqdm(label_files, desc="Converting labels"):
        # Đọc ảnh label
        label_path = os.path.join(input_dir, label_file)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # Ánh xạ lớp
        new_label = remap_labels(label, class_mapping)
        
        # Lưu ảnh label mới
        output_path = os.path.join(output_dir, label_file)
        cv2.imwrite(output_path, new_label)

# Chuyển đổi labels cho tập train
convert_labels(
    input_dir='data_traffic_supervision/labels/test',
    output_dir='data_safe_guide/labels/test',
    size=(512, 256)
)
