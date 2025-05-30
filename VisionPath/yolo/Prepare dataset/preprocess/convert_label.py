import json
import os
import cv2
import numpy as np
from scipy.ndimage import label
from pathlib import Path
from uuid import uuid4
from scipy.spatial.distance import cdist
import shutil
from multiprocessing import Pool, Manager
from functools import partial

# Đọc file config
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['labels']

# Lọc các lớp được chọn
def filter_classes(labels, selected_classes):
    return [lbl for lbl in labels if lbl['readable'] in selected_classes]

# Tạo mapping từ màu sắc sang class_id
def create_color_to_class_mapping(labels, selected_classes, start_id=0):
    color_to_class = {}
    readable_to_color = {lbl['readable']: tuple(lbl['color']) for lbl in labels}
    for class_id, class_name in enumerate(selected_classes, start=start_id):
        if class_name in readable_to_color:
            color = readable_to_color[class_name]
            color_to_class[color] = class_id
    return color_to_class


# Gộp các vùng gần nhau và tạo bounding box
def get_bounding_boxes(binary_mask, min_area=50, max_distance=50):
    labeled_array, num_features = label(binary_mask)
    if num_features == 0:
        return []
    regions = []
    for i in range(1, num_features + 1):
        region = (labeled_array == i)
        if np.sum(region) < min_area:
            continue
        y, x = np.where(region)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        regions.append((x_min, y_min, x_max, y_max))
    if not regions:
        return []
    merged_regions = []
    while regions:
        current = regions.pop(0)
        merged = [current]
        i = 0
        while i < len(regions):
            other = regions[i]
            box1 = np.array([[current[0], current[1]], [current[2], current[3]]])
            box2 = np.array([[other[0], other[1]], [other[2], other[3]]])
            distances = cdist(box1, box2, metric='euclidean')
            if distances.min() < max_distance:
                merged.append(regions.pop(i))
            else:
                i += 1
        x_min = min(r[0] for r in merged)
        y_min = min(r[1] for r in merged)
        x_max = max(r[2] for r in merged)
        y_max = max(r[3] for r in merged)
        merged_regions.append((x_min, y_min, x_max, y_max))
    return merged_regions

# Chuyển đổi mask sang định dạng YOLO
def mask_to_yolo(mask_path, color_to_class, img_width, img_height):
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Không thể đọc mask: {mask_path}")
        return []

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    height, width = mask_rgb.shape[:2]
    yolo_lines = []

    for color, class_id in color_to_class.items():
        binary_mask = np.all(mask_rgb == color, axis=2).astype(np.uint8)
        bboxes = get_bounding_boxes(binary_mask)
        for x_min, y_min, x_max, y_max in bboxes:
            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            box_width = (x_max - x_min) / width
            box_height = (y_max - y_min) / height
            if box_width > 0 and box_height > 0:
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    return yolo_lines

# Hàm xử lý một file mask (dùng trong multiprocessing)
def process_mask_file(args):
    mask_path, color_to_class, images_dir, output_labels, output_images = args
    mask_path = str(mask_path)
    
    # Đọc và kiểm tra mask
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Không thể đọc mask: {mask_path}")
        return 0
    
    img_height, img_width = mask.shape[:2]
    yolo_lines = mask_to_yolo(mask_path, color_to_class, img_width, img_height)
    
    if not yolo_lines:
        return 0  # Bỏ qua file không có nhãn
    
    # Ghi file label
    output_label_file = os.path.join(output_labels, f"{Path(mask_path).stem}.txt")
    with open(output_label_file, 'w') as f:
        f.write('\n'.join(yolo_lines))
    
    # Sao chép ảnh gốc cùng tên
    image_file = os.path.join(images_dir, f"{Path(mask_path).stem}.jpg")
    if os.path.exists(image_file):
        shutil.copy(str(image_file), os.path.join(output_images, Path(image_file).name))
        return 1
    else:
        print(f"Không tìm thấy ảnh gốc cho: {Path(mask_path).stem}.jpg")
        return 0

# Chuyển đổi toàn bộ folder với xử lý song song
def convert_mapillary_to_yolo_parallel(config_path, mask_dir, output_root, selected_classes, images_dir):
    output_root = Path(output_root)
    output_labels = output_root / "labels"
    output_images = output_root / "images"

    output_labels.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)

    # Đọc config và tạo mapping
    labels = load_config(config_path)
    filtered_labels = filter_classes(labels, selected_classes)
    color_to_class = create_color_to_class_mapping(filtered_labels, selected_classes, start_id=0)

    mask_dir = Path(mask_dir)
    images_dir = Path(images_dir)
    
    # Lấy danh sách file mask
    mask_files = list(mask_dir.glob("*.png"))
    if not mask_files:
        print(f"No mask files found in {mask_dir}")
        return

    # Chuẩn bị tham số cho multiprocessing
    args = [(mask_file, color_to_class, str(images_dir), str(output_labels), str(output_images)) for mask_file in mask_files]
    
    # Xử lý song song
    with Pool() as pool:
        results = pool.map(process_mask_file, args)
    
    count = sum(results)
    print(f"Tổng số ảnh đã được tạo label: {count}")

# Các lớp được chọn
SELECTED_CLASSES = [
    "Person", "Bicycle", "Car", "Motorcycle", "Vegetation", "Billboard", "Manhole", "Trash Can", "Barrier"
]

# Chạy
if __name__ == "__main__":
    CONFIG_PATH = "config.json"
    MASK_DIR = "masks"
    OUTPUT_DIR = "output"
    IMAGES_DIR = "images"

    convert_mapillary_to_yolo_parallel(CONFIG_PATH, MASK_DIR, OUTPUT_DIR, SELECTED_CLASSES, IMAGES_DIR)