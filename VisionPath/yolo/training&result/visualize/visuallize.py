import os
import cv2
import numpy as np
from pathlib import Path
import random

# Danh sách tên lớp tương ứng với ID từ 0 -> 8
CLASSES = [
    #"0", "1", "2", "3", "4", "5", "6", "7", "8", 
    "Person", "Bicycle", "Car", "Motorcycle", "Vegetation", "Billboard", "Manhole", "Trash Can", "Barrier"
]

def generate_colors(num_classes):
    """Tạo danh sách màu ngẫu nhiên cho các lớp."""
    random.seed(42)  # Để tái lập màu
    colors = []
    for _ in range(num_classes):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors

def read_yolo_label(label_path, img_width, img_height):
    """Đọc file nhãn YOLO và chuyển đổi sang tọa độ pixel."""
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes

    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            class_id = int(class_id)

            # Chuyển đổi từ tọa độ chuẩn hóa sang tọa độ pixel
            x_min = int((x_center - width / 2) * img_width)
            x_max = int((x_center + width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            y_max = int((y_center + height / 2) * img_height)

            bboxes.append((class_id, x_min, y_min, x_max, y_max))

    return bboxes

def draw_bboxes(image, bboxes, classes, colors):
    """Vẽ bounding box và nhãn lên ảnh."""
    for class_id, x_min, y_min, x_max, y_max in bboxes:
        if class_id >= len(classes):
            continue  # Bỏ qua nếu ID ngoài phạm vi

        color = colors[class_id]
        class_name = classes[class_id]

        # Vẽ bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Vẽ nhãn
        label = f"{class_name} ({class_id})"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x_min, y_min - label_height - baseline),
                      (x_min + label_width, y_min), color, -1)
        cv2.putText(image, label, (x_min, y_min - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

def visualize_yolo_results(images_dir, labels_dir, output_dir):
    """Xử lý toàn bộ thư mục và vẽ bounding box lên ảnh."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = generate_colors(len(CLASSES))
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)

    for label_file in labels_dir.glob("*.txt"):
        image_name = label_file.stem + ".jpg"
        image_path = images_dir / image_name
        print(f"Đang xử lý: {image_path}")
        if not image_path.exists():
            print(f"Không tìm thấy ảnh: {image_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Không thể đọc ảnh: {image_path}")
            continue

        img_height, img_width = image.shape[:2]
        bboxes = read_yolo_label(label_file, img_width, img_height)
        image_with_bboxes = draw_bboxes(image, bboxes, CLASSES, colors)

        output_path = output_dir / f"{label_file.stem}_bbox.jpg"
        cv2.imwrite(str(output_path), image_with_bboxes)
        print(f"Đã lưu kết quả: {output_path}")

if __name__ == "__main__":
    IMAGES_DIR = "visualize/images"
    LABELS_DIR = "visualize/labels"
    OUTPUT_DIR = "visualize/drawbb"

    visualize_yolo_results(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR)
