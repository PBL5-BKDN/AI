from ultralytics import YOLO
import os
import cv2
import glob

def auto_label_old_classes(
    image_dir,
    label_dir,
    model_path='yolov8n.pt',
    classes_old=['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light']
):
    """
    Tự động thêm nhãn cho các lớp cũ (class_id 0-6) vào file nhãn (không giới hạn số lượng).

    Args:
        image_dir (str): Thư mục chứa ảnh.
        label_dir (str): Thư mục chứa nhãn.
        model_path (str): Đường dẫn đến mô hình YOLOv8.
        classes_old (list): Danh sách lớp cũ.

    Returns:
        dict: Số lượng box đã gán trên mỗi lớp.
    """
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"❌ Error: Directory {image_dir} or {label_dir} does not exist.")
        return {}

    model = YOLO(model_path)
    coco_classes = model.names
    print("✅ Model classes:", coco_classes)

    # Map COCO class name → class_id trong model
    class_map = {coco_classes[i]: i for i in coco_classes if coco_classes[i] in classes_old}
    if not class_map:
        print("❌ Error: None of the classes_old found in model.names.")
        return {}

    missing_classes = [cls for cls in classes_old if cls not in class_map]
    if missing_classes:
        print(f"⚠️ Warning: Classes {missing_classes} not found in model.names")

    # Map lại tên lớp → class_id mới (0 đến N-1)
    new_class_map = {name: idx for idx, name in enumerate(classes_old)}
    print("🔄 New class map (0-based):", new_class_map)

    class_counts = {i: 0 for i in range(len(classes_old))}

    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return class_counts

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Warning: Could not read image {img_path}")
            continue

        results = model.predict(img_path, conf=0.3)
        new_labels = []

        for result in results:
            boxes = result.boxes.xywhn
            classes = result.boxes.cls
            for box, cls in zip(boxes, classes):
                cls_name = coco_classes[int(cls)]
                if cls_name in class_map:
                    new_class_id = new_class_map[cls_name]
                    x_center, y_center, w, h = box
                    new_labels.append(f"{new_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                    class_counts[new_class_id] += 1

        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        existing_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                existing_labels = f.readlines()
        else:
            print(f"📄 Creating label file: {label_path}")

        with open(label_path, 'w') as f:
            for label in existing_labels:
                f.write(label)
            for label in new_labels:
                f.write(f"{label}\n")

    print("📊 Bounding box counts for old classes:", class_counts)
    return class_counts

# --- Gọi hàm ---
image_dir = 'data/images'
label_dir = 'data/labels'
model_path = 'yolov8n.pt'

class_counts = auto_label_old_classes(
    image_dir=image_dir,
    label_dir=label_dir,
    model_path=model_path,
)
