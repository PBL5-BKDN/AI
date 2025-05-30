import os
import shutil
import random

images_dir = "data/images"  
labels_dir = "data/labels"
train_images_dir = "images/train"
train_labels_dir = "labels/train"
test_images_dir = "images/val"
test_labels_dir = "labels/val"

# Tạo các thư mục mới nếu chưa tồn tại
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Lấy danh sách tất cả ảnh (dựa trên file ảnh trong thư mục images)
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Xáo trộn danh sách ảnh để chia ngẫu nhiên
random.shuffle(image_files)

# Tính số lượng ảnh cho tập train và val (80% train, 20% val)
train_ratio = 0.8
train_count = int(len(image_files) * train_ratio)
train_files = image_files[:train_count]
test_files = image_files[train_count:]

# Sao chép file ảnh và nhãn vào thư mục train
for file_name in train_files:
    # Sao chép ảnh
    shutil.copy(os.path.join(images_dir, file_name), os.path.join(train_images_dir, file_name))
    # Sao chép nhãn (file .txt tương ứng)
    label_name = os.path.splitext(file_name)[0] + ".txt"
    if os.path.exists(os.path.join(labels_dir, label_name)):
        shutil.copy(os.path.join(labels_dir, label_name), os.path.join(train_labels_dir, label_name))

# Sao chép file ảnh và nhãn vào thư mục test
for file_name in test_files:
    # Sao chép ảnh
    shutil.copy(os.path.join(images_dir, file_name), os.path.join(test_images_dir, file_name))
    # Sao chép nhãn (file .txt tương ứng)
    label_name = os.path.splitext(file_name)[0] + ".txt"
    if os.path.exists(os.path.join(labels_dir, label_name)):
        shutil.copy(os.path.join(labels_dir, label_name), os.path.join(test_labels_dir, label_name))

print(f"Đã chia dữ liệu thành công: {len(train_files)} file train, {len(test_files)} file test.")