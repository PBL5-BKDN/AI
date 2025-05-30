import os
import shutil

label_dir = "output/labels"
image_dir = "output/images"

selected_label_dir = "selected_labels1"
selected_image_dir = "selected_images1"

target_classes = {1, 3, 6, 8, 9}

os.makedirs(selected_label_dir, exist_ok=True)
os.makedirs(selected_image_dir, exist_ok=True)

error_files = []
selected_count = 0

for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(label_dir, label_file)
    found = False

    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                    if class_id in target_classes:
                        found = True
                        break
                except ValueError:
                    continue
    except Exception as e:
        error_files.append((label_file, str(e)))
        continue

    if found:
        # Copy label
        shutil.copy(label_path, os.path.join(selected_label_dir, label_file))

        # Copy image nếu tồn tại
        base_name = os.path.splitext(label_file)[0]
        for ext in ['.jpg', '.png', '.jpeg']:
            image_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(image_path):
                shutil.copy(image_path, os.path.join(selected_image_dir, base_name + ext))
                break
        selected_count += 1

# Kết quả
print(f"\n✅ Đã chọn {selected_count} file hợp lệ.")

if error_files:
    print(f"\n⚠️ Có {len(error_files)} file bị lỗi format:")
    for filename, err in error_files:
        print(f"  - {filename}: {err}")
