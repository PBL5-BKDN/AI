# import os
# import random
# from collections import Counter
# from PIL import Image
# import numpy as np
# from sklearn.model_selection import train_test_split
# import shutil

# # Đường dẫn đến thư mục dữ liệu gốc
# image_dir = 'data/images/'
# mask_dir = 'data/labels/'

# # Lấy danh sách ảnh
# images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
# print(f"Total images found: {len(images)}")

# # Kiểm tra danh sách mask
# masks = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))]
# print(f"Total masks found: {len(masks)}")

# # Chỉ lấy những ảnh có mask tương ứng
# valid_images = []
# for img in images:
#     base_name = os.path.splitext(img)[0]
#     for ext in ('.png', '.jpg'):
#         if f"{base_name}{ext}" in masks:
#             valid_images.append(img)
#             break
# print(f"Total valid images (with masks): {len(valid_images)}")

# # Hàm tính phân bố lớp trong mask
# def get_class_distribution(mask_path):
#     mask = np.array(Image.open(mask_path))
#     return Counter(mask.flatten())

# # Chọn lọc 5,000 ảnh với trọng số dựa trên lớp hiếm
# weights = []
# for img in valid_images:
#     mask_path = os.path.join(mask_dir, os.path.splitext(img)[0] + '.png')
#     class_dist = get_class_distribution(mask_path)
#     score = sum(class_dist.get(i, 0) for i in [18, 20, 23])  # Lớp hiếm: traffic light, bicycle, motorcycle
#     weights.append(score + 1)

# selected_images = random.choices(valid_images, weights=weights, k=5000)
# print(f"Selected {len(selected_images)} images for training")

# # Chia tập dữ liệu: 80% train, 10% val, 10% test
# train_val_images, test_images = train_test_split(selected_images, test_size=0.1, random_state=42)
# train_images, val_images = train_test_split(train_val_images, test_size=0.111, random_state=42)

# # Tạo thư mục đích
# for split, img_list in [('train', train_images), ('val', val_images), ('test', test_images)]:
#     os.makedirs(f'selected/images/{split}', exist_ok=True)
#     os.makedirs(f'selected/labels/{split}', exist_ok=True)
#     print(f"Number of images in {split}: {len(img_list)}")
#     for img in img_list:
#         shutil.copy(os.path.join(image_dir, img), os.path.join(f'selected/images/{split}', img))
#         base_name = os.path.splitext(img)[0]
#         mask_file = None
#         for ext in ('.png', '.jpg'):
#             potential_mask = f"{base_name}{ext}"
#             if potential_mask in masks:
#                 mask_file = potential_mask
#                 break
#         if mask_file:
#             shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(f'selected/labels/{split}', mask_file))
#         else:
#             print(f"Mask not found for {img}")

# print("Dataset split completed successfully!")

import os
import random
import shutil
from tqdm import tqdm

def select_random_images(src_image_dir, src_label_dir, dest_image_dir, dest_label_dir, num_samples=1500):
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)

    # Lấy danh sách ảnh từ thư mục nguồn
    src_images = [f for f in os.listdir(src_image_dir) if f.endswith('.jpg')]

    # Lấy danh sách ảnh đã có trong thư mục đích
    dest_images = [f for f in os.listdir(dest_image_dir) if f.endswith('.jpg')]

    # Loại bỏ các ảnh đã có trong thư mục đích
    available_images = [f for f in src_images if f not in dest_images]

    # Kiểm tra xem có đủ ảnh để chọn không
    if len(available_images) < num_samples:
        raise ValueError(f"Không đủ ảnh để chọn! Chỉ có {len(available_images)} ảnh khả dụng, cần {num_samples}.")

    # Chọn ngẫu nhiên 1500 ảnh
    selected_images = random.sample(available_images, num_samples)

    # Sao chép ảnh và label sang thư mục đích
    for img_file in tqdm(selected_images, desc="Sao chép ảnh và label"):
        # Đường dẫn file ảnh
        src_img_path = os.path.join(src_image_dir, img_file)
        dest_img_path = os.path.join(dest_image_dir, img_file)

        # Đường dẫn file label (thay .jpg thành .png)
        label_file = img_file.replace('.jpg', '.png')
        src_label_path = os.path.join(src_label_dir, label_file)
        dest_label_path = os.path.join(dest_label_dir, label_file)

        # Kiểm tra xem file label có tồn tại không
        if not os.path.exists(src_label_path):
            print(f"Không tìm thấy label cho ảnh {img_file}, bỏ qua...")
            continue

        # Sao chép ảnh và label
        shutil.copy(src_img_path, dest_img_path)
        shutil.copy(src_label_path, dest_label_path)

    print(f"Đã sao chép {len(selected_images)} ảnh và label vào thư mục đích.")

# Sử dụng hàm
src_image_dir = 'data/images'
src_label_dir = 'data/labels'
dest_image_dir = 'selected/images/test'
dest_label_dir = 'selected/labels/test'

select_random_images(src_image_dir, src_label_dir, dest_image_dir, dest_label_dir, num_samples=100)