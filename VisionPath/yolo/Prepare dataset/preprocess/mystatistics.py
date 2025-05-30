# import os

# # Danh sách tên lớp (class names từ 0 đến 8)
# class_names = [
#     'person', 'bicycle', 'car', 'motorcycle', 'vegetation', 'billboard', 'manhole', 'trash Can', 'barrier'
# ]

# def count_bounding_boxes(label_dir, num_classes=8):
#     class_counts = [0] * num_classes

#     for label_file in os.listdir(label_dir):
#         if not label_file.endswith('.txt'):
#             continue
        
#         file_path = os.path.join(label_dir, label_file)
#         with open(file_path, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue  # bỏ dòng trống
                
#                 parts = line.split()
#                 if len(parts) < 1:
#                     continue  # dòng không hợp lệ
                
#                 try:
#                     class_id = int(parts[0])
#                     if 0 <= class_id < num_classes:
#                         class_counts[class_id] += 1
#                 except ValueError:
#                     print(f"Lỗi định dạng class_id trong file: {label_file}, dòng: {line}")
#                     continue

#     return class_counts

# # Thư mục chứa các file label (YOLO format)
# label_dir = 'selected_labels1'

# # Thống kê số lượng bounding box
# counts = count_bounding_boxes(label_dir, num_classes=len(class_names))

# vẽ biểu đồ ghi giá trị trên biểu đồ và lưu ảnh thống kê 
# Class 0 (person): 13863 bounding boxes
# Class 1 (bicycle): 3637 bounding boxes
# Class 2 (car): 29491 bounding boxes
# Class 3 (motorcycle): 3299 bounding boxes
# Class 4 (vegetation): 25959 bounding boxes
# Class 5 (billboard): 24246 bounding boxes
# Class 6 (manhole): 6660 bounding boxes
# Class 7 (trash Can): 4507 bounding boxes
# Class 8 (barrier): 6492 bounding boxes

import matplotlib.pyplot as plt
import numpy as np
def plot_class_distribution(class_counts, class_names):
    # Tạo biểu đồ cột
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts, color='skyblue')
    
    # Thêm tiêu đề và nhãn
    plt.title('Distribution of Bounding Boxes by Class')
    plt.xlabel('Class Names')
    plt.ylabel('Number of Bounding Boxes')
    
    # Hiển thị giá trị trên mỗi cột
    for i, count in enumerate(class_counts):
        plt.text(i, count + 1000, str(count), ha='center', va='bottom')

    # Lưu biểu đồ vào file
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution_2.png')
    
    # Hiển thị biểu đồ
    plt.show()
# Ví dụ về số lượng bounding box cho từng lớp
class_counts = [
    13863,  # person
    3637,   # bicycle
    29491,  # car
    3299,   # motorcycle
    25959,  # vegetation
    24246,  # billboard
    6660,   # manhole
    4507,   # trash Can
    6492    # barrier

]
# Danh sách tên lớp
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'vegetation', 
    'billboard', 'manhole', 'trash Can', 'barrier'
]
# Vẽ biểu đồ phân phối số lượng bounding box theo lớp
plot_class_distribution(class_counts, class_names)
