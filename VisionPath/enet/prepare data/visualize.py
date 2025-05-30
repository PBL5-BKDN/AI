import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

label_dir = "segmentation_data/labels"
image_dir = "segmentation_data/images"
save_dir = "segmentation_data/vis"

os.makedirs(save_dir, exist_ok=True)

for fname in os.listdir(label_dir):
    if not fname.endswith(".txt"):
        continue

    label_path = os.path.join(label_dir, fname)
    img_path = os.path.join(image_dir, fname.replace(".txt", ".jpg"))
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(fname.replace(".txt", ""))

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            # Chuyển tọa độ normalized (YOLO) sang pixel
            points = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
            polygon = Polygon(points, closed=True, edgecolor='lime', fill=False, linewidth=1.5)
            ax.add_patch(polygon)

    plt.axis("off")
    plt.savefig(os.path.join(save_dir, fname.replace(".txt", ".png")))
    plt.close()
