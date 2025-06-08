import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

# Màu cho từng lớp (theo số lớp của bạn)
COLORS = [
    (255, 255, 0),     # lớp 0
    (0, 255, 0),       # lớp 1
    (255, 0, 0),       # lớp 2
    (0, 0, 255),       # lớp 3
    (0, 0, 0),         # lớp 4 - nền
]

def decode_segmap(mask, num_classes):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(num_classes):
        color_mask[mask == cls] = COLORS[cls]
    return color_mask

def show_prediction(model, dataset, device, idx=0, num_classes=3):
    model.eval()
    with torch.no_grad():
        image, mask = dataset[idx]  # image: CxHxW, mask: HxW
        input_tensor = image.unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Unnormalize ảnh
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) +
                    np.array([0.485, 0.456, 0.406]))
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

        # Mask màu
        pred_color = decode_segmap(pred_mask, num_classes)
        mask_color = decode_segmap(mask.numpy(), num_classes)

        overlay = cv2.addWeighted(image_np, 0.6, pred_color, 0.4, 0)

        # Hiển thị
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(image_np)
        plt.title("Ảnh gốc")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(mask_color)
        plt.title("Mask Ground Truth")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(pred_color)
        plt.title("Mask Dự đoán")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

def analyze_position(pred):
    h, w = pred.shape
    bottom = pred[-h // 4:, :]  # 1/4 dưới ảnh
    unique_classes = np.unique(bottom)

    # Cảnh báo dựa vào lớp xuất hiện ở phần dưới
    if 2 in unique_classes:
        print("⚠️ Cảnh báo: Bạn đang đứng trên **đường xe chạy**!")
    elif 1 in unique_classes:
        print("🚸 Bạn đang đứng trên **vạch kẻ đường cho người đi bộ**.")
    elif 3 in unique_classes:
        print("✅ Bạn đang đứng trên **vỉa hè**.")
    else:
        print("❓ Không xác định được vị trí đứng.")

    # Xác định vị trí lớp 1 và lớp 3 trong toàn ảnh
    left = pred[:, :w//3]
    center = pred[:, w//3:2*w//3]
    right = pred[:, 2*w//3:]

    def find_position(region, cls):
        return cls in np.unique(region)

    for cls, name in [(1, "vạch kẻ đường"), (3, "vỉa hè")]:
        pos = []
        if find_position(left, cls):
            pos.append("bên trái")
        if find_position(center, cls):
            pos.append("phía trước")
        if find_position(right, cls):
            pos.append("bên phải")

        if pos:
            print(f"📍 Lớp {cls} ({name}) xuất hiện ở: {', '.join(pos)}.")
        else:
            print(f"📍 Lớp {cls} ({name}) **không xuất hiện** trong ảnh.")

def predict_image(model_path, image_path, output_path="overlay_output.jpg", device='cuda', num_classes=4):
    # Load model
    model = load_model(model_path, num_classes=num_classes)
    model.to(device)
    model.eval()

    # Load ảnh
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize về 512x384
    transform = A.Compose([
        A.Resize(384, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    aug = transform(image=img_rgb)
    input_tensor = aug['image'].unsqueeze(0).to(device)

    # Suy luận
    with torch.no_grad():
        start = time.time()
        output = model(input_tensor)  # (1, C, H, W)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        end = time.time()
        print(f"⏱ Thời gian suy luận: {(end - start)*1000:.2f} ms")

    # Phân tích logic đứng
    analyze_position(pred)

    # Overlay mask màu lên ảnh
    pred_color = decode_segmap(pred, num_classes)
    pred_color_bgr = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)

    img_resized = cv2.resize(img, (512, 384))  # Resize ảnh gốc cho khớp
    overlay = cv2.addWeighted(img_resized, 0.5, pred_color_bgr, 0.5, 0)

    cv2.imwrite(output_path, overlay)
    print(f"✅ Overlay đã lưu tại: {output_path}")

model_path = "deeplabv3plus_best.pth"
model = torch.load(model_path, map_location='cuda')
predict_image(model, "im1.jpg", device='cuda', num_classes=3, output_path="overlay_im1.jpg")
