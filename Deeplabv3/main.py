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

def predict_image(model, image_path, device='cuda', num_classes=3, output_path="overlay_output.jpg"):
    # Load ảnh và chuyển sang RGB
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize về 512x384
    transform = A.Compose([
        A.Resize(384, 512),  # (height, width)
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    aug = transform(image=img_rgb)
    input_tensor = aug['image'].unsqueeze(0).to(device)

    # Dự đoán và đo thời gian
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        infer_time = (time.time() - start_time) * 1000
        print(f"⏱ Inference time: {infer_time:.2f} ms")

        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Tạo mask màu và overlay
    pred_color = decode_segmap(pred, num_classes)
    pred_color_bgr = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)

    # Resize lại ảnh gốc nếu cần khớp kích thước
    img_resized = cv2.resize(img, (512, 384))  # (width, height)
    overlay = cv2.addWeighted(img_resized, 0.5, pred_color_bgr, 0.5, 0)

    cv2.imwrite(output_path, overlay)
    print(f"✅ Done: {output_path} saved")

model_path = "deeplabv3plus_best.pth"
model = torch.load(model_path, map_location='cuda')
predict_image(model, "im1.jpg", device='cuda', num_classes=3, output_path="overlay_im1.jpg")
