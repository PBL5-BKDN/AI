import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from gtts import gTTS
import os
from v1.model.enet import ENet  

# Xác định thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải mô hình
model = ENet(num_classes=10).to(device)
model.load_state_dict(torch.load('v1/enet_best.pth', map_location=device))
model.eval()

# Xác định tiền xử lý
input_size = (512, 256)  # (chiều rộng, chiều cao)
transform = transforms.Compose([
    transforms.Resize(input_size[::-1]),  # Resize mong đợi (chiều cao, chiều rộng)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Xác định tên lớp và bảng màu (gộp lớp 6 và 7 thành "Rào chắn")
class_names = [
    "Đường",              # Class ID: 0
    "Vạch kẻ đường",      # Class ID: 1
    "Làn xe chạy",        # Class ID: 2
    "Làn dịch vụ",        # Class ID: 3
    "Vạch qua đường",     # Class ID: 4
    "Lề đường",           # Class ID: 5
    "Rào chắn",           # Class ID: 6 (gộp Hàng rào và Barie)
    "Vỉa hè",             # Class ID: 7 (trước là 8)
    "Nền"                 # Class ID: 8 (trước là 9)
]
color_palette = [
    [0, 0, 0],       # Đường: Đen
    [255, 0, 0],     # Vạch kẻ đường: Đỏ
    [0, 255, 0],     # Làn xe chạy: Xanh lá
    [0, 0, 255],     # Làn dịch vụ: Xanh dương
    [255, 255, 0],   # Vạch qua đường: Vàng
    [255, 0, 255],   # Lề đường: Tím
    [0, 255, 255],   # Rào chắn: Xanh lam (màu của Hàng rào)
    [255, 128, 0],   # Vỉa hè: Cam
    [128, 0, 128]    # Nền: Tím đậm
]

# Hàm phân tích giúp xác định vùng nào chiếm ưu thế
def get_dominant_region(left_prop, center_prop, right_prop):
    safe_classes = [7, 4]  # Vỉa hè (7), Vạch qua đường (4)
    for cls in safe_classes:
        if left_prop[cls] > max(center_prop[cls], right_prop[cls]) and left_prop[cls] > 0.5:
            return "trái", cls
        if right_prop[cls] > max(center_prop[cls], left_prop[cls]) and right_prop[cls] > 0.5:
            return "phải", cls
        if center_prop[cls] > max(left_prop[cls], right_prop[cls]) and center_prop[cls] > 0.5:
            return "giữa", cls
    return None, None

# Hàm phân tích vùng và tạo hướng dẫn
def generate_guidance(pred):
    # Gộp lớp 7 (Barie) thành lớp 6 (Rào chắn) trong bản đồ phân đoạn
    pred = np.where(pred == 7, 6, pred)  # Chuyển lớp 7 thành lớp 6
    # Điều chỉnh các lớp > 7 để lấp chỗ trống
    pred = np.where(pred == 8, 7, pred)  # Vỉa hè từ 8 thành 7
    pred = np.where(pred == 9, 8, pred)  # Nền từ 9 thành 8

    # Chia khung hình thành các vùng
    height, width = pred.shape
    left_region = pred[:, :width//3]  # 1/3 bên trái
    center_region = pred[:, width//3:2*width//3]  # 1/3 giữa
    right_region = pred[:, 2*width//3:]  # 1/3 bên phải
    bottom_region = pred[2*height//3:, :]  # 1/3 dưới cùng

    # Hàm tính tỷ lệ lớp
    def get_proportions(region):
        total_pixels = region.size
        class_counts = np.bincount(region.flatten(), minlength=9)  # 9 lớp
        return class_counts / total_pixels

    left_prop = get_proportions(left_region)
    center_prop = get_proportions(center_region)
    right_prop = get_proportions(right_region)
    bottom_prop = get_proportions(bottom_region)

    # Logic hướng dẫn
    guidance = ""
    priority = None

    # Kiểm tra vùng dưới (gần người dùng)
    if bottom_prop[6] > 0.2:  # Rào chắn
        guidance = "Lùi lại, rào chắn phía trước."
        priority = "chướng ngại"
    elif bottom_prop[2] > 0.2 or bottom_prop[3] > 0.2:  # Làn xe đạp hoặc làn dịch vụ
        lane = "làn xe chạy" if bottom_prop[2] > bottom_prop[3] else "làn dịch vụ"
        guidance = f"Lùi lại, bạn đang ở {lane}. Tìm vỉa hè."
        priority = "không an toàn"
    elif bottom_prop[8] > 0.5:  # Nền
        guidance = "Lùi lại, khu vực không xác định. Tìm đường đã biết."
        priority = "không an toàn"
    elif bottom_prop[7] > 0.5:  # Vỉa hè
        guidance = "Tiến lên, bạn đang trên vỉa hè."
        priority = "an toàn"
    elif bottom_prop[4] > 0.2:  # Vạch qua đường
        guidance = "Tiến lên chậm, vạch qua đường phía trước. Đảm bảo an toàn."
        priority = "an toàn"
    elif bottom_prop[5] > 0.1:  # Lề đường
        guidance = "Tiến lên chậm, lề đường phía trước. Bước cẩn thận."
        priority = "thận trọng"
    elif bottom_prop[0] > 0.5:  # Đường
        # Kiểm tra trái/phải để tìm vỉa hè
        if left_prop[7] > 0.5 or left_prop[4] > 0.2:
            guidance = "Di chuyển sang trái, vỉa hè hoặc vạch qua đường ở đó."
            priority = "an toàn"
        elif right_prop[7] > 0.5 or right_prop[4] > 0.2:
            guidance = "Di chuyển sang phải, vỉa hè hoặc vạch qua đường ở đó."
            priority = "an toàn"
        else:
            guidance = "Lùi lại hoặc tìm vỉa hè."
            priority = "thận trọng"
    else:
        # Kiểm tra trái/phải/giữa để tìm hướng an toàn
        dominant_region, dominant_class = get_dominant_region(left_prop, center_prop, right_prop)
        if dominant_region == "trái":
            class_name = class_names[dominant_class]
            guidance = f"Di chuyển sang trái, {class_name} ở đó."
            priority = "an toàn"
        elif dominant_region == "phải":
            class_name = class_names[dominant_class]
            guidance = f"Di chuyển sang phải, {class_name} ở đó."
            priority = "an toàn"
        elif dominant_region == "giữa":
            class_name = class_names[dominant_class]
            guidance = f"Tiến lên, {class_name} phía trước."
            priority = "an toàn"
        elif center_prop[6] > 0.2:
            guidance = "Dừng lại, rào chắn phía trước."
            priority = "chướng ngại"
        else:
            guidance = "Dừng lại, khu vực phía trước không an toàn."
            priority = "không an toàn"

    return guidance, priority

# Hàm phát âm thanh hướng dẫn bằng gTTS
def speak_guidance(guidance):
    if guidance:
        tts = gTTS(text=guidance, lang='vi')
        tts.save("temp.mp3")
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load("temp.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
        os.remove("temp.mp3")  # Xóa file tạm

# Hàm xử lý và dự đoán cho ảnh
def predict_image(image_path, output_path=None):
    # Tải và tiền xử lý ảnh
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Ảnh {image_path} không tồn tại.")
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Chạy suy luận
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Gộp lớp 7 thành lớp 6 và điều chỉnh các lớp khác
    pred = np.where(pred == 7, 6, pred)  # Barie thành Rào chắn
    pred = np.where(pred == 8, 7, pred)  # Vỉa hè từ 8 thành 7
    pred = np.where(pred == 9, 8, pred)  # Nền từ 9 thành 8

    # Chuyển dự đoán thành bản đồ màu (RGB)
    pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id in range(9):  # 9 lớp
        pred_colored[pred == class_id] = color_palette[class_id]

    # Chuyển sang BGR cho OpenCV
    pred_colored_bgr = cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR)

    # Thay đổi kích thước ảnh gốc và chuyển sang BGR
    orig_image = image.resize(input_size, Image.BILINEAR)
    orig_image_bgr = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)

    # Tạo hướng dẫn
    guidance, _ = generate_guidance(pred)
    print(f"Hướng dẫn: {guidance}")  
    speak_guidance(guidance)

    # Tạo lớp phủ
    overlay = cv2.addWeighted(orig_image_bgr, 0.5, pred_colored_bgr, 0.5, 0)

    # Lưu kết quả
    if output_path:
        pred_image = Image.fromarray(pred_colored)
        pred_image.save(output_path + '_seg.png')
        Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).save(output_path + '_overlay.png')

    # Hiển thị kết quả
    cv2.imshow('Phân đoạn', pred_colored_bgr)
    cv2.imshow('Lớp phủ', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Demo với ảnh
image_path = 'v1/demo/test1.jpg'
output_path = 'v1/demo/test1'
predict_image(image_path, output_path)