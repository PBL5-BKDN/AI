import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as transforms
from gtts import gTTS
import os
import time
import pygame
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize ONNX Runtime session
onnx_model_path = 'model/enet.onnx'
try:
    session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    logging.info("ONNX model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load ONNX model: {e}")
    raise

input_name = session.get_inputs()[0].name

# Define preprocessing
input_size = (512, 256)  # (height, width)
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names and color palette
class_names = [
    "Đường", "Vạch kẻ đường", "Làn xe chạy", "Làn dịch vụ",
    "Vạch qua đường", "Lề đường", "Rào chắn", "Vỉa hè", "Nền"
]
color_palette = [
    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
    [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 128]
]

# Function to determine the dominant region
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

# Function to analyze regions and generate guidance
def generate_guidance(pred):
    pred = np.where(pred == 7, 6, pred)
    pred = np.where(pred == 8, 7, pred)
    pred = np.where(pred == 9, 8, pred)

    height, width = pred.shape
    left_region = pred[:, 2*width//3:]  # Right side (left from viewer's perspective)
    center_region = pred[:, width//3:2*width//3]
    right_region = pred[:, :width//3]  # Left side (right from viewer's perspective)
    bottom_region = pred[2*height//3:, :]  # Bottom third
    bottom_center_region = pred[2*height//3:, width//3:2*width//3]

    def get_proportions(region):
        total_pixels = region.size
        class_counts = np.bincount(region.flatten(), minlength=9)
        return class_counts / total_pixels

    left_prop = get_proportions(left_region)
    center_prop = get_proportions(center_region)
    right_prop = get_proportions(right_region)
    bottom_prop = get_proportions(bottom_region)
    bottom_center_prop = get_proportions(bottom_center_region)

    guidance = ""
    priority = None

    if bottom_prop[6] > 0.2 and bottom_center_prop[6] > 0.5:
        guidance = "Lùi lại, rào chắn phía trước."
        priority = "chướng ngại"
    elif (bottom_prop[2] > 0.2 and bottom_center_prop[2] > 0.5) or \
         (bottom_prop[3] > 0.2 and bottom_center_prop[3] > 0.5):
        lane = "làn xe chạy" if bottom_center_prop[2] > bottom_center_prop[3] else "làn dịch vụ"
        guidance = f"Lùi lại, bạn đang ở {lane}. Tìm vỉa hè."
        priority = "không an toàn"
    elif bottom_prop[8] > 0.5 and bottom_center_prop[8] > 0.5:
        guidance = "Lùi lại, khu vực không xác định. Tìm đường đã biết."
        priority = "không an toàn"
    elif bottom_prop[7] > 0.5 and bottom_center_prop[7] > 0.5:
        guidance = "Tiến lên, bạn đang trên vỉa hè."
        priority = "an toàn"
    elif bottom_prop[4] > 0.2 and bottom_center_prop[4] > 0.5:
        guidance = "Tiến lên chậm, vạch qua đường phía trước. Đảm bảo an toàn."
        priority = "an toàn"
    elif bottom_prop[5] > 0.1 and bottom_center_prop[5] > 0.5:
        guidance = "Tiến lên chậm, lề đường phía trước. Bước cẩn thận."
        priority = "thận trọng"
    elif bottom_prop[0] > 0.5 and bottom_center_prop[0] > 0.5:
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

# Function to play audio guidance
def speak_guidance(guidance):
    try:
        if guidance:
            tts = gTTS(text=guidance, lang='vi')
            tts.save("temp.mp3")
            pygame.mixer.init()
            pygame.mixer.music.load("temp.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.quit()
            os.remove("temp.mp3")
            logging.info(f"Played guidance: {guidance}")
    except Exception as e:
        logging.error(f"Failed to play audio: {e}")

# Function to process video
def predict_video(video_path, frame_skip=2):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Không thể mở video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Tổng số khung hình: {total_frames}, FPS: {fps}")

        last_guidance = ""
        last_speak_time = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to reduce load
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            input_tensor = transform(image).numpy()
            input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

            outputs = session.run(None, {input_name: input_tensor})[0]
            pred = np.argmax(outputs, axis=1).squeeze(0)

            pred = np.where(pred == 7, 6, pred)
            pred = np.where(pred == 8, 7, pred)
            pred = np.where(pred == 9, 8, pred)

            guidance, priority = generate_guidance(pred)
            if guidance != last_guidance and (time.time() - last_speak_time > 2):
                logging.info(f"Khung hình {frame_count}: {guidance}")
                print(f"Khung hình {frame_count}: {guidance}")
                speak_guidance(guidance)
                last_guidance = guidance
                last_speak_time = time.time()

            frame_count += 1
            logging.info(f"Đã xử lý khung hình {frame_count}/{total_frames}")

        cap.release()
        logging.info("Xử lý video hoàn tất.")
    except Exception as e:
        logging.error(f"Video processing failed: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        raise

# Run the demo
if __name__ == "__main__":
    video_path = 'demo/video1.mp4'
    try:
        predict_video(video_path, frame_skip=2)
    except Exception as e:
        logging.error(f"Program failed: {e}")