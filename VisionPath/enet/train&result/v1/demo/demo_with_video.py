import cv2
import numpy as np
import onnxruntime as ort
import os
import time
import threading
import pygame
import logging
from gtts import gTTS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load ONNX model
onnx_model_path = 'jetson_nano/model/enet.onnx'
try:
    session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    logging.info("ONNX model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load ONNX model: {e}")
    raise

input_name = session.get_inputs()[0].name

# Input size
input_size = (512, 256)  # height x width

# Class names and colors
class_names = ["Đường", "Vạch kẻ đường", "Làn xe chạy", "Làn dịch vụ",
               "Vạch qua đường", "Lề đường", "Rào chắn", "Vỉa hè", "Nền"]

color_palette = [
    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
    [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 128]
]

def decode_segmap(pred, color_palette):
    h, w = pred.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in enumerate(color_palette):
        color_image[pred == label_id] = color
    return color_image

def get_dominant_region(left_prop, center_prop, right_prop):
    safe_classes = [7, 4]  # Vỉa hè, Vạch qua đường
    for cls in safe_classes:
        if left_prop[cls] > max(center_prop[cls], right_prop[cls]) and left_prop[cls] > 0.5:
            return "trái", cls
        if right_prop[cls] > max(center_prop[cls], left_prop[cls]) and right_prop[cls] > 0.5:
            return "phải", cls
        if center_prop[cls] > max(left_prop[cls], right_prop[cls]) and center_prop[cls] > 0.5:
            return "giữa", cls
    return None, None

def generate_guidance(pred):
    height, width = pred.shape
    left_region = pred[:, 2*width//3:]
    center_region = pred[:, width//3:2*width//3]
    right_region = pred[:, :width//3]
    bottom_region = pred[2*height//3:, :]
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
        if dominant_region:
            class_name = class_names[dominant_class]
            directions = {"trái": "Di chuyển sang trái", "phải": "Di chuyển sang phải", "giữa": "Tiến lên"}
            guidance = f"{directions[dominant_region]}, {class_name} ở đó."
            priority = "an toàn"
        elif center_prop[6] > 0.2:
            guidance = "Dừng lại, rào chắn phía trước."
            priority = "chướng ngại"
        else:
            guidance = "Dừng lại, khu vực phía trước không an toàn."
            priority = "không an toàn"

    return guidance, priority

# Non-blocking TTS
def speak_guidance_async(guidance):
    def play():
        try:
            tts = gTTS(text=guidance, lang='vi')
            tts.save("temp.mp3")
            pygame.mixer.init()
            pygame.mixer.music.load("temp.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.quit()
            os.remove("temp.mp3")
            logging.info(f"Played guidance: {guidance}")
        except Exception as e:
            logging.error(f"Failed to play audio: {e}")
    threading.Thread(target=play, daemon=True).start()

def preprocess_frame(frame):
    resized = cv2.resize(frame, (input_size[1], input_size[0]))
    img = resized.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img, axis=0).astype(np.float32)

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

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            input_tensor = preprocess_frame(frame)
            outputs = session.run(None, {input_name: input_tensor})[0]
            pred = np.argmax(outputs, axis=1).squeeze(0)

            # Normalize labels
            pred = np.where(pred == 7, 6, pred)
            pred = np.where(pred == 8, 7, pred)
            pred = np.where(pred == 9, 8, pred)

            guidance, priority = generate_guidance(pred)

            if guidance != last_guidance and (time.time() - last_speak_time > 2):
                logging.info(f"Khung hình {frame_count}: {guidance}")
                print(f"Khung hình {frame_count}: {guidance}")
                speak_guidance_async(guidance)
                last_guidance = guidance
                last_speak_time = time.time()

            # Overlay
            seg_color = decode_segmap(pred, color_palette)
            seg_color = cv2.resize(seg_color, (frame.shape[1], frame.shape[0]))
            overlay = cv2.addWeighted(frame, 0.6, seg_color, 0.4, 0)
            cv2.imshow("Video Output", overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            logging.info(f"Đã xử lý khung hình {frame_count}/{total_frames}")

        cap.release()
        cv2.destroyAllWindows()
        logging.info("Xử lý video hoàn tất.")
    except Exception as e:
        logging.error(f"Video processing failed: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        raise

# Run
if __name__ == "__main__":
    video_path = 'jetson_nano/demo/test_2.mp4'
    try:
        predict_video(video_path, frame_skip=2)
    except Exception as e:
        logging.error(f"Program failed: {e}")
