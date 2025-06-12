import cv2
import zmq
import pickle
import requests
import time
import threading
import numpy as np
from navigation.speech.voice_mic import VoiceMic
from navigation.speech.voice_speaker import VoiceSpeaker

speaker_service = VoiceSpeaker(speaker_name="USB Audio Device")
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt(zmq.SUBSCRIBE, b"")
print("[ZMQ] Đã kết nối với socket camera server")
SEND_INTERVAL_MIN = 5
SEND_INTERVAL_MAX = 10

adaptive_interval = SEND_INTERVAL_MIN
BASE_URL = "http://14.245.164.135:3000"
SEND_INTERVAL = 12  # Giây, chỉ gửi ảnh mỗi 2 giây (giới hạn tần suất)
DIFF_THRESHOLD = 25  # Ngưỡng khác biệt, có thể điều chỉnh

last_sent_time = 0
lock = threading.Lock()
latest_frame = [None]
previous_frame = [None]

def frames_are_different(frame1, frame2, threshold=DIFF_THRESHOLD):
    if frame1 is None or frame2 is None:
        return True
    # Resize nhỏ lại để so sánh nhanh hơn, giảm nhiễu
    small1 = cv2.resize(frame1, (64, 64))
    small2 = cv2.resize(frame2, (64, 64))
    diff = cv2.absdiff(small1, small2)
    mean_diff = np.mean(diff)
    return mean_diff > threshold

def send_image_to_api(frame):
    try:
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print("[API] Lỗi mã hóa ảnh.")
            return
        files = {
            'image': ('obstacle.jpg', buffer.tobytes(), 'image/jpeg')
        }
        response = requests.post(f"{BASE_URL}/segment", files=files)
        data = response.json()
        print(f"[API] Phản hồi: {data}")
        message = data.get("data", "Lỗi không xác định.")
        speaker_service.speak(message)
    except Exception as e:
        print(f"[API] Lỗi gửi ảnh: {e}")

def api_sender_thread():
    global last_sent_time, adaptive_interval
    while True:
        time.sleep(0.05)
        now = time.time()
        with lock:
            frame = latest_frame[0]
            prev = previous_frame[0]
        if frame is not None and (now - last_sent_time >= adaptive_interval):
            if frames_are_different(frame, prev):  # nếu khác biệt lớn
                adaptive_interval = max(SEND_INTERVAL_MIN, adaptive_interval * 0.8)
                send_image_to_api(frame)
                with lock:
                    previous_frame[0] = frame.copy()
                last_sent_time = now
            else:  # nếu gần giống nhau, tăng interval
                adaptive_interval = min(SEND_INTERVAL_MAX, adaptive_interval * 1.2)

# Tạo thread riêng để gửi API
threading.Thread(target=api_sender_thread, daemon=True).start()

while True:
    try:
        data = socket.recv()
        frame = pickle.loads(data)
        with lock:
            latest_frame[0] = frame
    except Exception as e:
        print(f"[Camera ZMQ] Lỗi nhận frame: {e}")
        time.sleep(0.1)
