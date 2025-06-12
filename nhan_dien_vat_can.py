import cv2
import zmq
import pickle
import threading
import time
import board
import busio
import requests
import adafruit_vl53l1x
import time
from navigation.speech.voice_speaker import VoiceSpeaker
BASE_URL = "http://14.185.228.50:3000"
speaker_service = VoiceSpeaker(speaker_name="USB Audio Device")
# --- Biến toàn cục để lưu frame mới nhất nhận từ camera server ---
latest_frame = [None]  # dùng list để có thể gán trong thread

def zmq_camera_client_thread():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    while True:
        try:
            data = socket.recv()
            frame = pickle.loads(data)
            latest_frame[0] = frame  # cập nhật frame mới nhất
        except Exception as e:
            print(f"[Camera ZMQ] Lỗi nhận frame: {e}")
            time.sleep(0.1)

class ToFSensor:
    def __init__(self, i2c, sensor_id):
        self.sensor_id = sensor_id
        try:
            self.tof = adafruit_vl53l1x.VL53L1X(i2c)
            self.tof.distance_mode = 2
            self.tof.timing_budget = 200
            self.tof.start_ranging()
            print(f"[Cảm biến {sensor_id}] Khởi tạo thành công.")
        except Exception as e:
            print(f"[Cảm biến {sensor_id}] Lỗi khởi tạo: {e}")
            self.tof = None

    def read_distance(self):
        if self.tof and self.tof.data_ready:
            try:
                distance = self.tof.distance
                self.tof.clear_interrupt()
                return distance
            except OSError as e:
                print(f"[Cảm biến {self.sensor_id}] Lỗi đọc dữ liệu: {e}")
                self.stop()
                self.tof = None
        return None

    def stop(self):
        if self.tof:
            try:
                self.tof.stop_ranging()
            except Exception as e:
                print(f"[Cảm biến {self.sensor_id}] Lỗi khi dừng: {e}")

    def __del__(self):
        self.stop()

class ObstacleDetectionSystem:
    def __init__(self):
        self.sensors = []
        self.last_alert_time = 0
        self.alert_interval = 5

    def setup_sensors(self):
        try:
            i2c_buses = [
                busio.I2C(board.SCL, board.SDA),
                busio.I2C(board.SCL_1, board.SDA_1),
            ]
            self.sensors = [ToFSensor(i2c, idx+1) for idx, i2c in enumerate(i2c_buses)]
        except Exception as e:
            print(f"Lỗi khi khởi tạo các cảm biến: {e}")
            self.sensors = []

    def send_image_to_api_async(self, frame):
        try:
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                print("[API] Lỗi mã hóa ảnh.")
                return
            files = {
                'image': ('obstacle.jpg', buffer.tobytes(), 'image/jpeg')
            }
            response = requests.post(f"{BASE_URL}/detect", files=files)
            data = response.json()
            print(f"[API] Phản hồi: {data}")
            message = data.get("data", {}).get("data", "Không phát hiện vật cản")
            speaker_service.speak(message)
        except Exception as e:
            print(f"[API] Lỗi gửi ảnh: {e}")

    def detect_obstacles(self):
        distances = []
        for sensor in self.sensors:
            distance = sensor.read_distance()
            if distance:
                print(f"[Cảm biến {sensor.sensor_id}] Khoảng cách: {distance} cm")
                distances.append(distance)

        now = time.time()
        if any(100 <= d <= 150 for d in distances):
            if now - self.last_alert_time >= self.alert_interval:
                self.last_alert_time = now
                print("[Hệ thống] Phát hiện vật cản trong phạm vi 1–1.5m!")
                speaker_service.speak("Cảnh báo! Phát hiện vật cản gần phía trước.")

                frame = latest_frame[0]
                if frame is not None:
                    print(f"[Camera] Ảnh đã chụp thành công")
                    self.send_image_to_api_async(frame)
                else:
                    print("[Camera] Không có ảnh mới.")

    def run(self):
        self.setup_sensors()
        try:
            while True:
                self.detect_obstacles()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Dừng hệ thống.")
        finally:
            self.cleanup()

    def cleanup(self):
        for sensor in self.sensors:
            sensor.stop()

    def __del__(self):
        self.cleanup()

# --- Khởi động thread nhận frame từ camera server ---
camera_thread = threading.Thread(target=zmq_camera_client_thread, daemon=True)
camera_thread.start()

# --- Chạy hệ thống phát hiện vật cản ---
system = ObstacleDetectionSystem()
system.run()
