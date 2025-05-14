import cv2
import threading
from navigation.speech.voice import VoiceService
from navigation.navigation.navigator import Navigator
from navigation.services.gps import GPSService  
from navigation.services.api import APIService
from navigation.config.settings import MAX_REROUTE_ATTEMPTS
import time
import signal
import sys
import board
import io
import logging
sys.path.insert(0, "build/lib.linux-armv7l-2.7/")

import adafruit_vl53l1x
import requests
from utils import handle_take_photo
# Biến toàn cục để kiểm soát trạng thái chạy của các luồng
running = True
camera_lock = threading.Lock()

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
# Xử lý tín hiệu kết thúc (Ctrl+C)
def signal_handler(sig, frame):
    global running
    print("\nĐã nhận tín hiệu kết thúc. Đang dừng các luồng...")
    running = False
    sys.exit(0)

# Đăng ký handler cho tín hiệu SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def init_cam_bien_layser(voice_service):
    try:
        # Khởi tạo I2C và cảm biến VL53L1X
        i2c = board.I2C()  # Sử dụng I2C bus 1 trên Jetson Nano
        tof = adafruit_vl53l1x.VL53L1X(i2c)
        print("Đang khởi tạo cảm biến VL53L1X...")
        
        # Cấu hình khoảng cách tối đa (Long Range, lên đến 4m)
        tof.distance_mode = 2  # 1 = Short, 2 = Long
        tof.timing_budget = 300  # Thời gian đo 100ms cho độ chính xác cao
        tof.start_ranging()
        print("Cảm biến VL53L1X khởi tạo thành công, bắt đầu đo khoảng cách")
        
        while running:
            if tof.data_ready:
                distance = tof.distance
                print(f"Khoảng cách đo được: {distance} cm")
                if distance is not None and distance >= 0:
                    if 50 <= distance <= 100:
                        print("Vật cản trong phạm vi 0,5 - 1 mét!")
                        voice_service.speak("Cảnh báo: Vật cản trong phạm vi 0,5 - 1 mét!")
                        image_file = handle_take_photo(camera_lock, video_capture)
                        if image_file is None:
                            print("Không chụp được ảnh từ camera.")
                        try:
                            files = {'image':('obstacle.jpg', image_file, 'image/jpeg')}
                            response = requests.post("http://14.233.84.201:3000/detect", files=files)
                            data = response.json()
                            print(f"Dữ liệu từ API: {data}")
                            voice_service.speak(data["data"] if "data" in data else "Không phát hiện vật cản.")
                            time.sleep(10)
                        except requests.RequestException as e:
                            print(f"Lỗi gửi yêu cầu HTTP: {e}")
                else:
                    print("Khoảng cách không hợp lệ")
                tof.clear_interrupt()  # Xóa cờ ngắt
            time.sleep(1)  # Tần suất đo 1 lần/giây
    except Exception as e:
        print(f"Lỗi trong luồng cảm biến: {e}")
    finally:
        try:
            tof.stop_ranging()
            print("Đã dừng cảm biến VL53L1X")
        except Exception as e:
            print(f"Lỗi khi dừng cảm biến: {e}")    # Khởi tạo cảm biến


def init_phan_doan_lan_duong():
    print("Khởi tạo phân đoán làng đường")
    try:
        while running:
            print("Đang phán đoán làng đường...")
            time.sleep(5)
    except Exception as e:
        print(f"Lỗi trong luồng phân đoán làng đường: {e}")
    finally:
        print("Đã dừng luồng phân đoán làng đường")

def handle_ask_chatbot(voice_service, question):
    print("Xử lý yêu cầu chatbot")
    import  requests
    image_file = handle_take_photo(camera_lock, video_capture)
    if image_file is None:
        print("Không chụp được ảnh từ camera.")
    data = requests.post("http://14.233.84.201:3000/upload-image", data={
        "question": question,
        "file": {'image': ('obstacle.jpg', image_file, 'image/jpeg')},
    })
    data = data.json()
    print(data)
    voice_service.speak(data["text"])

def handle_navigtion(navigator, gps_service):
    print("navigation!")
    #Get destination from user
    destination = navigator.get_destination_from_user()
    
    # Get initial GPS position
    initial_lat, initial_lng = gps_service.wait_for_valid_location()
    
    # Check if GPS data is available
    if initial_lat is None or initial_lng is None:
        print("Không thể lấy vị trí GPS sau thời gian chờ")
        voice_service.speak("Không thể xác định vị trí của bạn. Vui lòng nói lại điểm đến.")
        return
    
    # Navigation loop
    reroute_attempts = 0
    
    while reroute_attempts < MAX_REROUTE_ATTEMPTS and running:
        print(f"Đang yêu cầu lộ trình từ API (Lần {reroute_attempts + 1})...")
        response_data = navigator.request_route(initial_lat, initial_lng, destination)
        
        if response_data and 'steps' in response_data and response_data['steps']:
            steps = response_data['steps']
            
            # Reset reroute counter when we get a valid route
            reroute_attempts = 0
            
            # Follow the route
            success, new_lat, new_lng = navigator.follow_route(steps, initial_lat, initial_lng)
            
            # If navigation completed successfully
            if success:
                break
                
            # If we need to reroute
            if new_lat and new_lng:
                initial_lat, initial_lng = new_lat, new_lng
                reroute_attempts += 1
                continue
        else:
            print("Không nhận được lộ trình hoặc lộ trình rỗng từ API.")
            reroute_attempts += 1
            
            if response_data:
                error_message = response_data.get('error', 'Không thể lấy lộ trình.')
                print(f"Lỗi API: {error_message}")
                voice_service.speak(f"Lỗi: {error_message}. Thử lại.")
            else:
                voice_service.speak("Không thể kết nối hoặc nhận dữ liệu từ máy chủ dẫn đường. Thử lại.")
            
            if reroute_attempts >= MAX_REROUTE_ATTEMPTS:
                print("Đã thử tìm lại đường quá nhiều lần. Bắt đầu lại từ đầu.")
                voice_service.speak("Không thể tìm được đường đi sau nhiều lần thử. Vui lòng nói lại điểm đến.")
                break


def xu_ly_yeu_cau(voice_service, gps_service, api_service): 
    print("Khởi tạo hệ thống dẫn đường")
    
    # Initialize navigator
    navigator = Navigator(gps_service, voice_service, api_service)
    try:
        while running:
            text = voice_service.recognize_speech()
            print(f"Đã nhận diện: {text}")
            if text and text.startswith("tâm"):
                handle_ask_chatbot(voice_service, text)
            elif text and text.startswith("nguyên"):
                handle_navigtion(navigator, gps_service)
            else:
                voice_service.speak("Yêu cầu không hợp lệ. Vui lòng thử lại.")   
            time.sleep(1)   
    except Exception as e:
        print(f"Lỗi trong luồng xử lý yêu cầu: {e}")
    finally:
        print("Đã dừng luồng xử lý yêu cầu")

def cleanup_resources(gps_service):
    # Dọn dẹp tài nguyên
    if gps_service:
        gps_service.cleanup()
    print("Đã dọn dẹp tài nguyên")

if __name__ == "__main__":    
    try:
        # Khởi tạo camera CSI
        video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        if not video_capture.isOpened():
            raise Exception("Không thể mở camera CSI")
        print("Camera CSI khởi tạo thành công")
        # Khởi tạo các dịch vụ dùng chung
        voice_service = VoiceService()
        gps_service = None
        api_service = None
        try:
            gps_service = GPSService()
            api_service = APIService()
        except Exception as e:
            print(f"Lỗi khi khởi tạo dịch vụ: {e}")
            raise
        
        # Tạo các thread riêng cho từng hàm
        t1 = threading.Thread(target=init_cam_bien_layser, args=(voice_service,), daemon=True)
        t2 = threading.Thread(target=init_phan_doan_lan_duong, daemon=True)
        t3 = threading.Thread(target=xu_ly_yeu_cau, args=(voice_service, gps_service, api_service), daemon=True)

        # Khởi động các thread
        t1.start()
        t2.start()
        t3.start()

        # Chờ các thread hoàn thành (sẽ không bao giờ kết thúc trừ khi có Ctrl+C)
        while running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nĐã nhận tín hiệu kết thúc từ bàn phím")
    except Exception as e:
        pass
    finally:
        # Đánh dấu dừng các luồng
        running = False
        # Dọn dẹp tài nguyên
        # cleanup_resources(gps_service)
        
        print("Chương trình đã kết thúc.")
