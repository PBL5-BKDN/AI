from navigation.speech.voice_speaker import VoiceSpeaker
from navigation.speech.voice_mic import VoiceMic
from navigation.services.gps import GPSService
from navigation.navigation.navigator import Navigator
import logging
import sys
import requests
import cv2 
import zmq
import pickle
import threading
import time
from navigation.services.api import APIService
import subprocess

# Cấp quyền cho cổng GPS trước khi sử dụng
try:
    subprocess.run(["sudo", "chmod", "666", "/dev/ttyTHS1"], check=True)
except Exception as e:
    print(f"Không thể cấp quyền cho /dev/ttyTHS1: {e}")

api_service = APIService()
gps_service = GPSService()
speaker_service = VoiceSpeaker()
mic_service = VoiceMic()
navigator = Navigator(gps_service, speaker_service, mic_service, api_service)
BASE_URL = "http://14.245.164.135:3000"
# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Biến toàn cục để lưu frame mới nhất nhận từ camera server ---
latest_frame = [None]  # dùng list để có thể gán trong thread
navigation_stop_event = threading.Event()

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
            
# --- Khởi động thread nhận frame từ camera server ---
camera_thread = threading.Thread(target=zmq_camera_client_thread, daemon=True)
camera_thread.start()

def handle_ask_chatbot(question):
    frame = latest_frame[0]
    if frame is None:
        print("[Camera] Không chụp được ảnh.")
        return
    print(f"[Camera] Ảnh đã chụp thành công")
    
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        print("[API] Lỗi mã hóa ảnh.")
        return

    try:   
        response = requests.post(
            f"{BASE_URL}/upload-image",
            data={"question": question},
            files={"file": ('obstacle.jpg', buffer.tobytes(), "image/jpeg")}
        )
        data = response.json()
        print(data)
        speaker_service.speak(data["data"])
    except Exception as e:
        print(f"Lỗi khi gọi API chatbot: {e}")
        speaker_service.speak("Lỗi khi kết nối đến hệ thống chatbot")

def handle_ask_chatbot_thread(question):
    thread = threading.Thread(target=handle_ask_chatbot, args=(question,))
    thread.start()

def handle_navigation(navigator, gps_service, textvoice=None):
    MAX_REROUTE_ATTEMPTS = 3
    running = True
    destination = textvoice
    print(f"Điểm đến: {destination}")    
    # Fake GPS to test
    # initial_lat = 16.0569047
    # initial_lng = 108.1815261
    initial_lat, initial_lng = gps_service.wait_for_valid_location()
    if initial_lat is None or initial_lng is None:
        print("Không thể lấy vị trí GPS sau thời gian chờ")
        speaker_service.speak("Không thể xác định vị trí của bạn. Mời nói lại điểm đến.")
        return
    
    reroute_attempts = 0
    while reroute_attempts < MAX_REROUTE_ATTEMPTS and running:
        if navigation_stop_event.is_set():
            print("Đã nhận lệnh dừng navigation.")
            speaker_service.speak("Đã dừng tìm đường.")
            break
        print(f"Đang yêu cầu lộ trình từ API (Lần {reroute_attempts + 1})...")
        try:
            response_data = navigator.request_route(initial_lat, initial_lng, destination)
        except Exception as e:
            print(f"Lỗi gửi yêu cầu đến API: {e}")
            speaker_service.speak("Lỗi kết nối đến máy chủ dẫn đường. Thử lại sau.")
            reroute_attempts += 1
            time.sleep(2)
            continue
        if response_data and 'steps' in response_data and response_data['steps']:
            steps = response_data['steps']
            reroute_attempts = 0
            try:
                success, new_lat, new_lng = navigator.follow_route(steps, initial_lat, initial_lng)
            except Exception as e:
                print(f"Lỗi khi theo dõi lộ trình: {e}")
                speaker_service.speak("Lỗi khi theo dõi lộ trình. Thử lại sau.")
                break
            if success:
                break
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
                speaker_service.speak(f"Lỗi: {error_message}. Thử lại.")
            else:
                speaker_service.speak("Không thể nhận dữ liệu từ máy chủ dẫn đường. Thử lại.")
            time.sleep(2)
            if reroute_attempts >= MAX_REROUTE_ATTEMPTS:
                print("Đã thử tìm lại đường quá nhiều lần. Bắt đầu lại từ đầu.")
                speaker_service.speak("Không thể tìm được đường đi. Mời nói lại điểm đến.")
                break

def handle_navigation_thread(navigator, gps_service, textvoice=None):
    navigation_stop_event.clear()  # reset event trước khi bắt đầu navigation mới
    thread = threading.Thread(target=handle_navigation, args=(navigator, gps_service, textvoice))
    thread.start()

def xu_ly_yeu_cau(): 
    logger.info("Khởi tạo luồng xử lý yêu cầu")
    speaker_service.speak("Nói câu bắt đầu bằng 'mi' để hỏi chatbot hoặc '3' để tìm đường.")
    try:
        while True:
            text = mic_service.recognize_speech()
            logger.info(f"Đã nhận diện: {text}")
            if text:
                words = text.strip().lower().split()
                if words:
                    first_word = words[0]
                    if first_word in ["mi", "me", "my", "mỹ", "mây"]:
                        textvoice = ' '.join(words[1:])
                        speaker_service.speak("Đang xử lý")
                        handle_ask_chatbot_thread(textvoice)
                        # speaker_service.speak("Đang lắng nghe yêu cầu tiếp theo")
                    elif first_word in ["3", "ba"]:
                        textvoice = ' '.join(words[1:])
                        navigator = Navigator(gps_service, speaker_service, mic_service, api_service)
                        handle_navigation_thread(navigator, gps_service, textvoice)
                        # speaker_service.speak("Đang lắng nghe yêu cầu tiếp theo")
                    elif text.strip() == "dừng lại":
                        navigation_stop_event.set()
                        speaker_service.speak("Đã dừng tìm đường. Bạn có thể nói lại điểm đến mới.")
                    elif text is not None and text.strip() != "":
                        speaker_service.speak("Yêu cầu không hợp lệ")
    except Exception as e:
        logger.error(f"Lỗi trong luồng xử lý yêu cầu: {e}")
    finally:
        pass

xu_ly_yeu_cau()
