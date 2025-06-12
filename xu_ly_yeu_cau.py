import logging
import sys
import requests
import cv2
import zmq
import pickle
import threading
import time
import subprocess
from navigation.speech.voice_speaker import VoiceSpeaker
from navigation.speech.voice_mic import VoiceMic
from navigation.services.gps import GPSService
from navigation.navigation.navigator import Navigator
from navigation.services.api import APIService

try:
    subprocess.run(["sudo", "chmod", "666", "/dev/ttyTHS1"], check=True)
except Exception as e:
    print(f"Không thể cấp quyền cho /dev/ttyTHS1: {e}")

api_service = APIService()
gps_service = GPSService()
speaker_service = VoiceSpeaker(speaker_name="USB Audio Device")
mic_service = VoiceMic(mic_name="USB Composite Device")
BASE_URL = "http://14.185.228.50:3000"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

latest_frame = [None]
navigation_stop_event = threading.Event()
navigation_thread = [None]  # allow reference update from main loop

def zmq_camera_client_thread():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    while True:
        try:
            data = socket.recv()
            frame = pickle.loads(data)
            latest_frame[0] = frame
        except Exception as e:
            print(f"[Camera ZMQ] Lỗi nhận frame: {e}")
            time.sleep(0.1)

def handle_ask_chatbot(question):
    frame = latest_frame[0]
    if frame is None:
        speaker_service.speak("Không chụp được ảnh từ camera.")
        return
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        speaker_service.speak("Lỗi mã hóa ảnh.")
        return
    try:
        response = requests.post(
            f"{BASE_URL}/upload-image",
            data={"question": question},
            files={"file": ('obstacle.jpg', buffer.tobytes(), "image/jpeg")}
        )
        data = response.json()
        speaker_service.speak(data.get("data", "Không có phản hồi từ chatbot."))
    except Exception as e:
        print(f"Lỗi khi gọi API chatbot: {e}")
        speaker_service.speak("Lỗi khi kết nối đến hệ thống chatbot")

def handle_ask_chatbot_thread(question):
    thread = threading.Thread(target=handle_ask_chatbot, args=(question,))
    thread.start()

def navigation_worker(destination):
    local_navigator = Navigator(gps_service, speaker_service, mic_service, api_service)
    handle_navigation(local_navigator, gps_service, destination)

def handle_navigation(navigator, gps_service, destination):
    MAX_REROUTE_ATTEMPTS = 3
    initial_lat, initial_lng = gps_service.wait_for_valid_location()
    if initial_lat is None or initial_lng is None:
        speaker_service.speak("Không thể xác định vị trí của bạn. Mời nói lại điểm đến.")
        return
    reroute_attempts = 0
    while reroute_attempts < MAX_REROUTE_ATTEMPTS:
        if navigation_stop_event.is_set():
            speaker_service.speak("Đã dừng tìm đường.")
            break
        try:
            response_data = navigator.request_route(initial_lat, initial_lng, destination)
        except Exception as e:
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
                speaker_service.speak("Lỗi khi theo dõi lộ trình. Thử lại sau.")
                break
            if success:
                break
            if new_lat and new_lng:
                initial_lat, initial_lng = new_lat, new_lng
                reroute_attempts += 1
                continue
        else:
            reroute_attempts += 1
            if response_data:
                error_message = response_data.get('error', 'Không thể lấy lộ trình.')
                speaker_service.speak(f"Lỗi: {error_message}. Thử lại.")
            else:
                speaker_service.speak("Không thể nhận dữ liệu từ máy chủ dẫn đường. Thử lại.")
            time.sleep(2)
            if reroute_attempts >= MAX_REROUTE_ATTEMPTS:
                speaker_service.speak("Không thể tìm được đường đi. Mời nói lại điểm đến.")
                break

def ask_chatbot_loop():
    while True:
        speaker_service.speak("Bạn muốn hỏi gì?")
        question = mic_service.recognize_speech()
        logger.info(f"Câu hỏi chatbot: {question}")
        if question:
            speaker_service.speak("Đang xử lý câu hỏi.")
            handle_ask_chatbot_thread(question)
            break
        else:
            None
def ask_destination_loop():
    while True:
        speaker_service.speak("Bạn muốn đi đâu?")
        destination = mic_service.recognize_speech()
        logger.info(f"Điểm đến navigation: {destination}")
        if destination:
            return destination
        else:
            None
def main_loop():
    speaker_service.speak("Hệ thống sẵn sàng. Hãy nói 'mi' để hỏi chatbot hoặc 'bi' để tìm đường.")
    while True:
        text = mic_service.recognize_speech()
        logger.info(f"Đã nhận diện: {text}")
        if not text:
            continue
        command = text.strip().lower()
        if command in ["mi", "me", "my", "mỹ", "mây"]:
            ask_chatbot_loop()
        elif command in ["b", "bi", "bì", "bí", "đi", "pi", "thi", "bee", "bị", "bỉ", "vy", "vi", "vuy", "v", "duy"]:
            # Nếu đang chỉ đường thì dừng thread cũ trước khi tạo thread mới
            if navigation_thread[0] is not None and navigation_thread[0].is_alive():
                navigation_stop_event.set()
                # Không join, chỉ set event để thread tự dừng!
            destination = ask_destination_loop()
            navigation_stop_event.clear()
            navigation_thread[0] = threading.Thread(target=navigation_worker, args=(destination,), daemon=True)
            navigation_thread[0].start()
            speaker_service.speak("Đang tìm đường đến điểm đến của bạn.")
        elif command == "dừng lại":
            if navigation_thread[0] is not None and navigation_thread[0].is_alive():
                navigation_stop_event.set()
                speaker_service.speak("Đã dừng tìm đường.")
            else:
                speaker_service.speak("Không có lộ trình nào đang được chỉ đường.")
        else:
            speaker_service.speak("Yêu cầu không hợp lệ. Hãy nói 'mi' hoặc 'bi'.")

# --- Khởi động thread nhận frame từ camera server ---
camera_thread = threading.Thread(target=zmq_camera_client_thread, daemon=True)
camera_thread.start()

main_loop()