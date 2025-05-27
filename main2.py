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
from jetson_nano.code.main import predict_camera
import adafruit_vl53l1x
import requests
from utils import handle_take_photo
# Biến toàn cục để kiểm soát trạng thái chạy của các luồng
running = True
camera_lock = threading.Lock()
import os
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
        "video/x-raw, format=(string)BGR ! "
        "appsink max-buffers=1 drop=true sync=false emit-signals=true"
        % (sensor_id, capture_width, capture_height,
           framerate, flip_method,
           display_width, display_height)
    )
# Xử lý tín hiệu kết thúc (Ctrl+C)
def signal_handler(sig, frame):
    global running
    print("\nĐã nhận tín hiệu kết thúc. Đang dừng các luồng...")
    running = False
    sys.exit(0)

# Đăng ký handler cho tín hiệu SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)
CAMERA_INDEX = 0


cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
def capture_image(filename="capture.jpg"):
   
    if not cap.isOpened():
        print("❌ Không mở được camera!")
        return None

    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Không đọc được frame từ camera!")
        cap.release()
        return None

    success = cv2.imwrite(filename, frame)
    time.sleep(1)

    if not success:
        print("❌ Không lưu được ảnh!")
        return None

    # Kiểm tra file có thực sự tồn tại
    if not os.path.exists(filename):
        print("❌ File ảnh không tồn tại sau khi lưu!")
        return None

    print(f"✅ Đã lưu ảnh vào {filename}")
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    return filename

def init_cam_bien_layser(voice_service):
    tof = None
    
    while running:
        try:
            # Nếu cảm biến chưa được khởi tạo, khởi tạo nó
            if tof is None:
                # Khởi tạo I2C và cảm biến VL53L1X
                print("Đang khởi tạo cảm biến VL53L1X...")
                i2c = board.I2C()
                tof = adafruit_vl53l1x.VL53L1X(i2c)
                
                # Cấu hình cảm biến
                tof.distance_mode = 2  # 1 = Short, 2 = Long
                tof.timing_budget = 200  # Thời gian đo 100ms cho độ chính xác cao
                tof.start_ranging()
                print("Cảm biến VL53L1X khởi tạo thành công, bắt đầu đo khoảng cách")
            
            # Đọc dữ liệu từ cảm biến
            if tof.data_ready:
                try:
                    distance = tof.distance
                    print(f"Khoảng cách đo được: {distance} cm")
                    
                    if distance is not None and distance >= 0:
                        if 50 <= distance <= 100:
                            print("Vật cản trong phạm vi 0,5 - 1 mét!")
                            voice_service.speak("Cảnh báo: Vật cản trong phạm vi 0,5 - 1 mét!")
                            
                            try:
                                image_file = capture_image()
                                if image_file is None:
                                    print("Không chụp được ảnh từ camera.")
                                else:
                                    try:
                                        with open(image_file, 'rb') as f:
                                            files = {'image': ('obstacle.jpg', f, 'image/jpeg')}
                                            response = requests.post("http://14.233.84.201:3000/detect", files=files)
                                            data = response.json()
                                            print(f"Dữ liệu từ API: {data}")
                                            voice_service.speak(data.get("data", "Không phát hiện vật cản."))
                                    except requests.RequestException as e:
                                        print(f"Lỗi gửi yêu cầu HTTP: {e}")
                            except Exception as cam_err:
                                print(f"Lỗi khi chụp ảnh: {cam_err}")
                    else:
                        print("Khoảng cách không hợp lệ")
                    
                    try:
                        tof.clear_interrupt()
                    except Exception as clear_err:
                        print(f"Lỗi khi xóa ngắt: {clear_err}")
                        raise OSError("Không thể xóa ngắt, cần tái khởi tạo")
                        
                except OSError as io_err:
                    print(f"Lỗi I/O khi đọc cảm biến: {io_err}")
                    if tof is not None:
                        try:
                            tof.stop_ranging()
                        except:
                            pass
                        tof = None
                    time.sleep(2)  # Chờ một lúc trước khi tái khởi tạo
                    continue
            
            time.sleep(0.5)  # Đợi 0.5 giây giữa các lần đọc
            
        except Exception as e:
            print(f"Lỗi trong luồng cảm biến: {e}")
            # Dọn dẹp cảm biến hiện tại nếu có lỗi
            if tof is not None:
                try:
                    tof.stop_ranging()
                except Exception:
                    pass
                tof = None
            time.sleep(2)  # Đợi trước khi thử lại
    
    # Dừng cảm biến khi thoát khỏi vòng lặp
    if tof is not None:
        try:
            tof.stop_ranging()
            print("Đã dừng cảm biến VL53L1X")
        except Exception as e:
            print(f"Lỗi khi dừng cảm biến: {e}")


def init_phan_doan_lan_duong(cap, camera_lock):
    print("Khởi tạo phân đoán làng đường")
    try:
        predict_camera(cap, camera_lock, running=True, frame_skip=2, capture_image_fn=capture_image)
    except Exception as e:
        print(f"Lỗi trong luồng phân đoán làng đường: {e}")
    finally:
        print("Đã dừng luồng phân đoán làng đường")

def handle_ask_chatbot(voice_service, question):
    print("Xử lý yêu cầu chatbot")
    image_file = capture_image("answer.jpg")
    if image_file is None:
        print("Không chụp được ảnh từ camera.")
        return

    try:
        with open(image_file, 'rb') as f:
            response = requests.post(
                "http://14.233.84.201:3000/upload-image",
                data={"question": question},
                files={"file": ("answer.jpg", f, "image/jpeg")}
            )
            data = response.json()
            print(data)
            voice_service.speak(data["text"])
    except Exception as e:
        print(f"Lỗi khi gọi API chatbot: {e}")
        voice_service.speak("Lỗi khi kết nối đến hệ thống trả lời câu hỏi.")


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
    try:
        while running:
            text = voice_service.recognize_speech()
            print(f"Đã nhận diện: {text}")
            if text:
                words = text.strip().lower().split()
                if words:
                    first_word = words[0]
                    if first_word in ["2", "hai"]:
                        textvoice = ' '.join(words[1:])
                        handle_ask_chatbot(voice_service, textvoice)
                    elif first_word in ["3", "ba"]:
                        navigator = Navigator(gps_service, voice_service, api_service)
                        handle_navigtion(navigator, gps_service)
                    else:
                        voice_service.speak("Yêu cầu không hợp lệ. Vui lòng thử lại.")
            time.sleep(1)
    except Exception as e:
        print(f"Lỗi trong luồng xử lý yêu cầu: {e}")
    finally:
        print("Đã dừng luồng xử lý yêu cầu")
def cleanup_resources():
    cap.release()
    # Dọn dẹp tài nguyên
    # if gps_service:
    #     gps_service.cleanup()
    print("Đã dọn dẹp tài nguyên")

if __name__ == "__main__":    
    video_capture = None
    try:
        # Khởi tạo camera CSI với retry mechanism
        # max_camera_retries = 5
        # for attempt in range(max_camera_retries):
        #     try:
        #         gst = gstreamer_pipeline(flip_method=0)
        #         print(f"GStreamer pipeline (lần thử {attempt + 1}): {gst}")
        #         video_capture = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                
        #         if video_capture.isOpened():
        #             # Test đọc frame để đảm bảo camera hoạt động
        #             ret, test_frame = video_capture.read()
        #             if ret and test_frame is not None:
        #                 print("Camera CSI khởi tạo thành công")
        #                 print(f"Độ phân giải camera: {video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)}x{video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        #                 print(f"FPS: {video_capture.get(cv2.CAP_PROP_FPS)}")
        #                 break
        #             else:
        #                 print(f"Camera mở nhưng không đọc được frame (lần thử {attempt + 1})")
        #                 video_capture.release()
        #                 video_capture = None
        #         else:
        #             print(f"Không thể mở camera CSI (lần thử {attempt + 1})")
        #             if video_capture:
        #                 video_capture.release()
        #                 video_capture = None
                        
        #         if attempt < max_camera_retries - 1:
        #             time.sleep(2)  # Đợi trước khi thử lại
                    
        #     except Exception as camera_err:
        #         print(f"Lỗi khởi tạo camera (lần thử {attempt + 1}): {camera_err}")
        #         if video_capture:
        #             video_capture.release()
        #             video_capture = None
        #         if attempt < max_camera_retries - 1:
        #             time.sleep(2)
        
        # if not video_capture or not video_capture.isOpened():
        #     raise Exception("Không thể khởi tạo camera CSI sau nhiều lần thử")
            
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
        #t1 = threading.Thread(target=init_cam_bien_layser, args=(voice_service,), daemon=True)
        t2 = threading.Thread(target=init_phan_doan_lan_duong, args=(cap, camera_lock), daemon=True)
        # t3 = threading.Thread(target=xu_ly_yeu_cau, args=(voice_service, gps_service, api_service), daemon=True)

        # Khởi động các thread
        # t1.start()
        t2.start()
        # t3.start()

        # Chờ các thread hoàn thành (sẽ không bao giờ kết thúc trừ khi có Ctrl+C)
        while running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nĐã nhận tín hiệu kết thúc từ bàn phím")
    except Exception as e:
        print(f"Lỗi chương trình: {e}")
    finally:
        running = False
        if video_capture and video_capture.isOpened():
            video_capture.release()
            print("Đã giải phóng camera")
        cleanup_resources()   
        print("Chương trình đã kết thúc.")
