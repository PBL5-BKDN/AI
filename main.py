import threading
from navigation.speech.voice import VoiceService
from navigation.navigation.navigator import Navigator
from navigation.services.gps import GPSService  
from navigation.services.api import APIService
from navigation.config.settings import MAX_REROUTE_ATTEMPTS
import time
import signal
import sys
# import VL53L1X

# Biến toàn cục để kiểm soát trạng thái chạy của các luồng
running = True

# Xử lý tín hiệu kết thúc (Ctrl+C)
def signal_handler(sig, frame):
    global running
    print("\nĐã nhận tín hiệu kết thúc. Đang dừng các luồng...")
    running = False
    sys.exit(0)

# Đăng ký handler cho tín hiệu SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def init_cam_bien_layser(voice_service):
    # Khởi tạo cảm biến
    # tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
    # tof.open()

    # # Thiết lập chế độ đo xa (Long Range)
    # tof.start_ranging(3)  # 1 = Short, 2 = Medium, 3 = Long
    print("Khởi tạo cảm biến")
    
    try:
        while running:
            print("Đang đo khoảng cách...")
            # distance = tof.get_distance()
            # print(f"Khoảng cách: {distance} mm")
            # if distance <= 2000:
            #     print("⚠️ Cảnh báo: Vật cản trong phạm vi 2 mét!")
                # Phát âm thanh cảnh báo
                # voice_service.speak("Cảnh báo: Vật cản trong phạm vi 2 mét!")
            time.sleep(5)
    except Exception as e:
        print(f"Lỗi trong luồng cảm biến: {e}")
    finally:
        # tof.stop_ranging()
        print("Đã dừng luồng cảm biến")

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

def xu_ly_yeu_cau(voice_service, gps_service, api_service): 
    print("Khởi tạo hệ thống dẫn đường")
    
    # Initialize navigator
    navigator = Navigator(gps_service, voice_service, api_service)
    
    try:
        while running:
            # Get destination from user
            destination = navigator.get_destination_from_user()
            
            # Get initial GPS position
            initial_lat, initial_lng = gps_service.wait_for_valid_location()
            
            # Check if GPS data is available
            if initial_lat is None or initial_lng is None:
                print("Không thể lấy vị trí GPS sau thời gian chờ")
                voice_service.speak("Không thể xác định vị trí của bạn. Vui lòng nói lại điểm đến.")
                continue
            
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
        # Khởi tạo các dịch vụ dùng chung
        voice_service = VoiceService()
        gps_service = GPSService()
        api_service = APIService()
        
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
        print(f"Lỗi chương trình chính: {e}")
    finally:
        # Đánh dấu dừng các luồng
        running = False
        
        # Dọn dẹp tài nguyên
        cleanup_resources(gps_service)
        
        print("Chương trình đã kết thúc.")
