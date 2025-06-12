"""
GPS service module for handling GPS communication
"""
import time
import threading
import pynmea2
import serial
from navigation.config.settings import GPS_PORT, BAUD_RATE

class GPSService:
    """Service for handling GPS location data"""
    
    def __init__(self):
        """
        Initialize the GPS service
        """
        self.serial_port = None
        self.current_lat = None
        self.current_lng = None
        self.update_thread = None
        self.running = False
        
        # Khởi động luồng cập nhật GPS
        self._start_gps_thread()
        
    def _start_gps_thread(self):
        """Start GPS update thread"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
    def _update_loop(self):
        """GPS update loop running in background thread"""
        try:
            self.serial_port = serial.Serial(GPS_PORT, BAUD_RATE, timeout=1)
            print(f"Đã kết nối với GPS trên cổng {GPS_PORT}")
        except Exception as e:
            print(f"Không thể kết nối với GPS: {str(e)}")
            return
            
        # Main GPS reading loop
        while self.running:
            try:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                
                if line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                    try:
                        msg = pynmea2.parse(line)
                        if msg.status == 'A':  # A = Active (valid position)
                            self.current_lat = msg.latitude
                            self.current_lng = msg.longitude
                            # print(f"📍 Vị trí GPS cập nhật: {self.current_lat}, {self.current_lng}")
                    except Exception as e:
                        print(f"Lỗi xử lý dữ liệu GPS: {str(e)}")
                        
            except Exception as e:
                print(f"Lỗi đọc dữ liệu GPS: {str(e)}")
                time.sleep(1)
    
    def get_location(self):
        """
        Get current GPS location
        
        Returns:
            tuple: (latitude, longitude) or (None, None) if no valid data
        """
        return self.current_lat, self.current_lng
    
    def wait_for_valid_location(self, timeout=300, navigation_stop_event=None):
        """
        Wait for valid GPS location data
        
        Args:
            timeout: Maximum time to wait in seconds (default: 5 minutes)
            navigation_stop_event: Event to check for stop signal
            
        Returns:
            tuple: (latitude, longitude) or (None, None) if timed out
        """
        print("Đang đợi dữ liệu GPS hợp lệ...")
        start_time = time.time()
        
        while True:
            # Kiểm tra nếu có yêu cầu dừng navigation
            if navigation_stop_event and navigation_stop_event.is_set():
                print("Đã nhận lệnh dừng, ngưng đợi GPS.")
                return None, None
                
            lat, lng = self.get_location()
            
            if lat is not None and lng is not None:
                print(f"Đã nhận được vị trí: {lat}, {lng}")
                return lat, lng
                
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print("Hết thời gian đợi GPS. Yêu cầu thử lại.")
                return None, None
                
            time.sleep(1)
            
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
            
        if self.serial_port:
            try:
                self.serial_port.close()
            except:
                pass