"""
Navigator module for handling navigation and route following
"""
from navigation.utils.distance import haversine
from navigation.config.settings import (
    COMPLETION_THRESHOLD_KM, 
    REROUTE_FACTOR, 
    SPEAK_THRESHOLD_FAR_KM, 
    SPEAK_THRESHOLD_NEAR_KM
)
import time

class Navigator:
    def __init__(self, gps_service, voice_service, api_service):
        self.gps_service = gps_service
        self.voice_service = voice_service
        self.api_service = api_service

    def get_destination_from_user(self):
        """Ask user for destination using voice"""
        destination = None
        while not destination:
            self.voice_service.speak("Mời bạn nói điểm đến.")
            destination = self.voice_service.recognize_speech()
            if not destination:
                self.voice_service.speak("Không nhận diện được giọng nói. Xin hãy thử lại.")
        
        print(f"📌 Điểm đến: {destination}")
        self.voice_service.speak(f"Đang tìm đường đến {destination}")
        return destination

    def request_route(self, lat, lng, destination):
        """Request route from API service"""
        print(f"🗺️ Đang yêu cầu lộ trình từ API...")
        return self.api_service.get_navigation_route(lat, lng, destination)

    def follow_route(self, steps, initial_lat, initial_lng):
        """Follow the navigation route step by step"""
        step_index = 0
        total_steps = len(steps)
        spoken_flags = [0] * total_steps
        
        print(f"✅ Nhận được {total_steps} bước hướng dẫn.")
        self.voice_service.speak(f"Đã tìm thấy lộ trình với {total_steps} bước.")
        
        step_start_lat, step_start_lng = initial_lat, initial_lng
        
        while step_index < total_steps:
            lat, lng = self.gps_service.get_location()
            
            if lat and lng:
                print(f"--- Bước {step_index + 1}/{total_steps} ---")
                print(f"📍 Vị trí hiện tại: Vĩ độ: {lat}, Kinh độ: {lng}")
                
                current_step = steps[step_index]
                instruction = current_step.get('instruction', 'Tiếp tục đi thẳng.')
                step_total_distance_km = current_step.get('distance', 0) / 1000.0
                print(f"📏 Tổng khoảng cách bước này: {step_total_distance_km * 1000:.1f} m")
                
                distance_traveled_in_step = haversine(step_start_lat, step_start_lng, lat, lng)
                print(f"🚶 Đã đi trong bước này: {distance_traveled_in_step * 1000:.1f} m")
                
                remaining_distance_km = max(0, step_total_distance_km - distance_traveled_in_step)
                print(f"🏁 Còn lại trong bước: {remaining_distance_km * 1000:.1f} m")
                
                # Check if off-route
                if (distance_traveled_in_step > step_total_distance_km * REROUTE_FACTOR 
                    and step_index > 0 and step_total_distance_km > 0.01):
                    print(f"⚠️ Phát hiện có thể chệch hướng!")
                    self.voice_service.speak("Bạn có thể đã đi chệch hướng. Đang tìm lại lộ trình.")
                    return False, lat, lng
                
                # Give voice instructions based on distance
                current_spoken_state = spoken_flags[step_index]
                if (current_spoken_state == 0 and remaining_distance_km < SPEAK_THRESHOLD_FAR_KM 
                    and step_total_distance_km > SPEAK_THRESHOLD_FAR_KM):
                    self.voice_service.speak(f"Sau khoảng {int(remaining_distance_km * 1000)} mét, {instruction}")
                    spoken_flags[step_index] = 1
                elif current_spoken_state <= 1 and remaining_distance_km < SPEAK_THRESHOLD_NEAR_KM:
                    self.voice_service.speak(f"{instruction} ngay.")
                    spoken_flags[step_index] = 2
                
                # Move to next step if completed current one
                if distance_traveled_in_step >= step_total_distance_km:
                    print(f"✅ Hoàn thành bước {step_index + 1}!")
                    step_index += 1
                    if step_index < total_steps:
                        step_start_lat, step_start_lng = lat, lng
                        print(f"--- Bắt đầu bước {step_index + 1} từ: {step_start_lat}, {step_start_lng} ---")
                    continue
            
            else:
                print("⚠️ Không thể lấy dữ liệu GPS, kiểm tra lại kết nối.")
                self.voice_service.speak("Mất tín hiệu GPS.")
            
            time.sleep(5)
        
        # Reached destination
        if step_index >= total_steps:
            print("🎉 Đã đến đích!")
            self.voice_service.speak("Bạn đã đến nơi.")
            return True, None, None
            
        return False, None, None
        
    def cleanup(self):
        pass