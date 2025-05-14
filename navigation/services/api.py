"""
API service module for communication with navigation backend
"""
import requests
from navigation.config.settings import API_URL

class APIService:
    def __init__(self, api_url=API_URL):
        self.api_url = api_url
        
    def get_navigation_route(self, latitude, longitude, destination):
        """
        Request navigation route from API
        
        Args:
            latitude (float): Current latitude
            longitude (float): Current longitude
            destination (str): Destination text
            
        Returns:
            dict: Response data or None if request failed
        """
        data = {
            "current_location": {"latitude": latitude, "longitude": longitude},
            "destination_text": destination
        }
        print(f"Yêu cầu API: {data}")
        
        try:
            response = requests.post(self.api_url, json=data)
            response.raise_for_status()
            print("📡 Phản hồi từ API:", response.json())
            return response.json()
        except Exception as e:
            print(f"Lỗi gửi yêu cầu đến API: {e}")
            return None