import cv2
import io
def handle_take_photo(camera_lock, video_capture):
    # Ghi hình ảnh từ camera
    with camera_lock:
        if video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                # Lưu ảnh tạm thời vào bộ nhớ
                _, buffer = cv2.imencode('.jpg', frame)
                image_file = io.BytesIO(buffer)
                image_file.name = 'obstacle.jpg'
                print("Đã chụp ảnh từ camera CSI")
                return image_file