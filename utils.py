import cv2
import io
import numpy as np

def handle_take_photo(camera_lock, video_capture):
    # Ghi hình ảnh từ camera
    with camera_lock:
        if video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture frame — có thể do buffer đầy hoặc xung đột lock")
                return None
            # Lưu ảnh tạm thời vào bộ nhớ
            _, buffer = cv2.imencode('.jpg', frame)
            image_file = io.BytesIO(buffer)
            image_file.name = 'obstacle.jpg'
            print("Đã chụp ảnh từ camera CSI")

            # Hiển thị ảnh vừa chụp lên màn hình
            image_file.seek(0)
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imshow("Obstacle Detected", img)
                cv2.waitKey(15000)  # Hiển thị 15 giây (15000 ms)
                cv2.destroyWindow("Obstacle Detected")
            else:
                print("Không thể giải mã ảnh để hiển thị.")
            image_file.seek(0)  # Đặt lại con trỏ để dùng tiếp
            return image_file
        else:
            print("Camera chưa sẵn sàng.")
            return None