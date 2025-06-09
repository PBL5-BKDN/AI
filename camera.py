import cv2
import time
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraStreamCapture:
    def __init__(self, udp_port=5000):
        self.pipeline = (
            f"udpsrc port={udp_port} ! application/x-rtp,encoding-name=H264,payload=96 ! "
            "rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=true max-buffers=1 sync=false"
        )
        print(f"Pipeline khởi tạo: {self.pipeline}")
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print(f"Không thể mở stream từ UDP! Pipeline: {self.pipeline}")
            raise RuntimeError("Không mở được stream từ UDP!")
        print("Đã kết nối tới stream UDP thành công.")

        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def _read_frames(self):
        try:
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.latest_frame = frame
                else:
                    # Đừng spam log, chỉ cảnh báo 1 lần mỗi 100 lần
                    if int(time.time() * 10) % 100 == 0:
                        print("Không nhận được frame từ stream!")
                time.sleep(0.01)
        except Exception as e:
            print(f"Lỗi trong thread đọc frame: {e}")

    def capture_image(self):
        with self.lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
            else:
                frame = None
        if frame is None:
            print("Không đọc được frame từ stream!")
            return None
        return frame

    def release(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join()
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        print("Đã giải phóng stream.")

    def __del__(self):
        self.release()

# Sử dụng:
if __name__ == "__main__":
    stream_service = CameraStreamCapture(udp_port=5000)
    try:
        while True:
            frame = stream_service.capture_image()
            if frame is not None:
                cv2.imshow("Test", frame)
                if cv2.waitKey(1) == 27:  # ESC
                    break
            else:
                print("Đang chờ frame...")
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stream_service.release()
        cv2.destroyAllWindows()