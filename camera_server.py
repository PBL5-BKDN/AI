import cv2
import zmq
import pickle

def main():
    # Đọc camera vật lý (có thể thay bằng pipeline GStreamer nếu dùng Jetson)
    pipeline = (
    "nvarguscamerasrc sensor-mode=4 ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")  # mở cổng cho các client kết nối vào

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Xoay frame 90 độ bằng OpenCV (chắc chắn xoay)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # Nén frame để gửi (bạn có thể nén JPEG để tiết kiệm băng thông)
        data = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
        socket.send(data)

    cap.release()

if __name__ == "__main__":
    main()