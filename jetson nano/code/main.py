"""
pip install pycuda opencv-python pyttsx3
sudo apt update
sudo apt install python3-pip libopencv-dev python3-opencv
pip3 install torch torchvision tensorrt pycuda pyttsx3
sudo apt install espeak
"""

## Load và chạy mô hình TensorRT
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
import queue
import threading
import pyttsx3
import time

# Định nghĩa tên lớp từ data.yaml (YOLOv5m)
yolo_classes = [
    "Ground Animal", "Person", "Bench", "Billboard", "Fire Hydrant",
    "Mailbox", "Manhole", "Pothole", "Traffic Sign (Front)", "Trash Can",
    "Bus", "Car", "Motorcycle", "Other Vehicle", "Truck"
]

# Định nghĩa lớp an toàn từ mapping.json (ENet)
safe_lane_ids = {7, 8, 10, 14}  # crosswalk - plain, curb cut, pedestrian-area, sidewalk

def load_trt_engine(engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    return runtime.deserialize_cuda_engine(engine_data)

# Load mô hình TensorRT
enet_engine = load_trt_engine("enet.trt")
yolo_engine = load_trt_engine("yolov5m.trt")

# Hàm suy luận TensorRT
def infer(engine, input_data):
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    np.copyto(inputs[0][0], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()
    return outputs[0][0].reshape(engine.get_binding_shape(engine[1]))

## Xử lý đầu vào từ camera
cap = cv2.VideoCapture(0)  # Camera mặc định
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

## Quản lý và phát thông báo qua loa
notification_queue = queue.Queue()
notification_lock = threading.Lock()
engine = pyttsx3.init()

def speak_notification():
    while True:
        if not notification_queue.empty() and notification_lock.acquire(blocking=False):
            message, repeat = notification_queue.get()
            engine.say(message)
            engine.runAndWait()
            if repeat:
                time.sleep(0.5)  # Đợi giữa hai lần lặp
                engine.say(message)
                engine.runAndWait()
            notification_lock.release()
        time.sleep(0.1)

# Khởi động luồng phát thông báo
threading.Thread(target=speak_notification, daemon=True).start()

# Thêm thông báo vào hàng đợi
def add_notification(message, priority=1, repeat=False):
    if priority == 1 and notification_lock.locked():
        with notification_queue.mutex:
            notification_queue.queue.clear()  # Xóa thông báo không ưu tiên
    notification_queue.put((message, repeat))

## Logic xử lý và phát thông báo
last_notification_frame = 0
frame_count = 0
is_in_safe_lane = True  # Trạng thái làn an toàn mặc định

# Ước lượng khoảng cách từ chiều rộng bounding box (giả định đơn giản)
def estimate_distance(box_width):
    # Giả định: chiều rộng bounding box lớn hơn 200 pixel là "gần" (<2m)
    return "close" if box_width > 200 else "far"

# Xác định hướng vật cản (trái, phải, giữa)
def determine_direction(box_center_x, frame_width=640):
    if box_center_x < frame_width // 3:
        return "trái"
    elif box_center_x > 2 * frame_width // 3:
        return "phải"
    else:
        return "giữa"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Chuẩn bị input cho mô hình (kích thước 640x640)
    input_frame = cv2.resize(frame, (640, 640)).transpose(2, 0, 1)[None].astype(np.float32)

    # Chạy ENet và YOLOv5m thay phiên nhau
    if frame_count % 2 == 0:  # Khung chẵn: Chạy ENet
        # Suy luận ENet
        enet_output = infer(enet_engine, input_frame)
        lane_mask = np.argmax(enet_output, axis=1)[0]  # Lấy lớp có xác suất cao nhất

        # Kiểm tra xem người dùng có trong làn an toàn không
        # Lấy vùng trung tâm của ảnh (giả định người dùng ở giữa)
        center_pixel = lane_mask[320, 320]  # Trung tâm ảnh 640x640
        is_in_safe_lane = center_pixel in safe_lane_ids

        # Nếu lệch làn an toàn, phát thông báo
        if not is_in_safe_lane and frame_count - last_notification_frame > 10:
            message = "Bạn đang ra khỏi làn an toàn, di chuyển sang trái để quay lại"
            add_notification(message, priority=2)
            last_notification_frame = frame_count

    else:  # Khung lẻ: Chạy YOLOv5m
        # Suy luận YOLOv5m
        yolo_output = infer(yolo_engine, input_frame)

        # Xử lý đầu ra YOLOv5m (giả định đầu ra có định dạng [batch, num_boxes, (x, y, w, h, conf, cls_probs)])
        # YOLOv5m thường trả về [1, num_boxes, 5 + num_classes], với num_classes = 15
        num_boxes = yolo_output.shape[1]
        for i in range(num_boxes):
            conf = yolo_output[0, i, 4]  # Confidence score
            if conf < 0.5:  # Ngưỡng confidence
                continue

            # Lấy lớp có xác suất cao nhất
            cls_probs = yolo_output[0, i, 5:20]  # 15 lớp (0-14)
            class_id = np.argmax(cls_probs)
            class_name = yolo_classes[class_id]

            # Lấy tọa độ bounding box
            x, y, w, h = yolo_output[0, i, 0:4]
            box_center_x = x
            box_width = w

            # Ước lượng khoảng cách và hướng
            distance = estimate_distance(box_width)
            direction = determine_direction(box_center_x)

            # Nếu vật cản gần (<2m), phát thông báo ưu tiên
            if distance == "close" and frame_count - last_notification_frame > 10:
                message = f"Vật cản {class_name} ở bên {direction}, di chuyển sang {'trái' if direction == 'phải' else 'phải'}"
                add_notification(message, priority=1, repeat=True)
                last_notification_frame = frame_count

    # Đảm bảo đồng bộ hóa
    cuda.Context.synchronize()

# Giải phóng tài nguyên
cap.release()