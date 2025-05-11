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
import pytesseract

# Định nghĩa tên lớp từ data.yaml (YOLOv5m)
yolo_classes = [
    "Ground Animal", "Person", "Bench", "Billboard", "Fire Hydrant",
    "Mailbox", "Manhole", "Pothole", "Traffic Sign (Front)", "Trash Can",
    "Bus", "Car", "Motorcycle", "Other Vehicle", "Truck"
]

# Định nghĩa lớp an toàn từ mapping.json (ENet)
safe_lane_ids = {7, 8, 10, 14}  # crosswalk - plain, curb cut, pedestrian-area, sidewalk

# Nhận input tên địa điểm từ người dùng
destination = input("Nhập tên địa điểm bạn muốn đến: ").strip().lower()
print(f"Địa điểm đích: {destination}")

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
current_priority = 0  # Ưu tiên của thông báo đang phát
last_notification_time = 0  # Thời gian phát thông báo cuối cùng
min_notification_interval = 1.0  # Khoảng thời gian tối thiểu giữa các thông báo (giây)

def speak_notification():
    global current_priority, last_notification_time
    while True:
        if not notification_queue.empty():
            with notification_lock:
                message, priority, repeat = notification_queue.get()
                # Ngắt thông báo hiện tại nếu có thông báo ưu tiên cao hơn
                if priority > current_priority:
                    engine.stop()
                current_priority = priority
                engine.say(message)
                engine.runAndWait()
                if repeat:
                    time.sleep(0.5)  # Đợi giữa hai lần lặp
                    engine.say(message)
                    engine.runAndWait()
                last_notification_time = time.time()
                current_priority = 0  # Đặt lại ưu tiên sau khi phát xong
        time.sleep(0.05)

# Khởi động luồng phát thông báo
threading.Thread(target=speak_notification, daemon=True).start()

# Thêm thông báo vào hàng đợi với logic thời gian thực
def add_notification(message, priority=1, repeat=False):
    global last_notification_time
    current_time = time.time()
    # Chỉ thêm thông báo nếu đã qua khoảng thời gian tối thiểu hoặc thông báo có ưu tiên cao hơn
    if current_time - last_notification_time >= min_notification_interval or priority > current_priority:
        with notification_lock:
            # Xóa hàng đợi cũ để giữ tính thời gian thực
            while not notification_queue.empty():
                notification_queue.get()
            notification_queue.put((message, priority, repeat))

## Logic xử lý và phát thông báo
last_notification_frame = 0
frame_count = 0
is_in_safe_lane = True  # Trạng thái làn an toàn mặc định
destination_reached = False  # Biến kiểm tra xem đã đến đích chưa

# Ước lượng khoảng cách từ chiều rộng bounding box (giả định đơn giản)
def estimate_distance(box_width):
    return "close" if box_width > 200 else "far"

# Xác định hướng vật cản (trái, phải, giữa)
def determine_direction(box_center_x, frame_width=640):
    if box_center_x < frame_width // 3:
        return "trái"
    elif box_center_x > 2 * frame_width // 3:
        return "phải"
    else:
        return "giữa"

# Đọc văn bản trên biển hiệu bằng OCR
def read_sign_text(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(thresh, config='--psm 6').strip().lower()
        return text
    except Exception as e:
        print(f"Error reading sign text: {e}")
        return ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Chuẩn bị input cho mô hình (kích thước 640x640)
    input_frame = cv2.resize(frame, (640, 640)).transpose(2, 0, 1)[None].astype(np.float32)

    # Chạy ENet và YOLOv5m thay phiên nhau
    if frame_count % 2 == 0:  # Khung chẵn: Chạy ENet
        enet_output = infer(enet_engine, input_frame)
        lane_mask = np.argmax(enet_output, axis=1)[0]
        center_pixel = lane_mask[320, 320]
        is_in_safe_lane = center_pixel in safe_lane_ids

        if not is_in_safe_lane and frame_count - last_notification_frame > 10:
            message = "Bạn đang ra khỏi làn an toàn, di chuyển sang trái để quay lại"
            add_notification(message, priority=2)
            last_notification_frame = frame_count

    else:  # Khung lẻ: Chạy YOLOv5m
        yolo_output = infer(yolo_engine, input_frame)
        num_boxes = yolo_output.shape[1]
        for i in range(num_boxes):
            conf = yolo_output[0, i, 4]
            if conf < 0.5:
                continue

            cls_probs = yolo_output[0, i, 5:20]
            class_id = np.argmax(cls_probs)
            class_name = yolo_classes[class_id]

            x, y, w, h = yolo_output[0, i, 0:4]
            box_center_x = x
            box_width = w
            box_center_y = y
            box_height = h

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(640, x2), min(640, y2)

            if class_id == 8 and not destination_reached:  # Traffic Sign (Front)
                sign_image = frame[y1:y2, x1:x2]
                if sign_image.size > 0:
                    sign_text = read_sign_text(sign_image)
                    print(f"Sign text detected: {sign_text}")
                    if sign_text and destination in sign_text:
                        message = f"Bạn đã đến {destination}!"
                        add_notification(message, priority=1, repeat=True)
                        destination_reached = True
                        last_notification_frame = frame_count
                        break

            distance = estimate_distance(box_width)
            direction = determine_direction(box_center_x)

            if distance == "close" and frame_count - last_notification_frame > 10:
                message = f"Vật cản {class_name} ở bên {direction}, di chuyển sang {'trái' if direction == 'phải' else 'phải'}"
                add_notification(message, priority=1, repeat=True)
                last_notification_frame = frame_count

    cuda.Context.synchronize()

# Giải phóng tài nguyên
cap.release()