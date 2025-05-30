# import cv2
# import onnxruntime as ort
# import numpy as np
# import time
# import csv

# # Load ENet ONNX model
# session = ort.InferenceSession('v2/model/enet.onnx')
# input_name = session.get_inputs()[0].name
# input_shape = session.get_inputs()[0].shape  # (1, 3, H, W)

# # Resize ảnh đầu vào đúng với input shape của model
# input_size = (input_shape[3], input_shape[2])  # (W, H)

# # Mở video (có thể thay bằng webcam)
# cap = cv2.VideoCapture('v2/demo/video1.mp4')  # hoặc cv2.VideoCapture(0) nếu dùng webcam

# latencies = []
# frame_count = 0

# # Tạo file CSV để lưu kết quả
# output_file = 'inference_latency.csv'
# with open(output_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Frame', 'Latency (ms)'])  # Tiêu đề cột

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_resized = cv2.resize(frame, input_size)
#         input_tensor = frame_resized.astype(np.float32).transpose(2, 0, 1) / 255.0  # HWC → CHW
#         input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 3, H, W)

#         # Đo thời gian bắt đầu
#         start_time = time.time()

#         # Inference
#         outputs = session.run(None, {input_name: input_tensor})

#         # Đo thời gian kết thúc
#         end_time = time.time()
#         latency = (end_time - start_time) * 1000  # ms
#         latencies.append(latency)

#         # Ghi thông tin latency của frame vào file CSV
#         writer.writerow([frame_count + 1, f"{latency:.2f}"])

#         # Hiển thị thông tin
#         frame_count += 1
#         if frame_count % 10 == 0:
#             print(f"Frame {frame_count} - Latency: {latency:.2f} ms")

#         # Nhấn 'q' để thoát sớm
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()

# # Tính toán độ trễ trung bình và FPS
# avg_latency = sum(latencies) / len(latencies)
# fps = 1000.0 / avg_latency

# # Ghi thông tin tổng kết vào file CSV
# with open(output_file, mode='a', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow([])  # Dòng trống
#     writer.writerow(['Average Latency (ms)', f"{avg_latency:.2f}"])
#     writer.writerow(['Estimated FPS', f"{fps:.2f}"])

# print(f"\n>>> Average Inference Latency: {avg_latency:.2f} ms")
# print(f">>> Estimated FPS: {fps:.2f} frames/sec")
# print(f">>> Results saved to {output_file}")



#=========== Đánh giá cho model PyTorch ============#

import cv2
import torch
import time
import numpy as np
import csv
from v2.model.enet import ENet

# Set number of CPU threads for PyTorch
torch.set_num_threads(4) 

# Load model ENet
try:
    # Instantiate the ENet model
    num_classes = 10  
    model = ENet(num_classes=num_classes)
    
    # Load the state dictionary
    state_dict = torch.load('v2/model/enet_best.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
except FileNotFoundError:
    print("Error: Model file 'v2/model/enet_best.pth' not found.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Input size for the model
input_size = (512, 256)  # (W, H)

# Open video
cap = cv2.VideoCapture('v2/demo/video1.mp4')
if not cap.isOpened():
    print("Error: Could not open video file 'v2/demo/video1.mp4'.")
    exit(1)

latencies = []
frame_count = 0

# Create CSV file for results
output_file = 'inference_latency_pytorch.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'Latency (ms)'])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame_resized = cv2.resize(frame, input_size, interpolation=cv2.INTER_LINEAR)
        input_array = frame_resized.astype(np.float32).transpose(2, 0, 1) / 255.0  # HWC → CHW
        input_tensor = torch.from_numpy(input_array).unsqueeze(0)  # (1, 3, H, W)

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        end_time = time.time()

        latency = (end_time - start_time) * 1000  # ms
        latencies.append(latency)

        # Write latency to CSV
        writer.writerow([frame_count + 1, f"{latency:.2f}"])

        # Display progress
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Frame {frame_count} - Latency: {latency:.2f} ms")

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Calculate average latency and FPS
if latencies:
    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency
else:
    avg_latency = 0
    fps = 0

# Append summary to CSV
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([])
    writer.writerow(['Average Latency (ms)', f"{avg_latency:.2f}"])
    writer.writerow(['Estimated FPS', f"{fps:.2f}"])

print(f"\n>>> Average Inference Latency: {avg_latency:.2f} ms")
print(f">>> Estimated FPS: {fps:.2f} frames/sec")
print(f">>> Results saved to {output_file}")