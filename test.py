import cv2
import torch

# Load YOLOv5m model từ .pt
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=False)
model.load_state_dict(torch.load('yolov5m.pt')['model'].float().state_dict())
model.eval()

# Mở video
cap = cv2.VideoCapture('video1.mp4')
if not cap.isOpened():
    print("Không thể mở video!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Lấy kết quả phát hiện
    detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, class]

    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Hiển thị
    cv2.imshow("YOLOv5m Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
