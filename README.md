# Jetson Navigation - Hỗ trợ người khiếm thị

Dự án sử dụng AI để hỗ trợ người khiếm thị di chuyển an toàn, tạo mô tả tiếng Việt cho hình ảnh và cảnh báo chướng ngại vật.

## Các thành phần chính

- **Mô hình Image Captioning**: Tạo mô tả tiếng Việt cho hình ảnh (ResNet-50 + LSTM)
- **Phát hiện đối tượng**: YOLOv5m/YOLOv8n để nhận diện người, xe cộ, chướng ngại vật
- **Phân đoạn làn đường**: ENet để phân biệt khu vực an toàn cho người đi bộ
- **Cảnh báo bằng giọng nói**: Thông báo về chướng ngại vật và hướng dẫn di chuyển

## Triển khai với Docker

### 1. Build và triển khai trên Jetson Nano

Dự án được đóng gói trong Docker để dễ dàng triển khai. Do phụ thuộc vào CUDA và TensorRT, việc build phải thực hiện trên Jetson Nano:

```bash
# Clone repository
git clone https://github.com/yourusername/jetson-navigation.git
cd jetson-navigation

# Cấp quyền thực thi cho script build
chmod +x build_on_jetson.sh

# Build và push image lên Docker Hub
./build_on_jetson.sh
```

### 2. Chạy container trên Jetson Nano

#### Sử dụng Docker Compose:

```bash
export DOCKER_USERNAME=dinhduc2004  # Hoặc username của bạn
docker-compose up -d
```

#### Hoặc sử dụng Docker trực tiếp:

```bash
docker run --runtime nvidia --network host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --device /dev/video0:/dev/video0 \
  -v $PWD/data:/app/data \
  dinhduc2004/jetson-nav:latest
```

## Cấu trúc dự án

- `main.py`: Mã nguồn chính của ứng dụng
- `test.py`: Script kiểm thử YOLOv5m
- `onnx_to_tensorRT.py`: Chuyển đổi mô hình ONNX sang TensorRT
- `yolov8n_api/`: API phát hiện đối tượng với YOLOv8n
- `Dockerfile`: Cấu hình Docker cho Jetson Nano
- `docker-compose.yml`: Cấu hình Docker Compose

## Yêu cầu hệ thống

- Jetson Nano với JetPack 4.6+
- Docker Engine
- NVIDIA Container Runtime
- Camera USB hoặc CSI

## Cấu hình nâng cao

Xem thêm chi tiết trong file `DOCKER_README.md`.
