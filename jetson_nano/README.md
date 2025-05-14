# Hệ thống Hỗ trợ Thị giác cho Jetson Nano

Ứng dụng phân đoạn ngữ nghĩa hình ảnh và tạo hướng dẫn âm thanh bằng tiếng Việt, chạy trên Jetson Nano.

## Yêu cầu hệ thống

-   Jetson Nano với JetPack 5.1.1 hoặc cao hơn
-   Docker và Docker Compose đã được cài đặt
-   Nvidia Container Runtime đã được cài đặt
-   Camera USB hoặc camera CSI được kết nối

## Cài đặt Docker và Docker Compose trên Jetson Nano

```bash
# Cài đặt Docker
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=arm64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Thêm người dùng hiện tại vào nhóm docker
sudo usermod -aG docker $USER

# Cài đặt Docker Compose
sudo apt install -y python3-pip
sudo pip3 install docker-compose

# Cài đặt Nvidia Container Runtime
sudo apt-get install -y nvidia-container-runtime

# Khởi động lại Docker để áp dụng thay đổi
sudo systemctl restart docker
```

## Chạy ứng dụng với Docker

1. Clone repository này:

```bash
git clone <repository_url>
cd jetson_nano
```

2. Xây dựng và chạy container:

```bash
docker-compose up -d --build
```

3. Xem logs:

```bash
docker-compose logs -f
```

4. Dừng container:

```bash
docker-compose down
```

## Cấu hình

-   Để thay đổi nguồn camera, chỉnh sửa biến `camera_source` trong file `code/main.py`
-   Để sử dụng camera CSI, thay đổi `device_id` thành chuỗi GStreamer phù hợp
-   Các tệp mô hình được lưu trong thư mục `model/`

## Cấu trúc dự án

```
jetson_nano/
├── code/                  # Mã nguồn chính
│   └── main.py            # Chương trình chính
├── model/                 # Chứa mô hình ONNX
│   └── enet.onnx          # Mô hình phân đoạn ENet
├── demo/                  # Tập lệnh và dữ liệu demo
├── test/                  # Tập lệnh và dữ liệu kiểm thử
├── Dockerfile             # Cấu hình Docker
├── docker-compose.yml     # Cấu hình Docker Compose
└── requirements.txt       # Các thư viện Python cần thiết
```

## Giải quyết sự cố

1. Nếu gặp lỗi khi truy cập camera:

    - Đảm bảo camera được kết nối đúng
    - Kiểm tra quyền truy cập thiết bị: `ls -l /dev/video*`
    - Thêm thiết bị camera vào docker-compose.yml: `- /dev/video1:/dev/video0`

2. Nếu gặp lỗi GPU:

    - Đảm bảo NVIDIA Container Runtime được cài đặt
    - Kiểm tra cài đặt JetPack: `sudo apt-cache show nvidia-jetpack`

3. Lỗi với ONNX Runtime:
    - Đảm bảo CUDA và cuDNN được cài đặt đúng
    - Kiểm tra xem ONNX Runtime có nhận diện được GPU không bằng cách kiểm tra logs

## Tối ưu hiệu suất

-   Điều chỉnh tham số `frame_skip` trong `code/main.py` để cân bằng giữa hiệu suất và tài nguyên
-   Thử nghiệm với các kích thước đầu vào khác nhau để tìm cấu hình tối ưu
-   Sử dụng ONNX Runtime với CUDAExecutionProvider để tăng tốc suy luận

## Sử dụng với camera CSI

Để sử dụng camera CSI trên Jetson Nano, thay đổi biến `camera_source` trong `code/main.py` thành chuỗi GStreamer:

```python
camera_source = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
```

Sau đó, cập nhật `docker-compose.yml` để thêm quyền truy cập vào camera CSI:

```yaml
devices:
    - /dev/video0:/dev/video0 # Camera USB
    - /dev/nvhost-ctrl
    - /dev/nvhost-ctrl-gpu
    - /dev/nvhost-prof-gpu
    - /dev/nvmap
    - /dev/nvhost-gpu
    - /dev/nvhost-as-gpu
```

## Lưu ý về ONNX Runtime

Dự án này sử dụng ONNX Runtime với GPU để tăng tốc suy luận. Dockerfile đã được cấu hình để cài đặt CUDA và cuDNN cần thiết cho ONNX Runtime GPU. Mô hình ONNX được sử dụng trực tiếp mà không cần chuyển đổi sang định dạng TensorRT.
