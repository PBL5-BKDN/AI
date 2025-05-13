# Hướng dẫn sử dụng Docker cho dự án Jetson Navigation

## Cấu trúc dự án

Dự án này được đóng gói trong Docker để dễ dàng triển khai trên thiết bị Jetson Nano. Cấu trúc bao gồm:

-   `Dockerfile`: Định nghĩa môi trường và cách build image
-   `docker-compose.yml`: Cấu hình để chạy container
-   `build_and_push.bat`: Script để build và đẩy image lên Docker Hub (Windows)
-   `build_and_push.sh`: Script để build và đẩy image lên Docker Hub (Linux)
-   `run_on_jetson.sh`: Script để pull và chạy container trên Jetson Nano

## Các bước thực hiện

### 1. Build và đẩy image lên Docker Hub

#### Trên Windows:

1. Chỉnh sửa file `build_and_push.bat` để thay `your_username` bằng tên người dùng Docker Hub của bạn
2. Chạy script:
    ```
    build_and_push.bat
    ```

#### Trên Linux:

1. Chỉnh sửa file `build_and_push.sh` để thay `your_username` bằng tên người dùng Docker Hub của bạn
2. Thêm quyền thực thi và chạy script:
    ```bash
    chmod +x build_and_push.sh
    ./build_and_push.sh
    ```

### 2. Chạy container trên Jetson Nano

#### Sử dụng script:

1. Sao chép file `run_on_jetson.sh` sang Jetson Nano
2. Chỉnh sửa file để thay `your_username` bằng tên người dùng Docker Hub của bạn
3. Thêm quyền thực thi và chạy script:
    ```bash
    chmod +x run_on_jetson.sh
    ./run_on_jetson.sh
    ```

#### Sử dụng Docker Compose:

1. Sao chép file `docker-compose.yml` sang Jetson Nano
2. Chỉnh sửa biến môi trường `DOCKER_USERNAME` hoặc trực tiếp thay đổi trong file
3. Chạy lệnh:
    ```bash
    export DOCKER_USERNAME=your_username
    docker-compose up -d
    ```

## Cấu hình nâng cao

### Tùy chỉnh volume

Container sử dụng volume để lưu trữ dữ liệu. Mặc định, thư mục `./data` trên máy chủ được ánh xạ vào `/app/data` trong container. Bạn có thể thay đổi cấu hình này trong `docker-compose.yml`.

### Truy cập camera

Container được cấu hình để truy cập camera trên thiết bị Jetson Nano thông qua `/dev/video0`. Nếu camera của bạn sử dụng thiết bị khác, hãy thay đổi cấu hình trong `docker-compose.yml` hoặc `run_on_jetson.sh`.

### Xử lý lỗi

Nếu gặp lỗi khi chạy container, bạn có thể kiểm tra logs:

```bash
docker logs jetson-smart-nav
```

## Yêu cầu hệ thống

-   Docker Engine
-   NVIDIA Container Runtime (cho Jetson Nano)
-   Docker Compose (tùy chọn)

## Lưu ý

-   Image Docker sử dụng base image `nvcr.io/nvidia/l4t-tensorrt:r8.2.1-cuda10.2-trt8.2.1.8`, phù hợp với Jetson Nano.
-   Đảm bảo rằng thiết bị Jetson Nano có đủ không gian lưu trữ cho Docker image (khoảng 2-3GB).
-   Mô hình TensorRT cần được tối ưu hóa cho kiến trúc ARM của Jetson Nano.
