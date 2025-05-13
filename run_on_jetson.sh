#!/bin/bash

# Thông tin Docker Hub
DOCKER_USERNAME="dinhduc2004"  # Thay thế bằng username Docker Hub của bạn
IMAGE_NAME="jetson-nav"
TAG="latest"
CONTAINER_NAME="jetson-smart-nav"

# Hiển thị thông báo
echo "Đang chuẩn bị chạy ứng dụng Jetson Navigation từ Docker Hub..."

# Kiểm tra và dừng container cũ nếu đang chạy
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container $CONTAINER_NAME đang chạy, đang dừng lại..."
    docker stop $CONTAINER_NAME
fi

# Xóa container cũ nếu tồn tại
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Đang xóa container cũ..."
    docker rm $CONTAINER_NAME
fi

# Pull image mới nhất từ Docker Hub
echo "Đang kéo image mới nhất từ Docker Hub..."
docker pull $DOCKER_USERNAME/$IMAGE_NAME:$TAG

# Chạy container với các tùy chọn phù hợp cho Jetson Nano
echo "Đang khởi động container..."
docker run -d \
  --name $CONTAINER_NAME \
  --runtime nvidia \
  --network host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --device /dev/video0:/dev/video0 \
  -v $PWD/data:/app/data \
  $DOCKER_USERNAME/$IMAGE_NAME:$TAG

# Kiểm tra kết quả
if [ $? -eq 0 ]; then
    echo "Container đã được khởi động thành công!"
    echo "Để xem logs, hãy sử dụng lệnh: docker logs $CONTAINER_NAME"
    echo "Để dừng container, hãy sử dụng lệnh: docker stop $CONTAINER_NAME"
else
    echo "Lỗi khi khởi động container."
fi 