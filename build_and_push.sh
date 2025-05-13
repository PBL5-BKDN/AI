#!/bin/bash

# Thông tin Docker Hub
DOCKER_USERNAME="dinhduc2004"  # Thay thế bằng username Docker Hub của bạn
IMAGE_NAME="jetson-nav"
TAG="latest"

# Hiển thị thông báo
echo "Bắt đầu build Docker image cho dự án Jetson Navigation..."

# Build Docker image
docker build -t $DOCKER_USERNAME/$IMAGE_NAME:$TAG .

# Kiểm tra kết quả build
if [ $? -eq 0 ]; then
    echo "Build thành công! Đang đẩy lên Docker Hub..."
    
    # Đăng nhập vào Docker Hub (bạn sẽ được yêu cầu nhập mật khẩu)
    docker login -u $DOCKER_USERNAME
    
    # Push image lên Docker Hub
    docker push $DOCKER_USERNAME/$IMAGE_NAME:$TAG
    
    if [ $? -eq 0 ]; then
        echo "Đã push image lên Docker Hub thành công!"
        echo "Để chạy trên Jetson Nano, hãy sử dụng lệnh:"
        echo "docker pull $DOCKER_USERNAME/$IMAGE_NAME:$TAG"
        echo "docker run --runtime nvidia --network host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=\$DISPLAY --device /dev/video0:/dev/video0 $DOCKER_USERNAME/$IMAGE_NAME:$TAG"
    else
        echo "Lỗi khi push image lên Docker Hub."
    fi
else
    echo "Lỗi khi build Docker image."
fi 