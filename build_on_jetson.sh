#!/bin/bash

# Thông tin Docker Hub
DOCKER_USERNAME="dinhduc2004"
IMAGE_NAME="jetson-nav"
TAG="latest"

# Hiển thị thông báo
echo "Bắt đầu build Docker image cho dự án Jetson Navigation..."

# Build Docker image sử dụng Dockerfile gốc
docker build -t $DOCKER_USERNAME/$IMAGE_NAME:$TAG .

if [ $? -eq 0 ]; then
    echo "Build thành công! Đang đẩy lên Docker Hub..."
    
    # Đăng nhập vào Docker Hub
    docker login -u $DOCKER_USERNAME
    
    # Push image lên Docker Hub
    docker push $DOCKER_USERNAME/$IMAGE_NAME:$TAG
    
    if [ $? -eq 0 ]; then
        echo "Đã push image lên Docker Hub thành công!"
    else
        echo "Lỗi khi push image lên Docker Hub."
    fi
else
    echo "Lỗi khi build Docker image."
fi 