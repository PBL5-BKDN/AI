@echo off
REM Thông tin Docker Hub
set DOCKER_USERNAME=dinhduc2004
set IMAGE_NAME=jetson-nav
set TAG=latest

echo Bat dau build Docker image cho du an Jetson Navigation...

REM Build Docker image sử dụng Dockerfile.build
docker build -t %DOCKER_USERNAME%/%IMAGE_NAME%:%TAG% -f Dockerfile.build .

if %ERRORLEVEL% EQU 0 (
    echo Build thanh cong! Dang day len Docker Hub...
    
    REM Đăng nhập vào Docker Hub (bạn sẽ được yêu cầu nhập mật khẩu)
    docker login -u %DOCKER_USERNAME%
    
    REM Push image lên Docker Hub
    docker push %DOCKER_USERNAME%/%IMAGE_NAME%:%TAG%
    
    if %ERRORLEVEL% EQU 0 (
        echo Da push image len Docker Hub thanh cong!
        echo De chay tren Jetson Nano, hay su dung lenh:
        echo docker pull %DOCKER_USERNAME%/%IMAGE_NAME%:%TAG%
        echo docker run --runtime nvidia --network host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --device /dev/video0:/dev/video0 %DOCKER_USERNAME%/%IMAGE_NAME%:%TAG%
    ) else (
        echo Loi khi push image len Docker Hub.
    )
) else (
    echo Loi khi build Docker image.
)

pause 