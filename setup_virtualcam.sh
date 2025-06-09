#!/bin/bash

# Tên thiết bị ảo và ID
VIDEO_DEV="/dev/video10"
LABEL="VirtualCam"

echo "[+] Kiểm tra module v4l2loopback..."
if ! lsmod | grep -q v4l2loopback; then
    echo "[+] Chưa có v4l2loopback, đang tải module..."
    sudo modprobe v4l2loopback video_nr=10 card_label="$LABEL" exclusive_caps=1
    sleep 1
else
    echo "[+] v4l2loopback đã được tải."
fi

# Kiểm tra thiết bị ảo có tồn tại chưa
if [ -e "$VIDEO_DEV" ]; then
    echo "[✅] Thiết bị ảo $VIDEO_DEV đã sẵn sàng."
else
    echo "[❌] Không tìm thấy $VIDEO_DEV. Có thể lỗi khi tải v4l2loopback."
    exit 1
fi

# Kiểm tra quyền truy cập
if [ ! -w "$VIDEO_DEV" ]; then
    echo "[⚠️] Không có quyền ghi vào $VIDEO_DEV. Đang cấp quyền..."
    sudo chmod 666 "$VIDEO_DEV"
fi

echo "[🚀] Virtual camera đã sẵn sàng để dùng!"
