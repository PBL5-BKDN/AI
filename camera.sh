#!/bin/bash
sudo systemctl restart nvargus-daemon
# Kill tiến trình camera_server.py cũ nếu còn chạy
pkill -f camera_server.py
sleep 1
python3 camera_server.py &
SERVER_PID=$!
sleep 2  # Đợi server khởi động (có thể tăng nếu cần)
python3 camera_client.py
kill $SERVER_PID