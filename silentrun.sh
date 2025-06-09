#!/bin/bash
# Tạo thư mục logs nếu chưa tồn tại
mkdir -p logs

# Thiết lập quyền cho cổng GPS
sudo chmod 666 /dev/ttyTHS1

# Xóa các file log cũ để khởi tạo lại
rm -f logs/stderr.log logs/stdout.log logs/app.log
touch logs/stderr.log logs/stdout.log logs/app.log

# Chạy ứng dụng với chuyển hướng tất cả output
# Tất cả log được ghi vào file log, không có gì hiển thị trên console
echo "Ứng dụng đang chạy trong chế độ im lặng. Các log được ghi vào thư mục logs/"
python3 main.py > logs/stdout.log 2> logs/stderr.log
