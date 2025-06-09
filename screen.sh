#!/bin/bash

# === THÔNG SỐ CƠ BẢN ===
RESOLUTION=$(xdpyinfo | grep dimensions | awk '{print $2}')  # Tự lấy độ phân giải màn hình
FPS=30
OUTPUT="$HOME/Videos/record_$(date +%Y%m%d_%H%M%S).mp4"
DURATION=""  # Để trống thì quay đến khi Ctrl+C

# === TUỲ CHỌN DURATION (NẾU MUỐN GIỚI HẠN) ===
# DURATION="00:10:00"  # Format: HH:MM:SS (bỏ comment nếu muốn giới hạn 10 phút chẳng hạn)

# === BẮT ĐẦU QUAY ===
echo "🎥 Bắt đầu quay màn hình với độ phân giải $RESOLUTION, fps=$FPS"
echo "💾 Video sẽ lưu tại: $OUTPUT"
echo "⏹️ Nhấn Ctrl+C để dừng quay bất kỳ lúc nào"

ffmpeg \
  -video_size "$RESOLUTION" \
  -framerate "$FPS" \
  -f x11grab \
  -i :0.0 \
  -c:v libx264 \
  -preset ultrafast \
  ${DURATION:+-t $DURATION} \
  "$OUTPUT"
