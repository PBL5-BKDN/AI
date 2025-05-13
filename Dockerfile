FROM nvcr.io/nvidia/l4t-tensorrt:r8.2.1-cuda10.2-trt8.2.1.8

# Thiết lập biến môi trường
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# Cài đặt hệ thống cần thiết
RUN apt update && apt install -y \
    python3-pip \
    python3-dev \
    libsm6 libxext6 libxrender-dev \
    espeak ffmpeg libespeak1 \
    tesseract-ocr tesseract-ocr-vie \
    libglib2.0-0 \
    libgtk2.0-dev \
    libcanberra-gtk* \
    libnss3-dev \
    libasound2 \
    git \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt pip packages
COPY requirement.txt /tmp/
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirement.txt && \
    pip3 install --no-cache-dir flask

# Tạo thư mục ứng dụng
WORKDIR /app

# Tạo thư mục cho mô hình
RUN mkdir -p /app/models

# Copy toàn bộ mã nguồn
COPY . /tmp/src/
RUN cp /tmp/src/*.py /app/ && \
    if [ -f /tmp/src/yolov8n_api/model.pt ]; then cp /tmp/src/yolov8n_api/model.pt /app/models/; fi && \
    cp /tmp/src/yolov8n_api/app.py /app/ && \
    cp /tmp/src/yolov8n_api/client.py /app/

# Copy các mô hình ONNX nếu có
RUN mkdir -p /app/onnx_models
COPY *.onnx /app/onnx_models/ 2>/dev/null || echo "No ONNX models found"

# Chuyển đổi mô hình ONNX sang TensorRT
RUN if [ -f /app/onnx_models/enet_simplified.onnx ]; then \
    echo "Converting enet_simplified.onnx to TensorRT..." && \
    python3 /app/onnx_to_tensorRT.py && \
    echo "Conversion completed successfully"; \
    else \
    echo "enet_simplified.onnx not found, skipping conversion"; \
    fi

# Mở cổng cho API
EXPOSE 5000

# Thiết lập quyền thực thi
RUN chmod +x /app/main.py

# Tạo volume để lưu trữ dữ liệu
VOLUME ["/app/data"]

# Thiết lập entrypoint
ENTRYPOINT ["python3", "main.py"]

# Hoặc sử dụng CMD nếu muốn linh hoạt hơn
# CMD ["python3", "main.py"]
