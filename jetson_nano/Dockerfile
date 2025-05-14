FROM nvcr.io/nvidia/l4t-base:r36.2.0

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libopencv-dev \
    libgtk-3-dev \
    libpng-dev \
    libjpeg-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    python3-pygame \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt CUDA và cuDNN để hỗ trợ onnxruntime-gpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-11-4 \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt các thư viện Python từ requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn
COPY . .

# Tạo thư mục logs
RUN mkdir -p logs

# Thiết lập biến môi trường
ENV PYTHONPATH=/app
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Thiết lập quyền thực thi
RUN chmod +x code/main.py

# Lệnh chạy mặc định
CMD ["python3", "code/main.py"]
