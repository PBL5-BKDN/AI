FROM nvcr.io/nvidia/l4t-tensorrt:r8.2.1-cuda10.2-trt8.2.1.8

# Cài đặt hệ thống cần thiết
RUN apt update && apt install -y \
    python3-pip \
    libsm6 libxext6 libxrender-dev \
    espeak ffmpeg libespeak1 \
    tesseract-ocr \
    libglib2.0-0 \
    libgtk2.0-dev \
    libcanberra-gtk* \
    libnss3-dev \
    libasound2 \
    && apt clean

# Cài pip packages
RUN pip3 install --upgrade pip
RUN pip3 install pyttsx3 pycuda opencv-python numpy pytesseract

# Copy code vào container
COPY . /app
WORKDIR /app

CMD ["python3", "main.py"]
