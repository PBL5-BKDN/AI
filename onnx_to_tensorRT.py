"""
sudo apt update
sudo apt install python3-pip libopencv-dev python3-opencv
pip3 install torch torchvision tensorrt pycuda
"""

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import os

def build_engine(onnx_file, output_file):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            print(f"ERROR: Failed to parse {onnx_file}")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
    
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 30
    builder.fp16_mode = True
    engine = builder.build_cuda_engine(network)
    
    with open(output_file, "wb") as f:
        f.write(engine.serialize())
    print(f"TensorRT engine saved to {output_file}")

if __name__ == "__main__":
    # Đường dẫn đến các file ONNX
    onnx_dir = "/app/onnx_models"
    
    # Đường dẫn đến file enet_simplified.onnx
    enet_onnx = os.path.join(onnx_dir, "enet_simplified.onnx")
    enet_trt = "/app/enet.trt"
    
    # Đường dẫn đến file my_yolov5m_simplified.onnx
    yolo_onnx = os.path.join(onnx_dir, "my_yolov5m_simplified.onnx")
    yolo_trt = "/app/yolov5m.trt"
    
    # Chuyển đổi ENet nếu file tồn tại
    if os.path.exists(enet_onnx):
        print(f"Converting {enet_onnx} to TensorRT...")
        build_engine(enet_onnx, enet_trt)
    else:
        print(f"File {enet_onnx} not found, skipping conversion")
    
    # Chuyển đổi YOLOv5m nếu file tồn tại
    if os.path.exists(yolo_onnx):
        print(f"Converting {yolo_onnx} to TensorRT...")
        build_engine(yolo_onnx, yolo_trt)
    else:
        print(f"File {yolo_onnx} not found, skipping conversion")
