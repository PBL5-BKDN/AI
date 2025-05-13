"""
sudo apt update
sudo apt install python3-pip libopencv-dev python3-opencv
pip3 install torch torchvision tensorrt pycuda
"""

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

def build_engine(onnx_file, output_file):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            print(f"ERROR: Failed to parse {onnx_file}")
            exit()
    
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 30
    builder.fp16_mode = True
    engine = builder.build_cuda_engine(network)
    
    with open(output_file, "wb") as f:
        f.write(engine.serialize())
    print(f"TensorRT engine saved to {output_file}")

build_engine("enet.onnx", "enet.trt")
