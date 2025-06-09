import onnxruntime as ort
import time
import numpy as np

def load_model_onnx(onnx_path):
    """
    Load ONNX model bằng onnxruntime, trả về session để infer
    """
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"ONNX model loaded with providers: {session.get_providers()}")
    return session

def infer_and_measure_onnx(session, input_tensor):
    """
    input_tensor: numpy array shape (1, 3, H, W), float32, normalized
    Trả về output logits numpy và thời gian inference (ms)
    """
    input_name = session.get_inputs()[0].name
    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    end = time.time()
    infer_time_ms = (end - start) * 1000
    return outputs[0], infer_time_ms
