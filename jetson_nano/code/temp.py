import torch
import segmentation_models_pytorch as smp

# Khởi tạo model giống lúc training
model = smp.DeepLabV3Plus(
    encoder_name='mobilenet_v2',
    encoder_weights=None,
    in_channels=3,
    classes=5
)

# Load trọng số đã huấn luyện
model.load_state_dict(torch.load("deeplabv3plus_best.pth", map_location="cpu"))
model.eval()
dummy_input = torch.randn(1, 3, 384, 512)  # Batch size = 1, đúng với input ảnh
torch.onnx.export(
    model,                       # model PyTorch
    dummy_input,                 # đầu vào mẫu
    "model.onnx",                # tên file ONNX đầu ra
    export_params=True,
    opset_version=11,            
    do_constant_folding=True,    # optimize các hằng số
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
