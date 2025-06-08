import torch
import segmentation_models_pytorch as smp
import time

def load_model(weights_path, num_classes):
    model = smp.DeepLabV3Plus(
        encoder_name='mobilenet_v2',
        encoder_weights=None,
        in_channels=3,
    )
    model.load_state_dict(torch.load(weights_path, map_location='cuda'))
    model.eval().to('cuda')
    return model

@torch.no_grad()
def infer_and_measure(model, image_tensor):
    """
    image_tensor: Tensor of shape (1, 3, H, W), already normalized & to(device)
    Returns: output prediction, and inference time in milliseconds
    """
    start = time.time()
    output = model(image_tensor)
    end = time.time()

    infer_time_ms = (end - start) * 1000  # milliseconds
    return output, infer_time_ms
