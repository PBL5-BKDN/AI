import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as transforms

# Load ONNX model
session = ort.InferenceSession("model/enet.onnx")
input_name = session.get_inputs()[0].name

# Define preprocessing
input_size = (512, 256)  # (height, width) as expected by the model
transform = transforms.Compose([
    transforms.Resize(input_size),  # Resize to (height, width) = (512, 256)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a color palette for 10 classes (RGB colors)
color_palette = [
    [0, 0, 0],      # Class 0: Black
    [255, 0, 0],    # Class 1: Red
    [0, 255, 0],    # Class 2: Green
    [0, 0, 255],    # Class 3: Blue
    [255, 255, 0],  # Class 4: Yellow
    [255, 0, 255],  # Class 5: Magenta
    [0, 255, 255],  # Class 6: Cyan
    [128, 128, 128],# Class 7: Gray
    [255, 128, 0],  # Class 8: Orange
    [128, 0, 128]   # Class 9: Purple
]

def preprocess(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).numpy()  # Shape: (3, 512, 256)
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # Shape: (1, 3, 512, 256)
    return input_tensor, image

def predict_image(image_path, output_path=None):
    # Preprocess
    input_tensor, orig_image = preprocess(image_path)

    # Run inference with ONNX Runtime
    outputs = session.run(None, {input_name: input_tensor})[0]  # Shape: (1, num_classes, 512, 256)
    pred = np.argmax(outputs, axis=1).squeeze(0)  # Shape: (512, 256)

    # Convert prediction to color map (RGB)
    pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id in range(10):
        pred_colored[pred == class_id] = color_palette[class_id]

    # Convert pred_colored to BGR for OpenCV
    pred_colored_bgr = cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR)

    # Resize original image to match output size and convert to NumPy array
    orig_image = orig_image.resize((input_size[1], input_size[0]), Image.BILINEAR)  # Resize to (width, height)
    orig_image = np.array(orig_image)  # Shape: (512, 256, 3), RGB

    # Convert orig_image to BGR for OpenCV
    orig_image_bgr = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

    # Verify shapes
    print(f"orig_image_bgr shape: {orig_image_bgr.shape}")
    print(f"pred_colored_bgr shape: {pred_colored_bgr.shape}")

    # Create overlay
    overlay = cv2.addWeighted(orig_image_bgr, 0.5, pred_colored_bgr, 0.5, 0)

    # Convert overlay back to RGB for saving as PNG
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Save results
    if output_path:
        # Save segmentation map (RGB)
        pred_image = Image.fromarray(pred_colored)
        pred_image.save(output_path + '_seg.png')
        # Save overlay (RGB)
        Image.fromarray(overlay_rgb).save(output_path + '_overlay.png')

    # Display results
    cv2.imshow('Segmentation', pred_colored_bgr)
    cv2.imshow('Overlay', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with an image
image_path = 'test/test3.jpg'
output_path = 'test/test3'
predict_image(image_path, output_path)