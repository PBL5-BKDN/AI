#pip install -U albumentations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from v1.model.enet import ENet
import csv
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import albumentations as A
import cv2

# Preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path)  
    return img

def preprocess_label(label_path):
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  
    return label
    
def normalize_image(image):
    return image.astype(np.float32) / 255.0

# Data augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=10, p=0.3),
    A.RandomCrop(height=256, width=512, p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
])

# Dataset
class MapillaryDataset(Dataset):
    def __init__(self, image_dir, label_dir, size=(512, 256), transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted(os.listdir(image_dir))
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.png'))
        image = preprocess_image(img_path)
        label = preprocess_label(label_path)
        if self.transform:
            augmented = transform(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']
        image = normalize_image(image.transpose(2, 0, 1))
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        return image, label

# DataLoader
train_dataset = MapillaryDataset(
    image_dir='data/images/train',
    label_dir='data/labels/train',
    transform=transform
)
val_dataset = MapillaryDataset(
    image_dir='data/images/val',
    label_dir='data/labels/val'
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Loss function
pixel_counts = [93662105, 3332873, 2979127, 19611636, 10701583, 5259916, 6064853, 4788531, 6659806, 375724157]
total_pixels = sum(pixel_counts)
weights = [1.0 / (count / total_pixels) if count > 0 else 0 for count in pixel_counts]
weights = [w / max(weights) for w in weights]
class_weights = torch.tensor(weights).cuda()

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.softmax(inputs, dim=1)
        targets = torch.nn.functional.one_hot(targets, num_classes=10).permute(0, 3, 1, 2).float()
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

criterion = lambda x, y: 0.5 * nn.CrossEntropyLoss(weight=class_weights)(x, y) + 0.5 * DiceLoss()(x, y)

# Định nghĩa device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Khởi tạo mô hình ENet
model = ENet(num_classes=10).to(device)
# Optimizer và scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Hàm tính Pixel Accuracy và IoU
def compute_metrics(outputs, labels, num_classes=10):
    preds = torch.argmax(outputs, dim=1)  # Dự đoán lớp cho mỗi pixel
    labels = labels.long()

    # Pixel Accuracy
    correct_pixels = (preds == labels).sum().item()
    total_pixels = labels.numel()
    pixel_acc = correct_pixels / total_pixels

    # IoU cho từng lớp
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()
        if union == 0:
            iou_per_class.append(0.0)  # Nếu không có pixel nào thuộc lớp này
        else:
            iou_per_class.append(intersection / union)
    mean_iou = np.mean(iou_per_class)

    return pixel_acc, mean_iou, iou_per_class

# Validation function (bao gồm thêm metrics)
def validate(model, loader, criterion, num_classes=10):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    pixel_acc_total = 0.0
    mean_iou_total = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Tính metrics
            pixel_acc, mean_iou, _ = compute_metrics(outputs, labels, num_classes)
            pixel_acc_total += pixel_acc
            mean_iou_total += mean_iou

            # Lưu dự đoán và nhãn để tính mAP
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            num_batches += 1

    # Tính trung bình
    val_loss = running_loss / num_batches
    pixel_acc = pixel_acc_total / num_batches
    mean_iou = mean_iou_total / num_batches

    # Tính mAP
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    precision, _, _, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=list(range(num_classes)))
    mean_ap = np.mean(precision)

    return val_loss, pixel_acc, mean_iou, mean_ap

# Training loop với lưu metrics
num_epochs = 50
best_val_loss = float('inf')

# metrics 
metrics = []
csv_file = 'v1/training_metrics.csv'

# save training metrics to CSV
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Pixel Accuracy', 'Mean IoU', 'mAP'])

for epoch in range(num_epochs):
    # Train
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    
    # Validate
    val_loss, pixel_acc, mean_iou, mean_ap = validate(model, val_loader, criterion, num_classes=10)
    
    # In kết quả
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
          f'Pixel Accuracy: {pixel_acc:.4f}, Mean IoU: {mean_iou:.4f}, mAP: {mean_ap:.4f}')
    
    # Lưu metrics vào danh sách
    metrics.append({
        'epoch': epoch+1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'pixel_acc': pixel_acc,
        'mean_iou': mean_iou,
        'mean_ap': mean_ap
    })
    
    # Ghi metrics vào file CSV
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, val_loss, pixel_acc, mean_iou, mean_ap])
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'enet_best.pth')
    
    scheduler.step(val_loss)

# Save final model
torch.save(model.state_dict(), 'enet_final.pth')

print("\nTraining Metrics Summary:")
for m in metrics:
    print(f"Epoch {m['epoch']}, Train Loss: {m['train_loss']:.4f}, Val Loss: {m['val_loss']:.4f}, "
          f"Pixel Accuracy: {m['pixel_acc']:.4f}, Mean IoU: {m['mean_iou']:.4f}, mAP: {m['mean_ap']:.4f}")
