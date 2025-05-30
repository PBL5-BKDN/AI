from ultralytics import YOLO

# Tải mô hình
model = YOLO("yolov8n.pt")

# Huấn luyện mô hình
model.train(
    data="/kaggle/working/data/data.yaml",  
    epochs=200,                                   
    patience=20,                                  # Patience để tránh early stopping sớm
    imgsz=640,                                    
    batch=16,                                     # Batch size
    device=0,                                     
    optimizer="AdamW",                            
    lr0=0.001,                                    # Learning rate ban đầu
    freeze = 0,
    augment=True,                                
    save_period=1,                                
    verbose=True,                                 
    project="/kaggle/working/runs/train",         
    name="exp"                                    
)

# Đánh giá mô hình
results = model.val(data="/kaggle/working/data/data.yaml")

# Kết quả đánh giá
print("Validation results:")
print(f"mAP@0.5: {results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {results.box.map:.4f}")
for i, name in enumerate(results.names.values()): 
    print(f"Class {i} ({name}): mAP@0.5 = {results.box.ap50[i]:.4f}")