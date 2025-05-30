import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv('v2/training_metrics.csv')

# Vẽ biểu đồ
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
plt.plot(df['Epoch'], df['Val Loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(df['Epoch'], df['Pixel Accuracy'], label='Pixel Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Pixel Accuracy')
plt.title('Pixel Accuracy')
plt.legend()
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(df['Epoch'], df['Mean IoU'], label='Mean IoU')
plt.xlabel('Epoch')
plt.ylabel('Mean IoU')
plt.title('Mean IoU')
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(df['Epoch'], df['mAP'], label='mAP')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('mAP')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('v2/metrics_plot.png')
plt.show()