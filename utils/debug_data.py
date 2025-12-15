import tensorflow as tf
import matplotlib.pyplot as plt
from app.utils import load_real_shoe_data, create_tf_dataset, TARGET_IMAGE_SIZE, IMAGE_CHANNELS

# Cấu hình
DATA_ROOT_DIR = "dataset" # Hoặc dataset_clean nếu anh đã lọc
BATCH_SIZE = 16
NUM_CLASSES = 1

print("🔍 Đang kiểm tra dữ liệu đầu vào...")

# 1. Load dữ liệu
paths, labels = load_real_shoe_data(DATA_ROOT_DIR, NUM_CLASSES, TARGET_IMAGE_SIZE, IMAGE_CHANNELS)
dataset = create_tf_dataset(paths, labels, BATCH_SIZE)

# 2. Lấy thử 1 lô (batch) ra xem
for images, labels in dataset.take(1):
    print(f"✅ Đã lấy được batch ảnh. Shape: {images.shape}")
    print(f"   Giá trị pixel min: {tf.reduce_min(images):.2f}, max: {tf.reduce_max(images):.2f}")
    
    # Vẽ ra màn hình
    fig = plt.figure(figsize=(10, 4))
    for i in range(min(8, BATCH_SIZE)): # Xem 8 tấm đầu
        ax = plt.subplot(2, 4, i + 1)
        # Chuyển từ [-1, 1] về [0, 1] để mắt người xem được
        img_show = (images[i] * 0.5) + 0.5
        plt.imshow(img_show)
        plt.title(f"Label: {int(labels[i])}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    break