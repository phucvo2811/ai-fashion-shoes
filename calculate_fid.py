import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm
from app.utils import load_real_shoe_data, create_tf_dataset, TARGET_IMAGE_SIZE, IMAGE_CHANNELS
from app.train_cgan import build_generator # Import model generator của anh

# --- CẤU HÌNH ---
DATA_DIR = "dataset"
MODEL_PATH = "models/generator_weights.h5" # Đường dẫn file trọng số tốt nhất
NUM_SAMPLES = 1000 # Số lượng ảnh để test (càng nhiều càng chuẩn nhưng lâu)
BATCH_SIZE = 32

print("🔄 Đang chuẩn bị tính FID...")

# 1. Load InceptionV3 (Mạng giám khảo chuyên nghiệp)
# Mạng này dùng để trích xuất đặc điểm của ảnh
inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

# 2. Hàm thay đổi kích thước ảnh về 299x299 (Chuẩn của Inception)
def scale_images(images, new_shape):
    images_list = []
    for image in images:
        new_image = tf.image.resize(image, new_shape)
        images_list.append(new_image)
    return tf.convert_to_tensor(images_list)

# 3. Hàm tính FID
def calculate_fid(model, images1, images2):
    # Tính toán đặc điểm (activations)
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    
    # Tính trung bình và hiệp phương sai
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    # Tính tổng bình phương sai số
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # Tính căn bậc hai của tích hiệp phương sai
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    # Công thức FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# --- THỰC THI ---
# A. Lấy ảnh thật
paths, labels = load_real_shoe_data(DATA_DIR, 1, TARGET_IMAGE_SIZE, IMAGE_CHANNELS)
ds = create_tf_dataset(paths, labels, BATCH_SIZE)
real_images = []
for img, _ in ds.take(NUM_SAMPLES // BATCH_SIZE):
    # Resize về 299x299 cho Inception
    img_resized = tf.image.resize(img, (299, 299))
    real_images.append(img_resized)
real_images = tf.concat(real_images, axis=0)
print(f"✅ Đã tải {real_images.shape[0]} ảnh thật.")

# B. Sinh ảnh giả
generator = build_generator()
generator.load_weights(MODEL_PATH)
noise = tf.random.normal([real_images.shape[0], 128])
fake_images = generator([noise, np.zeros((real_images.shape[0], 1))], training=False)
# Resize về 299x299 và chuyển về range [0, 255] rồi preprocess
fake_images = tf.image.resize(fake_images, (299, 299))
print(f"✅ Đã sinh {fake_images.shape[0]} ảnh giả.")

# C. Preprocess chuẩn cho Inception (-1 đến 1)
# Ảnh của mình đang là -1 đến 1 sẵn rồi, nhưng resize có thể làm lệch
# Đảm bảo đúng chuẩn preprocess_input của Keras
print("🧮 Đang tính toán điểm số (Có thể mất vài phút trên RTX 3050)...")
fid_score = calculate_fid(inception, real_images, fake_images)

print("-" * 30)
print(f"🏆 KẾT QUẢ FID: {fid_score:.4f}")
print("-" * 30)