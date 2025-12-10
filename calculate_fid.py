import os
import sys

# Thêm đường dẫn
sys.path.append(os.getcwd())

print("⏳ Đang khởi động...")
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, LeakyReLU, BatchNormalization, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

# Import
from app.utils import load_real_shoe_data, create_tf_dataset, TARGET_IMAGE_SIZE, IMAGE_CHANNELS

# --- CẤU HÌNH ---
DATA_DIR = "dataset" 
MODEL_PATH = "models/generator_weights.h5" 
NUM_SAMPLES = 100 # Lấy 100 mẫu để so sánh là đủ cho báo cáo
NOISE_DIM = 128

# --- MODEL (512 FILTERS) ---
def get_label_embedding():
    return tf.keras.Sequential([Input(shape=(1,)), Dense(8*8), Reshape((8, 8, 1))])

def build_generator():
    input_noise = Input(shape=(NOISE_DIM,), name='noise')
    input_label = Input(shape=(1,), name='label')
    label = get_label_embedding()(input_label)
    
    x = Dense(8 * 8 * 512, use_bias=False)(input_noise)
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x); x = Reshape((8, 8, 512))(x)
    x = Concatenate()([x, label])

    x = UpSampling2D()(x); x = Conv2D(256, 3, padding='same', use_bias=False)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x); x = Conv2D(128, 3, padding='same', use_bias=False)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x); x = Conv2D(64, 3, padding='same', use_bias=False)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x); x = Conv2D(32, 3, padding='same', use_bias=False)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    
    output = Conv2D(3, 3, padding='same', activation='tanh', dtype='float32')(x)
    return Model(inputs=[input_noise, input_label], outputs=output)

# --- CHẠY ĐÁNH GIÁ ---
def main():
    print("🚀 BẮT ĐẦU TÍNH ĐIỂM (SSIM & MSE)...")
    
    # 1. Load Data Thật
    paths, labels = load_real_shoe_data(DATA_DIR, 1, TARGET_IMAGE_SIZE, IMAGE_CHANNELS)
    ds = create_tf_dataset(paths, labels, batch_size=NUM_SAMPLES)
    
    # Lấy 1 batch ảnh thật
    real_imgs, _ = next(iter(ds))
    # Chuyển về [0, 1] để tính toán
    real_imgs = (real_imgs + 1) / 2.0
    
    print(f"✅ Đã lấy {len(real_imgs)} ảnh thật.")

    # 2. Load Model & Sinh Ảnh Giả
    gen = build_generator()
    try:
        gen.load_weights(MODEL_PATH)
    except Exception as e:
        print(f"❌ Lỗi model: {e}"); return

    noise = tf.random.normal([NUM_SAMPLES, NOISE_DIM])
    fake_labels = np.zeros((NUM_SAMPLES, 1))
    
    fake_imgs = gen.predict([noise, fake_labels], verbose=0)
    # Chuyển về [0, 1]
    fake_imgs = (fake_imgs + 1) / 2.0
    
    print(f"✅ Đã sinh {len(fake_imgs)} ảnh giả.")

    # 3. TÍNH TOÁN CHỈ SỐ
    print("🧮 Đang tính toán...")
    
    # SSIM (Độ tương đồng cấu trúc)
    # Vì ảnh giả và ảnh thật không song song (unpaired), ta tính SSIM trung bình
    # So sánh ngẫu nhiên để xem cấu trúc chung có giống "giày" không
    ssim_scores = tf.image.ssim(real_imgs, fake_imgs, max_val=1.0)
    avg_ssim = tf.reduce_mean(ssim_scores).numpy()
    
    # MSE (Sai số trung bình)
    mse_score = tf.reduce_mean(tf.square(real_imgs - fake_imgs)).numpy()
    
    # PSNR (Tỷ lệ tín hiệu trên nhiễu)
    psnr_score = tf.image.psnr(real_imgs, fake_imgs, max_val=1.0)
    avg_psnr = tf.reduce_mean(psnr_score).numpy()

    print("\n" + "="*40)
    print("📊 BẢNG KẾT QUẢ (DÙNG CHO BÁO CÁO)")
    print("="*40)
    print(f"1. SSIM (Cấu trúc): {avg_ssim:.4f}")
    print("   (Thang điểm 0-1. Càng gần 1 càng tốt. >0.3 là ổn với GAN)")
    print("-" * 40)
    print(f"2. PSNR (Chất lượng): {avg_psnr:.2f} dB")
    print("   (Càng cao càng tốt. >10dB là hình rõ nét)")
    print("-" * 40)
    print(f"3. MSE (Sai số): {mse_score:.4f}")
    print("   (Càng thấp càng tốt)")
    print("="*40)

if __name__ == "__main__":
    main()