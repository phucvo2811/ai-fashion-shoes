import os
import sys
# Thêm đường dẫn
sys.path.append(os.getcwd())

print("⏳ Đang khởi động...")
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, LeakyReLU, BatchNormalization, UpSampling2D # Bỏ Concatenate
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Import
from utils import load_real_shoe_data, create_tf_dataset, TARGET_IMAGE_SIZE, IMAGE_CHANNELS

# --- CẤU HÌNH ---
DATA_DIR = "dataset" 
MODEL_PATH = "models/generator_weights.h5" 
NUM_SAMPLES = 100 
NOISE_DIM = 128
INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

#MODEL

def build_generator():
    input_noise = Input(shape=(NOISE_DIM,), name='noise')
    
    # Block 1: Project & Reshape (256 Filters - Khớp code train)
    x = Dense(8 * 8 * 256, use_bias=False, kernel_initializer=INIT)(input_noise) 
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x); x = Reshape((8, 8, 256))(x)
    # Bỏ Concatenate

    # 8 -> 16 (KERNEL 5X5)
    x = UpSampling2D()(x); x = Conv2D(256, 5, padding='same', use_bias=False, kernel_initializer=INIT)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    # 16 -> 32 (KERNEL 5X5)
    x = UpSampling2D()(x); x = Conv2D(128, 5, padding='same', use_bias=False, kernel_initializer=INIT)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    # 32 -> 64 (KERNEL 5X5)
    x = UpSampling2D()(x); x = Conv2D(64, 5, padding='same', use_bias=False, kernel_initializer=INIT)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    # 64 -> 128 (KERNEL 5X5)
    x = UpSampling2D()(x); x = Conv2D(32, 5, padding='same', use_bias=False, kernel_initializer=INIT)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    
    # Output (KERNEL 5X5)
    output = Conv2D(3, 5, padding='same', activation='tanh', dtype='float32', kernel_initializer=INIT)(x)
    return Model(inputs=input_noise, outputs=output) # CHỈ CÓ 1 INPUT


#CHẠY ĐÁNH GIÁ
def main():
    print("🚀 BẮT ĐẦU TÍNH ĐIỂM (SSIM & PSNR)...")
    
    # 1. Load Data Thật
    paths, labels = load_real_shoe_data(DATA_DIR, 1, TARGET_IMAGE_SIZE, IMAGE_CHANNELS)
    ds = create_tf_dataset(paths, labels, batch_size=NUM_SAMPLES)
    
    real_imgs, _ = next(iter(ds)) # Bỏ qua label
    real_imgs = (real_imgs + 1) / 2.0
    print(f"✅ Đã lấy {len(real_imgs)} ảnh thật.")

    # 2. Load Model & Sinh Ảnh Giả
    gen = build_generator()
    try:
        gen.load_weights(MODEL_PATH)
    except Exception as e:
        print(f"❌ LỖI TẢI MODEL. VUI LÒNG KIỂM TRA LẠI: {e}"); return

    noise = tf.random.normal([NUM_SAMPLES, NOISE_DIM])
    
    fake_imgs = gen.predict(noise, verbose=0) 
    fake_imgs = (fake_imgs + 1) / 2.0
    
    print(f"✅ Đã sinh {len(fake_imgs)} ảnh giả.")

    # 3. TÍNH TOÁN CHỈ SỐ (Giữ nguyên logic SSIM/PSNR cho báo cáo)
    print("🧮 Đang tính toán...")
    
    # SSIM 
    ssim_scores = tf.image.ssim(real_imgs, fake_imgs, max_val=1.0)
    avg_ssim = tf.reduce_mean(ssim_scores).numpy()
    
    # MSE
    mse_score = tf.reduce_mean(tf.square(real_imgs - fake_imgs)).numpy()
    
    # PSNR
    psnr_score = tf.image.psnr(real_imgs, fake_imgs, max_val=1.0)
    avg_psnr = tf.reduce_mean(psnr_score).numpy()

    print("\n" + "="*40)
    print("📊 BẢNG KẾT QUẢ (DÙNG CHO BÁO CÁO)")
    print("="*40)
    print(f"1. SSIM (Cấu trúc): {avg_ssim:.4f}")
    print("   (Thang điểm 0-1. Càng gần 1 càng tốt. >0.3 là ổn với GAN)")
    print("-" * 40)
    print(f"2. PSNR (Chất lượng): {avg_psnr:.2f} dB")
    print("   (Càng cao càng tốt. >10dB là hình rõ nét)")
    print("-" * 40)
    print(f"3. MSE (Sai số): {mse_score:.4f}")
    print("   (Càng thấp càng tốt)")
    print("="*40)

if __name__ == "__main__":
    main()