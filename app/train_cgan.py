import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, UpSampling2D, GaussianNoise
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_real_shoe_data, create_tf_dataset, TARGET_IMAGE_SIZE, IMAGE_CHANNELS

# --- GPU ---
CUDA_HOME = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
if os.path.exists(os.path.join(CUDA_HOME, "bin")):
    os.environ["PATH"] += os.pathsep + os.path.join(CUDA_HOME, "bin")

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)
except: pass

# --- CẤU HÌNH ---
BATCH_SIZE = 16
NOISE_DIM = 128
NUM_CLASSES = 1
IMG_SHAPE = (*TARGET_IMAGE_SIZE, IMAGE_CHANNELS)
DATA_ROOT_DIR = "dataset" 
INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) # Chống đen hình

# --- MODEL (STANDARD GAN - NO LABELS) ---

def build_generator():
    # CHỈ CÓ INPUT NOISE, KHÔNG CÓ LABEL
    input_noise = Input(shape=(NOISE_DIM,), name='noise')

    # Dùng Kernel 5x5 để bắt dáng tốt hơn
    x = Dense(8 * 8 * 256, use_bias=False, kernel_initializer=INIT)(input_noise)
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x); x = Reshape((8, 8, 256))(x)

    # 8 -> 16
    x = UpSampling2D()(x)
    x = Conv2D(256, 5, padding='same', use_bias=False, kernel_initializer=INIT)(x)
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)

    # 16 -> 32
    x = UpSampling2D()(x)
    x = Conv2D(128, 5, padding='same', use_bias=False, kernel_initializer=INIT)(x)
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)

    # 32 -> 64
    x = UpSampling2D()(x)
    x = Conv2D(64, 5, padding='same', use_bias=False, kernel_initializer=INIT)(x)
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)

    # 64 -> 128
    x = UpSampling2D()(x)
    x = Conv2D(32, 5, padding='same', use_bias=False, kernel_initializer=INIT)(x)
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)

    output = Conv2D(3, 5, padding='same', activation='tanh', dtype='float32', kernel_initializer=INIT)(x)
    return Model(inputs=input_noise, outputs=output, name="Generator")

def build_discriminator():
    # CHỈ CÓ INPUT ẢNH, KHÔNG CÓ LABEL
    input_image = Input(shape=IMG_SHAPE, name='img')
    img = GaussianNoise(0.1)(input_image) # Nhiễu nhẹ giúp D không học vẹt
    
    # 128 -> 64 (Kernel 5x5)
    x = Conv2D(32, 5, strides=2, padding='same', kernel_initializer=INIT)(img)
    x = LeakyReLU(0.2)(x); x = Dropout(0.3)(x)

    # 64 -> 32 (KHÔNG DÙNG BATCH NORM Ở D ĐỂ ỔN ĐỊNH VỚI BATCH 16)
    x = Conv2D(64, 5, strides=2, padding='same', kernel_initializer=INIT)(x)
    x = LeakyReLU(0.2)(x); x = Dropout(0.3)(x)

    # 32 -> 16
    x = Conv2D(128, 5, strides=2, padding='same', kernel_initializer=INIT)(x)
    x = LeakyReLU(0.2)(x); x = Dropout(0.3)(x)

    # 16 -> 8
    x = Conv2D(256, 5, strides=2, padding='same', kernel_initializer=INIT)(x)
    x = LeakyReLU(0.2)(x); x = Dropout(0.3)(x)

    x = Flatten()(x)
    output = Dense(1, dtype='float32', kernel_initializer=INIT)(x)
    return Model(inputs=input_image, outputs=output, name="Discriminator")

# --- HUẤN LUYỆN ---
loss_obj = tf.keras.losses.MeanSquaredError()
g_opt = Adam(0.0002, 0.5); d_opt = Adam(0.0002, 0.5)

generator_optimizer = mixed_precision.LossScaleOptimizer(g_opt)
discriminator_optimizer = mixed_precision.LossScaleOptimizer(d_opt)

@tf.function
def train_step(images): # KHÔNG CẦN LABELS
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = 0.5 * loss_obj(tf.ones_like(fake_output), fake_output)
        disc_loss = 0.5 * (loss_obj(tf.ones_like(real_output), real_output) + loss_obj(tf.zeros_like(fake_output), fake_output))
        
        scale_g = generator_optimizer.get_scaled_loss(gen_loss)
        scale_d = discriminator_optimizer.get_scaled_loss(disc_loss)

    grad_g = gen_tape.gradient(scale_g, generator.trainable_variables)
    grad_d = disc_tape.gradient(scale_d, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_optimizer.get_unscaled_gradients(grad_g), generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_optimizer.get_unscaled_gradients(grad_d), discriminator.trainable_variables))
    return gen_loss, disc_loss

def generate_and_save_images(model, epoch):
    test_input = tf.random.normal([10, NOISE_DIM])
    # Không cần truyền labels nữa
    predictions = model(test_input, training=False)
    
    fig = plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow((predictions[i]*0.5)+0.5)
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.close(fig)

if __name__ == '__main__':
    paths, labels = load_real_shoe_data(DATA_ROOT_DIR, NUM_CLASSES, TARGET_IMAGE_SIZE, IMAGE_CHANNELS)
    dataset = create_tf_dataset(paths, labels, BATCH_SIZE)
    
    generator = build_generator(); discriminator = build_discriminator()
    
    print("🚀 Bắt đầu huấn luyện STANDARD LSGAN (SNEAKER 128x128)...")
    print("-> Đã bỏ Label Embedding (Giảm nhiễu)")
    print("-> Đã dùng Kernel 5x5 (Tăng độ nét)")
    
    for epoch in range(200):
        for image_batch, _ in dataset: # Bỏ qua label batch
            g_loss, d_loss = train_step(image_batch)
            
        print(f"Epoch {epoch+1} | G: {g_loss:.4f} D: {d_loss:.4f}")
        
        if (epoch+1) % 5 == 0: generate_and_save_images(generator, epoch+1)
        if (epoch+1) % 10 == 0:
            if not os.path.exists("models"): os.makedirs("models")
            generator.save_weights(f"models/generator_epoch_{epoch+1:04d}.h5")
            generator.save_weights("models/generator_weights.h5")