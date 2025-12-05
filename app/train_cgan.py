import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os

# Import các hàm đã sửa đổi từ utils.py
from utils import load_real_shoe_data, create_tf_dataset, TARGET_IMAGE_SIZE, IMAGE_CHANNELS, map_path_to_image

# --- BẮT ĐẦU CẤU HÌNH TƯƠNG THÍCH VÀ KHẮC PHỤC LỖI GPU/OOM ---
CUDA_HOME = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2" # Dùng v11.2 cho TF 2.10.0
CUBLAS_BIN = os.path.join(CUDA_HOME, "bin")

if os.path.exists(CUBLAS_BIN):
    os.environ["PATH"] += os.pathsep + CUBLAS_BIN
    os.environ["PATH"] += os.pathsep + os.path.join(CUDA_HOME, "lib", "x64")

tf.config.experimental.enable_optimize_for_graph = False
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False}) 

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Sử dụng GPU: {gpus[0].name}")
    else:
        print("Không tìm thấy GPU vật lý, đang chạy trên CPU.")
except RuntimeError as e:
    print(e)
# --- KẾT THÚC CẤU HÌNH GPU/OOM ---

BUFFER_SIZE = 10000 
BATCH_SIZE = 32      # Giảm Batch Size để ổn định VRAM 4GB
NOISE_DIM = 100
NUM_CLASSES = 10     
IMG_SHAPE = (*TARGET_IMAGE_SIZE, IMAGE_CHANNELS) 
DATA_ROOT_DIR = "dataset" 

print(f"Bắt đầu tải dữ liệu với kích thước ảnh: {IMG_SHAPE}")

train_image_paths, train_labels_array = load_real_shoe_data(
    data_dir=DATA_ROOT_DIR, 
    num_classes=NUM_CLASSES,
    target_size=TARGET_IMAGE_SIZE,
    channels=IMAGE_CHANNELS
)

train_dataset = create_tf_dataset(train_image_paths, train_labels_array, BATCH_SIZE)

if train_dataset is None:
    print("Dữ liệu không hợp lệ. Huấn luyện bị dừng.")
    exit()

# --- 2. Định nghĩa Mô hình (64x64) ---
# ... (Các hàm build_generator, build_discriminator giữ nguyên) ...
def get_label_embedding():
    model = tf.keras.Sequential([Input(shape=(1,)),Dense(8 * 8 * 1, use_bias=False),Reshape((8, 8, 1))], name='label_embedding')
    return model
def build_generator():
    input_noise = Input(shape=(NOISE_DIM,), name='noise_input'); input_label = Input(shape=(1,), name='label_input')
    label_embedding = get_label_embedding()(input_label)
    x = Dense(8 * 8 * 256, use_bias=False)(input_noise); x = BatchNormalization()(x); x = LeakyReLU()(x); x = Reshape((8, 8, 256))(x)
    merged_input = Concatenate()([x, label_embedding])
    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(merged_input); x = BatchNormalization()(x); x = LeakyReLU()(x)
    x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x); x = BatchNormalization()(x); x = LeakyReLU()(x)
    output_image = Conv2DTranspose(IMAGE_CHANNELS, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
    return Model(inputs=[input_noise, input_label], outputs=output_image, name='Generator')

def build_discriminator():
    input_image = Input(shape=IMG_SHAPE, name='image_input'); input_label = Input(shape=(1,), name='label_input')
    label_expanded = Dense(IMG_SHAPE[0] * IMG_SHAPE[1] * 1, use_bias=False)(input_label); label_embedding = Reshape(IMG_SHAPE[:-1] + (1,))(label_expanded) 
    merged_input = Concatenate()([input_image, label_embedding])
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(merged_input); x = LeakyReLU()(x); x = Dropout(0.3)(x)
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x); x = LeakyReLU()(x); x = Dropout(0.3)(x)
    x = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x); x = LeakyReLU()(x); x = Dropout(0.3)(x)
    x = Flatten()(x); output_prediction = Dense(1, activation='sigmoid')(x)
    return Model(inputs=[input_image, input_label], outputs=output_prediction, name='Discriminator')


# --- 3. Định nghĩa Hàm Loss và Optimizer (ĐÃ SỬA LEARNING RATE) ---
cross_entropy = tf.keras.losses.BinaryCrossentropy(); 
generator_optimizer = Adam(1e-4); # Giảm Learning Rate từ 2e-4 xuống 1e-4 để tăng ổn định
discriminator_optimizer = Adam(1e-4)

# Hàm Loss: THÊM LABEL SMOOTHING để tăng chất lượng ảnh
def discriminator_loss(real_output, fake_output):
    # Real Loss: Đánh giá ảnh thật là 0.9 (thay vì 1.0)
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output) 
    # Fake Loss: Đánh giá ảnh giả là 0.0 (Giữ nguyên)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# --- 4. Training Step (Giữ nguyên) ---
@tf.function
def train_step(images, labels):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM]); labels = tf.reshape(labels, [BATCH_SIZE, 1]) 
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True); real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True); gen_loss = generator_loss(fake_output); disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables); gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables)); discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

# --- 5. Hàm Huấn luyện và Trực quan hóa (Giữ nguyên) ---
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch, label_batch in dataset:
            g_loss, d_loss = train_step(image_batch, label_batch)
        print(f'Epoch {epoch + 1}/{epochs}, G_Loss: {g_loss.numpy():.4f}, D_Loss: {d_loss.numpy():.4f}')
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            generate_and_save_images(generator, epoch + 1)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                if not os.path.exists("models"): os.makedirs("models")
                generator.save_weights(os.path.join("models", "generator_weights.h5")); print("-> Đã lưu trọng số Generator.")

def generate_and_save_images(model, epoch):
    noise = tf.random.normal([NUM_CLASSES, NOISE_DIM]); labels = np.arange(0, NUM_CLASSES); predictions = model([noise, labels], training=False)
    is_color = predictions.shape[-1] == 3; fig = plt.figure(figsize=(10, 1.5))
    for i in range(predictions.shape[0]):
        ax = plt.subplot(1, NUM_CLASSES, i+1); img = (predictions[i] + 1) * 127.5; img = img.numpy().astype(np.uint8)
        if is_color: plt.imshow(img)
        else: plt.imshow(img[:, :, 0], cmap='gray')
        plt.title(f'Class {i}', fontsize=8); plt.axis('off')
    fig.savefig(f'image_at_epoch_{epoch:04d}.png'); plt.close(fig)

# --- 6. Thực thi chính ---
if __name__ == '__main__':
    generator = build_generator(); discriminator = build_discriminator(); EPOCHS = 100 
    print("Bắt đầu huấn luyện Conditional GAN...")
    train(train_dataset, EPOCHS)
    
    if not os.path.exists("models"): os.makedirs("models")
    generator.save_weights(os.path.join("models", "generator_weights.h5"))
    print("\n✅ Đã hoàn tất huấn luyện và lưu trọng số Generator vào models/generator_weights.h5")