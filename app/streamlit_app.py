import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.models import Model

# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="👟 AI Shoe Designer", layout="centered")
st.title("✨ AI Shoe Designer: Sáng Tạo Thiết Kế Giày")

# CÁC HẰNG SỐ
NOISE_DIM = 100
NUM_CLASSES = 10 
TARGET_IMAGE_SIZE = (64, 64) 
IMAGE_CHANNELS = 3 
IMG_SHAPE = (*TARGET_IMAGE_SIZE, IMAGE_CHANNELS)

CLASS_LABELS = {
    5: "Sandal (Dép/Giày hở)",
    7: "Sneaker (Giày thể thao)",
    9: "Ankle Boot (Giày cổ ngắn)",
}
SHOE_CLASSES = {k: v for k, v in CLASS_LABELS.items() if k in [5, 7, 9]} 

# --- 2. XÂY DỰNG KIẾN TRÚC MÔ HÌNH ---

def get_label_embedding():
    model = tf.keras.Sequential([
        Input(shape=(1,)),
        Dense(8 * 8 * 1, use_bias=False),
        Reshape((8, 8, 1))
    ], name='label_embedding')
    return model

def build_generator(noise_dim, num_classes):
    input_noise = Input(shape=(noise_dim,), name='noise_input')
    input_label = Input(shape=(1,), name='label_input')
    label_embedding = get_label_embedding()(input_label)

    x = Dense(8 * 8 * 256, use_bias=False)(input_noise)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Reshape((8, 8, 256))(x)

    merged_input = Concatenate()([x, label_embedding])

    # 8x8 -> 16x16
    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(merged_input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # 16x16 -> 32x32
    x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x) 

    # 32x32 -> 64x64
    output_image = Conv2DTranspose(IMAGE_CHANNELS, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)

    return Model(inputs=[input_noise, input_label], outputs=output_image, name='Generator')

# --- 3. TẢI TRỌNG SỐ ---

def load_cgan_model(noise_dim, num_classes):
    # 1. Xây dựng bộ khung
    model = build_generator(noise_dim, num_classes)
    
    # 2. Tìm file
    possible_paths = [
        "models/generator_weights.h5",       
        "../models/generator_weights.h5",    
        "generator_weights.h5",
        r"E:\AI-fashion-shoes\models\generator_weights.h5"
    ]
    
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break
            
    is_trained = False
    if found_path:
        try:
            model.load_weights(found_path)
            is_trained = True
        except Exception as e:
            print(f"Lỗi: {e}")
    
    return is_trained, model, found_path

# --- 4. KHỞI TẠO ---

is_trained, generator, loaded_path = load_cgan_model(NOISE_DIM, NUM_CLASSES)

if is_trained:
    st.success(f"✅ Đã kết nối thành công với bộ não AI! (Nguồn: {loaded_path})")
else:
    st.warning("⚠️ Không tìm thấy file trọng số. Đang chạy chế độ THỬ NGHIỆM.")

# --- 5. GIAO DIỆN & TƯƠNG TÁC ---

st.sidebar.header("⚙️ Tùy Chỉnh Thiết Kế")

selected_label_name = st.sidebar.selectbox(
    "1. Chọn Loại Giày:",
    options=list(SHOE_CLASSES.values()),
    index=1 
)
selected_label_key = [k for k, v in SHOE_CLASSES.items() if v == selected_label_name][0]

num_designs = st.sidebar.slider("2. Số Lượng Mẫu:", 1, 6, 3)

if st.sidebar.button("🎨 TẠO THIẾT KẾ MỚI"):
    
    st.markdown(f"### Kết quả cho: **{selected_label_name}**")
    
    with st.spinner('Đang vẽ...'):
        try:
            # Tạo dữ liệu đầu vào
            noise = tf.random.normal([num_designs, NOISE_DIM])
            labels = np.array([selected_label_key] * num_designs)
            
            # Chạy mô hình
            generated_images = generator([noise, labels], training=False)
            
            # Hiển thị ảnh
            cols = st.columns(num_designs)
            
            for i in range(num_designs):
                img_array = generated_images[i].numpy()
                img_array = (img_array + 1) * 127.5 
                img_array = img_array.astype(np.uint8)
                
                if img_array.shape[-1] == 1:
                    img_array = np.squeeze(img_array)
                
                # SỬA LỖI Ở ĐÂY: Dùng use_column_width thay vì use_container_width
                if num_designs > 1:
                    with cols[i]:
                        st.image(img_array, caption=f"Mẫu #{i+1}", use_column_width=True, clamp=True)
                else:
                    st.image(img_array, caption=f"Mẫu #{i+1}", width=300, clamp=True)
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi tạo ảnh: {e}")

else:
    st.info("👈 Chọn loại giày và bấm nút để bắt đầu!")

st.markdown("---")