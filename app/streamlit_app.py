import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, LeakyReLU, BatchNormalization, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

# --- CẤU HÌNH ---
st.set_page_config(page_title="👟 AI Sneaker Generator", layout="centered")
st.title("👟 AI Sneaker Generator (High-Fidelity 128px)")

NOISE_DIM = 128

# --- 1. KIẾN TRÚC GENERATOR (BẢN 512 FILTERS - KHỚP FILE TRAIN) ---
def get_label_embedding():
    return tf.keras.Sequential([
        Input(shape=(1,)), 
        Dense(8*8), 
        Reshape((8, 8, 1))
    ])

def build_generator():
    input_noise = Input(shape=(NOISE_DIM,), name='noise')
    input_label = Input(shape=(1,), name='label')
    
    # Xử lý nhãn
    label = get_label_embedding()(input_label)

    # Xử lý Noise (DÙNG 512 FILTERS ĐỂ KHỚP 32768)
    x = Dense(8 * 8 * 512, use_bias=False)(input_noise)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((8, 8, 512))(x)
    
    # Ghép Nhãn
    x = Concatenate()([x, label])

    # 4 Block Upsample (Giảm dần từ 512 -> 32)
    
    # Block 1: 8 -> 16 (Dùng 256 filters)
    x = UpSampling2D()(x)
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)

    # Block 2: 16 -> 32 (Dùng 128 filters)
    x = UpSampling2D()(x)
    x = Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)

    # Block 3: 32 -> 64 (Dùng 64 filters)
    x = UpSampling2D()(x)
    x = Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)

    # Block 4: 64 -> 128 (Dùng 32 filters)
    x = UpSampling2D()(x)
    x = Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)

    # Output
    output = Conv2D(3, 3, padding='same', activation='tanh', dtype='float32')(x)
    
    return Model(inputs=[input_noise, input_label], outputs=output, name="Generator")

# --- 2. LOAD MODEL ---
def find_checkpoints():
    if os.path.isdir("models"):
        files = [f for f in os.listdir("models") if f.endswith('.h5')]
        return sorted(files, reverse=True)
    return []

files = find_checkpoints()
selected_file = st.sidebar.selectbox("Chọn Epoch:", files) if files else None

@st.cache(allow_output_mutation=True)
def load_model(filename):
    model = build_generator()
    if filename:
        path = os.path.join("models", filename)
        if os.path.exists(path):
            try:
                model.load_weights(path)
                return True, model, path
            except Exception as e:
                return False, model, str(e)
    return False, model, "Không tìm thấy file"

is_ready, generator, msg = load_model(selected_file)

# --- 3. GIAO DIỆN ---
if is_ready:
    st.success(f"✅ Đã tải: {selected_file}")
else:
    st.error(f"⚠️ Lỗi tải: {msg}")

num = st.sidebar.slider("Số lượng:", 1, 4, 2)

if st.sidebar.button("🎨 VẼ GIÀY"):
    if is_ready:
        with st.spinner('Đang vẽ...'):
            noise = tf.random.normal([num, NOISE_DIM])
            # Fake nhãn 0 cho Sneaker
            labels = np.zeros((num, 1))
            
            imgs = generator([noise, labels], training=False)
            
            cols = st.columns(num)
            for i, col in enumerate(cols):
                img = (imgs[i].numpy() * 0.5) + 0.5
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                col.image(img, use_column_width=True, caption=f"Mẫu {i+1}")