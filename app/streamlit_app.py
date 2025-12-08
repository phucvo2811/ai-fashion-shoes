import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, LeakyReLU, BatchNormalization, Concatenate, UpSampling2D
from tensorflow.keras.models import Model

st.set_page_config(page_title="👟 AI Sneaker Designer", layout="centered")
st.title("👟 AI Sneaker Generator")

NOISE_DIM = 128
NUM_CLASSES = 1

# --- PHẢI KHỚP VỚI FILE TRAIN (512 FILTERS) ---
def get_label_embedding():
    return tf.keras.Sequential([Input(shape=(1,)), Dense(8*8), Reshape((8, 8, 1))])

def build_generator():
    input_noise = Input(shape=(NOISE_DIM,)); input_label = Input(shape=(1,))
    label = get_label_embedding()(input_label)

    # 512 Filters
    x = Dense(8 * 8 * 512, use_bias=False)(input_noise)
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x); x = Reshape((8, 8, 512))(x)
    x = Concatenate()([x, label])

    # 4 lớp Upsample (512->256->128->64->32)
    x = UpSampling2D()(x); x = Conv2D(256, 3, padding='same', use_bias=False)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x); x = Conv2D(128, 3, padding='same', use_bias=False)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x); x = Conv2D(64, 3, padding='same', use_bias=False)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x); x = Conv2D(32, 3, padding='same', use_bias=False)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)

    output = Conv2D(3, 3, padding='same', activation='tanh', dtype='float32')(x)
    return Model(inputs=[input_noise, input_label], outputs=output)

def find_checkpoints():
    if os.path.isdir("models"):
        return sorted([f for f in os.listdir("models") if f.endswith('.h5')], reverse=True)
    return []

files = find_checkpoints()
selected_file = st.sidebar.selectbox("Chọn Epoch:", files) if files else None

@st.cache(allow_output_mutation=True)
def load_model(filename):
    model = build_generator()
    if filename:
        path = os.path.join("models", filename)
        if os.path.exists(path):
            try: model.load_weights(path); return True, model
            except: pass
    return False, model

is_ready, generator = load_model(selected_file)

if is_ready: st.success(f"✅ Đang dùng: {selected_file}")
else: st.warning("⚠️ Chưa có file. Hãy train model trước.")

num = st.sidebar.slider("Số lượng:", 1, 4, 2)
if st.sidebar.button("🎨 VẼ GIÀY"):
    if is_ready:
        noise = tf.random.normal([num, NOISE_DIM])
        labels = np.zeros((num, 1))
        imgs = generator([noise, labels], training=False)
        st.image([(x.numpy()*0.5)+0.5 for x in imgs], width=300, caption=[f"Mẫu {i+1}" for i in range(num)])