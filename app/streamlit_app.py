import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.models import Model

# --- 1. Cấu hình và Tải Mô hình ---

st.set_page_config(page_title="👟 AI Shoe Designer", layout="centered")
st.title("✨ AI Shoe Designer: Sáng Tạo Thiết Kế Giày")

# Định nghĩa các hằng số (PHẢI KHỚP VỚI train_cgan.py)
NOISE_DIM = 100
NUM_CLASSES = 10 
TARGET_IMAGE_SIZE = (64, 64) 
IMAGE_CHANNELS = 3 # Đặt 3 hoặc 1
IMG_SHAPE = (*TARGET_IMAGE_SIZE, IMAGE_CHANNELS)

CLASS_LABELS = {
    5: "Sandal (Dép/Giày hở)",
    7: "Sneaker (Giày thể thao)",
    9: "Ankle Boot (Giày cổ ngắn)",
}
SHOE_CLASSES = {k: v for k, v in CLASS_LABELS.items() if k in [5, 7, 9]} 

# Hàm nhúng nhãn (get_label_embedding)
def get_label_embedding():
    # Phải khớp với 8x8 cho ảnh 64x64
    model = tf.keras.Sequential([
        Input(shape=(1,)),
        Dense(8 * 8 * 1, use_bias=False),
        Reshape((8, 8, 1))
    ], name='label_embedding')
    return model

# Hàm để xây dựng lại Generator (ĐÃ ĐIỀU CHỈNH cho ảnh 64x64)
@st.cache_resource
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