import streamlit as st
import tensorflow as tf
import numpy as np
import os
import io
from PIL import Image
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, LeakyReLU, BatchNormalization, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

# --- CẤU HÌNH ---
st.set_page_config(page_title="👟 Sneaker Report Tool", layout="wide") # Layout rộng để xem nhiều hình
st.title("👟 Công Cụ Lọc Ảnh Báo Cáo (Cherry-Picking)")
st.markdown("---")

NOISE_DIM = 128

# Cấu hình GPU/CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- 1. MODEL GENERATOR (BẢN 512 FILTERS - KHỚP 100% FILE SAVE) ---
def get_label_embedding():
    return tf.keras.Sequential([Input(shape=(1,)), Dense(8*8), Reshape((8, 8, 1))])

def build_generator():
    input_noise = Input(shape=(NOISE_DIM,), name='noise')
    input_label = Input(shape=(1,), name='label')
    label = get_label_embedding()(input_label)
    
    # Block 1: 512 Filters
    x = Dense(8 * 8 * 512, use_bias=False)(input_noise)
    x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x); x = Reshape((8, 8, 512))(x)
    x = Concatenate()([x, label])

    # Block 2: Upsample
    x = UpSampling2D()(x); x = Conv2D(256, 3, padding='same', use_bias=False)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x); x = Conv2D(128, 3, padding='same', use_bias=False)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x); x = Conv2D(64, 3, padding='same', use_bias=False)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x); x = Conv2D(32, 3, padding='same', use_bias=False)(x); x = BatchNormalization(momentum=0.8)(x); x = LeakyReLU(0.2)(x)
    
    output = Conv2D(3, 3, padding='same', activation='tanh', dtype='float32')(x)
    return Model(inputs=[input_noise, input_label], outputs=output)

# --- 2. HÀM LOAD MODEL ---
def find_checkpoints():
    if os.path.isdir("models"):
        return sorted([f for f in os.listdir("models") if f.endswith('.h5')], reverse=True)
    return []

# Sidebar Cấu hình
st.sidebar.header("⚙️ Bảng Điều Khiển")
files = find_checkpoints()
selected_file = st.sidebar.selectbox("Chọn Epoch (Nên chọn file mới nhất):", files) if files else None

@st.cache(allow_output_mutation=True)
def load_model(filename):
    model = build_generator()
    if filename:
        path = os.path.join("models", filename)
        if os.path.exists(path):
            try:
                model.load_weights(path)
                return True, model
            except: pass
    return False, model

is_ready, generator = load_model(selected_file)

if is_ready:
    st.sidebar.success(f"✅ Đã tải: {selected_file}")
else:
    st.sidebar.error("❌ Chưa tải được model")

# --- 3. GIAO DIỆN CHÍNH ---

# Quản lý trạng thái (Session State) để giữ hình không bị mất
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

# Số lượng ảnh muốn sinh
num_images = st.sidebar.slider("Số lượng ảnh muốn lọc:", 4, 20, 8, step=4)

# Nút Bấm
if st.sidebar.button("🎨 TẠO MẺ ẢNH MỚI"):
    if is_ready:
        with st.spinner(f'Đang vẽ {num_images} mẫu giày...'):
            noise = tf.random.normal([num_images, NOISE_DIM])
            labels = np.zeros((num_images, 1))
            
            # Sinh ảnh
            imgs = generator([noise, labels], training=False)
            
            # Chuyển về format chuẩn để hiển thị và lưu
            processed_imgs = []
            for i in range(num_images):
                img_array = (imgs[i].numpy() * 0.5) + 0.5
                img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
                processed_imgs.append(img_array)
            
            st.session_state.generated_images = processed_imgs

# --- 4. HIỂN THỊ VÀ TẢI VỀ ---
if st.session_state.generated_images:
    st.subheader(f"🔍 Kết quả: {len(st.session_state.generated_images)} mẫu")
    
    # Chia lưới 4 cột
    cols = st.columns(4)
    for i, img_array in enumerate(st.session_state.generated_images):
        with cols[i % 4]:
            # Hiển thị ảnh
            st.image(img_array, caption=f"Mẫu #{i+1}", use_column_width=True)
            
            # Tạo nút Download cho từng ảnh
            # Convert numpy array sang Bytes để tải về
            pil_img = Image.fromarray(img_array)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label=f"💾 Tải Mẫu #{i+1}",
                data=byte_im,
                file_name=f"sneaker_report_{i+1}.png",
                mime="image/png",
                key=f"dl_{i}" # Key duy nhất cho mỗi nút
            )
else:
    st.info("👈 Hãy bấm nút 'TẠO MẺ ẢNH MỚI' bên trái để bắt đầu lọc hình.")