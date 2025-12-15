import streamlit as st
import tensorflow as tf
import numpy as np
import os
import io
from PIL import Image
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Conv2D,
    LeakyReLU,
    BatchNormalization,
    UpSampling2D,
)
from tensorflow.keras.models import Model

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="👟 Sneaker Report Tool", layout="wide"
) 
st.title("👟 Công Cụ Lọc Ảnh Báo Cáo (Cherry-Picking)")
st.markdown("---")

NOISE_DIM = 128
INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


# Cấu hình GPU/CPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# --- 1. MODEL GENERATOR (STANDARD LSGAN - ĐÃ ĐỒNG BỘ 100%) ---
# Đã loại bỏ label inputs, Concatenate, và sửa Kernel thành 5x5
@st.cache(allow_output_mutation=True)
def build_generator():
    input_noise = Input(shape=(NOISE_DIM,), name='noise')
    
    # Block 1: Project & Reshape
    x = Dense(8 * 8 * 256, use_bias=False, kernel_initializer=INIT)(input_noise) 
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((8, 8, 256))(x)

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

    # Output 128x128x3 (Kernel 5x5)
    output = Conv2D(3, 5, padding='same', activation='tanh', dtype='float32', kernel_initializer=INIT)(x)
    return Model(inputs=input_noise, outputs=output, name="Generator")


# --- 2. HÀM LOAD MODEL ---
def find_checkpoints():
    if os.path.isdir("models"):
        return sorted(
            [f for f in os.listdir("models") if f.endswith(".h5")], reverse=True
        )
    return []

# Sidebar Cấu hình
st.sidebar.header("⚙️ Bảng Điều Khiển")
files = find_checkpoints()
selected_file = (
    st.sidebar.selectbox("Chọn Epoch (Nên chọn file mới nhất):", files)
    if files
    else None
)

@st.cache(allow_output_mutation=True)
def load_model(filename):
    model = build_generator()
    if filename:
        path = os.path.join("models", filename)
        if os.path.exists(path):
            try:
                # Load weights vào cấu trúc mới đã sửa
                model.load_weights(path) 
                return True, model
            except Exception as e:
                # Ghi lỗi nếu không load được (vì cấu trúc file .h5 cũ không khớp)
                st.sidebar.error(f"❌ Lỗi: {e}") 
                return False, model
    return False, model

is_ready, generator = load_model(selected_file)

if is_ready:
    st.sidebar.success(f"✅ Đã tải: {selected_file}")
elif not files:
    st.sidebar.info("📂 Chưa tìm thấy file weights (.h5) nào trong thư mục 'models'.")
else:
    st.sidebar.warning("⚠️ Lỗi tải model. Vui lòng kiểm tra file weights có đúng cấu trúc.")

# Quản lý trạng thái (Session State) để giữ hình không bị mất
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

# --- 3. GIAO DIỆN CHÍNH ---

# Số lượng ảnh muốn sinh
num_images = st.sidebar.slider("Số lượng ảnh muốn lọc:", 4, 20, 8, step=4)

# Nút Bấm
if st.sidebar.button("🎨 TẠO GIÀY MỚI"):
    if is_ready:
        with st.spinner(f"Đang vẽ {num_images} mẫu giày..."):
            noise = tf.random.normal([num_images, NOISE_DIM])
            
            # Sinh ảnh (Chỉ truyền noise, không cần labels)
            imgs = generator(noise, training=False) 

            # Chuyển về format chuẩn để hiển thị và lưu
            processed_imgs = []
            for i in range(num_images):
                img_array = (imgs[i].numpy() * 0.5) + 0.5
                img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
                processed_imgs.append(img_array)

            st.session_state.generated_images = processed_imgs
    else:
        st.error("Không thể sinh ảnh vì model chưa được tải thành công.")

# Hiển thị kết quả
if st.session_state.generated_images:
    st.subheader(f"🔍 Kết quả: {len(st.session_state.generated_images)} mẫu")

    # Chia lưới 4 cột
    cols = st.columns(4)
    for i, img_array in enumerate(st.session_state.generated_images):
        with cols[i % 4]:
            # Hiển thị ảnh
            st.image(img_array, caption=f"Mẫu #{i+1}", use_column_width=True)

            # Tạo nút Download cho từng ảnh
            pil_img = Image.fromarray(img_array)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label=f"💾 Tải Mẫu #{i+1}",
                data=byte_im,
                file_name=f"sneaker_report_{i+1}.png",
                mime="image/png",
                key=f"dl_{i}", 
            )
else:
    st.info("👈 Hãy bấm nút 'TẠO GIÀY MỚI' bên trái để bắt đầu lọc hình.")