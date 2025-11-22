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
IMAGE_CHANNELS = 3 
IMG_SHAPE = (*TARGET_IMAGE_SIZE, IMAGE_CHANNELS)

CLASS_LABELS = {
    5: "Sandal (Dép/Giày hở)",
    7: "Sneaker (Giày thể thao)",
    9: "Ankle Boot (Giày cổ ngắn)",
}
SHOE_CLASSES = {k: v for k, v in CLASS_LABELS.items() if k in [5, 7, 9]} 

# Hàm nhúng nhãn (get_label_embedding)
def get_label_embedding():
    
    model = tf.keras.Sequential([
        Input(shape=(1,)),
        Dense(8 * 8 * 1, use_bias=False),
        Reshape((8, 8, 1))
    ], name='label_embedding')
    return model

# Hàm để xây dựng lại Generator 
@st.cache(allow_output_mutation=True) # <-- Dùng @st.cache cũ và tham số bắt buộc
def build_generator(noise_dim, num_classes):
    input_noise = Input(shape=(noise_dim,), name='noise_input')
    input_label = Input(shape=(1,), name='label_input')
    label_embedding = get_label_embedding()(input_label)

    x = Dense(8 * 8 * 256, use_bias=False)(input_noise)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Reshape((8, 8, 256))(x)

    merged_input = Concatenate()([x, label_embedding])


    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(merged_input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)


    x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x) 

    output_image = Conv2DTranspose(IMAGE_CHANNELS, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)

    return Model(inputs=[input_noise, input_label], outputs=output_image, name='Generator')


# Hàm Tải Mô hình
@st.cache(allow_output_mutation=True) # <-- Dùng @st.cache cũ
def load_cgan_model(noise_dim, num_classes):
    """Tải trọng số mô hình Generator đã lưu."""
    # Tải mô hình bên ngoài cache để tránh lỗi lặp lại (đã khắc phục warning trước)
    model = build_generator(noise_dim, num_classes) 
    
    model_path = os.path.join("models", "generator_weights.h5")
    
    if os.path.exists(model_path):
        try:
            model.load_weights(model_path)
            return True, model # Trả về trạng thái True và model
        except Exception:
            return False, model # Trả về trạng thái False và model
    return False, model # Trả về False nếu không tìm thấy file

# Tải mô hình Generator và xử lý thông báo
status, generator = load_cgan_model(NOISE_DIM, NUM_CLASSES)

if status:
    st.success(f"✅ Đã tải mô hình Generator (Epoch {100}) thành công.")
else:
    st.warning("⚠️ Không tìm thấy trọng số đã huấn luyện. Đang sử dụng mô hình khởi tạo.")


# --- 2. Giao diện Người dùng và Logic Xử lý ---

# Thiết lập Sidebar để chọn điều kiện
st.sidebar.header("⚙️ Tùy Chỉnh Thiết Kế")

# 2.1. Chọn Loại Giày (Sử dụng nhãn lớp cGAN)
selected_label_name = st.sidebar.selectbox(
    "1. Chọn Loại Giày:",
    options=list(SHOE_CLASSES.values()),
    index=list(SHOE_CLASSES.values()).index("Sneaker (Giày thể thao)")
)
# Lấy label số học tương ứng
selected_label_key = [k for k, v in SHOE_CLASSES.items() if v == selected_label_name][0]

# 2.2. Chọn Số lượng thiết kế
num_designs = st.sidebar.slider(
    "2. Số Lượng Thiết Kế Muốn Tạo:",
    min_value=1, max_value=5, value=3
)

# 2.3. Tạo nút kích hoạt
if st.sidebar.button("🎨 TẠO THIẾT KẾ MỚI"):
    
    st.header(f"Kết quả tạo sinh ({selected_label_name})")
    
    # Sử dụng st.spinner để hiển thị trạng thái tải
    with st.spinner(f"AI đang thiết kế {num_designs} mẫu {selected_label_name}..."):
        
        # 1. Chuẩn bị đầu vào cho Generator
        noise = tf.random.normal([num_designs, NOISE_DIM])
        labels = np.array([selected_label_key] * num_designs)
        
        # 2. Tạo sinh hình ảnh bằng mô hình GAN
        generated_images = generator([noise, labels], training=False)
        
        # 3. Trực quan hóa và hiển thị
        cols = st.columns(num_designs)
        
        # Xác định kênh màu để hiển thị đúng
        is_color = IMAGE_CHANNELS == 3
        
        for i in range(num_designs):
            img_array = generated_images[i].numpy()
            img_array = (img_array + 1) * 127.5 # Rescale từ [-1, 1] về [0, 255]
            img_array = img_array.astype(np.uint8)
            
            # Tạo figure Matplotlib (để có thể hiển thị trong Streamlit)
            fig, ax = plt.subplots(figsize=(2, 2))
            
            if is_color:
                ax.imshow(img_array)
            else:
                ax.imshow(img_array[:, :, 0], cmap='gray')

            ax.axis('off')
            ax.set_title(f"Mẫu #{i+1}", fontsize=8)
            
            # Hiển thị trong cột tương ứng
            cols[i].pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.success("Tạo sinh hoàn tất!")
else:
    st.info("Hãy chọn loại giày và nhấn nút 'TẠO THIẾT KẾ MỚI' để bắt đầu!")

st.markdown(
    """
    ---
    *Ứng dụng này sử dụng Mô hình Conditional GAN (cGAN).*
    """
)