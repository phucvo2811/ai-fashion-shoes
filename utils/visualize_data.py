import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CẤU HÌNH ---
# Tên file ảnh gốc anh đã chuẩn bị sẵn ở thư mục gốc
INPUT_FILENAME = "anh_GOC.jpg" 
TARGET_SIZE = (128, 128)

def process_custom_image():
    # 1. Kiểm tra xem anh đã để file ảnh gốc ở đó chưa
    if not os.path.exists(INPUT_FILENAME):
        print(f"❌ LỖI: Không tìm thấy file '{INPUT_FILENAME}'!")
        print(f"👉 Anh vui lòng copy 1 tấm ảnh giày bất kỳ ra thư mục gốc và đổi tên nó thành '{INPUT_FILENAME}' trước nhé.")
        return

    print("\n" + "="*50)
    print(f"📸 ĐANG XỬ LÝ ẢNH CỦA ANH: {INPUT_FILENAME}")
    print("="*50)

    # --- BƯỚC 1: ĐỌC VÀ XỬ LÝ ẢNH ---
    try:
        # Đọc ảnh
        raw_img = tf.io.read_file(INPUT_FILENAME)
        # Tự động nhận diện jpg/png...
        img = tf.io.decode_image(raw_img, channels=3, expand_animations=False) 
    except:
        print("❌ Lỗi: File này không phải ảnh hợp lệ hoặc bị lỗi.")
        return
    
    # Resize Bicubic (Giống hệt quy trình train)
    img = tf.image.resize(img, TARGET_SIZE, method='bicubic')
    
    # Chuẩn hóa về [-1, 1] (Ra Tensor)
    img_tensor = (img / 127.5) - 1.0

    # --- BƯỚC 2: IN THÔNG TIN TENSOR (ĐỂ ANH CHỤP BÁO CÁO) ---
    print("\n📊 THÔNG TIN TENSOR (DỮ LIỆU MÁY HỌC):")
    print(f"Shape: {img_tensor.shape}")
    print(f"Values (Min/Max): {tf.reduce_min(img_tensor):.2f} / {tf.reduce_max(img_tensor):.2f}")
    print("Dữ liệu ma trận (Góc trên cùng bên trái):")
    print(img_tensor[0:3, 0:3, 0].numpy()) 

    # --- BƯỚC 3: LƯU ẢNH KẾT QUẢ ---
    # Chuyển ngược từ [-1, 1] về [0, 1] để lưu ảnh xem được
    display_img = (img_tensor * 0.5) + 0.5
    
    # Clip lại cho chắc chắn không bị lỗi hiển thị (phòng trường hợp Bicubic làm lố)
    display_img = tf.clip_by_value(display_img, 0.0, 1.0)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(display_img)
    plt.axis('off')
    
    output_filename = "anh_SAU_XU_LY.png"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    
    print("-" * 50)
    print(f"✅ Đã lưu ảnh kết quả thành công: '{output_filename}'")
    print("👉 XONG! Anh lấy 2 file này chèn vào Word là chuẩn bài.")

if __name__ == "__main__":
    process_custom_image()