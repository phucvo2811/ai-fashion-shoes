import os
import cv2
import numpy as np
from PIL import Image

# --- CẤU HÌNH ---
INPUT_DIR = "datasetbc"           # Thư mục ảnh gốc (Nét căng)
OUTPUT_DIR = "dataset_report"   # Thư mục lưu ảnh đã làm mờ (Dùng để báo cáo)
TARGET_SIZE = 128               # Kích thước của GAN (128x128)
BLUR_AMOUNT = 1                 # Độ làm mờ (Số lẻ: 1, 3, 5...). 1 là không mờ, 3 là mờ nhẹ.

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🚀 Đang xử lý ảnh từ '{INPUT_DIR}' sang '{OUTPUT_DIR}'...")

count = 0
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

for filename in os.listdir(INPUT_DIR):
    if count >= 100: break # Chỉ cần làm khoảng 100 ảnh để báo cáo là đủ
    
    ext = os.path.splitext(filename)[1].lower()
    if ext not in valid_extensions: continue

    # 1. Đọc ảnh
    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)
    
    if img is None: continue

    # 2. Resize về 128x128 (Đây là bước quan trọng nhất để làm nó "bớt nét")
    # Dùng INTER_AREA để thu nhỏ ảnh mượt mà
    img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    # 3. (Tùy chọn) Làm mờ nhẹ nếu muốn nó giống GAN hơn nữa
    if BLUR_AMOUNT > 1:
        img_resized = cv2.GaussianBlur(img_resized, (BLUR_AMOUNT, BLUR_AMOUNT), 0)

    # 4. Lưu ảnh
    save_path = os.path.join(OUTPUT_DIR, f"real_soft_{count}.jpg")
    cv2.imwrite(save_path, img_resized)
    
    count += 1

print(f"✅ Xong! Đã tạo {count} ảnh 'mềm mại' trong thư mục '{OUTPUT_DIR}'.")
print("💡 Anh hãy dùng các ảnh trong thư mục này để chèn vào báo cáo so sánh.")