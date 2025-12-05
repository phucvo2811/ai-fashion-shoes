import tensorflow as tf
import numpy as np
import os
# Không cần cv2 nữa, dùng tf.io
# import cv2 

# --- 1. Định nghĩa Hằng số Cấu hình ---
# Giữ nguyên 64x64 để đảm bảo ổn định VRAM
TARGET_IMAGE_SIZE = (64, 64) 
IMAGE_CHANNELS = 3 
DATA_ROOT_DIR = "dataset" 

# Ánh xạ tên thư mục chính sang nhãn số học
LABEL_MAP = {
    "Sandals": 5,
    "Shoes": 7,     
    "Sneakers": 7,  
    "Boots": 9,
    "Slippers": 0,
}

# --- 2. Hàm Tiền xử lý Ảnh (Sử dụng TensorFlow) ---
def map_path_to_image(image_path, label):
    """Ánh xạ đường dẫn file thành tensor ảnh đã tiền xử lý."""
    
    # 1. Tải và giải mã ảnh
    image = tf.io.read_file(image_path)
    # Lỗi thường xảy ra ở đây nếu file không phải JPEG, nên dùng decode_image
    image = tf.io.decode_image(image, channels=IMAGE_CHANNELS, expand_animations=False)
    
    # 2. Resize Ảnh
    image = tf.image.resize(image, TARGET_IMAGE_SIZE)
    
    # 3. Chuẩn hóa về [-1, 1]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1 # (image / 255) * 2 - 1
    
    return image, label

# --- 3. Hàm Tải Dữ liệu Lớn (Trả về LIST ĐƯỜNG DẪN) ---
def load_real_shoe_data(data_dir, num_classes, target_size=TARGET_IMAGE_SIZE, channels=IMAGE_CHANNELS):
    """Trả về danh sách đường dẫn và nhãn số học."""
    
    all_image_paths = []
    all_labels = []
    
    print(f"Bắt đầu quét tất cả ảnh trong thư mục: {data_dir}...")
    
    for root, _, files in os.walk(data_dir):
        assigned_label = -1
        
        for key, label in LABEL_MAP.items():
            if key.lower() in root.lower():
                assigned_label = label
                break
        
        if assigned_label == -1 or assigned_label >= num_classes:
            continue
            
        count = 0
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                all_image_paths.append(image_path)
                all_labels.append(assigned_label) 
                count += 1
        
        if count > 0:
            print(f"  -> Đã tìm thấy {count} ảnh từ {root} (Gán nhãn: {assigned_label})")
                    
    if not all_image_paths:
        print("Lỗi: Không tìm thấy ảnh nào hợp lệ. Huấn luyện bị dừng.")
        return [], np.array([])
        
    print(f"\n✅ Hoàn tất quét dữ liệu. Tổng số ảnh: {len(all_image_paths)}")
    
    return all_image_paths, np.array(all_labels)

# --- 4. Hàm Tạo Dataset TensorFlow ---
def create_tf_dataset(image_paths, labels_array, batch_size, buffer_size=10000):
    """Tạo đối tượng tf.data.Dataset từ danh sách đường dẫn."""
    if not image_paths:
        return None
        
    labels_tensor = tf.convert_to_tensor(labels_array, dtype=tf.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_tensor))
    
    # Map function tải ảnh từng file khi cần (sử dụng map_path_to_image đã sửa)
    dataset = dataset.map(map_path_to_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Áp dụng các kỹ thuật tối ưu luồng dữ liệu
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True) # Bỏ qua lô cuối
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) 
    
    return dataset
