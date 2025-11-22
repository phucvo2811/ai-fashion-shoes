import tensorflow as tf
import numpy as np
import os
import cv2 

# --- 1. Định nghĩa Hằng số Cấu hình ---
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

# --- 2. Hàm Tiền xử lý Ảnh (Từng ảnh) ---
def preprocess_image(image_path, target_size=TARGET_IMAGE_SIZE, channels=IMAGE_CHANNELS):
    """Tải, resize, và chuẩn hóa một ảnh về dải [-1, 1] cho GAN."""
    try:
        # Tải ảnh BGR hoặc Xám
        if channels == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if image is None:
            return None

        # Resize, Thêm Kênh, Chuẩn hóa
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
        if channels == 1:
            image = np.expand_dims(image, axis=-1)

        image = image.astype('float32')
        image = (image - 127.5) / 127.5
        
        return image
    except Exception:
        return None

# --- 3. Hàm Tải Dữ liệu Lớn (TRẢ VỀ LIST ĐƯỜNG DẪN ĐỂ TRÁNH OOM) ---
def load_real_shoe_data(data_dir, num_classes, target_size=TARGET_IMAGE_SIZE, channels=IMAGE_CHANNELS):
    """Tải tất cả dữ liệu ảnh từ data_dir và gán nhãn."""
    
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
        return [], []
        
    print(f"\n✅ Hoàn tất quét dữ liệu. Tổng số ảnh: {len(all_image_paths)}")
    
    return all_image_paths, np.array(all_labels)

# --- 4. Hàm Tạo Dataset TensorFlow (ĐÃ SỬA OOM BẰNG MAP FUNCTION) ---
# Tải ảnh từng file khi cần, không tải tất cả cùng lúc
def map_path_to_image(image_path, label):
    """Ánh xạ đường dẫn file thành tensor ảnh đã tiền xử lý."""
    
    # Hàm này chạy bên trong luồng dữ liệu của TensorFlow
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=IMAGE_CHANNELS) # Dùng tf.io thay vì cv2
    image = tf.image.resize(image, TARGET_IMAGE_SIZE)
    
    # Chuẩn hóa về [-1, 1]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    
    return image, label


def create_tf_dataset(image_paths, labels_array, batch_size, buffer_size=10000):
    """
    Tạo đối tượng tf.data.Dataset từ danh sách đường dẫn.
    """
    if not image_paths:
        return None
        
    # Tạo Dataset từ đường dẫn và nhãn
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_array))
    
    # Map function sẽ tải ảnh từng file khi cần, tránh OOM
    dataset = dataset.map(map_path_to_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 1. Shuffle (Xáo trộn)
    dataset = dataset.shuffle(buffer_size)
    
    # 2. Batch (Chia lô)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # 3. Prefetch (Tải trước): Tăng tốc độ
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) 
    
    return dataset

if __name__ == '__main__':
    print("utils.py chỉ là file hỗ trợ. Vui lòng chạy train_cgan.py để huấn luyện mô hình.")