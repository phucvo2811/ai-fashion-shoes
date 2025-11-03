import tensorflow as tf
import numpy as np
import os
import cv2 

# --- 1. Định nghĩa Hằng số Cấu hình ---
TARGET_IMAGE_SIZE = (64, 64) 
IMAGE_CHANNELS = 3 # Đặt 3 cho ảnh màu RGB, 1 cho ảnh xám
DATA_ROOT_DIR = "dataset" 

# Ánh xạ tên thư mục chính sang nhãn số học (SẼ DÙNG TRONG train_cgan.py và streamlit_app.py)
# Nhãn này được chọn để khớp với các lớp phổ biến trong Fashion MNIST (5, 7, 9)
LABEL_MAP = {
    "Sandals": 5,
    "Shoes": 7,     # Bao gồm các loại giày thông thường và Sneakers
    "Sneakers": 7,  # Nếu có thư mục Sneakers riêng, cũng gán là lớp 7
    "Boots": 9,
    "Slippers": 0,
    # Các nhãn khác trong cấu trúc của bạn (Ankle, Knee High) sẽ được ánh xạ qua từ khóa trên
}

# --- 2. Hàm Tiền xử lý Ảnh (Từng ảnh) ---
def preprocess_image(image_path, target_size=TARGET_IMAGE_SIZE, channels=IMAGE_CHANNELS):
    """Tải, resize, và chuẩn hóa một ảnh về dải [-1, 1] cho GAN."""
    try:
        if channels == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if image is None:
            return None

        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        if channels == 1:
            image = np.expand_dims(image, axis=-1)

        image = image.astype('float32')
        image = (image - 127.5) / 127.5
        
        return image
    
    except Exception as e:
        # Xử lý lỗi đọc ảnh mà không làm dừng toàn bộ chương trình
        # print(f"Lỗi xử lý ảnh {image_path}: {e}")
        return None

# --- 3. Hàm Tải Dữ liệu Lớn (ĐÃ SỬA: Quét nhiều cấp và Ánh xạ Nhãn) ---
def load_real_shoe_data(data_dir, num_classes, target_size=TARGET_IMAGE_SIZE, channels=IMAGE_CHANNELS):
    """
    Tải tất cả dữ liệu ảnh từ data_dir bằng cách quét các thư mục con 
    và gán nhãn dựa trên từ khóa trong đường dẫn.
    """
    
    all_images = []
    all_labels = []
    
    print(f"Bắt đầu quét tất cả ảnh trong thư mục: {data_dir}...")
    
    # Sử dụng os.walk để duyệt qua tất cả các thư mục con (Boots, Sandals, Ankle, adidas...)
    for root, _, files in os.walk(data_dir):
        
        # 1. TÌM NHÃN SỐ HỌC DỰA TRÊN ĐƯỜNG DẪN THƯ MỤC
        assigned_label = -1
        
        # Duyệt qua các từ khóa trong LABEL_MAP để tìm nhãn phù hợp
        for key, label in LABEL_MAP.items():
            # Kiểm tra xem từ khóa (ví dụ: 'Boots') có xuất hiện trong đường dẫn thư mục không
            if key.lower() in root.lower():
                assigned_label = label
                break # Lấy nhãn đầu tiên tìm thấy
        
        # Nếu nhãn không hợp lệ hoặc lớn hơn số lớp cho phép, bỏ qua thư mục này
        if assigned_label == -1 or assigned_label >= num_classes:
            continue
            
        # 2. Xử lý ảnh trong thư mục đã tìm thấy nhãn
        count = 0
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                processed_img = preprocess_image(image_path, target_size, channels)
                
                if processed_img is not None:
                    all_images.append(processed_img)
                    all_labels.append(assigned_label) 
                    count += 1
        
        if count > 0:
            print(f"  -> Đã tải {count} ảnh từ {root} (Gán nhãn: {assigned_label})")
                    
    if not all_images:
        print("Lỗi: Không tìm thấy ảnh nào hợp lệ. Vui lòng kiểm tra đường dẫn và cấu trúc file!")
        return np.array([]), np.array([])
        
    print(f"\n✅ Hoàn tất tải dữ liệu. Tổng số ảnh: {len(all_images)}")
    
    images_array = np.array(all_images)
    labels_array = np.array(all_labels)
    
    return images_array, labels_array

# --- 4. Hàm Tạo Dataset TensorFlow (Giữ nguyên) ---
def create_tf_dataset(images_array, labels_array, batch_size, buffer_size=10000):
    """Tạo đối tượng tf.data.Dataset từ mảng NumPy."""
    if images_array.size == 0:
        return None
        
    labels_tensor = tf.convert_to_tensor(labels_array, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((images_array, labels_tensor))
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    return dataset

if __name__ == '__main__':
    # Kiểm tra nhanh chức năng tải dữ liệu (chỉ hoạt động nếu thư mục 'dataset' tồn tại)
    print("Kiểm tra chức năng utils.py...")
    images, labels = load_real_shoe_data(DATA_ROOT_DIR, num_classes=10)
    if images.size > 0:
        print(f"Kiểm tra mảng ảnh: Shape={images.shape}, Dtype={images.dtype}")