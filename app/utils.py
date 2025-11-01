import tensorflow as tf
import numpy as np
import os
import cv2 

# --- 1. Định nghĩa Hằng số Cấu hình ---
TARGET_IMAGE_SIZE = (64, 64) 
IMAGE_CHANNELS = 3 # Đặt 3 cho ảnh màu RGB, 1 cho ảnh xám
DATA_ROOT_DIR = "dataset"

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
        print(f"Lỗi xử lý ảnh {image_path}: {e}")
        return None

# --- 3. Hàm Tải Dữ liệu Lớn (Batch Processing) ---
def load_real_shoe_data(data_dir, num_classes, target_size=TARGET_IMAGE_SIZE, channels=IMAGE_CHANNELS):
    """Tải tất cả dữ liệu ảnh giày và nhãn từ thư mục data_dir."""
    all_images = []
    all_labels = []
    print(f"Bắt đầu tải dữ liệu từ thư mục: {data_dir}...")
    
    for class_label in range(num_classes):
        class_name = f"class_{class_label}"
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue 
        
        count = 0
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_dir, filename)
                processed_img = preprocess_image(image_path, target_size, channels)
                
                if processed_img is not None:
                    all_images.append(processed_img)
                    all_labels.append(class_label)
                    count += 1
        
        if count > 0:
            print(f"  -> Đã tải {count} ảnh cho Lớp {class_label} ({class_name})")
                    
    if not all_images:
        print("Lỗi: Không tìm thấy ảnh nào hợp lệ. Kiểm tra đường dẫn và định dạng ảnh!")
        return np.array([]), np.array([])
        
    print(f"\n✅ Hoàn tất tải dữ liệu. Tổng số ảnh: {len(all_images)}")
    images_array = np.array(all_images)
    labels_array = np.array(all_labels)
    
    return images_array, labels_array

# --- 4. Hàm Tạo Dataset TensorFlow ---
def create_tf_dataset(images_array, labels_array, batch_size, buffer_size=10000):
    """Tạo đối tượng tf.data.Dataset từ mảng NumPy."""
    if images_array.size == 0:
        return None
        
    labels_tensor = tf.convert_to_tensor(labels_array, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((images_array, labels_tensor))
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    return dataset

if __name__ == '__main__':
    print("utils.py chỉ là file hỗ trợ. Vui lòng chạy train_cgan.py để huấn luyện mô hình.")