import tensorflow as tf
import numpy as np
import os

TARGET_IMAGE_SIZE = (128, 128) 
IMAGE_CHANNELS = 3 
DATA_ROOT_DIR = "dataset" 

# CHỈ LẤY SNEAKER (Gom cả Shoes vào làm 1 class)
LABEL_MAP = {
    "Sneakers": 0,  
    "Shoes": 0,     
}

def map_path_to_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=IMAGE_CHANNELS, expand_animations=False)
    image = tf.image.resize(image, TARGET_IMAGE_SIZE, method='bicubic') # Bicubic cho nét
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.random_flip_left_right(image) # Tăng dữ liệu
    return image, label

def load_real_shoe_data(data_dir, num_classes, target_size=TARGET_IMAGE_SIZE, channels=IMAGE_CHANNELS):
    all_image_paths = []
    all_labels = []
    print(f"Bắt đầu quét dữ liệu SNEAKER (128x128)...")
    
    for root, _, files in os.walk(data_dir):
        assigned_label = -1
        for key, label in LABEL_MAP.items():
            if key.lower() in root.lower():
                assigned_label = label; break
        
        if assigned_label != 0: continue
            
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(root, filename))
                all_labels.append(assigned_label) 

    if not all_image_paths: return [], np.array([])
    print(f"✅ Đã tìm thấy {len(all_image_paths)} ảnh Sneaker.")
    return all_image_paths, np.array(all_labels)

def create_tf_dataset(image_paths, labels_array, batch_size, buffer_size=10000):
    if not image_paths: return None
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_array))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(map_path_to_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) 
    return dataset