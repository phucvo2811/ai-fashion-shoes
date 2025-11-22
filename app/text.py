import tensorflow as tf

# 1. In ra phiên bản TensorFlow
print("TensorFlow Version:", tf.__version__)

# 2. Kiểm tra các thiết bị vật lý (Physical Devices) có sẵn
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"\n✅ Đã tìm thấy {len(gpus)} thiết bị GPU.")
    for gpu in gpus:
        print("Thiết bị GPU:", gpu)
else:
    print("\n❌ KHÔNG tìm thấy GPU nào. TensorFlow chỉ chạy trên CPU.")