import streamlit as st
from PIL import Image
import io

st.title("👟 AI Thiết kế Giày - Demo Giao diện")

prompt = st.text_input("Nhập mô tả ý tưởng giày của bạn:", "Giày thể thao màu trắng xanh kiểu tương lai")

if st.button("Tạo ảnh"):
    st.write(f"🎨 Đang tạo ảnh cho prompt: **{prompt}** ...")
    # Demo: tạo ảnh trống (sau này thay bằng ảnh AI sinh ra)
    img = Image.new("RGB", (512, 512), color=(120, 180, 255))
    st.image(img, caption="Ảnh mẫu demo", use_column_width=True)
