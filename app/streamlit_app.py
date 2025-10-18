import os
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

st.title("AI thiết kế giày — Demo")

# Lấy token Hugging Face (nếu dùng model private hoặc HF API)
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN == "":
    st.warning("Không tìm thấy HF_TOKEN — nếu dùng model Hugging Face private, export HF_TOKEN vào env.")

prompt = st.text_input("Mô tả mẫu giày (ví dụ: 'modern sneaker, white leather, chunky sole, minimal lines')", 
                       "modern sneaker, white leather, chunky sole, minimal lines")
steps = st.slider("Inference steps", 1, 50, 20)

if st.button("Sinh ảnh"):
    with st.spinner("Đang tải model... (lần đầu có thể lâu)"):
        # Ví dụ dùng runwayml/stable-diffusion-v1-5 (cần token nếu private)
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        image = pipe(prompt, num_inference_steps=steps).images[0]
        st.image(image, caption="Mẫu giày do AI sinh", output_format="PNG")
