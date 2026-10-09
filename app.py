import streamlit as st
import cv2
import torch
import timm
import numpy as np
import tempfile
import os
from PIL import Image
from torchvision import transforms

# --- 1. SET UP PAGE CONFIG (ใช้แบบ Wide เพื่อให้แสดง Grid สวยงาม) ---
st.set_page_config(
    page_title="ระบบตรวจจับคลิปวิดีโอ AI",
    page_icon="🎬",
    layout="wide" # ปรับหน้าจอให้กว้างเต็มตา
)

st.title("🎬 ระบบตรวจจับและคัดกรองคลิปวิดีโอ AI")
st.write("อัปโหลดคลิปวิดีโอ 10 วินาที ได้พร้อมกันหลายไฟล์ ระบบจะสรุปผลเป็นตารางการ์ดขนาดเล็ก")

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_detection_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_detection_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. HELPER FUNCTIONS ---
def extract_frames(video_path, frame_interval=15):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
                    
        count += 1
        
    cap.release()
    return frames

# --- 4. WEB UI INTERFACE (COMPACT GRID UI) ---
uploaded_files = st.file_uploader(
    "เลือกไฟล์วิดีโอ (.mp4, .mov, .avi) - เลือกพร้อมกันหลายไฟล์ได้", 
    type=["mp4", "mov", "avi"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 เลือกไว้ทั้งหมด {len(uploaded_files)} คลิป")
    
    if st.button("🔍 เริ่มกระบวนการสแกนตรวจจับทุกคลิป", type="primary"):
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์แบบกะทัดรัด:")
        
        # สร้างคอลัมน์แบบ Grid (3 คอลัมน์ต่อ 1 แถว)
        cols = st.columns(3)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # สลับแสดงผลลงในคอลัมน์ 0, 1, 2
            col = cols[idx % 3]
            
            with col:
                # สร้างกล่องการ์ดสำหรับแต่ละคลิป
                with st.container(border=True):
                    st.caption(f"🎬 คลิปที่ {idx+1}: **{uploaded_file.name[:20]}...**" if len(uploaded_file.name) > 20 else f"🎬 คลิปที่ {idx+1}: **{uploaded_file.name}**")
                    
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    
                    with st.spinner("กำลังสแกน..."):
                        frames = extract_frames(video_path)
                        
                        if not frames:
                            st.caption("⚠️ ไม่สามารถอ่านเฟรมได้")
                        else:
                            scores = []
                            with torch.no_grad():
                                for frame_np in frames:
                                    pil_img = Image.fromarray(frame_np)
                                    input_tensor = transform(pil_img).unsqueeze(0).to(device)
                                    output = model(input_tensor)
                                    probs = torch.softmax(output, dim=1)
                                    fake_prob = probs[0][1].item()
                                    scores.append(fake_prob)
                            
                            avg_score = float(np.mean(scores))
                            THRESHOLD = 0.70
                            percent_score = avg_score * 100
                            
                            # แสดงผลเป็นไอคอนเล็กขนาดกะทัดรัด
                            if avg_score >= THRESHOLD:
                                st.error(f"❌ **REJECT** ({percent_score:.0f}%)", icon="🚨")
                            else:
                                st.success(f"✅ **PASS** ({percent_score:.0f}%)", icon="🟢")
                                
                            st.progress(min(int(percent_score), 100))
                    
                    os.unlink(video_path)
