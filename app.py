import streamlit as st
import cv2
import torch
import timm
import numpy as np
import tempfile
import os
from PIL import Image
from torchvision import transforms

# --- 1. SET UP PAGE CONFIG ---
st.set_page_config(
    page_title="ระบบตรวจจับคลิปวิดีโอ AI",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 ระบบตรวจจับและคัดกรองคลิปวิดีโอ AI (รองรับหลายไฟล์)")
st.write("อัปโหลดคลิปวิดีโอ 10 วินาที ได้พร้อมกันหลายไฟล์ เพื่อสแกนหาความผิดปกติและคัดออกอัตโนมัติ")

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_detection_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    model = model.to(device)
    model.eval()
    
    # โหลดตัวตรวจจับใบหน้ามาตรฐานของ OpenCV (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade, device

model, face_cascade, device = load_detection_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. HELPER FUNCTIONS ---
def extract_and_crop_faces(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    cropped_faces = []
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # สแกนหาใบหน้าด้วย OpenCV
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face = rgb_frame[y:y+h, x:x+w]
                if face.shape[0] > 10 and face.shape[1] > 10:
                    cropped_faces.append(face)
                    
        count += 1
        
    cap.release()
    return cropped_faces

# --- 4. WEB UI INTERFACE (MULTI-FILE SUPPORT) ---
uploaded_files = st.file_uploader(
    "เลือกไฟล์วิดีโอ (.mp4, .mov, .avi) - เลือกพร้อมกันหลายไฟล์ได้", 
    type=["mp4", "mov", "avi"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 เลือกไว้ทั้งหมด {len(uploaded_files)} คลิป")
    
    if st.button("🔍 เริ่มกระบวนการสแกนตรวจจับทุกคลิป", type="primary"):
        st.divider()
        st.subheader("📊 ตารางสรุปผลการวิเคราะห์:")
        
        for idx, uploaded_file in enumerate(uploaded_files, 1):
            st.markdown(f"### 🎬 คลิปที่ {idx}: `{uploaded_file.name}`")
            
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            with st.spinner(f"กำลังสแกนคลิปที่ {idx}/{len(uploaded_files)}..."):
                faces = extract_and_crop_faces(video_path)
                
                if not faces:
                    st.warning("⚠️ ไม่พบใบหน้าบุคคลในวิดีโอ ไม่สามารถวิเคราะห์ได้")
                else:
                    scores = []
                    with torch.no_grad():
                        for face_np in faces:
                            pil_img = Image.fromarray(face_np)
                            input_tensor = transform(pil_img).unsqueeze(0).to(device)
                            output = model(input_tensor)
                            probs = torch.softmax(output, dim=1)
                            fake_prob = probs[0][1].item()
                            scores.append(fake_prob)
                    
                    avg_score = float(np.mean(scores))
                    THRESHOLD = 0.70
                    percent_score = avg_score * 100
                    
                    if avg_score >= THRESHOLD:
                        st.error(f"❌ **REJECT (คัดออก)** - ความแปลก AI: **{percent_score:.2f}%**")
                    else:
                        st.success(f"✅ **PASS (ผ่าน)** - ความแปลก AI: **{percent_score:.2f}%**")
                        
                    st.progress(min(int(percent_score), 100))
            
            os.unlink(video_path)
            st.divider()
