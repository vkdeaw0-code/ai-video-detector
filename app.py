import streamlit as st
import cv2
import torch
import timm
import numpy as np
import tempfile
import os
import pandas as pd
import subprocess
import imageio_ffmpeg
from scipy.io import wavfile
from PIL import Image
from torchvision import transforms

# --- 1. SET UP PAGE CONFIG ---
st.set_page_config(
    page_title="ระบบตรวจจับคลิปวิดีโอ AI",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 ระบบตรวจจับและคัดกรองคลิปวิดีโอ AI")
st.write("สแกนภาพรวม เสียงพากย์ และ Lip-Sync (ปรับเกณฑ์เข้มงวดขึ้นเล็กน้อย)")

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

def analyze_audio_and_lipsync(video_path, frames):
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    audio_risk = 0.0
    lip_sync_risk = 0.0
    
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            return 0.0, 0.0
            
        sample_rate, data = wavfile.read(audio_path)
        if len(data) == 0:
            os.unlink(audio_path)
            return 0.0, 0.0
            
        data_float = data.astype(np.float32)
        fft_data = np.abs(np.fft.rfft(data_float))
        
        fft_sum = np.sum(fft_data)
        if fft_sum > 0:
            normalized_fft = fft_data / fft_sum
            spectral_flatness = float(np.exp(np.mean(np.log(normalized_fft + 1e-12))))
            if spectral_flatness < 1e-6 or spectral_flatness > 1e-3:
                audio_risk = 0.08 # ปรับเพิ่มความเสี่ยงเสียง AI เป็น 8%
                
        rms_energy = np.sqrt(np.mean(data_float**2))
        
        if rms_energy > 500 and len(frames) > 2:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            prev_mouth = None
            motion_scores = []
            
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    mouth_roi = cv2.resize(gray[y + int(h/2):y+h, x:x+w], (64, 32))
                    
                    if prev_mouth is not None:
                        diff = cv2.absdiff(mouth_roi, prev_mouth)
                        motion_scores.append(np.mean(diff))
                    prev_mouth = mouth_roi
            
            if len(motion_scores) > 0:
                avg_mouth_motion = np.mean(motion_scores)
                if avg_mouth_motion < 2.0 and rms_energy > 1000:
                    lip_sync_risk = 0.10 # ปรับความเสี่ยงปากไม่ขยับเป็น 10%
                elif avg_mouth_motion < 4.0:
                    lip_sync_risk = 0.05
                    
        os.unlink(audio_path)
        return audio_risk, lip_sync_risk
        
    except Exception:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        return 0.0, 0.0

# --- 4. WEB UI INTERFACE ---
uploaded_files = st.file_uploader(
    "เลือกไฟล์วิดีโอ (.mp4, .mov, .avi) - เลือกพร้อมกันหลายไฟล์ได้", 
    type=["mp4", "mov", "avi"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 เลือกไว้ทั้งหมด {len(uploaded_files)} คลิป")
    
    if st.button("🔍 เริ่มกระบวนการสแกนตรวจจับทุกคลิป", type="primary"):
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์:")
        
        results_summary = []
        cols = st.columns(3)
        
        THRESHOLD = 0.70 # ปรับเกณฑ์ลดลงมาที่ 70% เพื่อให้คัดออกง่ายขึ้นนิดหน่อย
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 3]
            
            with col:
                with st.container(border=True):
                    st.caption(f"🎬 คลิปที่ {idx
