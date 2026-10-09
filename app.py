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

st.set_page_config(page_title="ระบบตรวจจับคลิปวิดีโอ AI", page_icon="🎬", layout="wide")
st.title("🎬 ระบบตรวจจับและคัดกรองคลิปวิดีโอ AI")
st.write("ระบบเน้นความเสถียร (คัดออกเมื่อมีวัตถุหายฉับพลัน มือบิดเบี้ยว หรือจุดพังรุนแรง)")

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

def extract_frames(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
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
                audio_risk = 0.03 
                
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
                    lip_sync_risk = 0.05 
                elif avg_mouth_motion < 4.0:
                    lip_sync_risk = 0.02
                    
        os.unlink(audio_path)
        return audio_risk, lip_sync_risk
    except Exception:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        return 0.0, 0.0

uploaded_files = st.file_uploader("เลือกไฟล์วิดีโอ (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"📁 เลือกไว้ทั้งหมด {len(uploaded_files)} คลิป")
    if st.button("🔍 เริ่มกระบวนการสแกนตรวจจับทุกคลิป", type="primary"):
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์:")
        
        results_summary = []
        cols = st.columns(3)
        THRESHOLD = 0.75 
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 3]
            with col:
                with st.container(border=True):
                    display_name = f"{uploaded_file.name[:20]}..." if len(uploaded_file.name) > 20 else uploaded_file.name
                    st.caption(f"🎬 คลิปที่ {idx+1}: **{display_name}**")
                    
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    
                    percent_score = 0.0
                    status = "ERROR"
                    
                    try:
                        with st.spinner("กำลังสแกน..."):
                            frames = extract_frames(video_path)
                            if not frames:
                                st.caption("⚠️ ไม่สามารถอ่านเฟรมได้")
                            else:
                                frame_scores = []
                                with torch.no_grad():
                                    for frame_np in frames:
                                        pil_img = Image.fromarray(frame_np)
                                        input_tensor = transform(pil_img).unsqueeze(0).to(device)
                                        output = model(input_tensor)
                                        probs = torch.softmax(output, dim=1)
                                        frame_scores.append(probs[0][1].item())
                                
                                mean_val = float(np.mean(frame_scores))
                                max_val = float(np.max(frame_scores))
                                std_val = float(np.std(frame_scores))
                                
                                audio_risk, lip_sync_risk = analyze_audio_and_lipsync(video_path, frames)
                                
                                # 💡 เพิ่มระบบดักจับวัตถุหาย (Temporal Glitch Detection)
                                temporal_penalty = 0.0
                                if len(frames) > 2:
                                    pixel_diffs = []
                                    for i in range(1, len(frames)):
                                        gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
                                        gray2 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                                        diff = np.mean(cv2.absdiff(gray1, gray2))
                                        pixel_diffs.append(diff)
                                    
                                    if len(pixel_diffs) > 0:
                                        max_diff = np.max(pixel_diffs)
                                        median_diff = np.median(pixel_diffs)
                                        # ถ้าจู่ๆ เฟรมมีการเปลี่ยนแปลงรุนแรง (วัตถุหาย) จะดันคะแนนขึ้นทันที 25%
                                        if median_diff < 15.0 and max_diff > (median_diff * 3.5):
                                            temporal_penalty = 0.25 
                                
                                base_score = (mean_val * 0.30) + (max_val * 0.15)
                                glitch_penalty = std_val * 1.8 
                                
                                # รวมคะแนนทั้งหมด
                                calibrated_score = base_score + glitch_penalty + audio_risk + lip_sync_risk + temporal_penalty
                                percent_score = float(np.clip(calibrated_score * 100, 2.0, 98.0))
                                
                                if (percent_score / 100.0) >= THRESHOLD:
                                    status = "REJECT"
                                    st.error(f"❌ **REJECT** ({percent_score:.0f}%)", icon="🚨")
                                else:
                                    status = "PASS"
                                    st.success(f"✅ **PASS** ({percent_score:.0f}%)", icon="🟢")
                                    
                                st.progress(min(int(percent_score), 100))
                    except Exception as e:
                        st.caption("⚠️ เกิดข้อผิดพลาด")
                    finally:
                        if os.path.exists(video_path):
                            os.unlink(video_path)
                    
                    results_summary.append({
                        "ลำดับ": idx + 1,
                        "ชื่อไฟล์": uploaded_file.name,
                        "คะแนนความเสี่ยง (%)": f"{percent_score:.2f}%",
                        "สถานะ": status
                    })
        
        st.divider()
        df_all = pd.DataFrame(results_summary)
        df_rejected = df_all[df_all["สถานะ"] == "REJECT"]
        
        m1, m2, m3 = st.columns(3)
        total_clips = len(df_all)
        rejected_count = len(df_rejected)
        pass_count = total_clips - rejected_count
        reject_rate = (rejected_count / total_clips * 100) if total_clips > 0 else 0
        
        m1.metric("จำนวนคลิปทั้งหมด", f"{total_clips} คลิป")
        m2.metric("จำนวนคลิปที่ผ่าน (PASS)", f"{pass_count} คลิป")
        m3.metric("จำนวนคลิปที่ถูกคัดออก (REJECT)", f"{rejected_count} คลิป", delta=f"{reject_rate:.1f}%")
        
        if not df_rejected.empty:
            st.error(f"⚠️ คลิปที่ไม่ผ่านเกณฑ์:")
            st.dataframe(df_rejected[["ลำดับ", "ชื่อไฟล์", "คะแนนความเสี่ยง (%)"]], use_container_width=True, hide_index=True)
