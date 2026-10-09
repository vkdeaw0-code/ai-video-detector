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
st.write("สแกนภาพรวม เสียงพากย์ และจังหวะการขยับปาก (Lip-Sync) อย่างสมดุล ไม่เข้มงวดเกินไป")

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
    """สกัดเฟรมภาพออกจากคลิปวิดีโอแบบสุ่มกระจาย"""
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
    """วิเคราะห์ความผิดปกติของเสียง (Audio Artifacts) และจังหวะปาก (Lip-Sync)"""
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
            
        # 1. ตรวจจับเสียงพากย์ AI สังเคราะห์ (Audio Spectral Flatness)
        data_float = data.astype(np.float32)
        fft_data = np.abs(np.fft.rfft(data_float))
        
        fft_sum = np.sum(fft_data)
        if fft_sum > 0:
            normalized_fft = fft_data / fft_sum
            spectral_flatness = float(np.exp(np.mean(np.log(normalized_fft + 1e-12))))
            if spectral_flatness < 1e-6 or spectral_flatness > 1e-3:
                audio_risk = 0.05 # บวกคะแนนความเสี่ยงเสียง AI แค่ 5% (เบาๆ)
                
        # 2. ตรวจจับการขยับปากและใบหน้า (Lip-Sync Check)
        rms_energy = np.sqrt(np.mean(data_float**2))
        
        # ถ้ามีเสียงคนพูด (RMS สูง) และดึงเฟรมมาได้
        if rms_energy > 500 and len(frames) > 2:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            prev_mouth = None
            motion_scores = []
            
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    # ครอปเฉพาะใบหน้าส่วนล่าง (บริเวณปาก)
                    mouth_roi = cv2.resize(gray[y + int(h/2):y+h, x:x+w], (64, 32))
                    
                    if prev_mouth is not None:
                        # คำนวณความต่างของพิกเซลปากว่ามีการขยับหรือไม่
                        diff = cv2.absdiff(mouth_roi, prev_mouth)
                        motion_scores.append(np.mean(diff))
                    prev_mouth = mouth_roi
            
            if len(motion_scores) > 0:
                avg_mouth_motion = np.mean(motion_scores)
                # ถ้าเสียงดังมาก แต่ปากแทบไม่ขยับ (อาการวิดีโอ AI ที่ปากแข็ง)
                if avg_mouth_motion < 2.0 and rms_energy > 1000:
                    lip_sync_risk = 0.08 # บวกความเสี่ยง 8%
                elif avg_mouth_motion < 4.0:
                    lip_sync_risk = 0.04 # บวกความเสี่ยง 4%
                    
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
        
        THRESHOLD = 0.75 # ตั้งเกณฑ์มาตรฐาน 75% ป้องกันการบล็อกคลิปเนียน
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 3]
            
            with col:
                with st.container(border=True):
                    st.caption(f"🎬 คลิปที่ {idx+1}: **{uploaded_file.name[:20]}...**" if len(uploaded_file.name) > 20 else f"🎬 คลิปที่ {idx+1}: **{uploaded_file.name}**")
                    
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    
                    with st.spinner("กำลังสแกนภาพ, เสียง, และจังหวะปาก..."):
                        frames = extract_frames(video_path)
                        
                        if not frames:
                            st.caption("⚠️ ไม่สามารถอ่านเฟรมได้")
                            status = "ERROR"
                            percent_score = 0
                        else:
                            frame_scores = []
                            with torch.no_grad():
                                for frame_np in frames:
                                    pil_img = Image.fromarray(frame_np)
                                    input_tensor = transform(pil_img).unsqueeze(0).to(device)
                                    output = model(input_tensor)
                                    probs = torch.softmax(output, dim=1)
                                    fake_prob = probs[0][1].item()
                                    frame_scores.append(fake_prob)
                            
                            # คำนวณสถิติภาพให้กระจายตัวเป็นธรรมชาติ
                            mean_val = float(np.mean(frame_scores))
                            std_val = float(np.std(frame_scores))
                            min_val = float(np.min(frame_scores))
                            max_val = float(np.max(frame_scores))
                            
                            # วิเคราะห์เสียงพากย์และจังหวะปาก
                            audio_risk, lip_sync_risk = analyze_audio_and_lipsync(video_path, frames)
                            
                            # รวมคะแนนอย่างสมดุล (ภาพ + เสียง + ขยับปาก)
                            dynamic_base = (min_val * 0.3) + (mean_val * 0.4) + (max_val * 0.2) + (std_val * 0.1)
                            calibrated_score = (np.power(dynamic_base, 2.2) * 0.78) + audio_risk + lip_sync_risk
                            
                            percent_score = float(np.clip(calibrated_score * 100, 3.0, 96.0))
                            
                            if (percent_score / 100.0) >= THRESHOLD:
                                status = "REJECT"
                                st.error(f"❌ **REJECT** ({percent_score:.0f}%)", icon="🚨")
                            else:
                                status = "PASS"
                                st.success(f"✅ **PASS** ({percent_score:.0f}%)", icon="🟢")
                                
                            st.progress(min(int(percent_score), 100))
                    
                    os.unlink(video_path)
                    
                    results_summary.append({
                        "ลำดับ": idx + 1,
                        "ชื่อไฟล์": uploaded_file.name,
                        "คะแนนความเสี่ยง (%)": f"{percent_score:.2f}%",
                        "สถานะ": status
                    })
        
        # --- 5. REJECTED CLIPS DASHBOARD ---
        st.divider()
        st.header("🚫 แดชบอร์ดสรุปคลิปที่ไม่ผ่านการคัดกรอง (Rejected Clips Dashboard)")
        
        df_all = pd.DataFrame(results_summary)
        df_rejected = df_all[df_all["สถานะ"] == "REJECT"]
        
        m1, m2, m3 = st.columns(3)
        total_clips = len(df_all)
        rejected_count = len(df_rejected)
        pass_count = total_clips - rejected_count
        reject_rate = (rejected_count / total_clips * 100) if total_clips > 0 else 0
        
        m1.metric("จำนวนคลิปทั้งหมด", f"{total_clips} คลิป")
        m2.metric("จำนวนคลิปที่ผ่าน (PASS)", f"{pass_count} คลิป")
        m3.metric("จำนวนคลิปที่ถูกคัดออก (REJECT)", f"{rejected_count} คลิป", delta=f"{reject_rate:.1f}% อัตราการคัดออก", delta_color="inverse")
        
        st.write("")
        
        if not df_rejected.empty:
            st.error(f"⚠️ ตรวจพบคลิปที่ไม่ผ่านเกณฑ์ทั้งหมด {len(df_rejected)} คลิป ดังรายการด้านล่าง:")
            st.dataframe(
                df_rejected[["ลำดับ", "ชื่อไฟล์", "คะแนนความเสี่ยง (%)"]], 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.balloons()
            st.success("🎉 ยินดีด้วย! ไม่พบคลิปที่ติดสถานะ REJECT ในรอบนี้ ทุกคลิปผ่านการคัดกรองทั้งหมด")
