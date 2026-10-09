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
import gc
from scipy.io import wavfile
from PIL import Image
from torchvision import transforms

st.set_page_config(page_title="AI Video Detector", page_icon="🎬", layout="wide")
st.title("🎬 ระบบตรวจจับคลิปวิดีโอ AI")
st.write("ระบบประเมินความสมจริง: เน้นตรวจสอบอวัยวะขาดหายและเสียงพูดผิดปกติ")

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

def extract_frames(video_path, target_fps=6): 
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30
    
    interval = int(video_fps / target_fps)
    if interval < 1: interval = 1
    
    frames = []
    motion_scores = []
    prev_gray = None
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % interval == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                diff = cv2.absdiff(gray_frame, prev_gray)
                motion_scores.append(np.mean(diff))
            else:
                motion_scores.append(0.0)
                
            prev_gray = gray_frame
            frames.append(rgb_frame)
        count += 1
    cap.release()
    return frames, motion_scores

def analyze_audio_and_lipsync(video_path):
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    audio_risk = 0.0
    audio_msg = ""
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            return 0.0, "ไม่มีเสียง"
            
        sample_rate, data = wavfile.read(audio_path)
        if len(data) == 0:
            return 0.0, "ไม่มีเสียง"
            
        data_float = data.astype(np.float32)
        fft_data = np.abs(np.fft.rfft(data_float))
        fft_sum = np.sum(fft_data)
        
        if fft_sum > 0:
            normalized_fft = fft_data / fft_sum
            spectral_flatness = float(np.exp(np.mean(np.log(normalized_fft + 1e-12))))
            if spectral_flatness < 1e-6 or spectral_flatness > 1e-3:
                audio_risk += 3.0 
                
        window_size = int(sample_rate * 0.1) 
        energies = np.array([np.sum(data_float[i:i+window_size]**2) for i in range(0, len(data_float), window_size)])
        if len(energies) > 0:
            energy_variance = np.var(energies) / (np.mean(energies) + 1e-6)
            if energy_variance < 0.5: 
                audio_risk += 5.0 
                
        if audio_risk >= 7.0:
            audio_msg = "เสียงฟังไม่รู้เรื่อง"
        elif audio_risk > 0:
            audio_msg = "เสียงเพี้ยนเล็กน้อย"
        
    except Exception:
        return 0.0, "ตรวจสอบเสียงไม่ได้"
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            
    return audio_risk, audio_msg

uploaded_files = st.file_uploader("เลือกไฟล์วิดีโอ (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"], accept_multiple_files=True)

if uploaded_files:
    if st.button("🔍 เริ่มสแกน", type="primary"):
        st.divider()
        results_summary = []
        cols = st.columns(3)
        THRESHOLD = 70.0 
        WARNING_THRESHOLD = 50.0
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 3]
            with col:
                with st.container(border=True):
                    display_name = f"{uploaded_file.name[:20]}..." if len(uploaded_file.name) > 20 else uploaded_file.name
                    st.caption(f"คลิป {idx+1}: **{display_name}**")
                    
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    
                    final_score = 0.0
                    status = "ERROR"
                    details = ""
                    
                    try:
                        with st.spinner("สแกน..."):
                            frames, motion_scores = extract_frames(video_path, target_fps=6)
                            
                            if not frames:
                                st.caption("⚠️ อ่านภาพไม่ได้")
                            else:
                                frame_scores = []
                                with torch.no_grad():
                                    for frame_np in frames:
                                        pil_img = Image.fromarray(frame_np)
                                        input_tensor = transform(pil_img).unsqueeze(0).to(device)
                                        output = model(input_tensor)
                                        probs = torch.softmax(output, dim=1)
                                        frame_scores.append(probs[0][1].item())
                                
                                mean_prob = float(np.mean(frame_scores))
                                median_prob = float(np.median(frame_scores))
                                
                                base_score = (median_prob * 30.0) + (mean_prob * 15.0) + 15.0 
                                glitch_thresh = min(0.92, max(0.85, median_prob + 0.20))
                                
                                glitch_penalty = 0.0
                                defect_count = 0
                                details_list = []
                                
                                early_probs = frame_scores[:8]
                                if sum([1 for p in early_probs if p > glitch_thresh]) >= 3:
                                    defect_count += 1
                                    glitch_penalty += 5.0
                                    details_list.append("ต้นคลิปผิดปกติ")
                                
                                avg_motion = np.mean(motion_scores) if len(motion_scores) > 0 else 0
                                morph_count = sum([1 for i in range(1, len(frame_scores)) if frame_scores[i] > glitch_thresh and motion_scores[i] > (avg_motion * 4.0)])
                                
                                if morph_count >= 2:
                                    defect_count += 1
                                    glitch_penalty += 8.0
                                    details_list.append("อวัยวะหาย/ผิดรูป")
                                
                                suspect_frames = [1 if score > glitch_thresh else 0 for score in frame_scores]
                                max_consecutive = 0
                                current_consecutive = 0
                                for is_suspect in suspect_frames:
                                    if is_suspect:
                                        current_consecutive += 1
                                        max_consecutive = max(max_consecutive, current_consecutive)
                                    else:
                                        current_consecutive = 0
                                
                                if max_consecutive >= 6: 
                                    defect_count += 1
                                    glitch_penalty += 8.0
                                    details_list.append("ภาพบิดเบี้ยว")
                                
                                audio_risk, audio_msg = analyze_audio_and_lipsync(video_path)
                                if audio_risk > 0:
                                    glitch_penalty += audio_risk
                                if audio_msg:
                                    details_list.append(audio_msg)
                                    
                                if defect_count >= 3:
                                    glitch_penalty += 12.0
                                    details_list.append("พังหลายจุด")
                                
                                details = " | ".join(list(dict.fromkeys(details_list))) if details_list else "ปกติ"
                                    
                                final_score = np.clip(base_score + glitch_penalty, 25.0, 85.0)
                                
                                if final_score >= THRESHOLD:
                                    status = "REJECT"
                                    st.error(f"❌ **ตก** ({final_score:.0f}%)")
                                    st.write(f"*{details}*")
                                elif final_score >= WARNING_THRESHOLD:
                                    status = "WARNING"
                                    st.warning(f"⚠️ **พอใช้** ({final_score:.0f}%)")
                                    st.write(f"*{details}*")
                                else:
                                    status = "PASS"
                                    st.success(f"✅ **ผ่าน** ({final_score:.0f}%)")
                                    st.write(f"*{details}*")
                                        
                                st.progress(min(int(final_score), 100))
                                
                    except Exception:
                        st.caption("⚠️ Error")
                    finally:
                        if os.path.exists(video_path):
                            os.unlink(video_path)
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    
                    results_summary.append({
                        "ลำดับ": idx + 1,
                        "ชื่อไฟล์": uploaded_file.name,
                        "คะแนน": f"{final_score:.0f}%",
                        "สถานะ": status,
                        "หมายเหตุ": details
                    })
        
        st.divider()
        df_all = pd.DataFrame(results_summary)
        
        df_pass = df_all[df_all["สถานะ"] == "PASS"]
        df_warning = df_all[df_all["สถานะ"] == "WARNING"]
        df_reject = df_all[df_all["สถานะ"] == "REJECT"]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ทั้งหมด", f"{len(df_all)}")
        m2.metric("ผ่าน", f"{len(df_pass)}")
        m3.metric("พอใช้", f"{len(df_warning)}")
        m4.metric("ตก", f"{len(df_reject)}")
        
        st.write("---")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.success("✅ ผ่าน (<50%)")
            if not df_pass.empty:
                st.dataframe(df_pass[["ชื่อไฟล์", "คะแนน"]], hide_index=True)
                
        with c2:
            st.warning("⚠️ พอใช้ (50-69%)")
            if not df_warning.empty:
                st.dataframe(df_warning[["ชื่อไฟล์", "คะแนน", "หมายเหตุ"]], hide_index=True)
                
        with c3:
            st.error("❌ ตก (≥70%)")
            if not df_reject.empty:
                st.dataframe(df_reject[["ชื่อไฟล์", "คะแนน", "หมายเหตุ"]], hide_index=True)
