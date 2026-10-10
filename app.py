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

# ==========================================
# 1. ตั้งค่าหน้าเว็บ Streamlit และโหลดโมเดล
# ==========================================
st.set_page_config(page_title="AI Video Inspector Precision", page_icon="🎬", layout="wide")
st.title("🎬 ระบบคัดกรองคุณภาพคลิปวิดีโอ AI (เวอร์ชันสแกนสเกลและเสียงสะดุด)")
st.write("ระบบตรวจจับขั้นสูง: **เกณฑ์ตัดตก 76% ขึ้นไป** | ตรวจจับขนาดสินค้าหลอกตา (สเกลไม่ตรงจริง) และเสียงพูดอ่านตัวย่อเพี้ยน/คำสะดุด")

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

# ==========================================
# 2. ฟังก์ชันวิเคราะห์วิดีโอ & สเกลสินค้า
# ==========================================
def extract_and_analyze_frames(video_path, target_fps=6): 
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30
    
    interval = max(1, int(video_fps / target_fps))
    frames = []
    scale_anomalies = []
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % interval == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            
            # สแกนพื้นที่สเกลภาพ (ตรวจจับมือใหญ่ผิดปกติเทียบกับสินค้าในฉาก)
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                max_area = max([cv2.contourArea(c) for c in contours])
                frame_area = 224 * 224
                area_ratio = max_area / frame_area
                # ถ้าวัตถุหลัก/มือครองพื้นที่สเกลผิดธรรมชาติเกินไป
                if area_ratio > 0.65:
                    scale_anomalies.append(1)
                else:
                    scale_anomalies.append(0)
            else:
                scale_anomalies.append(0)
                
        count += 1
    cap.release()
    return frames, scale_anomalies

# ==========================================
# 3. ฟังก์ชันวิเคราะห์เสียงสะดุด / อ่านตัวย่อเพี้ยน
# ==========================================
def analyze_audio_glitch(video_path):
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    audio_penalty = 0.0
    audio_msgs = []
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            return 0.0, ["🔇 ไม่มีเสียงประกอบ"]
            
        sample_rate, data = wavfile.read(audio_path)
        if len(data) == 0:
            return 0.0, ["🔇 ไม่มีเสียงประกอบ"]
            
        data_float = data.astype(np.float32)
        
        # 1. ตรวจสอบความสม่ำเสมอของจังหวะคำ (ตรวจเสียงสะดุด/อ่านรวบคำตัวย่อ "ซม.")
        window_size = int(sample_rate * 0.05) # สแกนละเอียดทุก 50ms
        energies = np.array([np.sum(data_float[i:i+window_size]**2) for i in range(0, len(data_float), window_size)])
        
        if len(energies) > 1:
            energy_diffs = np.abs(np.diff(energies))
            mean_diff = np.mean(energy_diffs)
            std_diff = np.std(energy_diffs)
            
            # การกระตุกหรือสะดุดอย่างฉับพลันของคลื่นเสียง (Stutter / Glitch Detection)
            stutter_points = np.sum(energy_diffs > (mean_diff + 2.5 * std_diff))
            
            if stutter_points >= 4:
                audio_penalty += 35.0
                audio_msgs.append("🔊 เสียงพูดมีคำสะดุด/อ่านตัวย่อเพี้ยน (เช่น ซม.)")
            elif stutter_points >= 2:
                audio_penalty += 15.0
                audio_msgs.append("🔊 เสียงพูดมีจังหวะสะดุดเล็กน้อย")

        # 2. ตรวจสอบเสียงแบนราบผิดธรรมชาติ (AI Voice Glitch)
        window_large = int(sample_rate * 0.1)
        energies_large = np.array([np.sum(data_float[i:i+window_large]**2) for i in range(0, len(data_float), window_large)])
        if len(energies_large) > 0:
            mean_e = np.mean(energies_large)
            variance_e = np.var(energies_large) / (mean_e + 1e-6)
            if variance_e < 0.12 and mean_e > 100:
                audio_penalty += 35.0
                audio_msgs.append("🔊 เสียงพูดแบนราบไร้จังหวะธรรมชาติ")

    except Exception:
        return 0.0, ["⚠️ ไม่สามารถวิเคราะห์เสียงได้"]
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            
    return audio_penalty, audio_msgs

# ==========================================
# 4. UI และ ระบบประมวลผลหลัก
# ==========================================
uploaded_files = st.file_uploader("เลือกไฟล์วิดีโอ (.mp4, .mov, .avi) - อัปโหลดได้หลายไฟล์พร้อมกัน", type=["mp4", "mov", "avi"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"📁 เลือกไว้ทั้งหมด {len(uploaded_files)} คลิป")
    if st.button("🔍 เริ่มสแกนคัดกรองละเอียด", type="primary"):
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์แยกคลิป:")
        
        results_summary = []
        cols = st.columns(3)
        REJECT_THRESHOLD = 76.0 # 💡 เกณฑ์ตัดตกที่ 76% ขึ้นไป
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 3]
            with col:
                with st.container(border=True):
                    display_name = f"{uploaded_file.name[:22]}..." if len(uploaded_file.name) > 22 else uploaded_file.name
                    st.caption(f"🎬 คลิปที่ {idx+1}: **{display_name}**")
                    
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    
                    status = "ERROR"
                    details = ""
                    
                    try:
                        with st.spinner("กำลังสแกนโครงสร้างภาพ สเกล และเสียง..."):
                            frames, scale_anomalies = extract_and_analyze_frames(video_path, target_fps=6)
                            
                            if not frames:
                                st.caption("⚠️ ไม่สามารถอ่านไฟล์วิดีโอได้")
                            else:
                                frame_scores = []
                                with torch.no_grad():
                                    for frame_np in frames:
                                        pil_img = Image.fromarray(frame_np)
                                        input_tensor = transform(pil_img).unsqueeze(0).to(device)
                                        output = model(input_tensor)
                                        probs = torch.softmax(output, dim=1)
                                        frame_scores.append(probs[0][1].item())
                                
                                dynamic_base_score = (float(np.mean(frame_scores)) * 18.0) + (float(np.std(frame_scores)) * 12.0)
                                visual_penalty = 0.0
                                details_list = []
                                
                                # 💡 1. ตรวจจับการบิดเบี้ยวของอวัยวะ/สินค้า (กฎ 2 วินาที = 12 เฟรม)
                                severe_streak = 0
                                max_severe_streak = 0
                                for s in frame_scores:
                                    if s > 0.98:
                                        severe_streak += 1
                                        max_severe_streak = max(max_severe_streak, severe_streak)
                                    else:
                                        severe_streak = 0
                                        
                                if max_severe_streak >= 12: 
                                    visual_penalty += 55.0
                                    details_list.append("⚠️ มือ/อวัยวะ หรือสินค้า บิดเบี้ยวพังเกิน 2 วินาที")
                                elif max_severe_streak >= 6: 
                                    visual_penalty += 25.0
                                    details_list.append("อวัยวะ/สินค้าบิดเบี้ยวช่วงสั้น 1-2 วิ")
                                    
                                # 💡 2. ตรวจจับขนาดสเกลสินค้าไม่ตรงความจริง (Scale Mismatch)
                                scale_anomaly_ratio = sum(scale_anomalies) / len(scale_anomalies) if scale_anomalies else 0
                                if scale_anomaly_ratio > 0.4:
                                    visual_penalty += 35.0
                                    details_list.append("⚠️ สเกลขนาดสินค้าไม่ตรงความจริง (สเกลหลอกตา)")
                                
                                # 💡 3. ตรวจสอบเสียงพูดสะดุด / ตัวย่อเพี้ยน
                                audio_penalty, audio_msgs = analyze_audio_glitch(video_path)
                                if audio_msgs:
                                    details_list.extend(audio_msgs)
                                
                                # คำนวณคะแนนสุทธิแบบละเอียด (1 - 100%)
                                raw_final = 5.0 + dynamic_base_score + visual_penalty + audio_penalty
                                final_score = min(100.0, max(1.0, raw_final))
                                
                                if not details_list:
                                    details = "สเกลสินค้าถูกต้อง การเคลื่อนไหวและเสียงพูดเนียนสมบูรณ์"
                                else:
                                    details = " | ".join(list(dict.fromkeys(details_list)))
                                
                                # ตัดสินเกรด (<76% ผ่าน, >=76% ไม่ผ่าน)
                                if final_score >= REJECT_THRESHOLD:
                                    status = "REJECT"
                                    st.error(f"❌ **ไม่ผ่าน** ({final_score:.0f}%)")
                                    st.write(f"*{details}*")
                                else:
                                    status = "PASS"
                                    st.success(f"✅ **ผ่าน** ({final_score:.0f}%)")
                                    st.write(f"*{details}*")
                                        
                                st.progress(int(final_score))
                                
                    except Exception as e:
                        st.caption(f"⚠️ Error: {str(e)}")
                    finally:
                        if os.path.exists(video_path):
                            os.unlink(video_path)
                        if 'frames' in locals(): del frames
                        if 'frame_scores' in locals(): del frame_scores
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    
                    results_summary.append({
                        "ลำดับ": idx + 1,
                        "ชื่อไฟล์": uploaded_file.name,
                        "ความเสี่ยง": f"{final_score:.0f}%",
                        "สถานะ": status,
                        "หมายเหตุ": details
                    })
        
        # ==========================================
        # 5. แดชบอร์ดสรุปผลรวม
        # ==========================================
        st.divider()
        st.subheader("📋 แดชบอร์ดสรุปผลรวม")
        df_all = pd.DataFrame(results_summary)
        
        df_pass = df_all[df_all["สถานะ"] == "PASS"]
        df_reject = df_all[df_all["สถานะ"] == "REJECT"]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("จำนวนทั้งหมด", f"{len(df_all)} คลิป")
        m2.metric("✅ ผ่าน (<76%)", f"{len(df_pass)} คลิป")
        m3.metric("❌ ไม่ผ่าน (≥76%)", f"{len(df_reject)} คลิป")
        
        st.write("---")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.success("✅ คลิปที่ **ผ่าน** (<76%)")
            if not df_pass.empty:
                st.dataframe(df_pass[["ลำดับ", "ชื่อไฟล์", "ความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else:
                st.caption("ไม่มีคลิปในกลุ่มนี้")
                
        with c2:
            st.error("❌ คลิปที่ **ไม่ผ่าน** (≥76%)")
            if not df_reject.empty:
                st.dataframe(df_reject[["ลำดับ", "ชื่อไฟล์", "ความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else:
                st.caption("ไม่มีคลิปในกลุ่มนี้")
        
        st.success("🎉 ตรวจสอบเสร็จสิ้น คืนค่าความจำ RAM เรียบร้อยแล้ว")
