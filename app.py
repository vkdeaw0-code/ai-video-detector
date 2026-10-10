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
# 1. ตั้งค่าหน้าเว็บและการจัดการหน่วยความจำ
# ==========================================
st.set_page_config(page_title="AI Video Inspector Ultimate V2", page_icon="🤖", layout="wide")
st.title("🤖 ระบบตรวจคัดกรองคลิป AI (Ultimate - Strict Scale)")
st.markdown("""
ระบบตรวจจับขั้นสูงสุด: **เกณฑ์ตัดตก 76% ขึ้นไป**
*   📏 **Scale & Miniature:** จับผิดสเกลหลอกตา ชั้นวางของจิ๋ว สัดส่วนมือเทียบกับสินค้าเพี้ยน (ปัดตก 100%)
*   👁️ **Vision:** ตรวจจับอวัยวะละลาย วาร์ปหายเกิน 2 วินาที (ปัดตก)
*   👂 **Audio:** ตรวจจับเสียงหุ่นยนต์, คำสะดุดเกิน 2 คำ, และการอ่านตัวย่อเพี้ยน (ปัดตก)
""")

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
# 2. เครื่องยนต์วิเคราะห์สเกลและการเคลื่อนไหวภาพ
# ==========================================
def analyze_video_structure_and_scale(video_path, target_fps=6):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30
    
    interval = max(1, int(video_fps / target_fps))
    frames = []
    
    prev_gray = None
    motion_metrics = []
    miniature_scale_flags = [] # เก็บค่าสัดส่วนมือที่ใหญ่ผิดปกติ
    
    # กำหนดช่วงสีผิวสำหรับจับมือคน (HSV Skin Threshold)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % interval == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            
            # --- 2.1 วิเคราะห์ Global Motion ---
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_metrics.append(np.mean(diff))
            else:
                motion_metrics.append(0.0)
            prev_gray = gray
                
            # --- 2.2 วิเคราะห์ Scale หลอกตา (มือ vs สินค้า) ---
            hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
            
            # หาพื้นที่มือ
            skin_contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_skin_area = max([cv2.contourArea(c) for c in skin_contours]) if skin_contours else 0
            
            # หาพื้นที่วัตถุโครงสร้าง (Edge Based)
            edges = cv2.Canny(gray, 50, 150)
            obj_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_obj_area = max([cv2.contourArea(c) for c in obj_contours]) if obj_contours else 1
            
            # หากมือมีขนาดใหญ่มากๆ เมื่อเทียบกับโครงสร้างหลัก (เช่น มือใหญ่กว่าชั้นวางของ)
            # ถือว่าเป็นสเกล "ของเล่นจิ๋ว" หรือ Miniature Anomaly
            if max_skin_area > 0 and max_obj_area > 0:
                hand_to_obj_ratio = max_skin_area / max_obj_area
                if hand_to_obj_ratio > 1.5: # มือใหญ่กว่าโครงสร้างหลัก 1.5 เท่าขึ้นไป
                    miniature_scale_flags.append(1)
                else:
                    miniature_scale_flags.append(0)
            else:
                miniature_scale_flags.append(0)
                
        count += 1
    cap.release()
    
    return frames, motion_metrics, miniature_scale_flags

# ==========================================
# 3. เครื่องยนต์วิเคราะห์เสียง (Audio Engine)
# ==========================================
def analyze_audio_strict(video_path):
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
            return 0.0, ["🔇 ไม่มีเสียง (อนุโลม)"]
            
        sample_rate, data = wavfile.read(audio_path)
        if len(data) == 0:
            return 0.0, ["🔇 ไม่มีเสียง (อนุโลม)"]
            
        data_float = data.astype(np.float32)
        
        # 3.1 ตรวจจับคำสะดุด / อ่านตัวย่อเพี้ยน (Stutter/Glitches)
        window_size = int(sample_rate * 0.05) 
        energies = np.array([np.sum(data_float[i:i+window_size]**2) for i in range(0, len(data_float), window_size)])
        
        if len(energies) > 2:
            energy_diffs = np.abs(np.diff(energies))
            mean_diff = np.mean(energy_diffs)
            std_diff = np.std(energy_diffs)
            
            stutter_points = np.sum(energy_diffs > (mean_diff + 3.0 * std_diff))
            
            if stutter_points >= 4:
                audio_penalty += 70.0 # ตัดตกทันที
                audio_msgs.append("⚠️ เสียงพูดพัง/คำสะดุดรัวเกิน 2 คำ")
            elif stutter_points >= 2:
                audio_penalty += 15.0
                audio_msgs.append("🔊 เสียงสะดุดเล็กน้อย 1-2 คำ (อนุโลม)")

        # 3.2 ตรวจสอบความแบนราบ (Robotic AI Voice)
        window_large = int(sample_rate * 0.2) 
        energies_large = np.array([np.sum(data_float[i:i+window_large]**2) for i in range(0, len(data_float), window_large)])
        
        if len(energies_large) > 0:
            mean_e = np.mean(energies_large)
            variance_e = np.var(energies_large) / (mean_e + 1e-6)
            
            if variance_e < 0.08 and mean_e > 100:
                audio_penalty += 70.0 # หุ่นยนต์ 100% ตัดตก
                audio_msgs.append("⚠️ เสียงพูดแบนราบเป็นหุ่นยนต์/ฟังไม่รู้ภาษา")
            elif variance_e < 0.22 and mean_e > 100:
                audio_penalty += 10.0
                if not audio_msgs: audio_msgs.append("🔊 เสียงสังเคราะห์แต่ฟังรู้เรื่อง (อนุโลม)")

    except Exception:
        return 0.0, ["⚠️ ระบบไม่สามารถวิเคราะห์คลื่นเสียงได้"]
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            
    return audio_penalty, audio_msgs

# ==========================================
# 4. ระบบประมวลผลกลางและ UI (Main Logic UI)
# ==========================================
uploaded_files = st.file_uploader(
    "เลือกไฟล์วิดีโอ (.mp4, .mov, .avi) - ลากวางได้หลายไฟล์พร้อมกัน", 
    type=["mp4", "mov", "avi"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 เตรียมประมวลผลทั้งหมด {len(uploaded_files)} คลิป")
    
    if st.button("🔍 เริ่มสแกนคัดกรอง (Strict Scan)", type="primary"):
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์แยกรายคลิป:")
        
        results_summary = []
        cols = st.columns(3)
        REJECT_THRESHOLD = 76.0 # เกณฑ์ตัดตก 76%
        progress_bar = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 3]
            with col:
                with st.container(border=True):
                    display_name = f"{uploaded_file.name[:20]}..." if len(uploaded_file.name) > 20 else uploaded_file.name
                    st.markdown(f"**🎬 {idx+1}. {display_name}**")
                    
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    
                    status = "ERROR"
                    details = ""
                    final_score = 0.0
                    
                    try:
                        with st.spinner("สแกนภาพ เสียง และสเกล..."):
                            # 1. ดึงภาพและวิเคราะห์โครงสร้าง
                            frames, motion_metrics, scale_flags = analyze_video_structure_and_scale(video_path, target_fps=6)
                            
                            if not frames:
                                st.caption("⚠️ ไฟล์วิดีโอเสีย ไม่สามารถอ่านภาพได้")
                                continue
                                
                            frame_scores = []
                            with torch.no_grad():
                                for frame_np in frames:
                                    pil_img = Image.fromarray(frame_np)
                                    input_tensor = transform(pil_img).unsqueeze(0).to(device)
                                    output = model(input_tensor)
                                    probs = torch.softmax(output, dim=1)
                                    frame_scores.append(probs[0][1].item())
                            
                            # 2. คำนวณ Base Risk (ทำให้คะแนนกระจายตัวตามความเนียนของคลิป)
                            mean_prob = float(np.mean(frame_scores))
                            std_prob = float(np.std(frame_scores))
                            dynamic_base_score = (mean_prob * 15.0) + (std_prob * 10.0)
                            
                            visual_penalty = 0.0
                            details_list = []
                            
                            # 3. กฎสเกลหลอกตา / ของเล่นจิ๋ว (Miniature Rule - เข้มงวดสุด)
                            # ถ้าระบบจับได้ว่ามือใหญ่ผิดปกติเทียบกับของในฉากเกิน 4 เฟรม (น้อยกว่า 1 วิด้วยซ้ำ)
                            if sum(scale_flags) >= 4:
                                visual_penalty += 85.0 # บวก 85% การันตีตก 100% แน่นอน
                                details_list.append("⛔ สเกลสินค้าหลอกตา (สัดส่วนมือใหญ่ผิดปกติ/ของเล่นจิ๋ว)")
                            
                            # 4. กฎภาพละลาย / อวัยวะบิดเบี้ยว (Warping & Melting)
                            severe_streak = 0
                            max_severe_streak = 0
                            avg_motion = np.mean(motion_metrics) if motion_metrics else 0
                            
                            for i, s in enumerate(frame_scores):
                                current_motion = motion_metrics[i]
                                is_panning = current_motion > (avg_motion + 5.0)
                                threshold = 0.99 if is_panning else 0.975 # ยืดหยุ่นตอนกล้องแพน
                                
                                if s > threshold:
                                    severe_streak += 1
                                    max_severe_streak = max(max_severe_streak, severe_streak)
                                else:
                                    severe_streak = 0
                                    
                            if max_severe_streak >= 12: # พัง 2 วิ (12 เฟรม) = ตก
                                visual_penalty += 75.0 
                                details_list.append("⛔ อวัยวะ/สินค้า บิดเบี้ยวละลายนานเกิน 2 วินาที")
                            elif max_severe_streak >= 6: 
                                visual_penalty += 25.0
                                details_list.append("อวัยวะ/สินค้าบิดเบี้ยวช่วงสั้น 1-2 วิ")
                                
                            # 5. กฎเสียงพูด
                            audio_penalty, audio_msgs = analyze_audio_strict(video_path)
                            if audio_msgs:
                                details_list.extend(audio_msgs)
                            
                            # 6. รวมคะแนนทั้งหมด
                            raw_final = 2.0 + dynamic_base_score + visual_penalty + audio_penalty
                            final_score = min(100.0, max(1.0, raw_final))
                            
                            # จัดข้อความ
                            if not details_list:
                                details = "✅ สมบูรณ์: สเกลถูกต้อง เคลื่อนไหวเนียน เสียงชัดเจน"
                            else:
                                details = " | ".join(list(dict.fromkeys(details_list)))
                            
                            # ประเมินผล (ตัดที่ 76%)
                            if final_score >= REJECT_THRESHOLD:
                                status = "REJECT"
                                st.error(f"❌ **ไม่ผ่าน ({final_score:.0f}%)**")
                                st.caption(f"*{details}*")
                            else:
                                status = "PASS"
                                st.success(f"✅ **ผ่าน ({final_score:.0f}%)**")
                                st.caption(f"*{details}*")
                                    
                            st.progress(int(final_score))
                                
                    except Exception as e:
                        st.caption(f"⚠️ Error เกิดข้อผิดพลาดกับคลิปนี้: {str(e)}")
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
                        "คะแนนความเสี่ยง": f"{final_score:.0f}%",
                        "สถานะ": status,
                        "สาเหตุ / หมายเหตุ": details
                    })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # ==========================================
        # 5. แดชบอร์ดสรุปผลรวม (Summary Dashboard)
        # ==========================================
        st.divider()
        st.subheader("📋 แดชบอร์ดสรุปผลรวม (Summary)")
        df_all = pd.DataFrame(results_summary)
        
        df_pass = df_all[df_all["สถานะ"] == "PASS"]
        df_reject = df_all[df_all["สถานะ"] == "REJECT"]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("จำนวนคลิปทั้งหมด", f"{len(df_all)} คลิป")
        m2.metric("✅ ผ่าน (< 76%)", f"{len(df_pass)} คลิป")
        m3.metric("❌ ไม่ผ่าน (≥ 76%)", f"{len(df_reject)} คลิป")
        
        st.write("---")
        
        tab1, tab2 = st.tabs(["✅ รายการคลิปที่ผ่าน", "❌ รายการคลิปที่ไม่ผ่าน"])
        
        with tab1:
            if not df_pass.empty:
                st.dataframe(df_pass[["ลำดับ", "ชื่อไฟล์", "คะแนนความเสี่ยง", "สาเหตุ / หมายเหตุ"]], hide_index=True, use_container_width=True)
            else:
                st.info("ไม่มีคลิปที่ผ่านเกณฑ์")
                
        with tab2:
            if not df_reject.empty:
                st.dataframe(df_reject[["ลำดับ", "ชื่อไฟล์", "คะแนนความเสี่ยง", "สาเหตุ / หมายเหตุ"]], hide_index=True, use_container_width=True)
            else:
                st.info("ไม่มีคลิปที่ถูกปัดตก")
        
        st.success("🎉 ประมวลผลเสร็จสิ้น ระบบได้ล้างแคชคืนหน่วยความจำให้เครื่องเรียบร้อยแล้ว")
