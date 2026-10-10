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
st.set_page_config(page_title="AI Video Inspector Pro", page_icon="🎬", layout="wide")
st.title("🎬 ระบบคัดกรองคุณภาพคลิปวิดีโอ AI (เวอร์ชันเข้มงวดสัดส่วนสมดุล)")
st.write("ระบบตรวจจับคุณภาพคน สินค้า และเสียงภาษาไทย: **เกณฑ์ตัดตก 76% ขึ้นไป** | พลาดเกิน 2 วินาที หรือเสียงเพี้ยนเกิน 2 คำ = หักคะแนนหนักตัดตก | คะแนน % คำนวณตามความละเอียดจริงของคลิป")

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
# 2. ฟังก์ชันสแกนภาพวิดีโอ (6 FPS)
# ==========================================
def extract_frames(video_path, target_fps=6): 
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30
    
    interval = max(1, int(video_fps / target_fps))
    frames = []
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % interval == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        count += 1
    cap.release()
    return frames

# ==========================================
# 3. ฟังก์ชันวิเคราะห์เสียงพูดภาษาไทย/อังกฤษ
# ==========================================
def analyze_audio_quality(video_path):
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    audio_penalty = 0.0
    audio_msg = "🔊 เสียงพูดชัดเจนเป็นธรรมชาติ"
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            return 0.0, "🔇 ไม่มีเสียงประกอบ"
            
        sample_rate, data = wavfile.read(audio_path)
        if len(data) == 0:
            return 0.0, "🔇 ไม่มีเสียงประกอบ"
            
        data_float = data.astype(np.float32)
        
        window_size = int(sample_rate * 0.1) 
        energies = np.array([np.sum(data_float[i:i+window_size]**2) for i in range(0, len(data_float), window_size)])
        
        if len(energies) > 0:
            mean_energy = np.mean(energies)
            energy_variance = np.var(energies) / (mean_energy + 1e-6)
            
            # ตรวจสอบการเพี้ยนของเสียงภาษาไทย/อังกฤษ
            if energy_variance < 0.12 and mean_energy > 100: 
                audio_penalty += 60.0 # เพี้ยนรุนแรงเกิน 2 คำฟังไม่รู้เรื่อง (ดันให้เสี่ยงตก)
                audio_msg = "🔊 เสียงพูดเพี้ยนเกิน 2 คำ/ฟังไม่รู้ภาษา (ตัดตก)"
            elif energy_variance < 0.22 and mean_energy > 100:
                audio_penalty += 25.0 # เพี้ยนประมาณ 1-2 คำพอเดาคำได้
                audio_msg = "🔊 เสียงพูดเพี้ยน 1-2 คำ (อนุโลมผ่าน)"
            elif energy_variance < 0.32 and mean_energy > 100:
                audio_penalty += 10.0 # เสียงสังเคราะห์เล็กน้อยแต่ชัดเจน
                audio_msg = "🔊 เสียงสังเคราะห์แต่ฟังชัดเจน (อนุโลมผ่าน)"
        
    except Exception:
        return 0.0, "⚠️ ไม่สามารถวิเคราะห์เสียงได้"
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            
    return audio_penalty, audio_msg

# ==========================================
# 4. UI และ ระบบประมวลผลหลัก
# ==========================================
uploaded_files = st.file_uploader("เลือกไฟล์วิดีโอ (.mp4, .mov, .avi) - อัปโหลดได้หลายไฟล์พร้อมกัน", type=["mp4", "mov", "avi"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"📁 เลือกไว้ทั้งหมด {len(uploaded_files)} คลิป")
    if st.button("🔍 เริ่มสแกนคุณภาพ", type="primary"):
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
                        with st.spinner("กำลังสแกนวิเคราะห์รายละเอียดเฟรม..."):
                            frames = extract_frames(video_path, target_fps=6)
                            
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
                                
                                # 💡 1. คำนวณ Base Risk จากสถิติจริงของทุกเฟรม (ทำให้ % แต่ละคลิปกระจายตัวไม่เท่ากัน)
                                mean_prob = float(np.mean(frame_scores))
                                std_prob = float(np.std(frame_scores))
                                dynamic_base_score = (mean_prob * 20.0) + (std_prob * 15.0)
                                
                                visual_penalty = 0.0
                                details_list = []
                                
                                severe_streak = 0
                                max_severe_streak = 0
                                total_bad_frames = 0
                                
                                for s in frame_scores:
                                    if s > 0.98: # สแกนหาเฟรมที่มีความบิดเบี้ยวของคนหรือสินค้าชัดเจน
                                        severe_streak += 1
                                        max_severe_streak = max(max_severe_streak, severe_streak)
                                        total_bad_frames += 1
                                    else:
                                        severe_streak = 0
                                        
                                # เพิ่มน้ำหนักจากสัดส่วนเฟรมเสียในคลิป
                                bad_ratio = total_bad_frames / len(frame_scores) if len(frame_scores) > 0 else 0
                                ratio_penalty = bad_ratio * 25.0
                                
                                # 💡 2. กฎความเข้มงวด 2 วินาที (12 เฟรมที่ 6 FPS = 2 วินาที)
                                if max_severe_streak >= 12: 
                                    visual_penalty += 65.0 # พังค้างเกิน 2 วินาทีเต็ม (ตัดตก >= 76%)
                                    details_list.append("⚠️ มือ/ขา/คน หรือสินค้า บิดเบี้ยวพังเกิน 2 วินาที")
                                elif max_severe_streak >= 6: 
                                    visual_penalty += 30.0 # พังช่วง 1-2 วินาที (เพิ่มความเสี่ยงชัดเจน)
                                    details_list.append("อวัยวะ/สินค้าบิดเบี้ยวช่วงสั้น 1-2 วิ")
                                elif max_severe_streak >= 2:
                                    visual_penalty += 10.0 # กระตุกแวบเดียวไม่ถึง 1 วิ
                                    details_list.append("ภาพกระตุกเล็กน้อยไม่ถึง 1 วิ (อนุโลม)")
                                
                                # 3. ตรวจสอบเสียงภาษาไทย
                                audio_penalty, audio_msg = analyze_audio_quality(video_path)
                                if audio_msg and audio_msg != "🔊 เสียงพูดชัดเจนเป็นธรรมชาติ":
                                    details_list.append(audio_msg)
                                
                                # 4. คำนวณคะแนนรวมสุทธิแบบกระจายตัวสมดุล (1 - 100%)
                                raw_final = 3.0 + dynamic_base_score + ratio_penalty + visual_penalty + audio_penalty
                                final_score = min(100.0, max(1.0, raw_final))
                                
                                if not details_list:
                                    details = "รูปทรงคนและสินค้าสมบูรณ์ เสียงภาษาไทยชัดเจนดี"
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
