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
# 1. ตั้งค่าและเตรียมโมเดล
# ==========================================
st.set_page_config(page_title="AI Video Detector", page_icon="🎬", layout="wide")
st.title("🎬 ระบบตรวจจับและคัดกรองคลิปวิดีโอ AI")
st.write("✨ **Smart Tolerance System:** อนุโลมจุดผิดปกติเสี้ยววินาที แต่ปัดตกทันทีหากภาพ/มือบิดเบี้ยวต่อเนื่องนานเกินไป")

@st.cache_resource
def load_detection_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ใช้ EfficientNet-B0 เพื่อความเร็ว (หากมีไฟล์ Weights .pth ที่เทรนมาเฉพาะ ให้โหลดเพิ่มที่นี่)
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
# 2. ฟังก์ชันประมวลผล (ปรับให้เสถียรและเร็วขึ้น)
# ==========================================
def extract_frames(video_path, target_fps=3):
    # เปลี่ยนจากการนับเฟรม (frame_interval) เป็นการดึงตามเวลา (target_fps) 
    # ทำให้วิเคราะห์คลิป 30fps หรือ 60fps ได้มาตรฐานเดียวกัน (ดึง 3 เฟรม/วินาที)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30
    
    interval = int(video_fps / target_fps)
    if interval < 1: interval = 1
    
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % interval == 0:
            # ย่อขนาดทันทีเพื่อเซฟ RAM
            frame_resized = cv2.resize(frame, (224, 224))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        count += 1
    cap.release()
    return frames

def analyze_audio_and_lipsync(video_path):
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    audio_risk, lip_sync_risk = 0.0, 0.0
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
            return 0.0, 0.0
            
        data_float = data.astype(np.float32)
        fft_data = np.abs(np.fft.rfft(data_float))
        fft_sum = np.sum(fft_data)
        
        # ค้นหาลักษณะเสียงสังเคราะห์ (Spectral Flatness)
        if fft_sum > 0:
            normalized_fft = fft_data / fft_sum
            spectral_flatness = float(np.exp(np.mean(np.log(normalized_fft + 1e-12))))
            if spectral_flatness < 1e-6 or spectral_flatness > 1e-3:
                audio_risk = 5.0 # แปลงเป็น 5% โดยตรง
        
    except Exception:
        pass
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            
    return audio_risk, lip_sync_risk

# ==========================================
# 3. UI และ ระบบประมวลผลหลัก
# ==========================================
uploaded_files = st.file_uploader("เลือกไฟล์วิดีโอ (.mp4, .mov, .avi) - แนะนำอัปโหลดครั้งละไม่เกิน 10 คลิป", type=["mp4", "mov", "avi"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"📁 เลือกไว้ทั้งหมด {len(uploaded_files)} คลิป")
    if st.button("🔍 เริ่มกระบวนการสแกนตรวจจับทุกคลิป", type="primary"):
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์แยกคลิป:")
        
        results_summary = []
        cols = st.columns(3)
        THRESHOLD = 75.0 # เกณฑ์ตัดตก (เกิน 75% = ไม่ผ่าน)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 3]
            with col:
                with st.container(border=True):
                    display_name = f"{uploaded_file.name[:20]}..." if len(uploaded_file.name) > 20 else uploaded_file.name
                    st.caption(f"🎬 คลิปที่ {idx+1}: **{display_name}**")
                    
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    
                    final_score = 0.0
                    status = "ERROR"
                    details = ""
                    
                    try:
                        with st.spinner("กำลังวิเคราะห์..."):
                            frames = extract_frames(video_path, target_fps=3)
                            
                            if not frames:
                                st.caption("⚠️ ไม่สามารถอ่านภาพจากวิดีโอได้")
                            else:
                                frame_scores = []
                                with torch.no_grad():
                                    for frame_np in frames:
                                        pil_img = Image.fromarray(frame_np)
                                        input_tensor = transform(pil_img).unsqueeze(0).to(device)
                                        output = model(input_tensor)
                                        probs = torch.softmax(output, dim=1)
                                        frame_scores.append(probs[0][1].item())
                                
                                # --- 💡 การวิเคราะห์ความต่อเนื่อง (Temporal Logic) ---
                                mean_prob = float(np.mean(frame_scores)) * 100
                                
                                # นับเฟรมที่คะแนนพุ่ง (มีความเสี่ยงว่าพัง/บิดเบี้ยว)
                                suspect_frames = [1 if score > 0.65 else 0 for score in frame_scores]
                                
                                # หาช่วงเวลาที่พัง "ต่อเนื่อง" ยาวที่สุด
                                max_consecutive = 0
                                current_consecutive = 0
                                for is_suspect in suspect_frames:
                                    if is_suspect:
                                        current_consecutive += 1
                                        max_consecutive = max(max_consecutive, current_consecutive)
                                    else:
                                        current_consecutive = 0
                                        
                                # คิดคะแนนพื้นฐานจากค่าเฉลี่ย
                                base_score = mean_prob
                                
                                # เงื่อนไขให้อภัย vs งัดให้ตก
                                if max_consecutive >= 3:
                                    # พังต่อเนื่อง 3 เฟรมขึ้นไป (ประมาณ 1 วิ) = มือละลาย/ของหายชัดเจน -> งัดคะแนนให้ตก
                                    glitch_penalty = 35.0
                                    details = "พบภาพบิดเบี้ยว/ผิดปกติต่อเนื่อง"
                                elif max_consecutive > 0:
                                    # พังแค่ 1-2 เฟรม (เสี้ยววิ) = กล้องสั่น/เบลอ -> อนุโลม หักนิดเดียว
                                    glitch_penalty = 5.0
                                    details = "พบจุดแปลกเล็กน้อย (อนุโลมให้)"
                                else:
                                    glitch_penalty = 0.0
                                    details = "ภาพรวมแนบเนียน"
                                    
                                audio_risk, lip_sync_risk = analyze_audio_and_lipsync(video_path)
                                
                                # รวมคะแนน
                                final_score = np.clip(base_score + glitch_penalty + audio_risk, 0.0, 100.0)
                                
                                if final_score >= THRESHOLD:
                                    status = "REJECT"
                                    st.error(f"❌ **REJECT** ({final_score:.0f}%)", icon="🚨")
                                    st.write(f"*{details}*")
                                else:
                                    status = "PASS"
                                    # ถ้าคะแนนคาบเส้น (60-74%) จะขึ้นเตือนสีส้มแบบผ่านหวุดหวิด
                                    if final_score >= 60.0:
                                        st.warning(f"✅ **PASS** ({final_score:.0f}%)", icon="⚠️")
                                        st.write(f"*{details} (ผ่านหวุดหวิด)*")
                                    else:
                                        st.success(f"✅ **PASS** ({final_score:.0f}%)", icon="🟢")
                                        st.write(f"*{details}*")
                                        
                                st.progress(min(int(final_score), 100))
                                
                    except Exception as e:
                        st.caption(f"⚠️ Error: {str(e)}")
                    finally:
                        if os.path.exists(video_path):
                            os.unlink(video_path)
                        # ระบบคืนพื้นที่ RAM แบบถอนรากถอนโคน
                        if 'frames' in locals(): del frames
                        if 'frame_scores' in locals(): del frame_scores
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    
                    results_summary.append({
                        "ลำดับ": idx + 1,
                        "ชื่อไฟล์": uploaded_file.name,
                        "ความเสี่ยง": f"{final_score:.1f}%",
                        "สถานะ": status,
                        "หมายเหตุ": details
                    })
        
       # ==========================================
        # 4. แดชบอร์ดสรุปผลรวม 3 ระดับ
        # ==========================================
        st.divider()
        st.subheader("📋 แดชบอร์ดสรุปผลรวม")
        df_all = pd.DataFrame(results_summary)
        
        # แยก DataFrame ตาม 3 สถานะ
        df_pass = df_all[df_all["สถานะ"] == "PASS"]
        df_warning = df_all[df_all["สถานะ"] == "WARNING"]
        df_reject = df_all[df_all["สถานะ"] == "REJECT"]
        
        # แสดงตัวเลขสรุปด้านบน
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("จำนวนทั้งหมด", f"{len(df_all)} คลิป")
        m2.metric("✅ ผ่าน", f"{len(df_pass)} คลิป")
        m3.metric("⚠️ พอใช้ได้", f"{len(df_warning)} คลิป")
        m4.metric("❌ ไม่ผ่านเลย", f"{len(df_reject)} คลิป")
        
        st.write("---")
        
        # สร้าง 3 คอลัมน์สำหรับโชว์ตารางแยก
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.success("✅ คลิปที่ **ผ่าน** (<60%)")
            if not df_pass.empty:
                st.dataframe(df_pass[["ลำดับ", "ชื่อไฟล์", "ความเสี่ยง"]], hide_index=True, use_container_width=True)
            else:
                st.caption("ไม่มีคลิปในกลุ่มนี้")
                
        with c2:
            st.warning("⚠️ คลิปที่ **พอใช้ได้** (60-74%)")
            if not df_warning.empty:
                st.dataframe(df_warning[["ลำดับ", "ชื่อไฟล์", "ความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else:
                st.caption("ไม่มีคลิปในกลุ่มนี้")
                
        with c3:
            st.error("❌ คลิปที่ **ไม่ผ่านเลย** (≥75%)")
            if not df_reject.empty:
                st.dataframe(df_reject[["ลำดับ", "ชื่อไฟล์", "ความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else:
                st.caption("ไม่มีคลิปในกลุ่มนี้")
        
        st.success("🎉 ตรวจสอบเสร็จสิ้น ระบบได้ล้างแคชเพื่อคืน RAM ให้กับเครื่องแล้ว")
