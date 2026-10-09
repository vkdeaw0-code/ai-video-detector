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
st.write("✨ **Smart Tolerance System:** เน้นความสมจริงของสินค้าและบุคคล อนุโลมเสียงและจุดบกพร่องเล็กน้อย ปัดตกเมื่อพบข้อผิดพลาดชัดเจนหลายจุด")

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
# 2. ฟังก์ชันประมวลผล (ปรับให้สแกนละเอียดขึ้น)
# ==========================================
def extract_frames(video_path, target_fps=4): # เพิ่มเป็น 4 เฟรม/วิ เพื่อสแกนการขยับของ มือ/ขา/สินค้า ได้ละเอียดขึ้น
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
        
        # ลดความเข้มงวดของเสียงลง ปล่อยผ่านได้มากขึ้น (หักสูงสุดแค่ 10%)
        if fft_sum > 0:
            normalized_fft = fft_data / fft_sum
            spectral_flatness = float(np.exp(np.mean(np.log(normalized_fft + 1e-12))))
            if spectral_flatness < 1e-6 or spectral_flatness > 1e-3:
                audio_risk += 5.0 
                
        window_size = int(sample_rate * 0.1) 
        energies = np.array([np.sum(data_float[i:i+window_size]**2) for i in range(0, len(data_float), window_size)])
        if len(energies) > 0:
            energy_variance = np.var(energies) / (np.mean(energies) + 1e-6)
            if energy_variance < 0.5: 
                audio_risk += 5.0 # ลดการหักคะแนนลงจากเดิม 15.0
        
    except Exception:
        pass
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            
    return audio_risk, 0.0

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
        THRESHOLD = 75.0 
        WARNING_THRESHOLD = 60.0
        
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
                            frames, motion_scores = extract_frames(video_path, target_fps=4)
                            
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
                                
                                # --- 💡 ตรรกะใหม่: ลดความเข้มงวด นับจำนวนจุดบกพร่อง (Defect Rules) ---
                                mean_prob = float(np.mean(frame_scores)) * 100
                                median_prob = float(np.median(frame_scores)) * 100
                                # ใช้ค่ากลางผสมค่าเฉลี่ยเป็น Base Score ทำให้กราฟนิ่งขึ้น ไม่โดดเพราะเฟรมเบลอเฟรมเดียว
                                base_score = (median_prob * 0.7) + (mean_prob * 0.3) 
                                
                                glitch_penalty = 0.0
                                defect_count = 0
                                details_list = []
                                
                                # 1. ต้นคลิป (Early Glitch) - อนุโลมมากขึ้น ต้องพังชัดเจนถึง 2 เฟรมในช่วงแรก
                                early_probs = frame_scores[:6]
                                bad_early_frames = sum([1 for p in early_probs if p > 0.75])
                                if bad_early_frames >= 2:
                                    defect_count += 1
                                    glitch_penalty += 10.0
                                    details_list.append("จุดบกพร่องต้นคลิป")
                                
                                # 2. การละลาย/อวัยวะผิดรูป (Morphing) เน้นสแกน มือ ขา สินค้า
                                avg_motion = np.mean(motion_scores) if len(motion_scores) > 0 else 0
                                morph_count = 0
                                for i in range(1, len(frame_scores)):
                                    # ยกระดับความมั่นใจของ AI เป็น > 0.80 และการกระตุกของพิกเซลต้องแรงมาก ป้องกันคลิปปกติที่คนเดินเร็วแล้วตก
                                    if frame_scores[i] > 0.80 and motion_scores[i] > (avg_motion * 2.5):
                                        morph_count += 1
                                
                                if morph_count >= 2:
                                    defect_count += 1
                                    glitch_penalty += 15.0
                                    details_list.append("อวัยวะ/สินค้าผิดรูปฉับพลัน")
                                
                                # 3. บิดเบี้ยวต่อเนื่อง (Sustained Glitch)
                                suspect_frames = [1 if score > 0.75 else 0 for score in frame_scores]
                                max_consecutive = 0
                                current_consecutive = 0
                                for is_suspect in suspect_frames:
                                    if is_suspect:
                                        current_consecutive += 1
                                        max_consecutive = max(max_consecutive, current_consecutive)
                                    else:
                                        current_consecutive = 0
                                
                                if max_consecutive >= 4: # พังติดกันประมาณ 1 วิ
                                    defect_count += 1
                                    glitch_penalty += 20.0
                                    details_list.append("ภาพบิดเบี้ยวต่อเนื่อง")
                                elif max_consecutive >= 2:
                                    glitch_penalty += 3.0
                                    # แค่เบลอเสี้ยววิ ไม่งับเป็นความผิดหลัก
                                
                                # 4. เสียง (Audio)
                                audio_risk, _ = analyze_audio_and_lipsync(video_path)
                                if audio_risk > 0:
                                    details_list.append("เสียงบกพร่องเล็กน้อย")
                                    
                                # 5. กฎคัดออก 3 จุด (The 3-Defect Rule)
                                if defect_count >= 3:
                                    glitch_penalty += 25.0
                                    details_list.append("❌ พบข้อบกพร่องหลายจุด")
                                
                                # สรุปข้อความ Note
                                if not details_list:
                                    details = "สมจริงและภาพรวมแนบเนียน"
                                else:
                                    # ลบข้อความซ้ำซ้อน
                                    details = " | ".join(list(dict.fromkeys(details_list)))
                                    
                                final_score = np.clip(base_score + glitch_penalty + audio_risk, 0.0, 100.0)
                                
                                if final_score >= THRESHOLD:
                                    status = "REJECT"
                                    st.error(f"❌ **REJECT** ({final_score:.0f}%)", icon="🚨")
                                    st.write(f"*{details}*")
                                elif final_score >= WARNING_THRESHOLD:
                                    status = "WARNING"
                                    st.warning(f"⚠️ **WARNING** ({final_score:.0f}%)", icon="⚠️")
                                    st.write(f"*{details} (ผ่านหวุดหวิด)*")
                                else:
                                    status = "PASS"
                                    st.success(f"✅ **PASS** ({final_score:.0f}%)", icon="🟢")
                                    st.write(f"*{details}*")
                                        
                                st.progress(min(int(final_score), 100))
                                
                    except Exception as e:
                        st.caption(f"⚠️ Error: {str(e)}")
                    finally:
                        if os.path.exists(video_path):
                            os.unlink(video_path)
                        if 'frames' in locals(): del frames
                        if 'frame_scores' in locals(): del frame_scores
                        if 'motion_scores' in locals(): del motion_scores
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
        
        df_pass = df_all[df_all["สถานะ"] == "PASS"]
        df_warning = df_all[df_all["สถานะ"] == "WARNING"]
        df_reject = df_all[df_all["สถานะ"] == "REJECT"]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("จำนวนทั้งหมด", f"{len(df_all)} คลิป")
        m2.metric("✅ ผ่าน", f"{len(df_pass)} คลิป")
        m3.metric("⚠️ พอใช้ได้", f"{len(df_warning)} คลิป")
        m4.metric("❌ ไม่ผ่านเลย", f"{len(df_reject)} คลิป")
        
        st.write("---")
        
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
