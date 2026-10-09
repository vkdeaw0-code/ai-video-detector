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
st.title("🎬 ระบบประเมินคุณภาพคลิปวิดีโอ AI")
st.write("ระบบวิเคราะห์ภาพรวม (0-100%): ประเมินผล 2 ระดับ (ผ่าน/ไม่ผ่าน) ตัดตกที่ 75% **[เวอร์ชันอัลกอริทึมแยกส่วน] จับผิดการละลายของอวัยวะขั้นเด็ดขาด และปกป้องคลิปปกติไม่ให้ถูกปัดตก**")

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
# 2. ฟังก์ชันประมวลผล 
# ==========================================
def extract_frames(video_path, target_fps=8): 
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
    audio_msg = "🔊 เสียงพูดเป็นธรรมชาติ"
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            return 0.0, "🔇 ไม่มีเสียง"
            
        sample_rate, data = wavfile.read(audio_path)
        if len(data) == 0:
            return 0.0, "🔇 ไม่มีเสียง"
            
        data_float = data.astype(np.float32)
        fft_data = np.abs(np.fft.rfft(data_float))
        fft_sum = np.sum(fft_data)
        
        if fft_sum > 0:
            normalized_fft = fft_data / fft_sum
            spectral_flatness = float(np.exp(np.mean(np.log(normalized_fft + 1e-12))))
            if spectral_flatness < 1e-6 or spectral_flatness > 1e-3:
                audio_risk += 4.0 
                
        window_size = int(sample_rate * 0.1) 
        energies = np.array([np.sum(data_float[i:i+window_size]**2) for i in range(0, len(data_float), window_size)])
        if len(energies) > 0:
            energy_variance = np.var(energies) / (np.mean(energies) + 1e-6)
            if energy_variance < 0.4: 
                audio_risk += 6.0 
                
        if audio_risk >= 10.0:
            audio_msg = "🔊 เสียงพูดเพี้ยนมาก/ฟังไม่รู้เรื่อง"
        elif audio_risk > 0:
            audio_msg = "🔊 พบการใช้เสียงสังเคราะห์ (AI Voice)"
        
    except Exception:
        return 0.0, "⚠️ ไม่สามารถวิเคราะห์เสียงได้"
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            
    return audio_risk, audio_msg

# ==========================================
# 3. UI และ ระบบประมวลผลหลัก
# ==========================================
uploaded_files = st.file_uploader("เลือกไฟล์วิดีโอ (.mp4, .mov, .avi) - แนะนำอัปโหลดครั้งละไม่เกิน 10 คลิป", type=["mp4", "mov", "avi"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"📁 เลือกไว้ทั้งหมด {len(uploaded_files)} คลิป")
    if st.button("🔍 เริ่มสแกน", type="primary"):
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์แยกคลิป:")
        
        results_summary = []
        cols = st.columns(3)
        THRESHOLD = 75.0 
        
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
                        with st.spinner("กำลังสแกนวิเคราะห์..."):
                            frames, motion_scores = extract_frames(video_path, target_fps=8)
                            
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
                                
                                mean_prob = float(np.mean(frame_scores))
                                median_prob = float(np.median(frame_scores))
                                
                                # 💡 1. กดยอด Base Score: ปกป้องคลิปที่เนียนให้เริ่มต้นที่ 0-25% เท่านั้น
                                base_score = min(25.0, (median_prob * 15.0) + (mean_prob * 10.0))
                                
                                glitch_penalty = 0.0
                                details_list = []
                                
                                # 💡 2. ระบบ Critical Detection (จับมือหาย/ตัวละลาย โดยไม่ง้อ Motion)
                                # ถ้าโมเดลฟันธงว่าเป็น Glitch แน่ๆ (>95%) ติดกัน 3 เฟรม ฟาดให้ตกทันที
                                severe_thresh = 0.95
                                severe_consecutive = 0
                                max_severe = 0
                                for score in frame_scores:
                                    if score > severe_thresh:
                                        severe_consecutive += 1
                                        max_severe = max(max_severe, severe_consecutive)
                                    else:
                                        severe_consecutive = 0
                                        
                                if max_severe >= 3:
                                    glitch_penalty += 55.0 # +55% (รวมฐาน 25% กลายเป็น 80% ตกชัวร์)
                                    details_list.append("⚠️ อวัยวะหรือเงาสะท้อนละลาย/ผิดรูปชัดเจน (Critical)")
                                elif max_severe >= 2:
                                    glitch_penalty += 20.0
                                    details_list.append("⚠️ พบจุดภาพละลายสั้นๆ")

                                # 💡 3. ระบบจับภาพกระตุกฉับพลัน (Warping)
                                # กรณีคะแนนพุ่งกระโดด (>85%) ร่วมกับมีการขยับเร็ว
                                glitch_thresh = min(0.90, max(0.80, median_prob + 0.15))
                                avg_motion = np.mean(motion_scores) if len(motion_scores) > 0 else 0
                                warp_count = sum([1 for i in range(1, len(frame_scores)) if frame_scores[i] > glitch_thresh and motion_scores[i] > max(avg_motion * 3.0, 5.0)])
                                
                                if warp_count >= 2 and max_severe < 3: # ถ้าโดนจับ Critical ไปแล้วจะไม่ซ้ำเติมคะแนนตรงนี้
                                    glitch_penalty += 15.0
                                    details_list.append("⚠️ ภาพหรืออวัยวะกระตุกผิดธรรมชาติ")
                                
                                # 💡 4. ภาพละลายต่อเนื่องทั่วไป (Sustained Glitch)
                                suspect_consecutive = 0
                                max_suspect = 0
                                for score in frame_scores:
                                    if score > glitch_thresh:
                                        suspect_consecutive += 1
                                        max_suspect = max(max_suspect, suspect_consecutive)
                                    else:
                                        suspect_consecutive = 0
                                
                                if max_suspect >= 8 and max_severe < 3: # แช่นานเกิน 1 วิ
                                    glitch_penalty += 25.0
                                    details_list.append("⚠️ ภาพพื้นหลังละลายต่อเนื่อง")
                                
                                # 💡 5. ประเมินเสียง
                                audio_risk, audio_msg = analyze_audio_and_lipsync(video_path)
                                if audio_risk > 0:
                                    glitch_penalty += audio_risk
                                if audio_msg:
                                    details_list.append(audio_msg)
                                        
                                final_score = np.clip(base_score + glitch_penalty, 0.0, 100.0)
                                
                                # จัดการข้อความ
                                if not details_list or (len(details_list) == 1 and "พบการใช้เสียงสังเคราะห์" in details_list[0]):
                                    details = "วิดีโอสมบูรณ์ (อนุโลมการใช้ AI Voice)" if audio_risk > 0 else "วิดีโอสมจริง เป็นธรรมชาติ"
                                else:
                                    details = " | ".join(list(dict.fromkeys(details_list)))
                                
                                # ตัดเกรด
                                if final_score >= THRESHOLD:
                                    status = "REJECT"
                                    st.error(f"❌ **ไม่ผ่าน** ({final_score:.0f}%)")
                                    st.write(f"*{details}*")
                                else:
                                    status = "PASS"
                                    st.success(f"✅ **ผ่าน** ({final_score:.0f}%)")
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
        # 4. แดชบอร์ดสรุปผลรวม 2 ระดับ
        # ==========================================
        st.divider()
        st.subheader("📋 แดชบอร์ดสรุปผลรวม")
        df_all = pd.DataFrame(results_summary)
        
        df_pass = df_all[df_all["สถานะ"] == "PASS"]
        df_reject = df_all[df_all["สถานะ"] == "REJECT"]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("จำนวนทั้งหมด", f"{len(df_all)} คลิป")
        m2.metric("✅ ผ่าน", f"{len(df_pass)} คลิป")
        m3.metric("❌ ไม่ผ่าน", f"{len(df_reject)} คลิป")
        
        st.write("---")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.success("✅ คลิปที่ **ผ่าน** (<75%)")
            if not df_pass.empty:
                st.dataframe(df_pass[["ลำดับ", "ชื่อไฟล์", "ความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else:
                st.caption("ไม่มีคลิปในกลุ่มนี้")
                
        with c2:
            st.error("❌ คลิปที่ **ไม่ผ่าน** (≥75%)")
            if not df_reject.empty:
                st.dataframe(df_reject[["ลำดับ", "ชื่อไฟล์", "ความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else:
                st.caption("ไม่มีคลิปในกลุ่มนี้")
        
        st.success("🎉 ตรวจสอบเสร็จสิ้น ระบบได้ล้างแคชเพื่อคืน RAM ให้กับเครื่องแล้ว")
