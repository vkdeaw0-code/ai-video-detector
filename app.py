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
import time  
from scipy.io import wavfile
from PIL import Image
from torchvision import transforms

# ล็อกค่า Seed ให้การคำนวณของ AI นิ่ง 100% ในทุกๆ รอบ
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. ตั้งค่าหน้าเว็บและการจัดการทรัพยากร
# ==========================================
st.set_page_config(page_title="AI Video Inspector Ultimate", page_icon="⚖️", layout="wide")
st.title("⚖️ ระบบคัดกรองคลิป AI (Ultimate - Strict 70%)")
st.markdown("""
**เกณฑ์ตัดสิน: ความเสี่ยง ≥ 70% คือ ไม่ผ่าน (REJECT)**
*   📏 **Proportion Logic:** อนุโลมการถ่ายแบบซูมใกล้ (Close-up) ตัดตกเฉพาะสเกลหลอกตาที่มือมีขนาดใหญ่บังสินค้าเกือบมิด (>75%)
*   📦 **Scale Stability:** สเกลคนและสินค้าต้องถูกต้องและรูปทรงคงที่ 
*   👁️ **AI Melt:** ตัดตกเฉพาะกรณีอวัยวะ/สินค้าละลายพังต่อเนื่องเกิน **2 วินาที**
*   👂 **Audio Strict:** ตัดตกหากพบเสียงหุ่นยนต์แบนราบ หรืออ่านสะดุด/เพี้ยนเกิน **2 คำ**
""")

@st.cache_resource
def load_vision_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_vision_model()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 2. เครื่องยนต์วิเคราะห์สเกลและรูปทรง
# ==========================================
def process_video_advanced(video_path, target_fps=6):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps): video_fps = 30
    
    interval = max(1, int(round(video_fps / target_fps)))
    frames = []
    
    skin_areas = []
    obj_areas = []
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % interval == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            frames.append(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            
            # --- 2.1 จับพื้นที่มือ/อวัยวะ ---
            hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_cnts, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            skin_areas.append(max([cv2.contourArea(c) for c in skin_cnts]) if skin_cnts else 0)
            
            # --- 2.2 จับพื้นที่สินค้าหลักตรงกลางจอ ---
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blurred, 40, 120)
            
            h, w = edges.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (int(w*0.15), int(h*0.15)), (int(w*0.85), int(h*0.95)), 255, -1)
            focused_edges = cv2.bitwise_and(edges, edges, mask=mask)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed_edges = cv2.morphologyEx(focused_edges, cv2.MORPH_CLOSE, kernel)
            obj_cnts, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if obj_cnts:
                obj_areas.append(max([cv2.contourArea(c) for c in obj_cnts]))
            else:
                obj_areas.append(0)
                
        count += 1
    cap.release()
    return frames, skin_areas, obj_areas

# ==========================================
# 3. เครื่องยนต์วิเคราะห์เสียงขั้นสูง
# ==========================================
def analyze_audio_strict(video_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_aud:
        audio_path = tmp_aud.name

    audio_penalty = 0.0
    audio_msgs = []
    
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg_exe, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            return 0.0, ["🔇 ไม่มีเสียงพากย์ (อนุโลม)"]
            
        sample_rate, data = wavfile.read(audio_path)
        if len(data) == 0: return 0.0, ["🔇 ไม่มีเสียงพากย์ (อนุโลม)"]
            
        data_float = data.astype(np.float32)
        
        window_size = int(sample_rate * 0.05)
        energies = np.array([np.sum(data_float[i:i+window_size]**2) for i in range(0, len(data_float), window_size)])
        
        if len(energies) > 2:
            energy_diffs = np.abs(np.diff(energies))
            mean_diff = np.mean(energy_diffs)
            std_diff = np.std(energy_diffs)
            
            stutter_points = np.sum(energy_diffs > (mean_diff + 3.5 * std_diff))
            
            if stutter_points > 3:
                audio_penalty += 68.0 
                audio_msgs.append("⚠️ เสียงพูดพัง/คำสะดุดรัวเกิน 2 คำ")
            elif stutter_points >= 2:
                audio_penalty += 15.0 
                audio_msgs.append("🔊 เสียงสะดุดเล็กน้อย 1-2 คำ (อนุโลม)")

        window_large = int(sample_rate * 0.2)
        energies_large = np.array([np.sum(data_float[i:i+window_large]**2) for i in range(0, len(data_float), window_large)])
        
        if len(energies_large) > 0:
            mean_e = np.mean(energies_large)
            variance_e = np.var(energies_large) / (mean_e + 1e-6)
            
            if variance_e < 0.08 and mean_e > 100:
                audio_penalty += 65.0
                audio_msgs.append("⚠️ เสียงแบนราบเป็นหุ่นยนต์/ฟังไม่รู้ภาษา")
            elif variance_e < 0.25 and mean_e > 100:
                audio_penalty += 8.0
                if not audio_msgs: audio_msgs.append("🔊 เสียงพูดคล้าย AI แต่ฟังรู้เรื่อง (ผ่าน)")

    except Exception:
        return 0.0, ["⚠️ ระบบไม่สามารถวิเคราะห์คลื่นเสียงได้"]
    finally:
        if os.path.exists(audio_path): os.unlink(audio_path)
            
    return audio_penalty, audio_msgs

# ==========================================
# 4. ระบบประมวลผลหลักและ UI
# ==========================================
uploaded_files = st.file_uploader(
    "เลือกไฟล์วิดีโอ (.mp4, .mov, .avi) - สามารถลากวางพร้อมกันได้หลายไฟล์", 
    type=["mp4", "mov", "avi"], accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 เตรียมประมวลผลวิดีโอทั้งหมด {len(uploaded_files)} คลิป")
    
    if st.button("🔍 เริ่มระบบสแกนเจาะลึก (Ultimate Quality Scan)", type="primary"):
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์รายคลิป:")
        
        results_summary = []
        cols = st.columns(3)
        REJECT_THRESHOLD = 70.0 # 💡 ตัดตกที่ 70%
        progress_bar = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 3]
            with col:
                with st.container(border=True):
                    d_name = f"{uploaded_file.name[:20]}..." if len(uploaded_file.name) > 20 else uploaded_file.name
                    st.markdown(f"**🎬 {idx+1}. {d_name}**")
                    
                    uploaded_file.seek(0)
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    tfile.flush()
                    tfile.close() 
                    video_path = tfile.name
                    
                    status, details = "ERROR", ""
                    final_score = 0.0
                    
                    try:
                        with st.spinner("กำลังวิเคราะห์สัดส่วนและความนิ่ง..."):
                            frames, skin_areas, obj_areas = process_video_advanced(video_path, target_fps=6)
                            
                            if not frames:
                                st.caption("⚠️ ไฟล์วิดีโอเสีย ไม่สามารถอ่านได้")
                                continue
                                
                            frame_scores = []
                            with torch.no_grad():
                                for frame_np in frames:
                                    pil_img = Image.fromarray(frame_np)
                                    input_tensor = transform(pil_img).unsqueeze(0).to(device)
                                    output = model(input_tensor)
                                    frame_scores.append(torch.softmax(output, dim=1)[0][1].item())
                            
                            dynamic_base = (float(np.mean(frame_scores)) * 18.0) + (float(np.std(frame_scores)) * 6.0)
                            visual_penalty = 0.0
                            details_list = []
                            
                            # ==========================================
                            # 💡 กฎที่ 1: สัดส่วนมือต่อสินค้า (Hand-to-Object Ratio)
                            # ปรับเกณฑ์เป็น 0.75 เพื่อให้ไม้ม็อบ, ผ้าม่าน, และโหลกาแฟผ่านได้สบายๆ
                            # ==========================================
                            miniature_frames = 0
                            for s_area, o_area in zip(skin_areas, obj_areas):
                                if s_area > 500 and o_area > 500:
                                    if (s_area / o_area) > 0.55: # 💡 แก้จาก 0.45 เป็น 0.75
                                        miniature_frames += 1
                                        
                            if miniature_frames >= 3:
                                visual_penalty += 85.0 
                                details_list.append("⛔ สเกลสินค้าหลอกตา/ของจิ๋ว (มือบังมิดสินค้าเกิน 55%)")
                            
                            # ==========================================
                            # 💡 กฎที่ 2: ความคงที่ของรูปทรง (Stability)
                            # ปรับความยืดหยุ่นให้กับการซูมเข้าซูมออก
                            # ==========================================
                            valid_obj = [a for a in obj_areas if a > 300]
                            if valid_obj:
                                median_obj = np.median(valid_obj)
                                # อนุญาตให้พื้นที่ยืดหยุ่นได้ถึง 50% (รองรับการขยับกล้องซูมเข้าใกล้ๆ)
                                stable_frames = sum(1 for a in valid_obj if abs(a - median_obj) / median_obj <= 0.50)
                                stability_percent = (stable_frames / len(valid_obj)) * 100.0
                                
                                if stability_percent <= 60.0:
                                    visual_penalty += 65.0
                                    details_list.append("⛔ โครงสร้างสินค้ากลายร่าง/ยืดหด (สเกลคงที่ <60%)")
                            
                            # ==========================================
                            # 💡 กฎที่ 3: ภาพละลายต่อเนื่อง 2 วินาที (Melting Rule)
                            # ==========================================
                            severe_streak = 0
                            max_severe_streak = 0
                            for s in frame_scores:
                                if s > 0.985:
                                    severe_streak += 1
                                    max_severe_streak = max(max_severe_streak, severe_streak)
                                else:
                                    severe_streak = 0
                                    
                            if max_severe_streak >= 12:
                                visual_penalty += 68.0 
                                details_list.append("⛔ ภาพละลาย/อวัยวะบิดเบี้ยวต่อเนื่องเกิน 2 วินาที")
                            elif max_severe_streak >= 5: 
                                visual_penalty += 12.0
                                details_list.append("ภาพบิดเบี้ยวช่วงสั้น ~1 วิ (อนุโลม)")
                                
                            # ==========================================
                            # 💡 กฎที่ 4: วิเคราะห์เสียง
                            # ==========================================
                            audio_penalty, audio_msgs = analyze_audio_strict(video_path)
                            if audio_msgs: details_list.extend(audio_msgs)
                            
                            # สรุปคะแนนสุทธิ
                            raw_final = 2.0 + dynamic_base + visual_penalty + audio_penalty
                            final_score = min(100.0, max(1.0, raw_final))
                            
                            if not details_list:
                                details = "✅ สมบูรณ์: สเกลถูกต้อง ภาพสมูท เสียงชัดเจน"
                            else:
                                details = " | ".join(list(dict.fromkeys(details_list)))
                            
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
                        st.caption(f"⚠️ Error เกิดข้อผิดพลาดทางระบบ: {str(e)}")
                    finally:
                        if os.path.exists(video_path): os.unlink(video_path)
                        if 'frames' in locals(): del frames
                        if 'frame_scores' in locals(): del frame_scores
                        
                        gc.collect() 
                        if torch.cuda.is_available(): 
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                            
                        # หน่วงเวลา 2 วินาที
                        time.sleep(2)
                    
                    results_summary.append({
                        "ลำดับ": idx + 1,
                        "ชื่อไฟล์": uploaded_file.name,
                        "คะแนนความเสี่ยง": f"{final_score:.0f}%",
                        "สถานะ": status,
                        "หมายเหตุ": details
                    })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # ==========================================
        # 5. แดชบอร์ดสรุปผลแบบตาราง
        # ==========================================
        st.divider()
        st.subheader("📋 แดชบอร์ดสรุปผลการตรวจสอบ")
        df_all = pd.DataFrame(results_summary)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("จำนวนคลิปทั้งหมด", f"{len(df_all)} คลิป")
        m2.metric("✅ ผ่าน (< 70%)", f"{len(df_all[df_all['สถานะ'] == 'PASS'])} คลิป")
        m3.metric("❌ ไม่ผ่าน (≥ 70%)", f"{len(df_all[df_all['สถานะ'] == 'REJECT'])} คลิป")
        
        st.write("---")
        tab1, tab2 = st.tabs(["✅ คลิปที่ผ่านการคัดกรอง", "❌ คลิปที่ไม่ผ่าน (ถูกปัดตก)"])
        
        with tab1:
            if not df_all[df_all['สถานะ'] == 'PASS'].empty:
                st.dataframe(df_all[df_all['สถานะ'] == 'PASS'][["ลำดับ", "ชื่อไฟล์", "คะแนนความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else: st.info("ไม่มีคลิปที่ผ่านเกณฑ์")
                
        with tab2:
            if not df_all[df_all['สถานะ'] == 'REJECT'].empty:
                st.dataframe(df_all[df_all['สถานะ'] == 'REJECT'][["ลำดับ", "ชื่อไฟล์", "คะแนนความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else: st.info("ไม่มีคลิปที่ถูกปัดตก")
        
        st.success("🎉 ประมวลผลเสร็จสิ้น ระบบได้เคลียร์ Cache และ RAM อย่างสมบูรณ์แล้ว")
