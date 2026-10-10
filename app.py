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

cv2.setNumThreads(4)
torch.set_num_threads(4)
torch.set_grad_enabled(False)

# ล็อกค่า Seed ให้การคำนวณของ AI นิ่ง 100% 
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. ตั้งค่าหน้าเว็บและการจัดการทรัพยากร
# ==========================================
st.set_page_config(page_title="AI Video Inspector Ultimate", page_icon="⚖️", layout="wide")
st.title("⚖️ ระบบคัดกรองคลิป AI (Context-Aware Scoring 71%)")
st.markdown("""
**เกณฑ์ตัดสิน: ความเสี่ยง ≥ 71% คือ ไม่ผ่าน (REJECT)**
*   🧠 **Context-Aware:** แยกโหมดอัตโนมัติระหว่างคลิป **"พรีเซนเตอร์ถือของ"** กับ **"เน้นสินค้า/ชี้ของจิ๋ว"**
*   🛍️ **Presenter Mode:** ถืออาหารสัตว์ผ่านฉลุย อนุโลมรอยยับและการแกว่งถุง
*   📦 **Strict Showcase Mode:** คลิปโชว์สินค้า/เห็นแต่มือ จะถูกคุมเข้มสเกล หากมือใหญ่เกินจริง หรือสินค้าบิดเบี้ยวจะถูกปัดตกทันที
*   ⚡ **Ultra-Fast Batch:** สแกนเร็วด้วย Multi-threading 
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
# 2. เครื่องยนต์วิเคราะห์แยกบริบทและสเกล
# ==========================================
def process_video_advanced(video_path, target_fps=6): 
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps): video_fps = 30
    
    interval = max(1, int(round(video_fps / target_fps)))
    frames = []
    skin_areas = []
    total_skin_areas = []
    skin_parts = []
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
            
            hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_cnts, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # เก็บข้อมูลพื้นที่ผิวเพื่อแยกโหมด (มีคนเต็มตัว vs มีแค่มือ)
            if skin_cnts:
                skin_areas.append(max([cv2.contourArea(c) for c in skin_cnts]))
                total_skin_areas.append(sum([cv2.contourArea(c) for c in skin_cnts]))
                skin_parts.append(sum(1 for c in skin_cnts if cv2.contourArea(c) > 500))
            else:
                skin_areas.append(0)
                total_skin_areas.append(0)
                skin_parts.append(0)
            
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
    return frames, skin_areas, total_skin_areas, skin_parts, obj_areas

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
        cmd = [ffmpeg_exe, "-y", "-threads", "4", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
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
                audio_penalty += 71.0 
                audio_msgs.append("⛔ เสียงพูดพัง/คำสะดุดรัวเกิน 2 คำ")
            elif stutter_points >= 2:
                audio_penalty += 15.0 
                audio_msgs.append("⚠️ เสียงสะดุดเล็กน้อย 1-2 คำ (หักคะแนน)")

        window_large = int(sample_rate * 0.2)
        energies_large = np.array([np.sum(data_float[i:i+window_large]**2) for i in range(0, len(data_float), window_large)])
        
        if len(energies_large) > 0:
            mean_e = np.mean(energies_large)
            variance_e = np.var(energies_large) / (mean_e + 1e-6)
            
            if variance_e < 0.08 and mean_e > 100:
                audio_penalty += 71.0
                audio_msgs.append("⛔ เสียงแบนราบเป็นหุ่นยนต์/ฟังไม่รู้ภาษา")
            elif variance_e < 0.25 and mean_e > 100:
                audio_penalty += 15.0
                if not audio_msgs: audio_msgs.append("⚠️ เสียงพูดคล้าย AI (หักคะแนน)")
    except Exception:
        return 0.0, ["⚠️ ระบบไม่สามารถวิเคราะห์คลื่นเสียงได้"]
    finally:
        if os.path.exists(audio_path): os.unlink(audio_path)
            
    return audio_penalty, audio_msgs

# ==========================================
# 4. ระบบประมวลผลหลัก
# ==========================================
uploaded_files = st.file_uploader(
    "เลือกไฟล์วิดีโอ (.mp4, .mov, .avi) - ลากวางพร้อมกันได้สูงสุด 50 คลิป", 
    type=["mp4", "mov", "avi"], accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 50:
        st.error(f"⚠️ ตรวจพบไฟล์ {len(uploaded_files)} คลิป! เซิร์ฟเวอร์ฟรีรองรับรวดเดียวสูงสุด 50 ไฟล์")
        st.stop()

    st.info(f"📁 เตรียมประมวลผลวิดีโอทั้งหมด {len(uploaded_files)} คลิป")
    
    if st.button("🔍 เริ่มระบบสแกนเจาะลึก (Context-Aware Scan)", type="primary"):
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์รายคลิป:")
        
        results_summary = []
        cols = st.columns(3)
        REJECT_THRESHOLD = 71.0 
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
                        with st.spinner("กำลังวิเคราะห์บริบท..."):
                            frames, skin_areas, total_skin_areas, skin_parts, obj_areas = process_video_advanced(video_path, target_fps=6)
                            
                            if not frames:
                                st.caption("⚠️ ไฟล์วิดีโอเสีย")
                                continue
                                
                            frame_scores = []
                            batch_size = 16 
                            
                            for i in range(0, len(frames), batch_size):
                                batch_frames = frames[i:i+batch_size]
                                tensors = [transform(Image.fromarray(f)) for f in batch_frames]
                                input_batch = torch.stack(tensors).to(device)
                                output = model(input_batch)
                                scores = torch.softmax(output, dim=1)[:, 1].tolist()
                                frame_scores.extend(scores)
                            
                            dynamic_base = (float(np.mean(frame_scores)) * 18.0) + (float(np.std(frame_scores)) * 6.0)
                            visual_penalty = 0.0
                            details_list = []
                            
                            # ==========================================
                            # 🧠 ระบบ AI คัดแยกโหมด: พรีเซนเตอร์ vs ชี้สินค้า
                            # ==========================================
                            median_total_skin = np.median([s for s in total_skin_areas if s > 0]) if any(s > 0 for s in total_skin_areas) else 0
                            median_skin_parts = np.median([p for p in skin_parts if p > 0]) if any(p > 0 for p in skin_parts) else 0
                            
                            # ถ้าเจอพื้นที่คนเยอะ (เกิน 8000) หรือเห็นอวัยวะหลายส่วนพร้อมกัน (หน้า+มือ) ให้เป็นโหมดพรีเซนเตอร์
                            is_presenter_mode = (median_total_skin > 8000) or (median_skin_parts >= 2)
                            
                            # ==========================================
                            # 💡 กฎที่ 1: สัดส่วนสเกลสัมพันธ์กับบริบท (Contextual Proportion)
                            # ==========================================
                            miniature_frames = 0
                            for s_area, o_area in zip(skin_areas, obj_areas):
                                if s_area > 500 and o_area > 500:
                                    if is_presenter_mode:
                                        # คนถือของ: อนุโลมให้ใช้สองมือถือของชิ้นใหญ่ สัดส่วนมือใหญ่ได้
                                        limit_ratio = 2.5 if o_area > 5000 else 1.5
                                    else:
                                        # ชี้สินค้าล้วนๆ (ไม่เห็นคน): มือต้องไม่ใหญ่กว่าของโครงสร้างหลัก (เช่น มือชี้ชั้นวาง มือต้องดูเล็ก)
                                        limit_ratio = 0.60 
                                        
                                    if (s_area / o_area) > limit_ratio: 
                                        miniature_frames += 1
                                        
                            if miniature_frames >= 3:
                                if is_presenter_mode:
                                    visual_penalty += 45.0
                                    details_list.append("⚠️ สเกลสินค้าหลอกตา (สัดส่วนมือพรีเซนเตอร์ใหญ่เกินจริง)")
                                else:
                                    visual_penalty += 71.0 # คลิปของจิ๋ว ปัดตกทันที
                                    details_list.append("⛔ สเกลของจิ๋วผิดธรรมชาติ (มือโผล่มาใหญ่กว่าสินค้ามาก)")
                            
                            # ==========================================
                            # 💡 กฎที่ 2: โครงสร้างและความแข็งแรง (Contextual Stability)
                            # ==========================================
                            valid_obj = [a for a in obj_areas if a > 300]
                            if valid_obj:
                                median_obj = np.median(valid_obj)
                                
                                if is_presenter_mode:
                                    # ถุงอาหารสัตว์อนุโลมให้ยับและขยับได้
                                    pass_percent = 30.0
                                    tolerance = 0.85
                                else:
                                    # ของโชว์เดี่ยวๆ (เช่น ชั้นเหล็ก ถังขยะ) ต้องนิ่งและโครงสร้างไม่กลายร่างเด็ดขาด!
                                    pass_percent = 60.0
                                    tolerance = 0.35 
                                    
                                stable_frames = sum(1 for a in valid_obj if abs(a - median_obj) / median_obj <= tolerance)
                                stability_percent = (stable_frames / len(valid_obj)) * 100.0
                                
                                if stability_percent <= pass_percent:
                                    if is_presenter_mode:
                                        visual_penalty += 45.0
                                        details_list.append(f"⚠️ ถุงสินค้าทรงไม่นิ่ง (ความนิ่ง {int(stability_percent)}%)")
                                    else:
                                        visual_penalty += 71.0 # ชั้นวางบิดเบี้ยว ปัดตกทันที
                                        details_list.append(f"⛔ โครงสร้างสินค้ากลายร่าง/ยืดหด (จับโป๊ะ AI ความนิ่ง {int(stability_percent)}%)")
                            
                            # ==========================================
                            # 💡 กฎที่ 3: ภาพละลายจับโป๊ะ AI (Contextual Melt)
                            # ==========================================
                            severe_streak = 0
                            max_severe_streak = 0
                            for s in frame_scores:
                                if s > 0.985:
                                    severe_streak += 1
                                    max_severe_streak = max(max_severe_streak, severe_streak)
                                else:
                                    severe_streak = 0
                                    
                            melt_limit = 12 if is_presenter_mode else 6
                            warn_limit = 5 if is_presenter_mode else 3
                            
                            if max_severe_streak >= melt_limit:
                                visual_penalty += 71.0 
                                details_list.append(f"⛔ ภาพละลายจับโป๊ะ AI ต่อเนื่อง ({max_severe_streak} เฟรม)")
                            elif max_severe_streak >= warn_limit: 
                                visual_penalty += 15.0 
                                details_list.append("⚠️ ภาพบิดเบี้ยวช่วงสั้น (หักคะแนน)")
                                
                            # กฎที่ 4: เสียง
                            audio_penalty, audio_msgs = analyze_audio_strict(video_path)
                            if audio_msgs: details_list.extend(audio_msgs)
                            
                            # สรุปคะแนนสุทธิ
                            raw_final = 2.0 + dynamic_base + visual_penalty + audio_penalty
                            final_score = min(100.0, max(1.0, raw_final))
                            
                            if not details_list:
                                details = "✅ สมบูรณ์: สเกลสมจริง โครงสร้างนิ่ง เสียงชัดเจน"
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
                        st.caption(f"⚠️ Error เกิดข้อผิดพลาด: {str(e)}")
                    finally:
                        if os.path.exists(video_path): os.unlink(video_path)
                        if 'frames' in locals(): del frames
                        if 'frame_scores' in locals(): del frame_scores
                        if 'input_batch' in locals(): del input_batch 
                        
                        gc.collect() 
                        if torch.cuda.is_available(): 
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        time.sleep(0.2) 
                    
                    results_summary.append({
                        "ลำดับ": idx + 1,
                        "ชื่อไฟล์": uploaded_file.name,
                        "คะแนนความเสี่ยง": f"{final_score:.0f}%",
                        "สถานะ": status,
                        "หมายเหตุ": details
                    })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        st.divider()
        st.subheader("📋 แดชบอร์ดสรุปผล")
        df_all = pd.DataFrame(results_summary)
        m1, m2, m3 = st.columns(3)
        m1.metric("จำนวนคลิปทั้งหมด", f"{len(df_all)} คลิป")
        m2.metric("✅ ผ่าน (< 71%)", f"{len(df_all[df_all['สถานะ'] == 'PASS'])} คลิป")
        m3.metric("❌ ไม่ผ่าน (≥ 71%)", f"{len(df_all[df_all['สถานะ'] == 'REJECT'])} คลิป")
        
        st.write("---")
        tab1, tab2 = st.tabs(["✅ คลิปที่ผ่าน", "❌ คลิปที่ไม่ผ่าน"])
        with tab1:
            if not df_all[df_all['สถานะ'] == 'PASS'].empty:
                st.dataframe(df_all[df_all['สถานะ'] == 'PASS'][["ลำดับ", "ชื่อไฟล์", "คะแนนความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else: st.info("ไม่มีคลิปที่ผ่านเกณฑ์")
        with tab2:
            if not df_all[df_all['สถานะ'] == 'REJECT'].empty:
                st.dataframe(df_all[df_all['สถานะ'] == 'REJECT'][["ลำดับ", "ชื่อไฟล์", "คะแนนความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else: st.info("ไม่มีคลิปที่ถูกปัดตก")
        
        st.success("🎉 ประมวลผลเสร็จสิ้น ระบบได้เคลียร์ Cache เรียบร้อยแล้ว")
