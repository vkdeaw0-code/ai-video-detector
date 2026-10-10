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

# เร่งความเร็วการประมวลผลขั้นสุด
cv2.setNumThreads(4)
torch.set_num_threads(4)
torch.set_grad_enabled(False)

# ล็อกค่า Seed ให้ AI นิ่ง 100%
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. ตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(page_title="AI Video Inspector Pro", page_icon="⚖️", layout="wide")
st.title("⚖️ ระบบคัดกรองคลิป AI (Height Silhouette Engine 71%)")
st.markdown("""
**เกณฑ์ตัดสิน: ความเสี่ยง ≥ 71% คือ ไม่ผ่าน (REJECT)**
*   🎯 **Height Silhouette Router:** แยกแยะคนยืนพรีเซนต์จริง (ต้องมีโครงสร้างความสูงจากหัวถึงลำตัว) ออกจากแค่มือยื่นมาจับของ
*   🐱 **Presenter Mode (คนยืนพรีเซนต์):** อนุโลมให้แพ็กเกจขยับ ยับได้ และมือถือของได้ตามธรรมชาติ (ผ่านฉลุย)
*   📦 **Showcase Mode (สินค้าตั้งโต๊ะ/มือชี้):** ล็อคเป้าจับโป๊ะสเกลมือยักษ์ และโครงสร้างบิดเบี้ยว (ปัดตกทันที)
*   ⚡ **Turbo Batch & Audio:** สแกนไวปรู๊ดปร๊าด พร้อมระบบตรวจสอบเสียงพากย์อัจฉริยะ
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
# 2. เครื่องยนต์วิเคราะห์โครงสร้างความสูงและสเกล
# ==========================================
def process_video_advanced(video_path, target_fps=6): 
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps): video_fps = 30
    
    interval = max(1, int(round(video_fps / target_fps)))
    frames = []
    skin_areas, obj_areas = [], []
    skin_widths, obj_widths = [], []
    silhouette_scores = []
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret: break
        
        if count % interval == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            frames.append(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            
            # --- สกัดผิวหนัง/มือ ---
            hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            kernel_skin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_skin)
            
            # 💡 Height Silhouette Logic: เช็คความต่อเนื่องของร่างกายจากบนลงล่าง
            # คนยืนจริง พิกเซลผิวหนังต้องกระจายตัวทั้งโซนบน (หัว) และโซนกลาง (ลำตัว)
            top_zone = np.sum(skin_mask[:75, :] > 0)     # โซนหัว/คอ
            middle_zone = np.sum(skin_mask[75:150, :] > 0) # โซนลำตัว/หน้าอก
            
            # ถ้ามีผิวหนังต่อเนื่องทั้งสองโซน แปลว่าเป็นคนยืนพรีเซนต์
            if top_zone > 200 and middle_zone > 300:
                silhouette_scores.append(1)
            else:
                silhouette_scores.append(0)
            
            skin_cnts, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_skin = [c for c in skin_cnts if cv2.contourArea(c) > 200]
            
            if valid_skin:
                largest_skin = max(valid_skin, key=cv2.contourArea)
                sx, sy, sw, sh = cv2.boundingRect(largest_skin)
                skin_widths.append(sw)
                skin_areas.append(cv2.contourArea(largest_skin))
            else:
                skin_widths.append(0)
                skin_areas.append(0)
            
            # --- สกัดวัตถุหลักในภาพ ---
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_frame, (7, 7), 0)
            edges = cv2.Canny(blurred, 40, 120)
            
            h, w = edges.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.95)), 255, -1)
            focused_edges = cv2.bitwise_and(edges, edges, mask=mask)
            
            kernel_obj = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed_edges = cv2.morphologyEx(focused_edges, cv2.MORPH_CLOSE, kernel_obj)
            obj_cnts, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if obj_cnts:
                largest_obj = max(obj_cnts, key=cv2.contourArea)
                ox, oy, ow, oh = cv2.boundingRect(largest_obj)
                obj_widths.append(ow)
                obj_areas.append(cv2.contourArea(largest_obj))
            else:
                obj_widths.append(0)
                obj_areas.append(0)
                
        count += 1
    cap.release()
    
    # 🧠 ตัดสินโหมดด้วยโครงสร้างความสูง (Silhouette Router):
    # ถ้าในคลิปมีเฟรมที่ตรวจพบคนยืน (มีหัว + ลำตัว) มากกว่า 20% ของวิดีโอ -> Presenter Mode
    # นอกนั้น (เช่น มีแต่วัตถุตั้งโต๊ะและมีแค่มือยื่นมาจิ้ม) -> Showcase Mode
    presenter_frames_ratio = sum(silhouette_scores) / max(1, len(silhouette_scores))
    is_presenter = presenter_frames_ratio >= 0.20
    
    return frames, skin_areas, obj_areas, skin_widths, obj_widths, is_presenter

# ==========================================
# 3. เครื่องยนต์เสียง
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
            return 0.0, []
            
        sample_rate, data = wavfile.read(audio_path)
        if len(data) == 0: return 0.0, []
            
        data_float = data.astype(np.float32)
        window_size = int(sample_rate * 0.05)
        energies = np.array([np.sum(data_float[i:i+window_size]**2) for i in range(0, len(data_float), window_size)])
        
        if len(energies) > 2:
            energy_diffs = np.abs(np.diff(energies))
            mean_diff = np.mean(energy_diffs)
            std_diff = np.std(energy_diffs)
            
            stutter_points = np.sum(energy_diffs > (mean_diff + 3.5 * std_diff))
            if stutter_points > 4:
                audio_penalty += 30.0 
                audio_msgs.append("⚠️ เสียงพูดสะดุดรัว (หักคะแนน)")
            elif stutter_points >= 2:
                audio_penalty += 5.0 
                audio_msgs.append("🔊 เสียงสะดุดเล็กน้อย (ผ่าน)")

        window_large = int(sample_rate * 0.2)
        energies_large = np.array([np.sum(data_float[i:i+window_large]**2) for i in range(0, len(data_float), window_large)])
        
        if len(energies_large) > 0:
            mean_e = np.mean(energies_large)
            variance_e = np.var(energies_large) / (mean_e + 1e-6)
            
            if variance_e < 0.08 and mean_e > 100:
                audio_penalty += 71.0
                audio_msgs.append("⛔ เสียงแบนราบเป็นหุ่นยนต์/AI 100%")
            elif variance_e < 0.25 and mean_e > 100:
                audio_penalty += 10.0
                if not audio_msgs: audio_msgs.append("⚠️ เสียงพูดโทนเดียวคล้าย AI")
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
        st.error(f"⚠️ ตรวจพบไฟล์ {len(uploaded_files)} คลิป! เซิร์ฟเวอร์รองรับสูงสุด 50 ไฟล์")
        st.stop()

    st.info(f"📁 เตรียมประมวลผลวิดีโอทั้งหมด {len(uploaded_files)} คลิป")
    
    if st.button("🔍 เริ่มระบบสแกนโครงสร้างอัจฉริยะ (Silhouette Scan)", type="primary"):
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
                    mode_label = ""
                    
                    try:
                        with st.spinner("กำลังตรวจสอบโครงสร้างคนยืนและสเกล..."):
                            frames, skin_areas, obj_areas, skin_widths, obj_widths, is_presenter = process_video_advanced(video_path, target_fps=6)
                            
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
                            
                            visual_penalty = 0.0
                            details_list = []
                            
                            if is_presenter:
                                mode_label = "🐱 โหมดพรีเซนต์ (คนยืนพรีเซนต์จริง)"
                                dynamic_base = (float(np.mean(frame_scores)) * 12.0) + (float(np.std(frame_scores)) * 5.0)
                            else:
                                mode_label = "📦 โหมดสินค้าตั้งโต๊ะ (มือชี้/สินค้าโชว์)"
                                dynamic_base = (float(np.mean(frame_scores)) * 55.0) + (float(np.std(frame_scores)) * 15.0)
                            
                            st.caption(f"_{mode_label}_")
                            
                            # ==========================================
                            # 💡 กฎที่ 1: ตรวจสอบสเกล (ทำงานเฉพาะโหมดสินค้าตั้งโต๊ะ)
                            # ==========================================
                            scale_violation_frames = 0
                            if not is_presenter:
                                for s_area, o_area, s_w, o_w in zip(skin_areas, obj_areas, skin_widths, obj_widths):
                                    if s_area > 200 and o_area > 500:
                                        if (s_w / (o_w + 1) > 0.40) or (s_area / (o_area + 1) > 0.30):
                                            scale_violation_frames += 1
                                            
                                if scale_violation_frames >= 2:
                                    visual_penalty += 71.0
                                    details_list.append("⛔ สเกลผิดธรรมชาติ (มือ/แขนใหญ่เกินสัดส่วนสินค้า)")
                            
                            # ==========================================
                            # 💡 กฎที่ 2: ความคงที่รูปทรงวัตถุ (Structural Stability)
                            # ==========================================
                            valid_obj = [a for a in obj_areas if a > 500]
                            if valid_obj:
                                median_obj = np.median(valid_obj)
                                
                                if is_presenter:
                                    pass_percent = 15.0
                                    tolerance = 0.85
                                else:
                                    pass_percent = 60.0
                                    tolerance = 0.15
                                    
                                stable_frames = sum(1 for a in valid_obj if abs(a - median_obj) / median_obj <= tolerance)
                                stability_percent = (stable_frames / len(valid_obj)) * 100.0
                                
                                if stability_percent < pass_percent and not is_presenter:
                                    visual_penalty += 71.0
                                    details_list.append("⛔ โครงสร้างสินค้าบิดเบี้ยว/ยืดหดผิดธรรมชาติ (จับโป๊ะ AI)")
                            
                            # ==========================================
                            # 💡 กฎที่ 3: ภาพละลายต่อเนื่อง (AI Melt)
                            # ==========================================
                            severe_streak = 0
                            max_severe_streak = 0
                            for s in frame_scores:
                                threshold = 0.985 if is_presenter else 0.920
                                if s > threshold:
                                    severe_streak += 1
                                    max_severe_streak = max(max_severe_streak, severe_streak)
                                else:
                                    severe_streak = 0
                                    
                            melt_limit = 12 if is_presenter else 4
                            warn_limit = 5 if is_presenter else 2
                            
                            if max_severe_streak >= melt_limit:
                                visual_penalty += 71.0
                                details_list.append("⛔ ภาพละลายบิดเบี้ยวต่อเนื่องชัดเจน")
                            elif max_severe_streak >= warn_limit:
                                visual_penalty += 15.0
                                details_list.append("⚠️ ภาพมีรอยบิดเบี้ยวช่วงสั้น")
                                
                            # กฎที่ 4: เสียง
                            audio_penalty, audio_msgs = analyze_audio_strict(video_path)
                            if audio_msgs: details_list.extend(audio_msgs)
                            
                            # สรุปคะแนนรวม
                            raw_final = 2.0 + dynamic_base + visual_penalty + audio_penalty
                            final_score = min(100.0, max(1.0, raw_final))
                            
                            if not details_list:
                                if is_presenter:
                                    details = "✅ สมบูรณ์ (Presenter Mode): โครงสร้างคนยืนและสินค้าถูกต้อง"
                                else:
                                    details = "✅ สมบูรณ์ (Showcase Mode): สเกลสมจริง โครงสร้างมั่นคง"
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
                        time.sleep(0.1) 
                    
                    results_summary.append({
                        "ลำดับ": idx + 1,
                        "ชื่อไฟล์": uploaded_file.name,
                        "หมวดหมู่": mode_label.split(" ")[1],
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
                st.dataframe(df_all[df_all['สถานะ'] == 'PASS'][["ลำดับ", "ชื่อไฟล์", "หมวดหมู่", "คะแนนความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else: st.info("ไม่มีคลิปที่ผ่านเกณฑ์")
        with tab2:
            if not df_all[df_all['สถานะ'] == 'REJECT'].empty:
                st.dataframe(df_all[df_all['สถานะ'] == 'REJECT'][["ลำดับ", "ชื่อไฟล์", "หมวดหมู่", "คะแนนความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else: st.info("ไม่มีคลิปที่ถูกปัดตก")
        
        st.success("🎉 ประมวลผลเสร็จสิ้น ระบบได้เคลียร์ Cache เรียบร้อยแล้ว")
