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
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. ตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(page_title="AI Video Inspector Ultimate", page_icon="⚖️", layout="wide")
st.title("⚖️ ระบบคัดกรองคลิป AI (Ultimate Hybrid Engine 71%)")
st.markdown("""
**เกณฑ์ตัดสิน: ความเสี่ยง ≥ 71% คือ ไม่ผ่าน (REJECT)**
*   👤 **Face & Spatial Switch:** ตรวจจับใบหน้าและมวลร่างกายช่วงบน เพื่อแยกหมวดหมู่วิดีโอแบบ 100%
*   🛍️ **Presenter Mode:** คนยืนขายอาหารสัตว์/ของใช้ ผ่านได้ลื่นไหล (อนุโลมการจับสินค้า ถุงขยับ)
*   📦 **Showcase Mode:** คลิปโชว์ของที่มีแต่มือชี้ (เช่น ชั้นวาง ถังขยะ) คุมเข้มสเกลมือและห้ามโครงสร้างบิดเบี้ยว
*   🎤 **Forgiving Audio:** เสียงพูดมีจังหวะหยุดพักหรือสะดุดเล็กน้อยจะไม่ถูกปัดตก
""")

@st.cache_resource
def load_vision_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    model = model.to(device)
    model.eval()
    
    # โหลดโมเดลตรวจจับใบหน้ามาตรฐานของ OpenCV
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    return model, device, face_cascade

model, device, face_cascade = load_vision_model()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 2. เครื่องยนต์วิเคราะห์สเกล & แยกหมวดหมู่ขั้นสูง
# ==========================================
def process_video_advanced(video_path, target_fps=6): 
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps): video_fps = 30
    
    interval = max(1, int(round(video_fps / target_fps)))
    frames = []
    skin_areas = []
    obj_areas = []
    
    # ตัวแปรสำหรับจับพฤติกรรมคน
    face_found_count = 0
    upper_body_skin_count = 0
    total_sampled = 0
    
    # คืนค่าโทนสีผิวให้กว้างขึ้น เพื่อให้จับพรีเซนเตอร์ในแสงห้างได้
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret: break
        
        if count % interval == 0:
            total_sampled += 1
            frame_resized = cv2.resize(frame, (224, 224))
            frames.append(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            # 1. ตรวจจับใบหน้า
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
            if len(faces) > 0:
                face_found_count += 1
                
            # 2. ตรวจจับพื้นที่ผิวหนัง (Skin Mask)
            hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # เช็คว่ามีผิวหนังอยู่ "ครึ่งบน" ของจอหรือไม่ (หน้า คอ ไหล่)
            upper_half_skin = np.sum(skin_mask[:112, :] > 0)
            if upper_half_skin > 1000: # ถ้ามีพิกเซลผิวหนังครึ่งบนเยอะ
                upper_body_skin_count += 1
                
            skin_cnts, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if skin_cnts:
                skin_areas.append(max([cv2.contourArea(c) for c in skin_cnts]))
            else:
                skin_areas.append(0)
            
            # 3. ตรวจจับสินค้า (Object Mask)
            blurred = cv2.GaussianBlur(gray_frame, (7, 7), 0)
            edges = cv2.Canny(blurred, 40, 120)
            
            h, w = edges.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (int(w*0.15), int(h*0.15)), (int(w*0.85), int(h*0.95)), 255, -1)
            focused_edges = cv2.bitwise_and(edges, edges, mask=mask)
            
            kernel_obj = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed_edges = cv2.morphologyEx(focused_edges, cv2.MORPH_CLOSE, kernel_obj)
            obj_cnts, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if obj_cnts:
                obj_areas.append(max([cv2.contourArea(c) for c in obj_cnts]))
            else:
                obj_areas.append(0)
                
        count += 1
    cap.release()
    
    # 🧠 ตัดสินใจเลือกโหมด (Router Logic):
    # ถ้าเจอหน้าคนเกิน 5% ของคลิป หรือ เจอผิวหนังช่วงบนจอเกิน 20% ของคลิป -> ให้เป็นโหมดพรีเซนเตอร์
    is_presenter = False
    if total_sampled > 0:
        if (face_found_count / total_sampled) >= 0.05 or (upper_body_skin_count / total_sampled) >= 0.20:
            is_presenter = True
            
    return frames, skin_areas, obj_areas, is_presenter

# ==========================================
# 3. เครื่องยนต์เสียง (ลดความโหดลง)
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
            
            # 💡 ลดการหักคะแนนเสียงสะดุดลง เพื่อให้คลิปคนพูดจริงผ่านได้
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
            
            # โทษเสียงหุ่นยนต์ยังคงหนักอยู่ (ถ้าเป็นหุ่นยนต์ชัดเจนตัดตก)
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
    
    if st.button("🔍 เริ่มระบบสแกนอัจฉริยะ (Hybrid Scan)", type="primary"):
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
                        with st.spinner("กำลังแยกแยะบริบทคลิป..."):
                            frames, skin_areas, obj_areas, is_presenter = process_video_advanced(video_path, target_fps=6)
                            
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
                            
                            # คำนวณความเสี่ยงจาก AI ละลาย (Baseline)
                            if is_presenter:
                                dynamic_base = (float(np.mean(frame_scores)) * 15.0) + (float(np.std(frame_scores)) * 5.0)
                            else:
                                # คลิปโชว์ของไม่มีคน คุม AI เข้มขึ้น
                                dynamic_base = (float(np.mean(frame_scores)) * 40.0) + (float(np.std(frame_scores)) * 10.0)
                            
                            # ==========================================
                            # 💡 กฎที่ 1: สเกลมือ (Hand Proportion)
                            # ==========================================
                            miniature_frames = 0
                            for s_area, o_area in zip(skin_areas, obj_areas):
                                if s_area > 300 and o_area > 300:
                                    if is_presenter:
                                        # คนถือของ: อนุโลมเต็มที่ สัดส่วนมือไม่จำกัด (เพราะบางทีกอดของ หรือของเล็ก)
                                        limit_ratio = 5.0 
                                    else:
                                        # ชี้ของอย่างเดียว: มือห้ามใหญ่เกิน 35% ของสินค้า
                                        limit_ratio = 0.35 
                                        
                                    if (s_area / o_area) > limit_ratio: 
                                        miniature_frames += 1
                                        
                            if miniature_frames >= 3:
                                if is_presenter:
                                    # พรีเซนเตอร์ถือของ ไม่หักคะแนนสเกลเลย (อนุโลมสุดๆ)
                                    pass
                                else:
                                    visual_penalty += 71.0 # คลิปชั้นวาง/ถังขยะ สเกลหลอกตา ปัดตกทันที
                                    details_list.append("⛔ สเกลผิดธรรมชาติ (สัดส่วนมือลอยใหญ่ผิดปกติ)")
                            
                            # ==========================================
                            # 💡 กฎที่ 2: ความคงที่รูปทรง (Structural Stability)
                            # ==========================================
                            valid_obj = [a for a in obj_areas if a > 500]
                            if valid_obj:
                                median_obj = np.median(valid_obj)
                                
                                if is_presenter:
                                    # ถุงอาหารสัตว์/ของในมือ: ยืดหดได้เต็มที่ (ผ่าน 100%)
                                    pass_percent = 20.0
                                    tolerance = 0.85
                                else:
                                    # โชว์ชั้นวาง/ถังขยะ: ต้องนิ่งมาก ห้ามยืดหดเกิน 20%
                                    pass_percent = 60.0
                                    tolerance = 0.20 
                                    
                                stable_frames = sum(1 for a in valid_obj if abs(a - median_obj) / median_obj <= tolerance)
                                stability_percent = (stable_frames / len(valid_obj)) * 100.0
                                
                                if stability_percent <= pass_percent:
                                    if is_presenter:
                                        # ไม่ทำโทษคลิปพรีเซนเตอร์
                                        pass
                                    else:
                                        visual_penalty += 71.0 # ชั้นวางยืดหด ปัดตก
                                        details_list.append(f"⛔ โครงสร้างสินค้าบิดเบี้ยว/ไม่เสถียร (จับโป๊ะ AI)")
                            
                            # ==========================================
                            # 💡 กฎที่ 3: ภาพละลาย (AI Melt)
                            # ==========================================
                            severe_streak = 0
                            max_severe_streak = 0
                            for s in frame_scores:
                                if s > 0.985:
                                    severe_streak += 1
                                    max_severe_streak = max(max_severe_streak, severe_streak)
                                else:
                                    severe_streak = 0
                                    
                            if max_severe_streak >= 12: # ละลายยาวเกิน 2 วิ โดนตกทุกโหมด
                                visual_penalty += 71.0 
                                details_list.append(f"⛔ ภาพละลายบิดเบี้ยวต่อเนื่องชัดเจน")
                            elif max_severe_streak >= 5 and not is_presenter: 
                                visual_penalty += 30.0 
                                details_list.append("⚠️ ภาพบิดเบี้ยวช่วงสั้น (หักคะแนน)")
                                
                            # กฎที่ 4: เสียง
                            audio_penalty, audio_msgs = analyze_audio_strict(video_path)
                            if audio_msgs: details_list.extend(audio_msgs)
                            
                            # สรุปคะแนน
                            raw_final = 2.0 + dynamic_base + visual_penalty + audio_penalty
                            final_score = min(100.0, max(1.0, raw_final))
                            
                            if not details_list:
                                if is_presenter:
                                    details = "✅ สมบูรณ์ (Presenter Mode): สเกลคนและสินค้าเป็นธรรมชาติ"
                                else:
                                    details = "✅ สมบูรณ์ (Showcase Mode): สเกลสมจริง โครงสร้างแข็งแรง"
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
