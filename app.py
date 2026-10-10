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
# 1. ตั้งค่าหน้าเว็บและการจัดการทรัพยากร
# ==========================================
st.set_page_config(page_title="AI Video Inspector (Scale & Focus Edition)", page_icon="🎯", layout="wide")
st.title("🎯 ระบบตรวจคัดกรองคลิป AI (โฟกัสสเกลคนและสินค้า > 60%)")
st.markdown("""
**เกณฑ์ตัดตก: 76% ขึ้นไป** | ระบบโฟกัสเฉพาะบุคคลและสินค้าหลักที่กำลังนำเสนอ
*   📏 **Scale Consistency:** สเกลคนและสินค้าต้องถูกต้องคงที่ **> 60% ของคลิป** (อนุโลมจังหวะขยับ 40%)
*   👁️ **Vision:** ตัดตกเฉพาะกรณีอวัยวะ/สินค้าละลายพังต่อเนื่องเกิน **2 วินาที**
*   👂 **Audio:** ตัดตกหากพบเสียงหุ่นยนต์แบนราบ หรืออ่านสะดุด/เพี้ยนเกิน **2 คำ**
""")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_models()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 2. เครื่องยนต์สกัดพื้นที่คนและสินค้า (Foreground Focus Area)
# ==========================================
def extract_focus_areas(video_path, target_fps=6):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30
    
    interval = max(1, int(video_fps / target_fps))
    frames = []
    
    skin_areas = []
    main_obj_areas = []
    
    # ช่วงสีผิว (Skin Tone HSV)
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
            
            # --- ก. โฟกัสตัวบุคคลและมือ (Skin Tracking) ---
            hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_cnts, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # ดึงเฉพาะพื้นที่ผิวที่ใหญ่ที่สุด (ตัดคนข้างหลังออก)
            skin_areas.append(max([cv2.contourArea(c) for c in skin_cnts]) if skin_cnts else 0)
            
            # --- ข. โฟกัสสินค้าด้านหน้า (Main Foreground Object) ---
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0) # เบลอเพิ่มขึ้นเพื่อลบรายละเอียดฉากหลัง
            edges = cv2.Canny(blurred, 60, 150)
            
            # ใช้ Morphological closing เพื่อเชื่อมเส้นขอบสินค้าให้เป็นก้อนเดียว
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            obj_cnts, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # ดึงเฉพาะวัตถุที่ใหญ่ที่สุด (ตัดสินค้าบนเชลฟ์ฉากหลังทิ้ง)
            main_obj_areas.append(max([cv2.contourArea(c) for c in obj_cnts]) if obj_cnts else 0)
            
        count += 1
    cap.release()
    return frames, skin_areas, main_obj_areas

# ==========================================
# 3. เครื่องยนต์วิเคราะห์เสียง (Audio Engine)
# ==========================================
def analyze_audio(video_path):
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    audio_penalty = 0.0
    audio_msgs = []
    
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg_exe, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            return 0.0, ["🔇 ไม่มีเสียง (อนุโลม)"]
            
        sample_rate, data = wavfile.read(audio_path)
        if len(data) == 0: return 0.0, ["🔇 ไม่มีเสียง (อนุโลม)"]
            
        data_float = data.astype(np.float32)
        
        # ตรวจคำสะดุด/อ่านรวบคำเพี้ยน
        window_size = int(sample_rate * 0.05) 
        energies = np.array([np.sum(data_float[i:i+window_size]**2) for i in range(0, len(data_float), window_size)])
        
        if len(energies) > 2:
            energy_diffs = np.abs(np.diff(energies))
            mean_diff, std_diff = np.mean(energy_diffs), np.std(energy_diffs)
            stutter_points = np.sum(energy_diffs > (mean_diff + 3.2 * std_diff))
            
            if stutter_points > 3: # สะดุดเกิน 2 ครั้ง (ตัดตก)
                audio_penalty += 65.0
                audio_msgs.append("⚠️ เสียงพูดพัง/คำสะดุดรัวเกิน 2 คำ")
            elif stutter_points >= 2: # เพี้ยน 1-2 ครั้ง (อนุโลม)
                audio_penalty += 15.0
                audio_msgs.append("🔊 เสียงสะดุดเล็กน้อย 1-2 คำ (อนุโลม)")

        # ตรวจเสียงแบนราบ
        window_large = int(sample_rate * 0.2) 
        energies_large = np.array([np.sum(data_float[i:i+window_large]**2) for i in range(0, len(data_float), window_large)])
        
        if len(energies_large) > 0:
            mean_e = np.mean(energies_large)
            variance_e = np.var(energies_large) / (mean_e + 1e-6)
            
            if variance_e < 0.08 and mean_e > 100:
                audio_penalty += 65.0
                audio_msgs.append("⚠️ เสียงแบนราบเป็นหุ่นยนต์/ฟังไม่เป็นภาษา")
            elif variance_e < 0.22 and mean_e > 100:
                audio_penalty += 10.0
                if not audio_msgs: audio_msgs.append("🔊 เสียงสังเคราะห์แต่ฟังรู้เรื่อง")

    except Exception:
        return 0.0, ["⚠️ ไม่สามารถวิเคราะห์คลื่นเสียงได้"]
    finally:
        if os.path.exists(audio_path): os.unlink(audio_path)
            
    return audio_penalty, audio_msgs

# ==========================================
# 4. ระบบประมวลผลกลางและ UI (Main Logic UI)
# ==========================================
uploaded_files = st.file_uploader(
    "เลือกไฟล์วิดีโอ (.mp4, .mov, .avi) - ลากวางได้หลายไฟล์พร้อมกัน", 
    type=["mp4", "mov", "avi"], accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 เตรียมประมวลผลทั้งหมด {len(uploaded_files)} คลิป")
    if st.button("🔍 เริ่มสแกนเจาะลึก", type="primary"):
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์แยกรายคลิป:")
        
        results_summary = []
        cols = st.columns(3)
        REJECT_THRESHOLD = 76.0 # 💡 ตัดตกที่ 76%
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
                    
                    status, details = "ERROR", ""
                    final_score = 0.0
                    
                    try:
                        with st.spinner("สแกนโครงสร้าง สเกล และเสียง..."):
                            # 1. ดึงภาพและดึงพื้นที่คน/สินค้าหลัก
                            frames, skin_areas, obj_areas = extract_focus_areas(video_path, target_fps=6)
                            
                            if not frames:
                                st.caption("⚠️ ไฟล์เสีย ไม่สามารถอ่านภาพได้")
                                continue
                                
                            frame_scores = []
                            with torch.no_grad():
                                for frame_np in frames:
                                    pil_img = Image.fromarray(frame_np)
                                    input_tensor = transform(pil_img).unsqueeze(0).to(device)
                                    output = model(input_tensor)
                                    frame_scores.append(torch.softmax(output, dim=1)[0][1].item())
                            
                            # 2. ฐานคะแนน (Base Risk)
                            dynamic_base_score = (float(np.mean(frame_scores)) * 18.0) + (float(np.std(frame_scores)) * 10.0)
                            visual_penalty = 0.0
                            details_list = []
                            
                            # 3. ตรวจจับการบิดเบี้ยว (Melting) - กฎ 2 วินาที
                            severe_streak = 0
                            max_severe_streak = 0
                            for s in frame_scores:
                                if s > 0.98:
                                    severe_streak += 1
                                    max_severe_streak = max(max_severe_streak, severe_streak)
                                else:
                                    severe_streak = 0
                                    
                            if max_severe_streak >= 12: # พังต่อเนื่องเกิน 2 วินาที (12 เฟรม)
                                visual_penalty += 65.0 
                                details_list.append("⛔ อวัยวะ/สินค้า บิดเบี้ยวละลายนานเกิน 2 วิ")
                            elif max_severe_streak >= 6: 
                                visual_penalty += 20.0
                                details_list.append("อวัยวะ/สินค้าบิดเบี้ยวช่วงสั้น 1-2 วิ")
                                
                            # 💡 4. กฎสเกลโดยรวม (Scale Consistency > 60%)
                            valid_skin = [a for a in skin_areas if a > 500]
                            valid_obj = [a for a in obj_areas if a > 500]
                            
                            skin_consistency = 100.0
                            obj_consistency = 100.0
                            
                            if valid_skin:
                                median_skin = np.median(valid_skin)
                                # อนุโลมให้ขนาดต่างจากค่ากลางได้ 45% (การขยับมือเข้า-ออกกล้อง)
                                consistent_skin = sum(1 for a in valid_skin if abs(a - median_skin) / median_skin <= 0.45)
                                skin_consistency = (consistent_skin / len(valid_skin)) * 100.0
                                
                            if valid_obj:
                                median_obj = np.median(valid_obj)
                                consistent_obj = sum(1 for a in valid_obj if abs(a - median_obj) / median_obj <= 0.45)
                                obj_consistency = (consistent_obj / len(valid_obj)) * 100.0
                                
                            # ถ้าระดับความคงที่ของสเกลคนหรือสินค้า ต่ำกว่าหรือเท่ากับ 60% ตลอดทั้งคลิป ถือว่าตกทันที
                            if skin_consistency <= 60.0 or obj_consistency <= 60.0:
                                visual_penalty += 65.0
                                details_list.append("⛔ สเกลคน/สินค้าผิดเพี้ยนเกิน 40% ของคลิป (สเกลแกว่ง)")
                                
                            # 5. วิเคราะห์เสียง
                            audio_penalty, audio_msgs = analyze_audio(video_path)
                            if audio_msgs: details_list.extend(audio_msgs)
                            
                            # 6. คำนวณคะแนนรวม
                            raw_final = 2.0 + dynamic_base_score + visual_penalty + audio_penalty
                            final_score = min(100.0, max(1.0, raw_final))
                            
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
                        st.caption(f"⚠️ Error เกิดข้อผิดพลาด: {str(e)}")
                    finally:
                        if os.path.exists(video_path): os.unlink(video_path)
                        if 'frames' in locals(): del frames
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    
                    results_summary.append({
                        "ลำดับ": idx + 1,
                        "ชื่อไฟล์": uploaded_file.name,
                        "คะแนนความเสี่ยง": f"{final_score:.0f}%",
                        "สถานะ": status,
                        "หมายเหตุ": details
                    })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # ==========================================
        # 5. แดชบอร์ดสรุปผลรวม
        # ==========================================
        st.divider()
        st.subheader("📋 แดชบอร์ดสรุปผลรวม")
        df_all = pd.DataFrame(results_summary)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("จำนวนคลิปทั้งหมด", f"{len(df_all)} คลิป")
        m2.metric("✅ ผ่าน (< 76%)", f"{len(df_all[df_all['สถานะ'] == 'PASS'])} คลิป")
        m3.metric("❌ ไม่ผ่าน (≥ 76%)", f"{len(df_all[df_all['สถานะ'] == 'REJECT'])} คลิป")
        
        st.write("---")
        tab1, tab2 = st.tabs(["✅ รายการคลิปที่ผ่าน", "❌ รายการคลิปที่ไม่ผ่าน"])
        
        with tab1:
            if not df_all[df_all['สถานะ'] == 'PASS'].empty:
                st.dataframe(df_all[df_all['สถานะ'] == 'PASS'][["ลำดับ", "ชื่อไฟล์", "คะแนนความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else: st.info("ไม่มีคลิปที่ผ่านเกณฑ์")
                
        with tab2:
            if not df_all[df_all['สถานะ'] == 'REJECT'].empty:
                st.dataframe(df_all[df_all['สถานะ'] == 'REJECT'][["ลำดับ", "ชื่อไฟล์", "คะแนนความเสี่ยง", "หมายเหตุ"]], hide_index=True, use_container_width=True)
            else: st.info("ไม่มีคลิปที่ถูกปัดตก")
        
        st.success("🎉 ประมวลผลเสร็จสิ้น คืนหน่วยความจำให้เครื่องเรียบร้อยแล้ว")
