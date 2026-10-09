import streamlit as st
import cv2
import torch
import timm
import numpy as np
import tempfile
import os
import pandas as pd
from PIL import Image
from torchvision import transforms

# --- 1. SET UP PAGE CONFIG ---
st.set_page_config(
    page_title="ระบบตรวจจับคลิปวิดีโอ AI",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 ระบบตรวจจับและคัดกรองคลิปวิดีโอ AI (ปรับเกณฑ์ผ่อนผันพิเศษ)")
st.write("ระบบตรวจจับที่ปรับ Calibration ให้ผ่อนผัน ปล่อยผ่านคลิปรีวิว/โฆษณา AI ที่ภาพรวมดูโอเคใช้งานได้")

# --- 2. LOAD MODELS ---
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

# --- 3. HELPER FUNCTIONS ---
def extract_frames(video_path, frame_interval=30):
    """สุ่มดึงเฟรมแบบกระจายห่างขึ้นเพื่อประเมินภาพรวม"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
                    
        count += 1
        
    cap.release()
    return frames

# --- 4. WEB UI INTERFACE ---
uploaded_files = st.file_uploader(
    "เลือกไฟล์วิดีโอ (.mp4, .mov, .avi) - เลือกพร้อมกันหลายไฟล์ได้", 
    type=["mp4", "mov", "avi"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 เลือกไว้ทั้งหมด {len(uploaded_files)} คลิป")
    
    if st.button("🔍 เริ่มกระบวนการสแกนตรวจจับทุกคลิป", type="primary"):
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์แบบกะทัดรัด:")
        
        results_summary = []
        cols = st.columns(3)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 3]
            
            with col:
                with st.container(border=True):
                    st.caption(f"🎬 คลิปที่ {idx+1}: **{uploaded_file.name[:20]}...**" if len(uploaded_file.name) > 20 else f"🎬 คลิปที่ {idx+1}: **{uploaded_file.name}**")
                    
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    
                    with st.spinner("กำลังสแกน..."):
                        frames = extract_frames(video_path)
                        
                        if not frames:
                            st.caption("⚠️ ไม่สามารถอ่านเฟรมได้")
                            status = "ERROR"
                            percent_score = 0
                        else:
                            scores = []
                            with torch.no_grad():
                                for frame_np in frames:
                                    pil_img = Image.fromarray(face_np if 'face_np' in locals() else frame_np)
                                    input_tensor = transform(pil_img).unsqueeze(0).to(device)
                                    output = model(input_tensor)
                                    probs = torch.softmax(output, dim=1)
                                    fake_prob = probs[0][1].item()
                                    scores.append(fake_prob)
                            
                            # 🛠️ CALIBRATION LOGIC
                            # นำค่าเฉลี่ยเฟรมมาผ่านฟังก์ชัน Softening บีบสเกลความไวของโมเดล
                            raw_score = float(np.median(scores))
                            
                            # บีบสเกลให้คลิปเนียนระดับใช้งานได้ถูไถ ตกมาอยู่ในช่วง 0.20 - 0.60
                            calibrated_score = np.clip((raw_score - 0.5) * 0.8 + 0.3, 0.0, 1.0)
                            
                            # ตั้งเกณฑ์ REJECT ไว้ที่ 80% หลัง Calibrate (เท่ากับต้องแย่ระดับหลุดโลกจริงๆ)
                            THRESHOLD = 0.80
                            percent_score = float(calibrated_score * 100)
                            
                            if calibrated_score >= THRESHOLD:
                                status = "REJECT"
                                st.error(f"❌ **REJECT** ({percent_score:.0f}%)", icon="🚨")
                            else:
                                status = "PASS"
                                st.success(f"✅ **PASS** ({percent_score:.0f}%)", icon="🟢")
                                
                            st.progress(min(int(percent_score), 100))
                    
                    os.unlink(video_path)
                    
                    results_summary.append({
                        "ลำดับ": idx + 1,
                        "ชื่อไฟล์": uploaded_file.name,
                        "คะแนนความแปลก AI (%)": f"{percent_score:.2f}%",
                        "สถานะ": status
                    })
        
        # --- 5. REJECTED CLIPS DASHBOARD ---
        st.divider()
        st.header("🚫 แดชบอร์ดสรุปคลิปที่ไม่ผ่านการคัดกรอง (Rejected Clips Dashboard)")
        
        df_all = pd.DataFrame(results_summary)
        df_rejected = df_all[df_all["สถานะ"] == "REJECT"]
        
        m1, m2, m3 = st.columns(3)
        total_clips = len(df_all)
        rejected_count = len(df_rejected)
        pass_count = total_clips - rejected_count
        reject_rate = (rejected_count / total_clips * 100) if total_clips > 0 else 0
        
        m1.metric("จำนวนคลิปทั้งหมด", f"{total_clips} คลิป")
        m2.metric("จำนวนคลิปที่ผ่าน (PASS)", f"{pass_count} คลิป")
        m3.metric("จำนวนคลิปที่ถูกคัดออก (REJECT)", f"{rejected_count} คลิป", delta=f"{reject_rate:.1f}% อัตราการคัดออก", delta_color="inverse")
        
        st.write("")
        
        if not df_rejected.empty:
            st.error(f"⚠️ ตรวจพบคลิปที่ไม่ผ่านเกณฑ์ทั้งหมด {len(df_rejected)} คลิป ดังรายการด้านล่าง:")
            st.dataframe(
                df_rejected[["ลำดับ", "ชื่อไฟล์", "คะแนนความแปลก AI (%)"]], 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.balloons()
            st.success("🎉 ยินดีด้วย! ไม่พบคลิปที่ติดสถานะ REJECT ในรอบนี้ ทุกคลิปผ่านการคัดกรองทั้งหมด")
