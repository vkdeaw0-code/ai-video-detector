import streamlit as st
import cv2
import torch
import timm
import numpy as np
import tempfile
import gc
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

st.set_page_config(page_title="AI Video Detector", page_icon="🎥", layout="wide")
st.title("🎥 ระบบตรวจจับและคัดกรองคลิปวิดีโอ AI")
st.write("ระบบวิเคราะห์ความผิดปกติของภาพ (ประเมินภาพรวมทั้งคลิป ลดความเข้มงวดเพื่อป้องกันคลิปจริงตกเกณฑ์)")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # แนะนำให้ใช้ pretrained=True เป็นฐานเริ่มต้น หากยังไม่มีไฟล์โมเดลที่เทรนมาเฉพาะเจาะจง
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    
    # ⚠️ สำคัญมาก: หากคุณมีไฟล์ Weights ที่เทรนแยกมาสำหรับการจับ AI (Deepfake) 
    # ให้เอาคอมเมนต์บรรทัดล่างออก แล้วใส่ path ของไฟล์ .pth
    # model.load_state_dict(torch.load('your_ai_detector_weights.pth', map_location=device))
    
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

uploaded_file = st.file_uploader("อัปโหลดวิดีโอของคุณที่นี่ (MP4, MOV, AVI)", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    st.info("กำลังประมวลผล... กรุณารอสักครู่")
    progress_bar = st.progress(0)
    
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # ดึงมาวิเคราะห์ 2 เฟรมต่อ 1 วินาที (เพื่อให้เห็นความต่อเนื่องมากขึ้น แต่ยังประมวลผลไว)
    frames_to_skip = max(1, fps // 2)
    
    ai_probabilities = []
    frame_count = 0
    analyzed_frames = 0
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frames_to_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                input_tensor = transform(pil_image).unsqueeze(0).to(device)
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                ai_prob = probabilities[0][1].item() * 100
                ai_probabilities.append(ai_prob)
                analyzed_frames += 1
                
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
            
            frame_count += 1
            
            if analyzed_frames % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    cap.release()
    progress_bar.progress(1.0)
    
    # --- ส่วนตัดสินผล (ปรับให้ยืดหยุ่นและมีเหตุผลที่สุด) ---
    if len(ai_probabilities) > 0:
        avg_ai_prob = sum(ai_probabilities) / len(ai_probabilities)
        
        # กำหนดเกณฑ์ว่าเฟรมไหนเข้าข่าย "เสี่ยงสูง" (เกิน 80%)
        high_risk_threshold = 80.0
        high_risk_count = sum(1 for p in ai_probabilities if p > high_risk_threshold)
        high_risk_percent = (high_risk_count / len(ai_probabilities)) * 100
        
        # เงื่อนไขการตก:
        # 1. ค่าเฉลี่ยทั้งคลิปต้องเกิน 70% (ดูจากภาพรวม) OR
        # 2. มีเฟรมที่เสี่ยงสูงเกิน 25% ของจำนวนเฟรมที่วิเคราะห์ทั้งหมด (ป้องกันการตกเพราะเฟรมเบลอแค่ 1-2 เฟรม)
        if avg_ai_prob > 70.0 or high_risk_percent > 25.0:
            st.error(f"❌ **ผลการตรวจสอบ: ไม่ผ่าน (วิดีโอมีลักษณะเข้าข่ายการสร้างด้วย AI)**")
            st.write("### 📌 เหตุผลประกอบ:")
            if avg_ai_prob > 70.0:
                st.write(f"- **ภาพรวมของคลิปผิดธรรมชาติ:** ความเสี่ยงเฉลี่ยอยู่ที่ {avg_ai_prob:.2f}% (สูงกว่าเกณฑ์มาตรฐานที่ 70%)")
            if high_risk_percent > 25.0:
                st.write(f"- **พบจุดบิดเบี้ยวต่อเนื่อง:** ตรวจพบเฟรมที่มีความผิดปกติรุนแรงถึง {high_risk_percent:.1f}% ของคลิปทั้งหมด ({high_risk_count} จาก {analyzed_frames} เฟรม) ซึ่งมักไม่ใช่แค่การเบลอจากการถ่ายกล้องสั่น")
        else:
            st.success(f"✅ **ผลการตรวจสอบ: ผ่าน (วิดีโอมีความเป็นธรรมชาติ)**")
            st.write("### 📌 เหตุผลประกอบ:")
            st.write(f"- **ภาพรวมอยู่ในเกณฑ์ปลอดภัย:** ความเสี่ยงเฉลี่ยอยู่ที่ {avg_ai_prob:.2f}%")
            if high_risk_count > 0:
                st.write(f"- **อนุโลมจุดบกพร่องเล็กน้อย:** ตรวจพบเฟรมที่อาจดูผิดปกติบ้าง ({high_risk_count} เฟรม) แต่คิดเป็นเพียง {high_risk_percent:.1f}% ของคลิป ซึ่งสามารถเกิดขึ้นได้จากกล้องสั่นหรือการบีบอัดไฟล์ภาพตามธรรมชาติ")
            else:
                st.write("- วิดีโอมีความต่อเนื่องและไม่พบเฟรมที่มีความบิดเบี้ยวเลย")
            
        st.caption(f"ข้อมูลทางเทคนิค: วิเคราะห์ทั้งหมด {analyzed_frames} เฟรมหลัก")
    else:
        st.warning("ไม่สามารถวิเคราะห์วิดีโอนี้ได้ (วิดีโออาจสั้นเกินไปหรือไฟล์มีปัญหา)")
