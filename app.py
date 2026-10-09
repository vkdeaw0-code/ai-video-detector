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

# 1. ตั้งค่าหน้าเพจ
st.set_page_config(page_title="AI Video Detector", page_icon="🎥", layout="wide")
st.title("🎥 ระบบตรวจจับและคัดกรองคลิปวิดีโอ AI")
st.write("ระบบวิเคราะห์ความผิดปกติของภาพ (เน้นจุดที่บิดเบี้ยวหรือเฟรมภาพที่ไม่ต่อเนื่อง)")

# 2. โหลดโมเดล (ใช้ Cache เพื่อความรวดเร็วและไม่โหลดซ้ำ)
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ใช้ EfficientNet-B0 ซึ่งกินสเปคน้อยและประมวลผลไว
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
    # หากคุณมีไฟล์ Weights (.pth) ของตัวเอง ให้ใส่โค้ดโหลดตรงนี้ เช่น model.load_state_dict(torch.load('weights.pth'))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# 3. เตรียมการแปลงรูปภาพ (ขนาด 224x224 ตามมาตรฐาน EfficientNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. ส่วนอัปโหลดไฟล์
uploaded_file = st.file_uploader("อัปโหลดวิดีโอของคุณที่นี่ (MP4, MOV, AVI)", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    st.info("กำลังประมวลผล... กรุณารอสักครู่")
    progress_bar = st.progress(0)
    
    # บันทึกไฟล์ชั่วคราวเพื่อใช้กับ OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # เทคนิคเพิ่มความไว: ตรวจสอบแค่ 1 เฟรม ทุกๆ 1 วินาที (ข้ามเฟรม)
    frames_to_skip = fps if fps > 0 else 30 
    
    ai_probabilities = []
    frame_count = 0
    analyzed_frames = 0
    
    with torch.no_grad(): # ปิดการคำนวณ Gradient เพื่อประหยัด Memory มหาศาล
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # วิเคราะห์เฉพาะเฟรมที่กำหนด
            if frame_count % frames_to_skip == 0:
                # แปลงสี BGR (OpenCV) เป็น RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # นำเข้าโมเดล
                input_tensor = transform(pil_image).unsqueeze(0).to(device)
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # สมมติว่า Class 0 = Real, Class 1 = AI
                ai_prob = probabilities[0][1].item() * 100
                ai_probabilities.append(ai_prob)
                analyzed_frames += 1
                
                # อัปเดตหลอดโหลด
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
            
            frame_count += 1
            
            # ล้างหน่วยความจำทุกๆ 50 เฟรมที่วิเคราะห์ ป้องกันเว็บร่วง
            if analyzed_frames % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    cap.release()
    progress_bar.progress(1.0)
    
    # 5. สรุปผลลัพธ์ (ตรรกะแบบไม่เข้มงวดมาก)
    if len(ai_probabilities) > 0:
        avg_ai_prob = sum(ai_probabilities) / len(ai_probabilities)
        max_ai_prob = max(ai_probabilities)
        
        # เงื่อนไขการตัดสิน: เฉลี่ย > 60% หรือ มีบางเฟรมพังหนักมาก (> 90%)
        if avg_ai_prob > 60.0 or max_ai_prob > 90.0:
            st.error(f"❌ **ผลการตรวจสอบ: ไม่ผ่าน (ตรวจพบความเป็นไปได้ของ AI สูง)**")
            st.write("### 📌 เหตุผลประกอบ:")
            st.write(f"- **ความเสี่ยงเฉลี่ยทั้งคลิป:** {avg_ai_prob:.2f}% (เกินเกณฑ์ 60%)")
            if max_ai_prob > 90.0:
                st.write(f"- **พบเฟรมที่มีความผิดเพี้ยนรุนแรง:** ตรวจพบความเป็นไปได้ถึง {max_ai_prob:.2f}% ในบางช่วงของวิดีโอ ซึ่งมักเกิดจากจุดที่บิดเบี้ยว มือแปลก หรือวัตถุหายฉับพลัน")
        else:
            st.success(f"✅ **ผลการตรวจสอบ: ผ่าน (วิดีโอมีความเป็นธรรมชาติสูง)**")
            st.write("### 📌 เหตุผลประกอบ:")
            st.write(f"- **ความเสี่ยงเฉลี่ยทั้งคลิป:** {avg_ai_prob:.2f}% (อยู่ในเกณฑ์ปลอดภัย)")
            st.write("- ไม่พบเฟรมที่มีความผิดปกติหรือการบิดเบี้ยวอย่างรุนแรงตลอดทั้งคลิป")
            
        st.write(f"*(ดึงข้อมูลมาวิเคราะห์ทั้งหมด {analyzed_frames} เฟรมหลัก จากความยาววิดีโอทั้งหมด)*")
    else:
        st.warning("ไม่สามารถวิเคราะห์วิดีโอนี้ได้ (วิดีโออาจสั้นเกินไปหรือไฟล์มีปัญหา)")
