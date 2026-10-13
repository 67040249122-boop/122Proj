# 2. เขียนโค้ดแอปพลิเคชัน Streamlit ลงในไฟล์ app.py
streamlit_code = """import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Student Course Completion Predictor",
    page_icon="🎓",
    layout="centered"
)

# โหลดโมเดลและ Scaler
@st.cache_resource
def load_assets():
    model = joblib.load('student_completion_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_assets()
    st.title("🎓 ระบบทำนายการเรียนสำเร็จคอร์สออนไลน์")
    st.write("กรุณากรอกข้อมูลพฤติกรรมการเรียนของผู้เรียนเพื่อวิเคราะห์โอกาสในการเรียนสำเร็จ")
    
    st.markdown("---")
    
    # รับข้อมูลจากผู้ใช้งาน (Input fields)
    col1, col2 = st.columns(2)
    
    with col1:
        login_days = st.number_input("จำนวนวันที่ล็อกอินเข้าสู่ระบบ (วัน)", min_value=0, max_value=30, value=15, step=1)
        video_hours = st.number_input("จำนวนชั่วโมงที่ดูวิดีโอ (ชั่วโมง)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
        quiz_average = st.number_input("คะแนนสอบควิซเฉลี่ย (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        
    with col2:
        assignments_submitted = st.number_input("จำนวนการบ้านที่ส่ง (งาน)", min_value=0, max_value=10, value=5, step=1)
        forum_posts = st.number_input("จำนวนกระทู้ที่เขียนในฟอรัม (โพสต์)", min_value=0, max_value=50, value=2, step=1)
        course_level = st.selectbox(
            "ระดับความยากของหลักสูตร",
            options=["Beginner", "Intermediate", "Advanced"]
        )
    
    # แปลงระดับหลักสูตรเป็นตัวเลขตัวสเกล (Ordinal encoding)
    level_mapping = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
    course_level_encoded = level_mapping[course_level]
    
    st.markdown("---")
    
    # ปุ่มกดคำนวณทำนายผล
    if st.button("🔮 ทำนายผลลัพธ์", use_container_width=True):
        # จัดกลุ่มข้อมูลนำเข้า
        input_data = pd.DataFrame([[
            login_days, 
            video_hours, 
            quiz_average, 
            assignments_submitted, 
            forum_posts, 
            course_level_encoded
        ]], columns=['login_days', 'video_hours', 'quiz_average', 'assignments_submitted', 'forum_posts', 'course_level_encoded'])
        
        # สเกลข้อมูลเข้าตามมาตรฐานโมเดล
        input_scaled = scaler.transform(input_data)
        
        # ทำนายผล
        prediction = model.predict(input_scaled)[0]
        prediction_prob = model.predict_proba(input_scaled)[0][1]
        
        # แสดงผลลัพธ์
        st.subheader("📊 ผลการทำนาย")
        if prediction == 1:
            st.success(f"🟢 ผู้เรียนรายนี้มีแนวโน้มที่จะ **'เรียนจบ' (Completed)** สูงมาก! (โอกาสสำเร็จ: {prediction_prob*100:.2f}%)")
        else:
            st.error(f"🔴 ผู้เรียนรายนี้มีความเสี่ยงที่จะ **'เรียนไม่จบ' (Not Completed)** (โอกาสสำเร็จเพียง: {prediction_prob*100:.2f}%)")
            
except FileNotFoundError:
    st.error("⚠️ ไม่พบไฟล์โมเดลหรือสเกลเลอร์ กรุณาตรวจสอบให้แน่ใจว่าได้อัปโหลดไฟล์ 'student_completion_model.joblib' และ 'scaler.joblib' เรียบร้อยแล้ว")
"""

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(streamlit_code)

print("สร้างไฟล์แอปพลิเคชัน 'app.py' เรียบร้อยแล้ว!")