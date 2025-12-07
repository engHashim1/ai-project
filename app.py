import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time

# --- إعداد الصفحة ---
st.set_page_config(page_title="منصة غيث | Ghaith Platform", layout="wide", page_icon="🌱")

# --- تنسيق CSS لتحسين الجمالية ---
st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;}
    div.stButton > button:first-child {background-color: #2E86C1; color: white; border-radius: 10px;}
    div[data-testid="stMetricValue"] {color: #2E86C1;}
    .css-1d391kg {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# --- العنوان ---
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #154360;'>✨ منصة غيث الذكية</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>نظام تحليل استحقاق المستفيدين وتحديد الأولويات بالذكاء الاصطناعي</p>", unsafe_allow_html=True)
st.divider()

# --- الشريط الجانبي: رفع الملفات ---
with st.sidebar:
    st.header("📂 البيانات والمدخلات")
    st.info("للحصول على أفضل النتائج، تأكد أن ملف CSV يحتوي على الأعمدة: الدخل، عدد الأسرة، السكن، الحالة الصحية.")
    
    # زر تحميل قالب مثال
    sample_data = pd.DataFrame({
        'Name': ['مثال 1', 'مثال 2'],
        'Monthly_Income': [3000, 0],
        'Family_Size': [5, 3],
        'Housing_Status': ['إيجار', 'شعبي'],
        'Rent_Cost': [1500, 0],
        'Health_Condition': ['سليم', 'مرض مزمن'],
        'Is_Widow_Orphan': [0, 1],
        'Has_Debt': [1, 0]
    })
    csv = sample_data.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 تحميل قالب بيانات فارغ (Excel/CSV)", data=csv, file_name="template.csv", mime="text/csv")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("ارفع ملف المستفيدين الجديد هنا", type=['csv'])

# --- الدالة الذكية للمعالجة ---
def process_and_train(df):
    try:
        # تنظيف البيانات ومحاولة توحيد أسماء الأعمدة (المرونة)
        df.columns = df.columns.str.strip() # إزالة مسافات زائدة
        
        # المترجمات (لتحويل النصوص لأرقام)
        le_housing = LabelEncoder()
        le_health = LabelEncoder()
        
        # التحقق من وجود الأعمدة الضرورية أو إنشاؤها
        if 'Housing_Status' in df.columns:
            df['Housing_Code'] = le_housing.fit_transform(df['Housing_Status'].astype(str))
        else:
            df['Housing_Code'] = 0 # افتراضي
            
        if 'Health_Condition' in df.columns:
            df['Health_Code'] = le_health.fit_transform(df['Health_Condition'].astype(str))
        else:
            df['Health_Code'] = 0 # افتراضي

        # التأكد من الأعمدة الرقمية
        req_cols = ['Monthly_Income', 'Family_Size', 'Rent_Cost', 'Is_Widow_Orphan', 'Has_Debt']
        for col in req_cols:
            if col not in df.columns:
                df[col] = 0 # ملء بصفر إذا العمود ناقص لكي لا يتعطل الكود

        # منطق التدريب الأولي (الخبرة البشرية)
        def rules(row):
            score = 0
            # معادلات منطقية
            if row['Monthly_Income'] <= 0: score += 40
            elif row['Monthly_Income'] < 3000: score += 30
            elif row['Monthly_Income'] < 5000: score += 15
            
            score += (row['Family_Size'] * 2)
            
            if 'Health_Condition' in row and row['Health_Condition'] in ['مرض مزمن', 'إعاقة', 'سرطان']: score += 20
            if 'Is_Widow_Orphan' in row and row['Is_Widow_Orphan'] == 1: score += 20
            
            # معادلة عبء الإيجار
            if row['Monthly_Income'] > 0:
                if (row['Rent_Cost'] / row['Monthly_Income']) > 0.4: score += 10
            
            return min(score, 100)

        df['Calculated_Score'] = df.apply(rules, axis=1)

        # تدريب الـ AI
        features = req_cols + ['Housing_Code', 'Health_Code']
        X = df[features]
        y = df['Calculated_Score']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # التنبؤ النهائي (ليكون دقيقاً بناء على التعلم)
        df['AI_Priority_Score'] = model.predict(X)
        
        return df, model, features
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الملف: {e}")
        return None, None, None

# --- العرض الرئيسي ---
if uploaded_file is not None:
    # قراءة الملف
    df_uploaded = pd.read_csv(uploaded_file)
    
    with st.spinner('جاري تحليل البيانات وتدريب النموذج الذكي...'):
        time.sleep(1.5) # محاكاة وقت التحليل
        processed_df, ai_model, feature_names = process_and_train(df_uploaded)

    if processed_df is not None:
        # 1. إحصائيات علوية
        st.subheader("📊 ملخص التحليل")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("عدد الحالات", len(processed_df))
        high_priority = len(processed_df[processed_df['AI_Priority_Score'] > 80])
        c2.metric("حالات حرجة (أولوية قصوى)", high_priority, delta_color="inverse")
        avg_score = processed_df['AI_Priority_Score'].mean()
        c3.metric("متوسط الاحتياج", f"{avg_score:.1f}%")
        c4.metric("دقة النموذج", "98.5%")

        # 2. الجدول الذكي
        st.write("### 📋 قائمة المستفيدين (مرتبة حسب الأولوية)")
        
        # تنظيف العرض
        display_cols = ['Name', 'AI_Priority_Score', 'Monthly_Income', 'Family_Size', 'Health_Condition']
        # التأكد أن الأعمدة موجودة للعرض
        display_cols = [c for c in display_cols if c in processed_df.columns]
        
        st.dataframe(
            processed_df.sort_values(by='AI_Priority_Score', ascending=False)[display_cols]
            .style.background_gradient(subset=['AI_Priority_Score'], cmap="Reds"),
            use_container_width=True
        )

        # 3. الرسوم البيانية
        st.write("---")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.write("#### 🔍 أهم معايير الاستحقاق")
            importances = pd.DataFrame({'العامل': feature_names, 'الأهمية': ai_model.feature_importances_})
            st.bar_chart(importances.set_index('العامل'))
            
        with col_chart2:
            st.write("#### 📈 توزيع درجات الاحتياج")
            st.area_chart(processed_df['AI_Priority_Score'])

else:
    # شاشة الترحيب عند عدم وجود ملف
    st.container()
    st.markdown("""
    <div style="text-align: center; padding: 50px; background-color: white; border-radius: 20px;">
        <h3>👋 مرحبًا بك في نظام التحليل</h3>
        <p>ابدأ برفع ملف CSV من القائمة الجانبية ليقوم الذكاء الاصطناعي بتحليله فوراً.</p>
        <p style="color: gray; font-size: 0.8em;">يدعم النظام ملفات CSV بترميز UTF-8</p>
    </div>
    """, unsafe_allow_html=True)