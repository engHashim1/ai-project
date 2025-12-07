import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# --- 1. إعداد الصفحة والتصميم (CSS Injection) ---
st.set_page_config(page_title="منصة مُيسّر | Ghaith AI", layout="wide", page_icon="🌱")

# تخصيص التصميم ليصبح احترافياً جداً
st.markdown("""
<style>
    /* تغيير الخطوط والخلفيات */
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
    }
    
    /* تنظيف الواجهة العلوية */
    header {visibility: hidden;}
    
    /* تنسيق الكروت الإحصائية */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    /* تلوين القوائم */
    .stSidebar {
        background-color: #f8f9fa;
    }
    
    /* زر التحليل */
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        width: 100%;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. بناء نموذج الذكاء الاصطناعي (Core Logic) ---
@st.cache_data
def train_model(df):
    # تنظيف وتجهيز البيانات
    df.columns = df.columns.str.strip()
    le_dict = {} # لحفظ المترجمات
    
    # معالجة النصوص
    for col in ['Housing_Status', 'Health_Condition', 'Region']:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_Code'] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
    
    # الأعمدة المطلوبة (التعامل بمرونة)
    expected_cols = ['Monthly_Income', 'Family_Size', 'Rent_Cost', 'Is_Widow_Orphan', 'Has_Debt']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
            
    # القواعد المنطقية لتدريب الـ AI (Ground Truth Generation)
    def calculate_rules(row):
        score = 0
        if row['Monthly_Income'] == 0: score += 40
        elif row['Monthly_Income'] < 3000: score += 30
        elif row['Monthly_Income'] < 5000: score += 15
        
        score += (row['Family_Size'] * 2.5) # وزن عالٍ للأسرة
        
        if 'Health_Condition' in row and row['Health_Condition'] in ['مرض مزمن', 'إعاقة', 'سرطان']: score += 20
        if 'Is_Widow_Orphan' in row and row['Is_Widow_Orphan'] == 1: score += 15
        
        # معادلة الفقر المدقع (بعد خصم الإيجار)
        disposable = row['Monthly_Income'] - row['Rent_Cost']
        if disposable < 500: score += 10
        
        return min(score, 100)

    df['Training_Target'] = df.apply(calculate_rules, axis=1)
    
    # الميزات التي سيتعلم منها الـ AI
    features = expected_cols + [c for c in df.columns if '_Code' in c]
    
    X = df[features]
    y = df['Training_Target']
    
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X, y)
    
    return model, features, le_dict

# --- 3. القائمة الجانبية الاحترافية ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913520.png", width=80)
    st.markdown("### منصة غيث الذكية")
    
    selected = option_menu(
        menu_title=None,
        options=["لوحة القيادة", "تحليل الملفات", "محاكي السيناريو"],
        icons=["speedometer2", "cloud-upload", "sliders"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#2E86C1", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "right", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#2E86C1"},
        }
    )
    st.markdown("---")
    st.info("💡 **نصيحة:** استخدم 'محاكي السيناريو' لتجربة ذكاء النظام لحظياً.")

# --- 4. الصفحة الأولى: لوحة القيادة (Dashboard) ---
if selected == "لوحة القيادة":
    st.title("📊 النظرة الشاملة (Overview)")
    st.markdown("ملخص أداء الجمعية وتوزيع احتياجات المستفيدين")
    
    # بيانات وهمية للعرض الجمالي (Dashboard Dummy Data)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("إجمالي المستفيدين", "1,240", "+12%")
    col2.metric("الحالات الحرجة", "85", "6.4%")
    col3.metric("ميزانية الدعم", "450K SAR", "-2%")
    col4.metric("دقة الـ AI", "99.1%", "+0.5%")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("توزيع الفئات حسب الاحتياج")
        # رسم بياني تفاعلي (Donut Chart)
        labels = ['أولوية قصوى', 'احتياج متوسط', 'احتياج منخفض']
        values = [15, 55, 30]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker_colors=['#E74C3C', '#F1C40F', '#2ECC71'])])
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    with col_g2:
        st.subheader("اتجاه طلبات الدعم (شهرياً)")
        # رسم بياني خطي (Line Chart)
        months = ['يناير', 'فبراير', 'مارس', 'أبريل', 'مايو']
        requests = [120, 150, 130, 200, 250]
        fig2 = px.area(x=months, y=requests, color_discrete_sequence=['#2E86C1'])
        fig2.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300, xaxis_title="", yaxis_title="عدد الطلبات")
        st.plotly_chart(fig2, use_container_width=True)

# --- 5. الصفحة الثانية: تحليل الملفات (Core Feature) ---
elif selected == "تحليل الملفات":
    st.title("📂 المعالجة الذكية للبيانات")
    
    # تحميل القالب
    sample_csv = "Name,Monthly_Income,Family_Size,Rent_Cost,Housing_Status,Health_Condition,Is_Widow_Orphan,Has_Debt\nمستفيد 1,0,5,1500,إيجار,مرض مزمن,0,1\nمستفيد 2,8000,3,0,ملك,سليم,0,0"
    st.download_button("📥 تحميل قالب Excel فارغ", sample_csv, "template.csv", "text/csv")
    
    uploaded_file = st.file_uploader("ارفع ملف المستفيدين (CSV)", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        with st.spinner('جاري تشغيل خوارزميات الذكاء الاصطناعي...'):
            model, features, _ = train_model(df)
            
            # تجهيز البيانات للتوقع
            X_pred = df[features].fillna(0)
            df['AI_Score'] = model.predict(X_pred)
            
            # عرض النتائج
            st.success("✅ تم التحليل بنجاح!")
            
            tab1, tab2 = st.tabs(["📋 القائمة المرتبة", "🔍 تفسير النتائج"])
            
            with tab1:
                # تلوين الجدول حسب الأولوية
                styled_df = df.sort_values(by='AI_Score', ascending=False)[['Name', 'AI_Score', 'Monthly_Income', 'Family_Size']]
                st.dataframe(
                    styled_df.style.background_gradient(subset=['AI_Score'], cmap="Reds"),
                    use_container_width=True,
                    height=500
                )
            
            with tab2:
                # رسم بياني لأهم العوامل (Feature Importance)
                importance_df = pd.DataFrame({'Factor': features, 'Importance': model.feature_importances_})
                importance_df = importance_df.sort_values(by='Importance', ascending=True)
                
                fig_imp = px.bar(importance_df, x='Importance', y='Factor', orientation='h', title="ماذا أثر في قرار الـ AI؟", color='Importance', color_continuous_scale='Blues')
                st.plotly_chart(fig_imp, use_container_width=True)

# --- 6. الصفحة الثالثة: محاكي السيناريو (The WOW Factor) ---
elif selected == "محاكي السيناريو":
    st.title("🎛️ محاكي الذكاء الاصطناعي (Interactive Simulator)")
    st.markdown("جرب تغيير بيانات مستفيد افتراضي لترى كيف يتخذ النظام قراره لحظياً.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        val_income = st.slider("💰 الدخل الشهري", 0, 15000, 3000, step=500)
        val_rent = st.number_input("🏠 تكلفة الإيجار", 0, 5000, 1500)
    with c2:
        val_family = st.slider("👨‍👩‍👧‍👦 عدد أفراد الأسرة", 1, 15, 5)
        val_debt = st.checkbox("عليه ديون؟")
    with c3:
        val_health = st.selectbox("🏥 الحالة الصحية", ["سليم", "مرض مزمن", "إعاقة"])
        val_orphan = st.checkbox("أيتام / أرامل؟")

    # المنطق الخلفي للمحاكاة (Rule-based approximation for demo)
    # ملاحظة: هنا نستخدم القواعد مباشرة للمحاكاة السريعة بدون إعادة تدريب
    score = 0
    if val_income == 0: score += 40
    elif val_income < 3000: score += 30
    elif val_income < 5000: score += 15
    
    score += (val_family * 2.5)
    
    if val_health != "سليم": score += 20
    if val_orphan: score += 15
    if val_debt: score += 5
    if (val_income - val_rent) < 500: score += 10
    
    final_score = min(score, 100)
    
    st.divider()
    
    # عرض النتيجة بشكل عداد سرعة (Gauge Chart)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = final_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "نسبة الأولوية (AI Score)"},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2E86C1"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#D5F5E3'},
                {'range': [50, 80], 'color': '#FCF3CF'},
                {'range': [80, 100], 'color': '#FADBD8'}],
        }))
    
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # رسالة النظام
    if final_score > 80:
        st.error("⚠️ القرار: منح فوري (حالة طارئة)")
    elif final_score > 50:
        st.warning("⚠️ القرار: وضع في قائمة الانتظار")
    else:
        st.success("✅ القرار: حالة مكتفية (أولوية منخفضة)")
