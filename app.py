import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="NE-EpiGuard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a3a5c;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a3a5c, #2e6da4);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .disease-high {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
    }
    .disease-low {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
    }
    .doctor-box {
        background: #fff3cd;
        border-left: 5px solid #f39c12;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .advisory-box {
        background: #f0f8ff;
        border-left: 5px solid #1a3a5c;
        padding: 1.2rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model (cached) ─────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    from src.pipeline.predict_pipeline import PredictPipeline
    return PredictPipeline()

@st.cache_resource
def load_langchain():
    from src.langchain_helper import extract_data_from_pdf, generate_health_advisory
    return extract_data_from_pdf, generate_health_advisory

# ─── Header ──────────────────────────────────────────────────────
st.markdown('<div class="main-title">🦠 NE-EpiGuard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Waterborne Disease Prediction System — Northeast India</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png", width=80)
    st.markdown("### NE-EpiGuard")
    st.markdown("**Version**: 1.0.0")
    st.markdown("**Model**: Stacking Ensemble")
    st.markdown("**Macro F1**: 0.9212")
    st.markdown("**States**: 8 NE States")
    st.markdown("**Diseases**: 8 Classes")
    st.markdown("---")
    st.markdown("### Sample Reports")
    st.markdown("Download test PDFs:")
    for report in ["sample_report_cholera.pdf", "sample_report_typhoid.pdf", "sample_report_healthy.pdf"]:
        path = os.path.join("sample_reports", report)
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(
                    label=f"📄 {report.replace('sample_report_', '').replace('.pdf', '').title()}",
                    data=f,
                    file_name=report,
                    mime="application/pdf",
                    key=report
                )
    st.markdown("---")
    st.markdown("⚠️ *For educational purposes only. Not a substitute for medical advice.*")

# ─── Session State ───────────────────────────────────────────────
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = None
if 'advisory' not in st.session_state:
    st.session_state.advisory = None

# ─── Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Manual Input",
    "📄 PDF Upload",
    "📊 Prediction Results",
    "🤖 AI Health Advisory"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Manual Input
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Patient & Environmental Data Input")
    st.markdown("Fill in the details below and click **Predict Disease**.")

    with st.form("manual_input_form"):

        # Row 1 — Patient Info
        st.markdown("#### 👤 Patient & Location")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            state = st.selectbox("State", [
                "Assam", "Manipur", "Meghalaya", "Mizoram",
                "Nagaland", "Arunachal Pradesh", "Sikkim", "Tripura"
            ])
        with c2:
            age = st.number_input("Age", min_value=0, max_value=100, value=28)
        with c3:
            season = st.selectbox("Season", ["Monsoon", "Post-Monsoon", "Summer", "Winter"])
        with c4:
            month = st.slider("Month", 1, 12, 7)

        # Row 2 — Water Source
        st.markdown("#### 💧 Water Source & Sanitation")
        c1, c2, c3 = st.columns(3)
        with c1:
            water_source = st.selectbox("Water Source", [
                "River", "Pond", "Open Well", "Rainwater", "Tanker", "Borewell", "Piped"
            ])
        with c2:
            water_treatment = st.selectbox("Water Treatment", [
                "Untreated", "Boiled", "Filtered", "Chlorinated"
            ])
        with c3:
            handwashing = st.selectbox("Handwashing Practice", ["Never", "Sometimes", "Always"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            toilet_access = st.selectbox("Toilet Access", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with c2:
            open_defecation = st.number_input("Open Defecation Rate (%)", 0.0, 100.0, 35.0)
        with c3:
            sewage_pct = st.number_input("Sewage Treatment (%)", 0.0, 100.0, 15.0)
        with c4:
            flooding = st.selectbox("Flooding", [0, 1], format_func=lambda x: "Yes" if x else "No")

        # Row 3 — Water Quality
        st.markdown("#### 🔬 Water Quality Parameters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            wqi    = st.number_input("Water Quality Index", 0.0, 100.0, 25.0)
            ph     = st.number_input("pH", 0.0, 14.0, 6.5)
            turb   = st.number_input("Turbidity (NTU)", 0.0, 100.0, 30.0)
        with c2:
            do_    = st.number_input("Dissolved O₂ (mg/L)", 0.0, 20.0, 4.0)
            bod    = st.number_input("BOD (mg/L)", 0.0, 50.0, 18.0)
            tds    = st.number_input("TDS (mg/L)", 0.0, 3000.0, 900.0)
        with c3:
            fecal  = st.number_input("Fecal Coliform (CFU/100ml)", 0, 10000, 2000)
            total  = st.number_input("Total Coliform (CFU/100ml)", 0, 15000, 4500)
            nitrate= st.number_input("Nitrate (mg/L)", 0.0, 100.0, 40.0)
        with c4:
            fluoride = st.number_input("Fluoride (mg/L)", 0.0, 5.0, 1.2)
            arsenic  = st.number_input("Arsenic (µg/L)", 0.0, 100.0, 25.0)

        # Row 4 — Climate
        st.markdown("#### 🌧️ Climate Data")
        c1, c2, c3 = st.columns(3)
        with c1:
            temp = st.number_input("Avg Temperature (°C)", 0.0, 45.0, 28.0)
        with c2:
            rain = st.number_input("Avg Rainfall (mm)", 0.0, 1200.0, 600.0)
        with c3:
            humidity = st.number_input("Avg Humidity (%)", 0.0, 100.0, 85.0)

        # Row 5 — Symptoms
        st.markdown("#### 🩺 Reported Symptoms")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            s_diarrhea  = st.checkbox("Diarrhea")
            s_vomiting  = st.checkbox("Vomiting")
        with c2:
            s_fever     = st.checkbox("Fever")
            s_abdominal = st.checkbox("Abdominal Pain")
        with c3:
            s_dehydration = st.checkbox("Dehydration")
            s_jaundice    = st.checkbox("Jaundice")
        with c4:
            s_bloody = st.checkbox("Bloody Stool")
            s_rash   = st.checkbox("Skin Rash")

        submitted = st.form_submit_button("🔍 Predict Disease", use_container_width=True, type="primary")

    if submitted:
        patient_dict = {
            "state": state, "age": age, "season": season, "month": month,
            "water_source": water_source, "water_treatment": water_treatment,
            "handwashing_practice": handwashing, "toilet_access": toilet_access,
            "open_defecation_rate": open_defecation, "sewage_treatment_pct": sewage_pct,
            "flooding": flooding, "water_quality_index": wqi, "ph": ph,
            "turbidity_ntu": turb, "dissolved_oxygen_mg_l": do_,
            "bod_mg_l": bod, "fecal_coliform_per_100ml": fecal,
            "total_coliform_per_100ml": total, "tds_mg_l": tds,
            "nitrate_mg_l": nitrate, "fluoride_mg_l": fluoride, "arsenic_ug_l": arsenic,
            "avg_temperature_c": temp, "avg_rainfall_mm": rain, "avg_humidity_pct": humidity,
            "symptom_diarrhea": int(s_diarrhea), "symptom_vomiting": int(s_vomiting),
            "symptom_fever": int(s_fever), "symptom_abdominal_pain": int(s_abdominal),
            "symptom_dehydration": int(s_dehydration), "symptom_jaundice": int(s_jaundice),
            "symptom_bloody_stool": int(s_bloody), "symptom_skin_rash": int(s_rash),
        }

        with st.spinner("Predicting disease..."):
            try:
                pipeline = load_pipeline()
                df_input = pd.DataFrame([patient_dict])
                result   = pipeline.predict(df_input)
                st.session_state.prediction_result = result
                st.session_state.patient_data      = patient_dict
                st.session_state.advisory          = None
                st.success("✅ Prediction complete! Go to **📊 Prediction Results** tab.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ════════════════════════════════════════════════════════════════
# TAB 2 — PDF Upload
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Upload Patient Health Report (PDF)")
    st.markdown("Upload a structured health report PDF. Gemini AI will auto-extract all fields and run the prediction.")

    st.info("📥 Download sample PDFs from the sidebar to test this feature.")

    uploaded_file = st.file_uploader("Upload PDF Report", type=["pdf"])

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        if st.button("🔍 Extract & Predict", use_container_width=True, type="primary"):
            with st.spinner("Extracting data from PDF using Gemini AI..."):
                try:
                    extract_fn, advisory_fn = load_langchain()
                    patient_dict = extract_fn(uploaded_file)
                    st.markdown("#### Extracted Data")
                    st.json(patient_dict)
                except Exception as e:
                    st.error(f"PDF extraction failed: {e}")
                    patient_dict = None

            if patient_dict:
                with st.spinner("Running disease prediction..."):
                    try:
                        pipeline = load_pipeline()
                        df_input = pd.DataFrame([patient_dict])
                        result   = pipeline.predict(df_input)
                        st.session_state.prediction_result = result
                        st.session_state.patient_data      = patient_dict
                        st.session_state.advisory          = None
                        st.success("✅ Prediction complete! Go to **📊 Prediction Results** tab.")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

# ════════════════════════════════════════════════════════════════
# TAB 3 — Prediction Results
# ════════════════════════════════════════════════════════════════
with tab3:
    if st.session_state.prediction_result is None:
        st.info("No prediction yet. Please use **📋 Manual Input** or **📄 PDF Upload** tab first.")
    else:
        result = st.session_state.prediction_result
        disease = result['disease']
        prob    = result['probability']

        st.markdown("### Prediction Results")

        # Main result card
        css_class = "disease-high" if disease != "No_Disease" else "disease-low"
        display_disease = disease.replace("_", " ")
        st.markdown(
            f'<div class="{css_class}">🦠 {display_disease}<br><small style="font-size:1rem">Confidence: {prob:.1f}%</small></div>',
            unsafe_allow_html=True
        )
        st.markdown("")

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Predicted Disease", display_disease)
        with c2:
            st.metric("Confidence", f"{prob:.1f}%")
        with c3:
            symptom_count = sum([
                st.session_state.patient_data.get(f"symptom_{s}", 0)
                for s in ['diarrhea','vomiting','fever','abdominal_pain',
                          'dehydration','jaundice','bloody_stool','skin_rash']
            ])
            st.metric("Symptoms Present", f"{symptom_count}/8")
        with c4:
            urgency = "🔴 HIGH" if disease in ['Cholera', 'Leptospirosis'] else \
                      "🟡 MEDIUM" if disease != "No_Disease" else "🟢 LOW"
            st.metric("Urgency Level", urgency)

        # Doctor recommendation
        if result.get('doctor_info') and disease != 'No_Disease':
            st.markdown("---")
            st.markdown("#### 👨‍⚕️ Doctor Recommendation")
            doc = result['doctor_info']
            st.markdown(f"""
<div class="doctor-box">
<b>⚠️ High Confidence Prediction — Medical Consultation Recommended</b><br><br>
<b>Recommended Specialist:</b> {doc['specialist']}<br>
<b>Urgency:</b> {doc['urgency']}<br><br>
<b>Emergency Numbers:</b> 📞 <b>108</b> (Ambulance) &nbsp;|&nbsp; 📞 <b>104</b> (Health Helpline)
</div>
""", unsafe_allow_html=True)

        # Probability chart
        st.markdown("---")
        st.markdown("#### 📊 Disease Probability Distribution")
        probs = result['probabilities']
        prob_df = pd.DataFrame({
            'Disease': list(probs.keys()),
            'Probability (%)': list(probs.values())
        }).sort_values('Probability (%)', ascending=True)

        colors = ['#e74c3c' if d == disease else '#3498db' for d in prob_df['Disease']]
        fig = go.Figure(go.Bar(
            x=prob_df['Probability (%)'],
            y=prob_df['Disease'],
            orientation='h',
            marker_color=colors,
            text=[f"{v:.1f}%" for v in prob_df['Probability (%)']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Predicted Probability per Disease Class",
            xaxis_title="Probability (%)",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Patient summary
        if st.session_state.patient_data:
            with st.expander("📋 View Patient Data Summary"):
                pd_data = st.session_state.patient_data
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Location & Environment**")
                    st.write(f"State: {pd_data.get('state', 'N/A')}")
                    st.write(f"Season: {pd_data.get('season', 'N/A')}")
                    st.write(f"Flooding: {'Yes' if pd_data.get('flooding', 0) else 'No'}")
                    st.write(f"Water Source: {pd_data.get('water_source', 'N/A')}")
                    st.write(f"Water Treatment: {pd_data.get('water_treatment', 'N/A')}")
                    st.write(f"Handwashing: {pd_data.get('handwashing_practice', 'N/A')}")
                with c2:
                    st.markdown("**Key Water Quality**")
                    st.write(f"WQI: {pd_data.get('water_quality_index', 'N/A')}")
                    st.write(f"Fecal Coliform: {pd_data.get('fecal_coliform_per_100ml', 'N/A')} CFU/100ml")
                    st.write(f"Turbidity: {pd_data.get('turbidity_ntu', 'N/A')} NTU")
                    st.write(f"pH: {pd_data.get('ph', 'N/A')}")
                    st.write(f"Dissolved O₂: {pd_data.get('dissolved_oxygen_mg_l', 'N/A')} mg/L")

# ════════════════════════════════════════════════════════════════
# TAB 4 — AI Health Advisory
# ════════════════════════════════════════════════════════════════
with tab4:
    if st.session_state.prediction_result is None:
        st.info("No prediction yet. Please run a prediction first.")
    else:
        result  = st.session_state.prediction_result
        disease = result['disease']
        prob    = result['probability']

        st.markdown("### 🤖 AI Health Advisory")
        st.markdown(f"**Predicted Disease:** {disease.replace('_', ' ')} &nbsp;|&nbsp; **Confidence:** {prob:.1f}%")
        st.markdown("---")

        if st.session_state.advisory is None:
            if st.button("🤖 Generate WHO Health Advisory", use_container_width=True, type="primary"):
                with st.spinner("Generating advisory using Gemini AI + WHO guidelines..."):
                    try:
                        _, advisory_fn = load_langchain()
                        advisory = advisory_fn(
                            disease=disease,
                            probability=prob,
                            patient_data=st.session_state.patient_data
                        )
                        st.session_state.advisory = advisory
                        st.rerun()
                    except Exception as e:
                        st.error(f"Advisory generation failed: {e}")
        else:
            st.markdown(
                f'<div class="advisory-box">{st.session_state.advisory}</div>',
                unsafe_allow_html=True
            )
            st.markdown("")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Advisory",
                    data=st.session_state.advisory,
                    file_name=f"advisory_{disease}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                if st.button("🔄 Regenerate Advisory", use_container_width=True):
                    st.session_state.advisory = None
                    st.rerun()

        # Helpline quick reference
        st.markdown("---")
        st.markdown("#### 📞 Emergency Helplines")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.error("🚑 **108** — Emergency Ambulance\n\nFree | 24x7 | All NE States")
        with c2:
            st.warning("📞 **104** — Health Helpline\n\nFree | 24x7 | Medical Advice")
        with c3:
            st.info("🏥 **011-23921401** — NCDC\n\nNational Centre for Disease Control")

# ─── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem'>"
    "NE-EpiGuard v1.0 &nbsp;|&nbsp; Built by Sumeet Kumar Pal &nbsp;|&nbsp; "
    "⚠️ For educational purposes only — not a substitute for medical diagnosis"
    "</div>",
    unsafe_allow_html=True
)
