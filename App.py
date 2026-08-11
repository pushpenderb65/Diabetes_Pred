
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f9ff 0%, #ffffff 45%);
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
        padding: 2rem 1.75rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.75rem;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.25);
    }
    .hero h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 800;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        opacity: 0.92;
        font-size: 0.98rem;
    }

    /* Section card */
    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.4rem 1.5rem 0.6rem 1.5rem;
        border: 1px solid #eef2f7;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.4rem;
    }
    .section-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #1e293b;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb, #06b6d4);
        color: white;
        border: none;
        padding: 0.75rem 0;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.05rem;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3);
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        color: white;
        border: none;
    }

    /* Result cards */
    .result-card {
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .result-risk {
        background: #fef2f2;
        border: 1.5px solid #fecaca;
    }
    .result-safe {
        background: #f0fdf4;
        border: 1.5px solid #bbf7d0;
    }
    .result-title {
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .result-risk .result-title { color: #b91c1c; }
    .result-safe .result-title { color: #15803d; }
    .result-sub {
        color: #475569;
        font-size: 0.95rem;
    }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load saved artifacts (model, scaler, training column order)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgboost_balanced_model.joblib")
    scaler = joblib.load("scaler.joblib")
    training_columns = joblib.load("training_columns.joblib")
    return model, scaler, training_columns

model, scaler, training_columns = load_artifacts()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🩺 About")
    st.write(
        "Apni basic health details daalkar turant jaanein ki aapko "
        "diabetes ka risk kitna ho sakta hai — AI model ke through "
        "kuch second me analysis."
    )
    st.markdown("---")
    st.markdown("### 📋 Kaise use karein")
    st.write(
        "1. Apni health details bharein\n"
        "2. Threshold adjust karein (optional)\n"
        "3. **Predict** button dabayein"
    )
    st.markdown("---")
    st.caption("⚠️ Ye tool sirf educational/demo purpose ke liye hai. Medical advice ke liye doctor se consult karein.")

# ------------------------------------------------------------------
# Hero header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🩺 Diabetes Risk Predictor</h1>
        <p>Apni health details daalein aur AI-powered risk prediction turant paayein</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Personal details
# ------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Personal Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
with col2:
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
with col3:
    smoking_history = st.selectbox(
        "Smoking History",
        ["never", "No Info", "current", "former", "ever", "not current"],
    )
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Medical history
# ------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">❤️ Medical History</div>', unsafe_allow_html=True)

col4, col5 = st.columns(2)
with col4:
    hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
with col5:
    heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Vitals
# ------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Vitals</div>', unsafe_allow_html=True)

know_bmi = st.checkbox("Mujhe apna BMI pata hai", value=True)

if know_bmi:
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=22.0, step=0.1)
else:
    st.caption("BMI pata nahi? Height aur weight daalein, hum calculate kar denge 👇")
    hcol, wcol = st.columns(2)
    with hcol:
        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5)
    with wcol:
        weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=65.0, step=0.5)

    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1) if height_m > 0 else 0.0
    st.metric("Calculated BMI", f"{bmi}")

col7, col8 = st.columns(2)
with col7:
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
with col8:
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=100, step=1)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Threshold
# ------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">⚙️ Advanced Setting</div>', unsafe_allow_html=True)
threshold = st.slider(
    "Decision threshold — is value se kam probability pe bhi 'Diabetic' predict hoga",
    min_value=0.05, max_value=0.95, value=0.5, step=0.05,
)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Preprocess: EXACTLY wahi steps jo notebook me train time pe kiye the
# ------------------------------------------------------------------
def preprocess_input(raw: dict, training_columns, scaler) -> np.ndarray:
    row = pd.DataFrame([raw])

    bin_edges = [0, 18.5, 25.0, 30.0, float("inf")]
    bin_labels = ["Underweight", "Normal weight", "Overweight", "Obesity"]
    row["bins"] = pd.cut(row["bmi"], bins=bin_edges, labels=bin_labels, right=False)

    cat_cols = ["gender", "smoking_history", "bins"]
    row = pd.get_dummies(row, columns=cat_cols, drop_first=True)

    row = row.reindex(columns=training_columns, fill_value=0)
    row = row.astype(int)

    scaled = scaler.transform(row)
    return scaled

# ------------------------------------------------------------------
# Predict button + result
# ------------------------------------------------------------------
predict_clicked = st.button("🔍 Predict Diabetes Risk", type="primary")

if predict_clicked:
    raw_input = {
        "gender": gender,
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
    }

    with st.spinner("Model analyzing your data..."):
        X_input = preprocess_input(raw_input, training_columns, scaler)
        proba = float(model.predict_proba(X_input)[0, 1])  # numpy float32 -> python float
        prediction = int(proba >= threshold)

    st.markdown("### 📈 Result")

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-card result-risk">
                <div class="result-title">⚠️ Diabetic Risk Detected</div>
                <div class="result-sub">Predicted probability of diabetes: <b>{proba:.1%}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-safe">
                <div class="result-title">✅ Low Diabetes Risk</div>
                <div class="result-sub">Predicted probability of diabetes: <b>{proba:.1%}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    m1, m2 = st.columns(2)
    m1.metric("Diabetes Probability", f"{proba:.1%}")
    m2.metric("Threshold Used", f"{threshold:.2f}")

    # ---------------- Gauge chart: risk probability ----------------
    gauge_color = "#dc2626" if prediction == 1 else "#16a34a"
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 34, "color": gauge_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#1e293b", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    gauge_fig.update_layout(
        height=260,
        margin=dict(t=20, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1e293b", "family": "sans-serif"},
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    # ---------------- Bar chart: vitals vs normal range ----------------
    st.markdown("#### 🔬 Aapke Vitals vs Normal Range")

    vitals_df = pd.DataFrame(
        {
            "Metric": ["BMI", "HbA1c (%)", "Glucose (mg/dL)"],
            "Aapki Value": [bmi, hba1c, glucose],
            "Normal Upper Limit": [25.0, 5.7, 140],
        }
    )

    bar_fig = go.Figure()
    bar_fig.add_trace(
        go.Bar(
            x=vitals_df["Metric"],
            y=vitals_df["Aapki Value"],
            name="Aapki Value",
            marker_color="#2563eb",
            text=vitals_df["Aapki Value"],
            textposition="outside",
        )
    )
    bar_fig.add_trace(
        go.Bar(
            x=vitals_df["Metric"],
            y=vitals_df["Normal Upper Limit"],
            name="Normal Upper Limit",
            marker_color="#94a3b8",
            text=vitals_df["Normal Upper Limit"],
            textposition="outside",
        )
    )
    bar_fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font={"color": "#1e293b", "family": "sans-serif"},
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    st.info(
        "ℹ️ Ye prediction ek machine learning model ka estimate hai, "
        "final diagnosis ke liye kripya certified doctor se consult karein.",
        icon="ℹ️",
    )
