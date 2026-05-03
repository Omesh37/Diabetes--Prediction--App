import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# Page config
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

if 'splash_done' not in st.session_state:
    st.session_state.splash_done = False

if not st.session_state.splash_done:
    st.markdown(
        """
        <style>
        .splash {
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background:
                radial-gradient(circle at 50% 40%, rgba(255, 79, 79, 0.10) 0%, rgba(14, 17, 23, 0.92) 60%),
                radial-gradient(circle at 25% 75%, rgba(88, 166, 255, 0.08) 0%, rgba(14, 17, 23, 0.0) 55%);
            backdrop-filter: blur(14px);
            z-index: 9999;
            margin: 0;
            padding: 0;
            pointer-events: none;
            animation: fadeOutSplash 2.8s ease forwards;
        }
        .splash-inner {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            animation: splashIn 420ms ease-out both;
        }
        .heart-wrap {
            position: relative;
            width: 180px;
            height: 180px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .ring {
            position: absolute;
            inset: 0;
            border-radius: 999px;
            border: 2px solid rgba(255, 79, 79, 0.22);
            filter: drop-shadow(0 0 12px rgba(255, 79, 79, 0.18));
            animation: ringPulse 1.15s ease-out infinite;
            opacity: 0.95;
        }
        .ring.r2 {
            animation-delay: 0.58s;
            opacity: 0.75;
        }
        @keyframes fadeOutSplash {
            0% { opacity: 1; visibility: visible; }
            68% { opacity: 1; }
            100% { opacity: 0; visibility: hidden; }
        }
        @keyframes splashIn {
            from { opacity: 0; transform: translateY(10px) scale(0.98); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes heartBeat {
            0% { transform: scale(1); }
            10% { transform: scale(1.24); }
            20% { transform: scale(1); }
            34% { transform: scale(1.16); }
            100% { transform: scale(1); }
        }
        @keyframes ringPulse {
            0% { transform: scale(0.25); opacity: 0; }
            18% { opacity: 0.55; }
            100% { transform: scale(1.20); opacity: 0; }
        }
        @keyframes ecgMove {
            0% { background-position: 0% 0; opacity: 0.45; }
            50% { opacity: 0.75; }
            100% { background-position: 200% 0; opacity: 0.45; }
        }
        .heart {
            font-size: 7.2rem;
            animation: heartBeat 1.05s cubic-bezier(0.2, 0.9, 0.2, 1) infinite;
            filter: drop-shadow(0 0 26px rgba(255, 79, 79, 0.55));
        }
        .ecg {
            width: 300px;
            height: 2px;
            border-radius: 999px;
            background: linear-gradient(
                90deg,
                rgba(88, 166, 255, 0) 0%,
                rgba(88, 166, 255, 0) 20%,
                rgba(88, 166, 255, 0.95) 50%,
                rgba(88, 166, 255, 0) 80%,
                rgba(88, 166, 255, 0) 100%
            );
            background-size: 200% 100%;
            animation: ecgMove 1.1s linear infinite;
            opacity: 0.65;
        }
        .splash-title {
            margin-top: 2px;
            color: #c9d1d9;
            font-weight: 800;
            letter-spacing: 0.3px;
        }
        .splash-subtitle {
            color: #8b949e;
            font-size: 0.95rem;
        }
        @media (prefers-reduced-motion: reduce) {
            .ring, .heart, .ecg { animation: none !important; }
            .splash { animation-duration: 1ms !important; }
        }
        [data-testid="stHeader"] {
            display: none;
        }
        </style>
        <div class="splash">
            <div class="splash-inner">
                <div class="heart-wrap">
                    <div class="ring r1"></div>
                    <div class="ring r2"></div>
                    <div class="heart">❤️</div>
                </div>
                <div class="ecg"></div>
                <div class="splash-title">Loading</div>
                <div class="splash-subtitle">Preparing predictor…</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.session_state.splash_done = True

# Custom UI styling
st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            height: 100%;
            margin: 0;
            padding: 0;
            background: #0E1117;
            background-attachment: fixed;
        }
        .stApp {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 32px 20px;
            background: #0E1117;
        }

        /* Fancy prediction result */
        .result-card {
            margin-top: 18px;
            padding: 18px 18px;
            border-radius: 18px;
            border: 1px solid rgba(88, 166, 255, 0.18);
            background: linear-gradient(135deg, rgba(22, 27, 34, 0.78) 0%, rgba(13, 17, 23, 0.62) 100%);
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
            position: relative;
            overflow: hidden;
            animation: resultPop 420ms ease-out both;
        }
        .result-card.fullscreen {
            width: min(980px, 100%);
            padding: 30px 32px;
            border-radius: 24px;
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.45);
            animation: resultFloatIn 520ms cubic-bezier(0.2, 0.9, 0.2, 1) both;
        }
        .result-overlay {
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 28px 20px;
            background:
                radial-gradient(circle at 15% 20%, rgba(88, 166, 255, 0.18) 0%, rgba(14, 17, 23, 0.2) 40%),
                radial-gradient(circle at 85% 80%, rgba(255, 123, 114, 0.14) 0%, rgba(14, 17, 23, 0.25) 45%),
                rgba(10, 12, 16, 0.88);
            backdrop-filter: blur(10px);
            z-index: 9990;
            animation: overlayFadeIn 320ms ease-out both;
        }
        .back-button-wrap {
            margin-top: 18px;
            display: flex;
            justify-content: center;
        }
        .back-button {
            background: linear-gradient(135deg, rgba(88, 166, 255, 0.95), rgba(88, 166, 255, 0.7));
            color: #0b1117;
            border: 0;
            border-radius: 999px;
            padding: 10px 22px;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }
        .back-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.4);
        }
        .result-card::before {
            content: "";
            position: absolute;
            inset: -2px;
            background: radial-gradient(circle at 20% 20%, rgba(88, 166, 255, 0.18) 0%, rgba(88, 166, 255, 0.0) 55%);
            pointer-events: none;
            opacity: 0.9;
        }
        .result-card::after {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            opacity: 0.9;
        }
        .result-card.result-high { border-color: rgba(255, 123, 114, 0.35); }
        .result-card.result-high::after { background: linear-gradient(90deg, rgba(255, 123, 114, 0.0) 0%, rgba(255, 123, 114, 0.95) 50%, rgba(255, 123, 114, 0.0) 100%); }
        .result-card.result-low { border-color: rgba(63, 185, 80, 0.35); }
        .result-card.result-low::after { background: linear-gradient(90deg, rgba(63, 185, 80, 0.0) 0%, rgba(63, 185, 80, 0.95) 50%, rgba(63, 185, 80, 0.0) 100%); }

        .result-header {
            position: relative;
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .result-icon {
            font-size: 2.2rem;
            line-height: 1;
            animation: iconPop 520ms ease-out both;
        }
        .result-title {
            color: #c9d1d9;
            font-size: 1.15rem;
            font-weight: 850;
            margin: 0;
        }
        .result-subtitle {
            color: #8b949e;
            margin-top: 2px;
            font-size: 0.95rem;
        }
        .verdict-banner {
            margin-top: 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 12px 14px;
            border-radius: 14px;
            border: 1px solid rgba(88, 166, 255, 0.2);
            background: rgba(88, 166, 255, 0.08);
            animation: verdictReveal 650ms cubic-bezier(0.2, 0.9, 0.2, 1) both;
        }
        .verdict-label {
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #8b949e;
            font-weight: 700;
        }
        .verdict-text {
            font-size: clamp(1.05rem, 3vw, 1.45rem);
            font-weight: 900;
            letter-spacing: 0.2px;
            color: #c9d1d9;
        }
        .result-card.result-high .verdict-banner {
            border-color: rgba(255, 123, 114, 0.35);
            background: rgba(255, 123, 114, 0.12);
        }
        .result-card.result-high .verdict-text { color: #ff7b72; }
        .result-card.result-low .verdict-banner {
            border-color: rgba(63, 185, 80, 0.35);
            background: rgba(63, 185, 80, 0.12);
        }
        .result-card.result-low .verdict-text { color: #3fb950; }
        .pill {
            display: inline-block;
            margin-left: 8px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 850;
            letter-spacing: 0.2px;
            vertical-align: middle;
        }
        .pill-high {
            background: rgba(255, 123, 114, 0.12);
            color: #ff7b72;
            border: 1px solid rgba(255, 123, 114, 0.35);
        }
        .pill-low {
            background: rgba(63, 185, 80, 0.12);
            color: #3fb950;
            border: 1px solid rgba(63, 185, 80, 0.35);
        }

        .risk-label-row {
            position: relative;
            margin-top: 14px;
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 10px;
            color: #8b949e;
            font-size: 0.92rem;
        }
        .risk-value {
            color: #c9d1d9;
            font-weight: 800;
        }
        .risk-meter {
            position: relative;
            margin-top: 8px;
            height: 10px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.06);
            overflow: hidden;
        }
        .risk-fill {
            height: 100%;
            width: 100%;
            transform-origin: left;
            transform: scaleX(var(--risk, 0));
            background: linear-gradient(90deg, #3fb950 0%, #d29922 55%, #ff7b72 100%);
            animation: fillGrow 900ms cubic-bezier(0.2, 0.9, 0.2, 1) both;
        }
        .risk-caption {
            position: relative;
            margin-top: 10px;
            color: rgba(139, 148, 158, 0.95);
            font-size: 0.86rem;
        }
        .result-note {
            position: relative;
            margin-top: 10px;
            color: rgba(139, 148, 158, 0.9);
            font-size: 0.82rem;
        }

        @keyframes resultPop {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes resultFloatIn {
            from { opacity: 0; transform: translateY(18px) scale(0.98); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes overlayFadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes fillGrow {
            from { transform: scaleX(0); }
            to { transform: scaleX(var(--risk, 0)); }
        }
        @keyframes iconPop {
            0% { transform: scale(0.78); opacity: 0; }
            60% { transform: scale(1.12); opacity: 1; }
            100% { transform: scale(1); }
        }
        @keyframes verdictReveal {
            from { opacity: 0; transform: translateY(12px) scale(0.98); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        .hero {
            text-align: center;
            padding: 10px 0 18px 0;
            animation: pulseGlow 2.8s ease-in-out infinite;
        }
        .hero h1 {
            margin-bottom: 6px;
            color: #58a6ff;
            font-size: 2.4rem;
        }
        .hero p {
            margin-top: 0;
            color: #8b949e;
            font-size: 1rem;
        }
        .metric-box {
            background: linear-gradient(135deg, rgba(30, 40, 52, 0.8) 0%, rgba(22, 27, 34, 0.6) 100%);
            border-radius: 14px;
            padding: 14px 16px;
            border: 1px solid rgba(88, 166, 255, 0.2);
            box-shadow: 0 6px 18px rgba(88, 166, 255, 0.1);
            text-align: center;
            font-weight: 600;
            color: #58a6ff;
            min-height: 56px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseGlow {
            0%, 100% { transform: translateY(0px); filter: drop-shadow(0 0 0 rgba(88, 166, 255, 0.0)); }
            50% { transform: translateY(-2px); filter: drop-shadow(0 8px 12px rgba(88, 166, 255, 0.2)); }
        }
        .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #0969da 0%, #1f6feb 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 1rem;
            font-weight: 700;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 18px rgba(88, 166, 255, 0.3);
        }
        @media (max-width: 640px) {
            .result-header {
                flex-direction: column;
                align-items: flex-start;
            }
            .verdict-banner {
                flex-direction: column;
                align-items: flex-start;
            }
            .risk-label-row {
                flex-direction: column;
                align-items: flex-start;
                gap: 4px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load saved objects
model = joblib.load("random_forest_diabetes.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.markdown(
    """
    <div class="hero">
        <h1>🩺 Diabetes Prediction App</h1>
        <p>Enter a few health details to get a quick prediction</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-box">⚡ Fast prediction</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box">📊 Model-backed</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box">🔒 Private input</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("Provide the following details:")
# Example input fields (adapt to your dataset)
age = st.slider("Age", 18, 100, 30)
smoking_history = st.selectbox("Smoking History", ['current','ever','former','never','not current','No Info'])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
blood_glucose_level = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
hypertension = st.selectbox("Hypertension", ['NO','YES'])
heart_disease = st.selectbox("Heart Disease", ['NO','YES'])
gender = st.selectbox("Gender", ["Male", "Female"])
HbA1c_level = st.slider("HbA1c_level(%)",5.0,10.0,5.7,0.1)
# Convert categorical to numeric
gender_num = 1 if gender == "Male" else 0

if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "result_html" not in st.session_state:
    st.session_state.result_html = ""

result_slot = st.empty()


if st.button("Predict"):
    with st.spinner("Analyzing..."):
        time.sleep(0.15)

    # Build raw input dict
    raw_input = {
        'age': age,
        'bmi': bmi,
        'blood_glucose_level': blood_glucose_level,
        'hypertension': 1 if hypertension == "YES" else 0,
        'heart_disease': 1 if heart_disease == "YES" else 0,
        'HbA1c_level': HbA1c_level,
    }

    # One-hot encode gender
    raw_input['gender_Male'] = 1 if gender == "Male" else 0
    raw_input['gender_Female'] = 1 if gender == "Female" else 0

    # One-hot encode smoking_history
    for cat in ['current','ever','former','never','not current','No Info']:
        raw_input[f'smoking_history_{cat}'] = 1 if smoking_history == cat else 0

    # Convert to DataFrame
    input_data = pd.DataFrame([raw_input])

    # Align with training columns
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[columns]

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    proba_diabetes = None
    if hasattr(model, "predict_proba"):
        try:
            proba_diabetes = float(model.predict_proba(scaled_input)[0][1])
        except Exception:
            proba_diabetes = None

    risk = proba_diabetes if proba_diabetes is not None else (1.0 if int(prediction) == 1 else 0.0)
    risk = float(np.clip(risk, 0.0, 1.0))
    risk_pct = int(round(risk * 100))

    if int(prediction) == 1:
        result_class = "result-high"
        pill_class = "pill-high"
        icon = "⚠️"
        verdict = "Diabetic"
        subtitle = "Higher diabetes risk signal detected by the model."
    else:
        result_class = "result-low"
        pill_class = "pill-low"
        icon = "✅"
        verdict = "Not Diabetic"
        subtitle = "Lower diabetes risk signal detected by the model."

    caption = (
        f"{risk_pct}% estimated probability (model output)"
        if proba_diabetes is not None
        else "Estimated probability unavailable for this model"
    )

    st.session_state.result_html = f"""
        <div class="result-overlay">
            <div class="result-card {result_class} fullscreen">
                <div class="result-header">
                    <div class="result-icon">{icon}</div>
                    <div>
                        <div class="result-title">Prediction <span class="pill {pill_class}">{verdict}</span></div>
                        <div class="result-subtitle">{subtitle}</div>
                    </div>
                </div>
                <div class="verdict-banner">
                    <span class="verdict-label">Result</span>
                    <span class="verdict-text">{verdict}</span>
                </div>
                <div class="risk-label-row">
                    <span>Estimated risk</span>
                    <span class="risk-value">{risk_pct}%</span>
                </div>
                <div class="risk-meter" style="--risk:{risk:.3f}">
                    <div class="risk-fill"></div>
                </div>
                <div class="risk-caption">{caption}</div>
                <div class="result-note">For informational use only — not medical advice.</div>
                <div class="result-actions">
                    <form method="get">
                        <input type="hidden" name="back" value="1" />
                        <button type="submit" class="back-button">Back</button>
                    </form>
                </div>
            </div>
        </div>
        """
    st.session_state.show_result = True

if st.session_state.show_result:
    params = st.query_params
    if params.get("back") in (["1"], "1"):
        st.session_state.show_result = False
        st.session_state.result_html = ""
        st.query_params.clear()
        result_slot.empty()
        st.experimental_rerun()
    result_slot.markdown(st.session_state.result_html, unsafe_allow_html=True)
    if st.session_state.get("back_button"):
        st.session_state.show_result = False
        st.session_state.result_html = ""
        st.session_state.pop("back_button", None)
        result_slot.empty()
        st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)
