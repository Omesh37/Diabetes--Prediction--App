import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

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


if st.button("Predict"):
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

    if prediction == 1:
        st.error("⚠️ The model predicts: Diabetic")
    else:
        st.success("✅ The model predicts: Not Diabetic")

st.markdown('</div>', unsafe_allow_html=True)
