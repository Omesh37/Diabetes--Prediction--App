import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved objects
model = joblib.load("random_forest_diabetes.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Diabetes Prediction App")
st.markdown("Provide the following details:- ")
# Example input fields (adapt to your dataset)
age = st.slider("Age", 18, 100, 30)
smoking_history = st.selectbox("Somking History",['current','ever','former','never','not current'])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
blood_glucose_level = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
hypertension = st.selectbox("Hypertension", ['NO','YES'])
heart_disease = st.selectbox("Heart Disease", ['NO','YES'])
gender = st.selectbox("Gender", ["Male", "Female"])
HbA1c_level = st.slider("HbA1c_level(%)",5.0,10.0,5.7)
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
        st.success("⚠️ The model predicts: Diabetic")
    else:
        st.success("✅ The model predicts: Not Diabetic")
