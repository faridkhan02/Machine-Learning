"""
Streamlit front-end for the Diabetes Prediction model.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

MODEL_PATH = os.path.join("model", "diabetes_logreg_model.pkl")
SCALER_PATH = os.path.join("model", "diabetes_scaler.pkl")

FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


st.title("🩺 Diabetes Risk Predictor")
st.write(
    "Fill in the health details below. If you're not sure about a value "
    "(like Insulin or Skin Thickness), leave it at the default — it's set "
    "to a typical average value."
)

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python diabetes_prediction.py` first "
        "to generate `model/diabetes_logreg_model.pkl` and `model/diabetes_scaler.pkl`."
    )
    st.stop()

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0, max_value=20, value=2, step=1,
        help="Number of times the patient has been pregnant. Enter 0 if none."
    )
    glucose = st.slider(
        "Glucose (mg/dL)",
        min_value=0, max_value=250, value=120,
        help="Blood sugar level from a glucose tolerance test. "
             "Normal is roughly 70–140. Higher values raise diabetes risk."
    )
    blood_pressure = st.slider(
        "Blood Pressure (mm Hg)",
        min_value=0, max_value=150, value=70,
        help="Diastolic blood pressure (the lower number in a BP reading). "
             "A healthy range is roughly 60–80."
    )
    skin_thickness = st.slider(
        "Skin Thickness (mm)",
        min_value=0, max_value=100, value=20,
        help="Triceps skinfold thickness, used to estimate body fat. "
             "Not sure? Leave at the default (20)."
    )

with col2:
    insulin = st.slider(
        "Insulin (mu U/mL)",
        min_value=0, max_value=900, value=80,
        help="2-Hour serum insulin level. Not sure? Leave at the default (80)."
    )
    bmi = st.number_input(
        "BMI",
        min_value=0.0, max_value=70.0, value=28.0, step=0.1, format="%.1f",
        help="Body Mass Index = weight(kg) / height(m)². "
             "Under 25 = healthy weight, 25–30 = overweight, 30+ = obese."
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=2.5, value=0.5, step=0.01, format="%.2f",
        help="A score estimating diabetes risk based on family history. "
             "Higher = stronger family history. Not sure? Leave at 0.5 (average)."
    )
    age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)

st.divider()

if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
    patient_df = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])[FEATURE_NAMES]

    patient_scaled = scaler.transform(patient_df)
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0, 1]

    st.subheader("Result")

    if probability >= 0.66:
        risk_label = "High risk"
    elif probability >= 0.33:
        risk_label = "Moderate risk"
    else:
        risk_label = "Low risk"

    if prediction == 1:
        st.error(f"### {risk_label} of Diabetes\nEstimated probability: **{probability:.0%}**")
    else:
        st.success(f"### {risk_label} of Diabetes\nEstimated probability: **{probability:.0%}**")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "⚠️ This is a demo model for educational purposes only — it is **not** a "
        "medical diagnosis. Please consult a doctor for any real health concerns."
    )

with st.expander("About this model"):
    st.write(
        "Logistic Regression trained via GridSearchCV (5-fold stratified CV, "
        "optimized for ROC-AUC) on the Pima Indians Diabetes dataset. "
        "Test accuracy: **70.78%**, Test ROC-AUC: **0.827**."
    )