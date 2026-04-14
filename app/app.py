import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("models/model.pkl")

st.set_page_config(page_title="Wine Health Predictor", layout="centered")

st.title("🍷 Wine Health Risk Predictor")
st.write("Adjust the values below to estimate potential health risk.")

# Input sliders (based on important features)
fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.0)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.5)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.08)
free_sulfur = st.slider("Free Sulfur Dioxide", 1, 80, 15)
total_sulfur = st.slider("Total Sulfur Dioxide", 6, 300, 46)
density = st.slider("Density", 0.9900, 1.0050, 0.9968, step=0.0001, format="%.4f")
pH = st.slider("pH", 2.8, 4.0, 3.3)
sulphates = st.slider("Sulphates", 0.3, 1.5, 0.6)
alcohol = st.slider("Alcohol (%)", 5.0, 15.0, 10.0)

# Predict button
if st.button("Predict"):
    # Full feature array (12 features expected)
    data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur,
        total_sulfur,
        density,
        pH,
        sulphates,
        alcohol,
        
    ]])

    prediction = model.predict(data)[0]

    st.subheader("Prediction Result:")

    if prediction == 0:
        st.success("🟢 Low Health Risk")
    elif prediction == 1:
        st.warning("🟡 Moderate Health Risk")
    else:
        st.error("🔴 High Health Risk")