import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('Diabetes Prediction App')

st.write("""
This app predicts the likelihood of diabetes based on patient data.
""")

# Input fields
pregnancies = st.slider('Pregnancies', 0, 17, 3)
glucose = st.slider('Glucose Level', 0, 200, 117)
blood_pressure = st.slider('Blood Pressure (mm Hg)', 0, 122, 72)
skin_thickness = st.slider('Skin Thickness (mm)', 0, 99, 23)
insulin = st.slider('Insulin Level (mu U/ml)', 0, 846, 30)
bmi = st.slider('BMI', 0.0, 67.1, 32.0)
dpf = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
age = st.slider('Age', 21, 81, 29)

# Predict button
if st.button('Predict Diabetes'):
    # Create input array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scale the input
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]
    
    # Display results
    st.subheader('Prediction Results')
    if prediction[0] == 1:
        st.error(f'High risk of diabetes (Probability: {probability:.2%})')
    else:
        st.success(f'Low risk of diabetes (Probability: {probability:.2%})')

    # Interpretation
    st.subheader('Interpretation')
    st.write("""
    - Probability > 50% suggests higher risk of diabetes
    - Probability < 50% suggests lower risk of diabetes
    - This is not a medical diagnosis. Please consult a healthcare professional.
    """)