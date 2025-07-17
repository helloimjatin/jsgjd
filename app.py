# app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained model and label encoder
st.write("✅ Streamlit app is running!")
clf = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")

# List of symptoms (must match training features)
symptoms = [
    'fever', 'cough', 'headache', 'fatigue', 'chills',
    'rash', 'nausea', 'sore_throat', 'joint_pain', 'runny_nose'
]

# Streamlit UI setup
st.set_page_config(page_title="Simple Disease Predictor", page_icon="🩺")
st.title("🩺 Simple Disease Predictor")
st.markdown("Check the symptoms you are experiencing to predict a possible disease. \
This tool is for educational use only.")

# Create checkboxes for symptoms
user_input = {}
for symptom in symptoms:
    user_input[symptom] = st.checkbox(symptom.replace('_', ' ').capitalize())

# When user clicks the Predict button
if st.button("🔍 Predict Disease"):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Predict using loaded model
    prediction = clf.predict(input_df)[0]
    predicted_disease = le.inverse_transform([prediction])[0]

    # Show result
    st.success(f"🩺 Based on the symptoms, the predicted disease is: **{predicted_disease}**")

# Footer
st.markdown("---")
st.caption("⚠️ This tool does not provide a medical diagnosis. Always consult a doctor.")
