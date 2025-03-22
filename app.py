import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load("salary_model1.pkl")  # Ensure this file exists
scaler = joblib.load("scaler.pkl")  # Ensure scaler is saved during training

# Function to preprocess user input
def preprocess_input(age, gender, experience, education):
    # Convert gender to numeric
    gender = 1 if gender == "Male" else 0

    # One-Hot Encoding for Education
    education_levels = ["Bachelors", "Masters", "PhD", "BTech"]
    education_encoded = [1 if education == level else 0 for level in education_levels[1:]]  # Drop first category

    # Create feature array
    features = np.array([age, gender, experience] + education_encoded).reshape(1, -1)
    features_scaled = scaler.transform(features)  # Use saved scaler

    return features_scaled

# Streamlit UI
st.title("💼 Employee Salary Prediction")
st.write("Enter employee details to predict the expected salary.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=70, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=2)
education = st.selectbox("Education", ["Bachelors", "Masters", "PhD", "BTech"])

# Predict button
if st.button("Predict Salary"):
    # Preprocess input
    input_features = preprocess_input(age, gender, experience, education)

    # Predict salary
    predicted_salary = model.predict(input_features)[0]

    # Display prediction
    st.success(f"💰 Predicted Salary: ₹{predicted_salary:,.2f}")
