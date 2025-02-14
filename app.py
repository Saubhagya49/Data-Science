import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load feature names (if saved during training)
try:
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    feature_names = None  # Handle case where feature names are not stored

# Streamlit App UI
st.title("🚀 SpaceX Falcon 9 Landing Prediction")
st.write("Predict whether a Falcon 9 booster will land successfully.")

# User Inputs
flight_number = st.number_input("Flight Number", min_value=1, step=1)
payload_mass = st.number_input("Payload Mass (kg)", min_value=0, step=100)
orbit = st.selectbox("Orbit Type", ["LEO", "GTO", "MEO", "ISS"])
launch_site = st.selectbox("Launch Site", ["CCAFS", "KSC", "VAFB"])

# Create DataFrame for User Input
input_data = pd.DataFrame([[flight_number, payload_mass, orbit, launch_site]], 
                          columns=["Flight Number", "Payload Mass", "Orbit", "Launch Site"])

# One-Hot Encoding (Convert Categorical to Numerical)
input_data = pd.get_dummies(input_data)

# Ensure Input Data has Same Columns as Model Training
if feature_names:
    # Add missing columns as 0
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match model's expected input
    input_data = input_data[feature_names]

# Convert to NumPy Array
input_array = input_data.to_numpy().reshape(1, -1)

# Debugging: Check Input Shape
st.write(f"Input Shape: {input_array.shape}")

# Predict Button
if st.button("Predict Landing Success"):
    try:
        # Make prediction
        prediction = model.predict(input_array)

        # Display result
        result = "Successful Landing 🏆" if prediction[0] == 1 else "Landing Failure ❌"
        st.success(f"Predicted Outcome: {result}")

    except Exception as e:
        st.error(f"Error: {e}")
