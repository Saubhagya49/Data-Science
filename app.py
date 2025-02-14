import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the feature names used during training (if available)
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

# One-Hot Encoding (If Used in Training)
input_data = pd.DataFrame([[flight_number, payload_mass, orbit, launch_site]], 
                          columns=["Flight Number", "Payload Mass", "Orbit", "Launch Site"])

# Apply Same Preprocessing as in Jupyter Notebook
if feature_names:
    # One-Hot Encode Categorical Features
    input_data = pd.get_dummies(input_data)

    # Ensure Input Data has Same Columns as During Training
    missing_cols = [col for col in feature_names if col not in input_data.columns]
    for col in missing_cols:
        input_data[col] = 0  # Add missing columns with value 0

    # Reorder Columns to Match Training Order
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
