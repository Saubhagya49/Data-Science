import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit App UI
st.title("🚀 SpaceX Falcon 9 Landing Prediction")
st.write("Predict whether a Falcon 9 booster will land successfully.")

# User Inputs
flight_number = st.number_input("Flight Number", min_value=1, step=1)
payload_mass = st.number_input("Payload Mass (kg)", min_value=0, step=100)
orbit = st.selectbox("Orbit Type", ["LEO", "GTO", "MEO", "ISS"])
launch_site = st.selectbox("Launch Site", ["CCAFS", "KSC", "VAFB"])

# Encode categorical variables (replace with your model's encoding)
orbit_mapping = {"LEO": 0, "GTO": 1, "MEO": 2, "ISS": 3}
launch_site_mapping = {"CCAFS": 0, "KSC": 1, "VAFB": 2}

orbit_encoded = orbit_mapping[orbit]
launch_site_encoded = launch_site_mapping[launch_site]

# Convert input to DataFrame for prediction
input_data = np.array([[flight_number, payload_mass, orbit_encoded, launch_site_encoded]])

# Debugging: Print input shape
st.write(f"Input shape: {input_data.shape}")

# Predict Button
if st.button("Predict Landing Success"):
    try:
        # Make prediction
        prediction = model.predict(input_data)

        # Display result
        result = "Successful Landing 🏆" if prediction[0] == 1 else "Landing Failure ❌"
        st.success(f"Predicted Outcome: {result}")

    except Exception as e:
        st.error(f"Error: {e}")
