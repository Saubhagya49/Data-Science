import streamlit as st
import pandas as pd
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

# Convert input to DataFrame for prediction
input_data = pd.DataFrame([[flight_number, payload_mass, orbit, launch_site]], 
                          columns=["Flight Number", "Payload Mass", "Orbit", "Launch Site"])

# Predict Button
if st.button("Predict Landing Success"):
    import numpy as np

    # Convert input to NumPy array and reshape
    input_data = np.array(input_data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    result = "Successful Landing 🏆" if prediction[0] == 1 else "Landing Failure ❌"
    st.success(f"Predicted Outcome: {result}")

