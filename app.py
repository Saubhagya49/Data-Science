import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("spacex_fixed_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("🚀 SpaceX Landing Prediction")
st.write("Enter details about the rocket launch to predict if it will successfully land.")

# Input Fields
payload_mass = st.number_input("Payload Mass (kg)", min_value=0, step=100)

# Orbit Selection
orbit_options = ['Orbit_LEO', 'Orbit_ISS', 'Orbit_PO', 'Orbit_GTO', 'Orbit_ES-L1', 
                 'Orbit_SSO', 'Orbit_HEO', 'Orbit_MEO', 'Orbit_VLEO', 'Orbit_SO', 'Orbit_GEO']
orbit = st.selectbox("Orbit Type", orbit_options)

# Launch Site Selection
launch_site_options = ['LaunchSite_CCAFS SLC 40', 'LaunchSite_VAFB SLC 4E', 'LaunchSite_KSC LC 39A']
launch_site = st.selectbox("Launch Site", launch_site_options)

# Prepare Input Data
input_data = np.zeros((1, len(orbit_options) + len(launch_site_options) + 1))  # +1 for PayloadMass
input_data[0, 0] = payload_mass  # Set PayloadMass

# Set Orbit one-hot encoding
orbit_index = orbit_options.index(orbit) + 1  # +1 because PayloadMass is at index 0
input_data[0, orbit_index] = 1

# Set Launch Site one-hot encoding
launch_site_index = launch_site_options.index(launch_site) + len(orbit_options) + 1  # +1 for PayloadMass
input_data[0, launch_site_index] = 1

# Make Prediction on Button Click
if st.button("Predict"):
    prediction = model.predict(input_data)
    
    # Display the result
    if prediction[0] == 1:
        st.success("🟢 The rocket is likely to **land successfully!**")
    else:
        st.error("🔴 The rocket is likely to **fail the landing.**")
