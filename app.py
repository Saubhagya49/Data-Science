import streamlit as st
import pickle
import numpy as np
import pandas as pd
import random

# Load trained model
with open("best_spacex_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load real failure samples from dataset
with open("failed_feature_samples.pkl", "rb") as f:
    failed_samples = pd.read_pickle(f)

# Title
st.title("🚀 SpaceX Landing Prediction")
st.write("Enter rocket launch details to predict landing success.")

# Input Fields
payload_mass = st.number_input("Payload Mass (kg)", min_value=0, step=100)
orbit = st.selectbox("Orbit Type", ['LEO', 'ISS', 'PO', 'GTO', 'ES-L1', 'SSO', 'HEO', 'MEO', 'VLEO', 'SO', 'GEO'])
launch_site = st.selectbox("Launch Site", ['CCAFS SLC 40', 'VAFB SLC 4E', 'KSC LC 39A'])

# Encode Orbit Type
orbit_mapping = {'LEO': 0, 'ISS': 1, 'PO': 2, 'GTO': 3, 'ES-L1': 4, 'SSO': 5, 'HEO': 6, 'MEO': 7, 'VLEO': 8, 'SO': 9, 'GEO': 10}
orbit_encoded = orbit_mapping[orbit]

# Encode Launch Site
launch_site_mapping = {'CCAFS SLC 40': 0, 'VAFB SLC 4E': 1, 'KSC LC 39A': 2}
launch_site_encoded = launch_site_mapping[launch_site]

# Prepare Input Data with 3 user inputs
input_data = np.array([[payload_mass, orbit_encoded, launch_site_encoded]])

# Randomly select a failed case for missing features (20% probability)
if random.random() < 0.2:
    random_row = failed_samples.sample(n=1).to_numpy()
else:
    random_row = failed_samples.mean().to_numpy().reshape(1, -1)  # Use average failed data

# Replace the first 3 features with user input
random_row[0, :3] = input_data

# Make Prediction on Button Click
if st.button("Predict"):
    prediction = model.predict(random_row)
    
    # Display the result
    if prediction[0] == 1:
        st.success("🟢 The rocket is likely to **land successfully!**")
    else:
        st.error("🔴 The rocket is likely to **fail the landing.**")
