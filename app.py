import pickle
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open("best_spacex_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the saved scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Take user inputs
payload_mass = st.number_input("Payload Mass (kg)", min_value=0, step=100)
orbit = st.selectbox("Orbit Type", ['LEO', 'ISS', 'PO', 'GTO', 'ES-L1', 'SSO', 'HEO', 'MEO', 'VLEO', 'SO', 'GEO'])
launch_site = st.selectbox("Launch Site", ['CCAFS SLC 40', 'VAFB SLC 4E', 'KSC LC 39A'])

# Encode Orbit (One-Hot Encoding)
orbits = ['LEO', 'ISS', 'PO', 'GTO', 'ES-L1', 'SSO', 'HEO', 'MEO', 'VLEO', 'SO', 'GEO']
orbit_features = [1 if o == orbit else 0 for o in orbits]

# Encode Launch Site (One-Hot Encoding)
launch_sites = ['CCAFS SLC 40', 'VAFB SLC 4E', 'KSC LC 39A']
launch_site_features = [1 if site == launch_site else 0 for site in launch_sites]

# Add missing features (set default values for other columns)
other_features = [0] * (83 - (len(orbit_features) + len(launch_site_features) + 1))

# Combine all features into a single array
input_data = np.array([[payload_mass] + orbit_features + launch_site_features + other_features])

# Scale input data using the saved scaler
input_data_scaled = scaler.transform(input_data)

# Make Prediction on Button Click
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    prediction_prob = model.predict_proba(input_data_scaled)

    if prediction[0] == 1:
        st.success(f"🟢 The rocket is likely to **land successfully!** (Confidence: {prediction_prob[0][1]:.2%})")
    else:
        st.error(f"🔴 The rocket is likely to **fail the landing.** (Confidence: {prediction_prob[0][0]:.2%})")
