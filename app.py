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

# --- User Inputs ---
payload_mass = st.number_input("Payload Mass (kg)", min_value=0, step=100)

orbit = st.selectbox("Orbit Type", ['LEO', 'ISS', 'PO', 'GTO', 'ES-L1', 'SSO', 'HEO', 'MEO', 'VLEO', 'SO', 'GEO'])

launch_site = st.selectbox("Launch Site", ['CCAFS SLC 40', 'VAFB SLC 4E', 'KSC LC 39A'])

flights = st.selectbox("Number of Previous Flights", [1, 2, 3, 4, 5, 6])

block = st.slider("Block Version", min_value=1.0, max_value=5.0, step=1.0)

grid_fins = st.checkbox("Grid Fins", value=False)
reused = st.checkbox("Reused Booster", value=False)
legs = st.checkbox("Landing Legs", value=False)

if reused:
    reused_count = st.selectbox("Reused Count", [1, 2, 3, 4, 5])
else:
    reused_count = 0

# --- Automatically Handle Remaining Data ---
# 'booster_version' can be treated as a fixed numeric value if it's constant.
booster_version = 0  # Fixed value for "Falcon 9"
serial = ""  # If serial is used in the model, handle it as needed

# --- One-Hot Encoding for Categorical Inputs ---
orbits = ['LEO', 'ISS', 'PO', 'GTO', 'ES-L1', 'SSO', 'HEO', 'MEO', 'VLEO', 'SO', 'GEO']
orbit_features = [1 if o == orbit else 0 for o in orbits]

launch_sites = ['CCAFS SLC 40', 'VAFB SLC 4E', 'KSC LC 39A']
launch_site_features = [1 if site == launch_site else 0 for site in launch_sites]

# Convert boolean inputs to integers
grid_fins_int = int(grid_fins)
reused_int = int(reused)
legs_int = int(legs)

# --- Fill Missing Features with Zeros (Ensure 83 Features in Total) ---
# Fill the missing features (those that aren't needed) with zeros.
missing_features = [0] * (83 - len(orbit_features) - len(launch_site_features) - 7)  # Fill to make up the missing features

# Combine user inputs, manually added features, and zero-padded missing features
other_features = [flights, block, grid_fins_int, reused_int, reused_count, legs_int, booster_version]  # Add other features as needed

# Final input array with padding (make sure it has exactly 83 features)
input_data = np.array([[payload_mass] + orbit_features + launch_site_features + other_features + missing_features])

# --- Scaling ---
input_data_scaled = scaler.transform(input_data)

# --- Prediction ---
if st.button("Predict"):
    try:
        # Make the prediction using the model
        prediction = model.predict(input_data_scaled)
        prediction_prob = model.predict_proba(input_data_scaled)

        if prediction[0] == 1:
            st.success(f"🟢 The rocket is likely to **land successfully!** (Confidence: {prediction_prob[0][1]:.2%})")
        else:
            st.error(f"🔴 The rocket is likely to **fail the landing.** (Confidence: {prediction_prob[0][0]:.2%})")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
