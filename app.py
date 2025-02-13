import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("best_spacex_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("🚀 SpaceX Landing Prediction")
st.write("Enter details about the rocket launch to predict if it will successfully land.")

# Input Fields
payload_mass = st.number_input("Payload Mass (kg)", min_value=0, step=100)

# Orbit Selection
orbit = st.selectbox("Orbit Type", 
                      ['LEO', 'ISS', 'PO', 'GTO', 'ES-L1', 'SSO', 'HEO', 'MEO', 'VLEO', 'SO', 'GEO'])

# Launch Site Selection
launch_site = st.selectbox("Launch Site", 
                           ['CCAFS SLC 40', 'VAFB SLC 4E', 'KSC LC 39A'])

# Encode Orbit Type
orbit_mapping = {
    'LEO': 0, 'ISS': 1, 'PO': 2, 'GTO': 3, 'ES-L1': 4, 
    'SSO': 5, 'HEO': 6, 'MEO': 7, 'VLEO': 8, 'SO': 9, 'GEO': 10
}
orbit_encoded = orbit_mapping[orbit]

# Encode Launch Site
launch_site_mapping = {
    'CCAFS SLC 40': 0, 
    'VAFB SLC 4E': 1, 
    'KSC LC 39A': 2
}
launch_site_encoded = launch_site_mapping[launch_site]

# Prepare Input Data for Prediction
input_data = np.array([[payload_mass, orbit_encoded, launch_site_encoded]])

# Make Prediction on Button Click
if st.button("Predict"):
    prediction = model.predict(input_data)
    
    # Display the result
    if prediction[0] == 1:
        st.success("🟢 The rocket is likely to **land successfully!**")
    else:
        st.error("🔴 The rocket is likely to **fail the landing.**")
