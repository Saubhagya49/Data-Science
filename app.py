# Apply One-Hot Encoding to match training data
input_data = pd.get_dummies(input_data)

# Ensure input has same features as training data
for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0  # Add missing columns with default value

# Reorder columns to match training data
input_data = input_data[feature_names]

# Convert to NumPy Array
input_array = input_data.to_numpy().reshape(1, -1)

# 🚀 Fix: Ensure correct number of features before applying scaler
if input_array.shape[1] != len(feature_names):
    st.error(f"Feature mismatch! Expected {len(feature_names)} features, but got {input_array.shape[1]}.")
else:
    # Apply scaling
    input_scaled = scaler.transform(input_array)

    # Predict Button
    if st.button("Predict Landing Success"):
        try:
            # Make prediction
            prediction = model.predict(input_scaled)

            # Display result
            result = "Successful Landing 🏆" if prediction[0] == 1 else "Landing Failure ❌"
            st.success(f"Predicted Outcome: {result}")

        except Exception as e:
            st.error(f"Error: {e}")
