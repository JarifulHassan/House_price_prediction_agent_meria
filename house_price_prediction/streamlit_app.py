import streamlit as st
import pickle
import numpy as np

# Load the trained XGBoost model
with open("best_house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🏡 House Price Prediction App")

st.write("Enter house details below to predict the price:")

# User inputs
size = st.number_input("Size (sqft)", min_value=500, max_value=10000, step=50)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)
house_age = st.number_input("House Age (years)", min_value=0, max_value=100, step=1)

# Location Encoding
location = st.selectbox("Location", ["CityA", "CityB", "CityC", "CityD"])
location_encoding = {"CityA": [0, 0, 0], "CityB": [1, 0, 0], "CityC": [0, 1, 0], "CityD": [0, 0, 1]}

# Condition Encoding
condition = st.selectbox("Condition", ["Good", "New", "Poor"])
condition_encoding = {"Good": [1, 0, 0], "New": [0, 1, 0], "Poor": [0, 0, 1]}

# House Type Encoding
house_type = st.selectbox("House Type", ["Single Family", "Townhouse"])
house_type_encoding = {"Single Family": [1, 0], "Townhouse": [0, 1]}

# Button to predict
if st.button("Predict Price"):
    # Create input data array
    input_data = np.array([
        size, bedrooms, bathrooms, house_age,
        location_encoding[location][0], location_encoding[location][1], location_encoding[location][2],
        condition_encoding[condition][0], condition_encoding[condition][1], condition_encoding[condition][2],
        house_type_encoding[house_type][0], house_type_encoding[house_type][1]
    ]).reshape(1, -1)

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    st.success(f"🏠 Predicted House Price: ${predicted_price:,.2f}")
