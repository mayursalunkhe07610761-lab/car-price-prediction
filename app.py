import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("models/car_model.pkl", "rb"))

st.title("🚗 Car Price Prediction")

present_price = st.number_input("Present Price", 0.0, 100.0, 5.0)

kms_driven = st.number_input("Kilometers Driven", 0, 500000, 10000)

owner = st.selectbox("Owners", [0, 1, 2, 3])

car_age = st.slider("Car Age", 0, 20, 5)

if st.button("Predict"):

    features = np.array([
        [present_price, kms_driven, owner, car_age]
    ])

    prediction = model.predict(features)

    st.success(f"Estimated Price: ₹ {prediction[0]:.2f} Lakhs")