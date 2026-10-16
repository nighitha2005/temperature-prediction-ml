
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Temperature Prediction",
    page_icon="🌡️"
)

st.title("🌡️ Temperature Prediction Using Machine Learning")
st.write("Enter the weather conditions to predict temperature.")

model_path = "temperature_prediction_model.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(model_path)

humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
pressure = st.number_input("Pressure (hPa)", 950.0, 1050.0, 1010.0)
wind_speed = st.number_input("Wind Speed", 0.0, 50.0, 5.0)
cloud_cover = st.number_input("Cloud Cover (%)", 0.0, 100.0, 50.0)

if st.button("Predict Temperature"):
    input_data = pd.DataFrame({
        "Humidity": [humidity],
        "Pressure": [pressure],
        "Wind_Speed": [wind_speed],
        "Cloud_Cover": [cloud_cover]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Temperature: {prediction:.2f} °C"
    )