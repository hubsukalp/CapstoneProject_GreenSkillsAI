import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# App Configuration
st.set_page_config(page_title="GreenGuard AQI Forecast", layout="centered")
st.title("🌿 GreenGuard: AI-Based AQI Prediction Dashboard")

# Try importing TensorFlow model loader and joblib
try:
    from tensorflow.keras.models import load_model
    import joblib
except ImportError:
    st.error("🚨 Missing packages. Please install TensorFlow and Joblib:\n`pip install tensorflow joblib`")
    st.stop()

# Required file list
required_files = ['greenguard_lstm_model.h5', 'scaler.save', 'aqi_sample_data.csv']
missing_files = [f for f in required_files if not os.path.exists(f)]

# File Check
if missing_files:
    st.error(f"❌ Missing file(s): {', '.join(missing_files)}\nPlease ensure all required files are in the same folder as this app.")
    st.stop()

# Load model, scaler, and dataset
try:
    model = load_model('greenguard_lstm_model.h5')
    scaler = joblib.load('scaler.save')
    df = pd.read_csv('aqi_sample_data.csv')
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# Check if DataFrame is empty
if df.empty or 'AQI' not in df.columns:
    st.error("AQI data is missing or empty in the CSV file.")
    st.stop()

# Display Current AQI
st.subheader("📍 Today's AQI")
current_aqi = df['AQI'].iloc[-1]
st.metric("Current AQI", round(current_aqi, 2))

# Predict Next Day AQI
SEQ_LENGTH = 14
try:
    data_scaled = scaler.transform(df[['AQI']])
    X_input = data_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    y_pred = model.predict(X_input)
    predicted_aqi = scaler.inverse_transform(y_pred)[0][0]
    st.success(f"✅ Predicted AQI for Tomorrow: {round(predicted_aqi, 2)}")
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

# Plot AQI Trend
st.subheader("📈 AQI Trend (Past 30 Days + Forecast)")
try:
    fig, ax = plt.subplots()
    ax.plot(df['AQI'].tail(30).values, label='Past AQI', marker='o')
    ax.axhline(y=predicted_aqi, color='red', linestyle='--', label='Forecast AQI')
    ax.legend()
    ax.set_xlabel('Days')
    ax.set_ylabel('AQI')
    ax.grid(True)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Plotting error: {e}")

# Sustainability Tip
st.subheader("💡 Sustainability Tip")
if predicted_aqi > 150:
    st.warning("⚠️ High AQI expected. Recommend limiting traffic, promoting remote work and indoor air purifiers.")
elif predicted_aqi > 100:
    st.info("Moderate AQI forecasted. Use public transport and avoid outdoor exertion.")
else:
    st.success("Great air quality forecasted! Keep supporting green practices and low emissions.")

# Footer
st.markdown("---")
st.caption("Developed as part of Capstone Project | GreenGuard 🌿")
