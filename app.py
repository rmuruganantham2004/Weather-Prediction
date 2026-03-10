"""Streamlit application for Weather Prediction Dashboard."""
import os
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Weather Prediction Dashboard",
    page_icon="🌦️",
    layout="wide")

st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .metric-card {
        background-color: white; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-bottom: 20px;
    }
    .main-header { font-family: 'Inter', sans-serif; color: #1e3a8a; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("🌦️ 7-Day Weather Forecast Dashboard")
st.markdown(
    "Predicts the temperature and weather condition for the next 7 days using Machine Learning.")

st.sidebar.header("Settings")
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "🌦️ **Weather Prediction Dashboard**\n\n"
    "This application uses Machine Learning (Random Forest) to forecast temperature and weather conditions for the next 7 days based on localized historical data.\n\n"
    "Built with Python, Scikit-Learn, and Streamlit."
)
st.sidebar.markdown("---")

DATA_DIR = "data"
MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    st.error("Models directory not found. Please run the training script first.")
    st.stop()

try:
    locations_files = [f for f in os.listdir(
        DATA_DIR) if f.endswith('_weather.csv')]
    app_locations = [
        f.replace(
            '_weather.csv',
            '').replace(
            '_',
            ' ').title() for f in locations_files]
except FileNotFoundError:
    st.error("Data directory not found. Please run the data generation script.")
    st.stop()

if not app_locations:
    st.error("No location data available.")
    st.stop()

selected_location = st.sidebar.selectbox(
    "Select Location", sorted(app_locations))
loc_safe = selected_location.replace(' ', '_').lower()


@st.cache_resource
def load_models(location):
    """Load pre-trained models and label encoder for a given location."""
    temp_model_path = os.path.join(MODEL_DIR, f"{location}_temp_model.pkl")
    cond_model_path = os.path.join(MODEL_DIR, f"{location}_cond_model.pkl")
    le_path = os.path.join(MODEL_DIR, f"{location}_label_encoder.pkl")

    if os.path.exists(temp_model_path) and os.path.exists(cond_model_path):
        loc_temp_model = joblib.load(temp_model_path)
        loc_cond_model = joblib.load(cond_model_path)
        loc_le = joblib.load(le_path)
        return loc_temp_model, loc_cond_model, loc_le
    return None, None, None


loaded_temp_model, loaded_cond_model, loaded_le = load_models(loc_safe)

if not loaded_temp_model:
    st.error(f"Models for {selected_location} not found.")
    st.stop()

# pylint: disable=too-many-locals


def generate_forecast(location_safe, loc_temp_model, loc_cond_model, loc_le):
    """Generate a 7-day forecast using trained models."""
    df = pd.read_csv(os.path.join(DATA_DIR, f"{location_safe}_weather.csv"))
    last_row = df.iloc[-1]

    current_features = np.array([[
        last_row['temperature'], last_row['humidity'],
        last_row['wind_speed'], last_row['atmospheric_pressure']
    ]])

    forecast_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]
    forecast_temps = []
    forecast_conds = []

    for _ in range(7):
        next_temp = loc_temp_model.predict(current_features)[0]
        next_cond_encoded = loc_cond_model.predict(current_features)[0]
        next_cond = loc_le.inverse_transform([next_cond_encoded])[0]

        forecast_temps.append(next_temp)
        forecast_conds.append(next_cond)

        next_hum = np.clip(
            current_features[0][1] +
            np.random.normal(
                0,
                5),
            10,
            100)
        next_wind = np.clip(
            current_features[0][2] +
            np.random.normal(
                0,
                2),
            0,
            50)
        next_pres = current_features[0][3] + np.random.normal(0, 2)

        current_features = np.array(
            [[next_temp, next_hum, next_wind, next_pres]])

    return pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
        'Day': [d.strftime('%A') for d in forecast_dates],
        'Temperature (°C)': [round(t, 2) for t in forecast_temps],
        'Condition': [c.title() for c in forecast_conds]
    })


loaded_forecast_df = generate_forecast(
    loc_safe,
    loaded_temp_model,
    loaded_cond_model,
    loaded_le)

st.markdown("### 📊 Model Performance Metrics")
try:
    metrics_df = pd.read_csv(
        os.path.join(
            MODEL_DIR,
            'metrics.csv'),
        index_col=0)
    if selected_location in metrics_df.index:
        loc_metrics = metrics_df.loc[selected_location]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="metric-card"><h4>Temperature MAE</h4><h1>{
                    loc_metrics["mae_temperature"]:.2f}°C</h1></div>',
                unsafe_allow_html=True)
        with col2:
            st.markdown(
                f'<div class="metric-card"><h4>Temperature MSE</h4><h1>{
                    loc_metrics["mse_temperature"]:.2f}</h1></div>',
                unsafe_allow_html=True)
        with col3:
            st.markdown(
                f'<div class="metric-card"><h4>Condition Accuracy</h4><h1>{
                    loc_metrics["accuracy_condition"] *
                    100:.1f}%</h1></div>',
                unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Metrics file not found.")

st.markdown(f"### 📅 7-Day Forecast for {selected_location}")
col_chart, col_table = st.columns([2, 1])

with col_table:
    def add_emoji(cond):
        """Append an emoji corresponding to the weather condition."""
        cond = cond.lower()
        if 'sunny' in cond:
            return '☀️ ' + cond.title()
        if 'rainy' in cond:
            return '🌧️ ' + cond.title()
        if 'cloudy' in cond:
            return '☁️ ' + cond.title()
        return cond.title()

    display_df = loaded_forecast_df.copy()
    display_df['Condition'] = display_df['Condition'].apply(add_emoji)
    st.dataframe(display_df[['Day',
                             'Temperature (°C)',
                             'Condition']],
                 use_container_width=True,
                 hide_index=True)

with col_chart:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        loaded_forecast_df['Day'],
        loaded_forecast_df['Temperature (°C)'],
        marker='o',
        linestyle='-',
        color='#1e3a8a',
        linewidth=2,
        markersize=8)
    ax.set_title(
        'Temperature Trend',
        fontsize=14,
        fontweight='bold',
        color='#1e3a8a')
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.fill_between(
        loaded_forecast_df['Day'],
        loaded_forecast_df['Temperature (°C)'],
        alpha=0.2,
        color='#3b82f6')
    ax.set_ylim(min(loaded_forecast_df['Temperature (°C)']) - 5,
                max(loaded_forecast_df['Temperature (°C)']) + 5)

    for idx, temp_val in enumerate(loaded_forecast_df['Temperature (°C)']):
        ax.annotate(
            f"{temp_val}°C",
            (idx,
             temp_val),
            textcoords="offset points",
            xytext=(
                0,
                10),
            ha='center',
            fontsize=10)

    st.pyplot(fig)
