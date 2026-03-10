import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Weather Prediction Dashboard", page_icon="🌦️", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1e3a8a;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌦️ 7-Day Weather Forecast Dashboard")
st.markdown("Predicts the temperature and weather condition for the next 7 days using Machine Learning.")

# Sidebar
st.sidebar.header("Settings")
data_dir = "data"
model_dir = "models"

if not os.path.exists(model_dir):
    st.error("Models directory not found. Please run the training script first.")
    st.stop()

# Load available locations
try:
    locations_files = [f for f in os.listdir(data_dir) if f.endswith('_weather.csv')]
    locations = [f.replace('_weather.csv', '').replace('_', ' ').title() for f in locations_files]
except FileNotFoundError:
    st.error("Data directory not found. Please run the data generation script.")
    st.stop()

if not locations:
    st.error("No location data available.")
    st.stop()
    
selected_location = st.sidebar.selectbox("Select Location", sorted(locations))
loc_safe = selected_location.replace(' ', '_').lower()

# Load Models
@st.cache_resource
def load_models(location):
    temp_model_path = os.path.join(model_dir, f"{location}_temp_model.pkl")
    cond_model_path = os.path.join(model_dir, f"{location}_cond_model.pkl")
    le_path = os.path.join(model_dir, f"{location}_label_encoder.pkl")
    
    if os.path.exists(temp_model_path) and os.path.exists(cond_model_path):
        temp_model = joblib.load(temp_model_path)
        cond_model = joblib.load(cond_model_path)
        le = joblib.load(le_path)
        return temp_model, cond_model, le
    return None, None, None

temp_model, cond_model, le = load_models(loc_safe)

if not temp_model:
    st.error(f"Models for {selected_location} not found.")
    st.stop()

# Generate 7-day forecast
def generate_forecast(location_safe, temp_model, cond_model, le):
    # Get the latest row of data for the location to use as starting point
    df = pd.read_csv(os.path.join(data_dir, f"{location_safe}_weather.csv"))
    last_row = df.iloc[-1]
    
    current_features = np.array([[
        last_row['temperature'],
        last_row['humidity'],
        last_row['wind_speed'],
        last_row['atmospheric_pressure']
    ]])
    
    forecast_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]
    forecast_temps = []
    forecast_conds = []
    
    for _ in range(7):
        # Predict next day
        next_temp = temp_model.predict(current_features)[0]
        next_cond_encoded = cond_model.predict(current_features)[0]
        next_cond = le.inverse_transform([next_cond_encoded])[0]
        
        forecast_temps.append(next_temp)
        forecast_conds.append(next_cond)
        
        # We need to simulate the features for the next prediction
        # (For a real system we'd forecast these features too, or use API data)
        # Here we just use a random walk bounded variation for simulation purpose
        next_hum = np.clip(current_features[0][1] + np.random.normal(0, 5), 10, 100)
        next_wind = np.clip(current_features[0][2] + np.random.normal(0, 2), 0, 50)
        next_pres = current_features[0][3] + np.random.normal(0, 2)
        
        current_features = np.array([[next_temp, next_hum, next_wind, next_pres]])
        
    forecast_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
        'Day': [d.strftime('%A') for d in forecast_dates],
        'Temperature (°C)': [round(t, 2) for t in forecast_temps],
        'Condition': [c.title() for c in forecast_conds]
    })
    
    return forecast_df

forecast_df = generate_forecast(loc_safe, temp_model, cond_model, le)

# Top metrics
st.markdown("### 📊 Model Performance Metrics")
try:
    metrics_df = pd.read_csv(os.path.join(model_dir, 'metrics.csv'), index_col=0)
    if selected_location in metrics_df.index:
        loc_metrics = metrics_df.loc[selected_location]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h4>Temperature MAE</h4><h1>{loc_metrics["mae_temperature"]:.2f}°C</h1></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h4>Temperature MSE</h4><h1>{loc_metrics["mse_temperature"]:.2f}</h1></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h4>Condition Accuracy</h4><h1>{loc_metrics["accuracy_condition"]*100:.1f}%</h1></div>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Metrics file not found.")


# Charts
st.markdown(f"### 📅 7-Day Forecast for {selected_location}")

col_chart, col_table = st.columns([2, 1])

with col_table:
    # Adding emojis to conditions
    def add_emoji(cond):
        cond = cond.lower()
        if 'sunny' in cond: return '☀️ ' + cond.title()
        if 'rainy' in cond: return '🌧️ ' + cond.title()
        if 'cloudy' in cond: return '☁️ ' + cond.title()
        return cond.title()
        
    display_df = forecast_df.copy()
    display_df['Condition'] = display_df['Condition'].apply(add_emoji)
    st.dataframe(display_df[['Day', 'Temperature (°C)', 'Condition']], use_container_width=True, hide_index=True)

with col_chart:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecast_df['Day'], forecast_df['Temperature (°C)'], marker='o', linestyle='-', color='#1e3a8a', linewidth=2, markersize=8)
    ax.set_title('Temperature Trend', fontsize=14, fontweight='bold', color='#1e3a8a')
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Fill under line for aesthetics
    ax.fill_between(forecast_df['Day'], forecast_df['Temperature (°C)'], alpha=0.2, color='#3b82f6')
    ax.set_ylim(min(forecast_df['Temperature (°C)']) - 5, max(forecast_df['Temperature (°C)']) + 5)
    
    # Annotate points
    for i, txt in enumerate(forecast_df['Temperature (°C)']):
        ax.annotate(f"{txt}°C", (i, txt), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
        
    st.pyplot(fig)
