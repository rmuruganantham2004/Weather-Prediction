import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_weather_data(locations, num_days=1000, output_dir="data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    start_date = datetime.now() - timedelta(days=num_days)
    
    for loc in locations:
        dates = [start_date + timedelta(days=i) for i in range(num_days)]
        
        # Base values depending on location roughly (for variation)
        base_temp = np.random.uniform(10, 25)
        
        # Adding seasonality (sine wave)
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        seasonality = 10 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Generate features
        temp = base_temp + seasonality + np.random.normal(0, 3, num_days)
        humidity = np.clip(np.random.normal(60, 15, num_days), 10, 100)
        wind_speed = np.clip(np.random.normal(15, 8, num_days), 0, 50)
        pressure = np.random.normal(1013, 10, num_days)
        
        # Logic for weather condition based on features
        conditions = []
        for i in range(num_days):
            t, h, w, p = temp[i], humidity[i], wind_speed[i], pressure[i]
            if h > 80 and p < 1010:
                conditions.append('rainy')
            elif h < 50 and p > 1015:
                conditions.append('sunny')
            else:
                conditions.append('cloudy')
                
        # Creating DataFrame
        df = pd.DataFrame({
            'date': dates,
            'location': loc,
            'temperature': temp,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'atmospheric_pressure': pressure,
            'weather_condition': conditions
        })
        
        # Target for next day's temp and condition (for training)
        df['target_temp_next_day'] = df['temperature'].shift(-1)
        df['target_condition_next_day'] = df['weather_condition'].shift(-1)
        
        # Drop last row since it doesn't have a target
        df = df.dropna()
        
        filename = os.path.join(output_dir, f"{loc.replace(' ', '_').lower()}_weather.csv")
        df.to_csv(filename, index=False)
        print(f"Generated data for {loc} -> {filename}")

if __name__ == "__main__":
    locations = ["San Francisco", "New York", "London", "Tokyo", "Sydney", "Mumbai"]
    generate_synthetic_weather_data(locations, num_days=1500)
