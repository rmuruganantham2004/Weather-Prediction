import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate_models(data_dir="data", model_dir="models"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    metrics = {}
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue
            
        loc = filename.replace('_weather.csv', '').replace('_', ' ').title()
        filepath = os.path.join(data_dir, filename)
        
        # Load dataset
        df = pd.read_csv(filepath)
        
        # We'll use these features to predict next day targets
        features = ['temperature', 'humidity', 'wind_speed', 'atmospheric_pressure']
        X = df[features]
        y_temp = df['target_temp_next_day']
        y_cond = df['target_condition_next_day']
        
        # Encode conditions to numerical
        le = LabelEncoder()
        
        # Collect all possible conditions from current and target to fit encoder robustly
        all_conditions = pd.concat([df['weather_condition'], y_cond]).unique()
        le.fit(all_conditions)
        
        y_cond_encoded = le.transform(y_cond)
        
        # Train-test split
        X_train, X_test, y_temp_train, y_temp_test, y_cond_train, y_cond_test = train_test_split(
            X, y_temp, y_cond_encoded, test_size=0.2, random_state=42
        )
        
        # Models
        temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        cond_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train Temperature Model
        temp_model.fit(X_train, y_temp_train)
        y_temp_pred = temp_model.predict(X_test)
        mae = mean_absolute_error(y_temp_test, y_temp_pred)
        mse = mean_squared_error(y_temp_test, y_temp_pred)
        
        # Train Condition Model
        cond_model.fit(X_train, y_cond_train)
        y_cond_pred = cond_model.predict(X_test)
        accuracy = accuracy_score(y_cond_test, y_cond_pred)
        
        # Store metrics
        metrics[loc] = {
            'mae_temperature': mae,
            'mse_temperature': mse,
            'accuracy_condition': accuracy
        }
        
        # Save models and encoder
        loc_safe = loc.replace(' ', '_').lower()
        joblib.dump(temp_model, f"{model_dir}/{loc_safe}_temp_model.pkl")
        joblib.dump(cond_model, f"{model_dir}/{loc_safe}_cond_model.pkl")
        joblib.dump(le, f"{model_dir}/{loc_safe}_label_encoder.pkl")
        
        print(f"Trained models for {loc}. Temp MAE: {mae:.2f}, Cond Acc: {accuracy:.2%}")
        
    # Save overall metrics
    pd.DataFrame(metrics).T.to_csv(os.path.join(model_dir, 'metrics.csv'))
    print("All models trained and saved!")

if __name__ == "__main__":
    train_and_evaluate_models()
