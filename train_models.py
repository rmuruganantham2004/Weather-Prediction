"""Module for training weather prediction models."""
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

# pylint: disable=too-many-locals


def train_and_evaluate_models(data_dir="data", model_dir="models"):
    """Train and evaluate models for weather prediction."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    metrics = {}

    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue

        loc = filename.replace('_weather.csv', '').replace('_', ' ').title()
        filepath = os.path.join(data_dir, filename)

        df = pd.read_csv(filepath)
        features = [
            'temperature',
            'humidity',
            'wind_speed',
            'atmospheric_pressure']
        x_features = df[features]
        y_temp = df['target_temp_next_day']
        y_cond = df['target_condition_next_day']

        le = LabelEncoder()
        all_conditions = pd.concat([df['weather_condition'], y_cond]).unique()
        le.fit(all_conditions)

        y_cond_encoded = le.transform(y_cond)

        x_train, x_test, y_temp_train, y_temp_test, y_cond_train, y_cond_test = train_test_split(
            x_features, y_temp, y_cond_encoded, test_size=0.2, random_state=42)

        temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        cond_model = RandomForestClassifier(n_estimators=100, random_state=42)

        temp_model.fit(x_train, y_temp_train)
        y_temp_pred = temp_model.predict(x_test)
        mae = mean_absolute_error(y_temp_test, y_temp_pred)
        mse = mean_squared_error(y_temp_test, y_temp_pred)

        cond_model.fit(x_train, y_cond_train)
        y_cond_pred = cond_model.predict(x_test)
        accuracy = accuracy_score(y_cond_test, y_cond_pred)

        metrics[loc] = {
            'mae_temperature': mae,
            'mse_temperature': mse,
            'accuracy_condition': accuracy
        }

        loc_safe = loc.replace(' ', '_').lower()
        joblib.dump(temp_model, f"{model_dir}/{loc_safe}_temp_model.pkl")
        joblib.dump(cond_model, f"{model_dir}/{loc_safe}_cond_model.pkl")
        joblib.dump(le, f"{model_dir}/{loc_safe}_label_encoder.pkl")

        print(
            f"Trained models for {loc}. Temp MAE: {
                mae:.2f}, Cond Acc: {
                accuracy:.2%}")

    pd.DataFrame(metrics).T.to_csv(os.path.join(model_dir, 'metrics.csv'))
    print("All models trained and saved!")


if __name__ == "__main__":
    train_and_evaluate_models()
