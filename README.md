# 🌦️ Weather Prediction Dashboard

An end-to-end Machine Learning web application that predicts the temperature and weather condition (Sunny, Cloudy, Rainy) for the next 7 days based on synthetic historical training records. 

Built using **Python**, **Scikit-Learn**, and **Streamlit**.

## Features
- **Data Generation:** Generates synthetic historical weather data with features like temperature, humidity, wind speed, and atmospheric pressure.
- **Machine Learning Models:** 
  - `RandomForestRegressor` to predict continuous temperature values.
  - `RandomForestClassifier` to predict discrete weather conditions.
- **Interactive Dashboard:** A responsive Streamlit application to visualize 7-day temperature trends using Matplotlib charts and display real-time model accuracy metrics (MAE, MSE, Accuracy).

## Project Structure
- `data_generation.py`: Script to generate synthetic historical weather datasets for various cities.
- `train_models.py`: Script to train and save the Random Forest models.
- `app.py`: The Streamlit dashboard application.
- `requirements.txt`: Python dependencies.
- `data/`: Directory where generated datasets are stored (ignored via `.gitignore`).
- `models/`: Directory where trained models and encoders are serialized to (ignored via `.gitignore`).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rmuruganantham2004/Weather-Prediction.git
cd Weather-Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Generate Data:**  
   *(Optional if `data/` already exists)* Run the data generation script to create datasets for the test cities.
   ```bash
   python data_generation.py
   ```

2. **Train Models:**  
   *(Optional if `models/` already exists)* Train the Machine Learning models on the generated data.
   ```bash
   python train_models.py
   ```

3. **Run the Dashboard:**  
   Launch the Streamlit interactive dashboard.
   ```bash
   streamlit run app.py
   ```

## Technologies Used
- **Pandas** & **NumPy** for Data Manipulation
- **Scikit-Learn** for Machine Learning
- **Matplotlib** for Data Visualization
- **Streamlit** for the Frontend Dashboard
- **Joblib** for Model Serialization

## License
MIT License
