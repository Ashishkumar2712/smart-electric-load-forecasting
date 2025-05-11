import pandas as pd
import pickle
from datetime import timedelta

def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def preprocess_data(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['rolling_mean'] = df['load'].rolling(window=3, min_periods=1).mean()

    df = df.dropna()
    features = ['hour', 'dayofweek', 'month', 'rolling_mean']
    return df, features

def predict_load(model, df, horizon, features):
    last_row = df.iloc[-1]
    forecast = []

    for i in range(horizon):
        next_datetime = last_row['datetime'] + timedelta(hours=1)
        next_features = {
            'hour': next_datetime.hour,
            'dayofweek': next_datetime.dayofweek,
            'month': next_datetime.month,
            'rolling_mean': df['load'].rolling(3, min_periods=1).mean().iloc[-1],
        }

        X_pred = pd.DataFrame([next_features])
        y_pred = model.predict(X_pred)[0]

        forecast.append({
            'datetime': next_datetime,
            'load': None,
            'predicted_load': y_pred
        })

        # Update last_row and df for next iteration
        last_row = pd.Series({
            'datetime': next_datetime,
            'load': y_pred
        })
        df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)

    forecast_df = pd.DataFrame(forecast)
    return forecast_df

def send_email(to_email, df):
    # Placeholder function for email
    print(f"Pretending to send email to {to_email}")
    return True
