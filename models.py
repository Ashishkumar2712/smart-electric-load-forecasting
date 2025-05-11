import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Load your dataset
df = pd.read_csv("your_data.csv")  # Replace with your real CSV filename
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

# Feature engineering
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['rolling_mean'] = df['load'].rolling(window=3, min_periods=1).mean()

df = df.dropna()

features = ['hour', 'dayofweek', 'month', 'rolling_mean']
target = 'load'

X = df[features]
y = df[target]

# Model pipeline with scaler
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Train
model.fit(X, y)

# Save to model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to model.pkl")
