import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("all_sources.csv")

features = ["ra", "dec", "teff", "logg", "fe_h", "snr", "parallax"]
target = ["z"]

for col in features + target:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col] = 0.0

X = df[features].values
y = df[target].values.ravel()

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

os.makedirs("../../datasets/ml_preprocessed/models", exist_ok=True)
joblib.dump(model, "../../datasets/ml_preprocessed/models/redshift_model.pkl")
joblib.dump(scaler_x, "../../datasets/ml_preprocessed/models/scaler_x.pkl")
joblib.dump(scaler_y, "../../datasets/ml_preprocessed/models/scaler_y.pkl")

print("Model and scalers saved.")
