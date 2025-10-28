
import os
import sys
import pickle
import numpy as np

try:
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    print(" All required libraries available")
except ImportError as e:
    print(f" Missing library: {e}")
    sys.exit(1)


def main():
    print("=" * 80)
    print("EXPORTING GRADIENT BOOSTING MODEL FOR PRODUCTION")
    print("=" * 80)

    data_path = "datasets/ml_preprocessed/stellar_redshift_training.csv"
    output_dir = "datasets/ml_preprocessed/models"

    if not os.path.exists(data_path):
        print(f" Training data not found: {data_path}")
        sys.exit(1)

    print(f"\nLoading training data...")
    df = pd.read_csv(data_path)
    print(f" Loaded {len(df)} samples")

    features = ["ra", "dec", "teff", "logg", "fe_h", "snr", "parallax"]
    target = "z"

    df_clean = df.dropna(subset=[target])
    df_clean = df_clean.dropna(subset=features)

    X = df_clean[features].values.astype(np.float32)
    y = df_clean[target].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")

    print(f"\nTraining Gradient Boosting model...")
    model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        verbose=0,
    )

    model.fit(X_train, y_train)
    print(f" Model trained successfully")

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)

    print(f"\nModel Performance:")
    print(f"  Train RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}")
    print(f"  Val RMSE:   {val_rmse:.6f}, R²: {val_r2:.4f}")

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(X_train)
    scaler_y.fit(y_train.reshape(-1, 1))

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "production_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"\n Saved: production_model.pkl")

    with open(os.path.join(output_dir, "production_scaler_x.pkl"), "wb") as f:
        pickle.dump(scaler_x, f)
    print(f" Saved: production_scaler_x.pkl")

    with open(os.path.join(output_dir, "production_scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)
    print(f" Saved: production_scaler_y.pkl")

    metadata = {
        "model_type": "gradient_boosting",
        "features": features,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "n_estimators": 500,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
    }

    with open(os.path.join(output_dir, "production_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print(f" Saved: production_metadata.pkl")

    print(f"\n" + "=" * 80)
    print("TESTING MODEL WITH SAMPLE INPUTS")
    print("=" * 80)

    test_cases = [
        {
            "name": "Nearby Sun-like star",
            "values": [200.12, -47.33, 5800, 4.3, 0.0, 100, 7.2],
        },
        {"name": "Distant hot star", "values": [85.3, -15.8, 8500, 4.0, 0.2, 60, 1.2]},
        {"name": "Red giant", "values": [150.5, 30.2, 4200, 2.5, -0.5, 80, 3.5]},
        {"name": "Metal-poor star", "values": [310.7, 60.4, 5500, 4.5, -2.0, 40, 5.8]},
    ]

    for test in test_cases:
        x_test = np.array([test["values"]], dtype=np.float32)
        z_pred = model.predict(x_test)[0]
        print(f"\n{test['name']}:")
        print(f"  Input: {test['values']}")
        print(f"  Predicted z: {z_pred:.6f}")

    print(f"\n" + "=" * 80)
    print("EXPORT COMPLETE!")
    print("=" * 80)
    print(f"\nProduction-ready files saved in: {output_dir}/")
    print(f"  - production_model.pkl (Gradient Boosting)")
    print(f"  - production_scaler_x.pkl")
    print(f"  - production_scaler_y.pkl")
    print(f"  - production_metadata.pkl")
    print(f"\nThese files work with standard Python (no torch required)")
    print(f"Model performance: R² = {val_r2:.4f}")


if __name__ == "__main__":
    main()
