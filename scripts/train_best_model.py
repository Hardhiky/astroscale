
import os
import sys
import numpy as np
import pickle
from pathlib import Path

try:
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    print(" All required libraries loaded successfully")
except ImportError as e:
    print(f" Missing library: {e}")
    print("Install with: pip install pandas numpy scikit-learn torch")
    sys.exit(1)

try:
    import xgboost as xgb

    HAS_XGB = True
    print(" XGBoost available")
except ImportError:
    HAS_XGB = False
    print(" XGBoost not available (optional)")

try:
    import lightgbm as lgb

    HAS_LGB = True
    print(" LightGBM available")
except ImportError:
    HAS_LGB = False
    print(" LightGBM not available (optional)")


class ImprovedNeuralNet(nn.Module):

    def __init__(self, input_dim=7, hidden_dims=[256, 128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


def load_and_prepare_data(data_path):
    print(f"\n{'=' * 80}")
    print("LOADING AND PREPARING DATA")
    print(f"{'=' * 80}")

    df = pd.read_csv(data_path)
    print(f" Loaded {len(df)} samples from {data_path}")

    features = ["ra", "dec", "teff", "logg", "fe_h", "snr", "parallax"]
    target = "z"

    print(f"\nData quality check:")
    print(f"  Total rows: {len(df)}")
    print(f"  Missing z values: {df[target].isna().sum()}")

    df_clean = df.dropna(subset=[target])
    print(f"  Clean rows: {len(df_clean)}")

    df_clean = df_clean.dropna(subset=features)
    print(f"  Final rows: {len(df_clean)}")

    if len(df_clean) < 100:
        raise ValueError("Not enough clean data for training!")

    X = df_clean[features].values.astype(np.float32)
    y = df_clean[target].values.astype(np.float32)

    print(f"\nTarget (z) statistics:")
    print(f"  Mean: {y.mean():.6f}")
    print(f"  Std:  {y.std():.6f}")
    print(f"  Min:  {y.min():.6f}")
    print(f"  Max:  {y.max():.6f}")

    return X, y, features


def train_neural_network(
    X_train, y_train, X_val, y_val, scaler_x, scaler_y, device="cpu"
):
    print(f"\n{'=' * 80}")
    print("TRAINING NEURAL NETWORK")
    print(f"{'=' * 80}")

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)

    model = ImprovedNeuralNet(input_dim=X_train.shape[1]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    patience = 30
    patience_counter = 0
    batch_size = 256

    print(f"\nTraining configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: 0.001")
    print(f"  Early stopping patience: {patience}")

    for epoch in range(200):
        model.train()

        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0

        for i in range(0, len(X_train_t), batch_size):
            batch_idx = indices[i : i + batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t).squeeze()
            val_loss = criterion(val_outputs, y_val_t).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1:3d}: Train Loss = {epoch_loss / len(X_train):.6f}, Val Loss = {val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        train_pred_scaled = model(X_train_t).cpu().numpy()
        val_pred_scaled = model(X_val_t).cpu().numpy()

    train_pred = scaler_y.inverse_transform(train_pred_scaled)
    val_pred = scaler_y.inverse_transform(val_pred_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)

    print(f"\n Neural Network Results:")
    print(f"  Train RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}")
    print(f"  Val RMSE:   {val_rmse:.6f}, R²: {val_r2:.4f}")

    return model, val_rmse, val_r2


def train_gradient_boosting_models(X_train, y_train, X_val, y_val):
    models = {}

    if HAS_XGB:
        print(f"\n{'=' * 80}")
        print("TRAINING XGBOOST")
        print(f"{'=' * 80}")

        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
        )

        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

        train_pred = xgb_model.predict(X_train)
        val_pred = xgb_model.predict(X_val)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)

        print(f"\n XGBoost Results:")
        print(f"  Train RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}")
        print(f"  Val RMSE:   {val_rmse:.6f}, R²: {val_r2:.4f}")

        models["xgboost"] = {"model": xgb_model, "val_rmse": val_rmse, "val_r2": val_r2}

    if HAS_LGB:
        print(f"\n{'=' * 80}")
        print("TRAINING LIGHTGBM")
        print(f"{'=' * 80}")

        lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        train_pred = lgb_model.predict(X_train)
        val_pred = lgb_model.predict(X_val)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)

        print(f"\n LightGBM Results:")
        print(f"  Train RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}")
        print(f"  Val RMSE:   {val_rmse:.6f}, R²: {val_r2:.4f}")

        models["lightgbm"] = {
            "model": lgb_model,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
        }

    print(f"\n{'=' * 80}")
    print("TRAINING GRADIENT BOOSTING (sklearn)")
    print(f"{'=' * 80}")

    gb_model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        verbose=1,
    )

    gb_model.fit(X_train, y_train)

    train_pred = gb_model.predict(X_train)
    val_pred = gb_model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)

    print(f"\n Gradient Boosting Results:")
    print(f"  Train RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}")
    print(f"  Val RMSE:   {val_rmse:.6f}, R²: {val_r2:.4f}")

    models["gradient_boosting"] = {
        "model": gb_model,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
    }

    return models


def main():
    print(f"\n{'=' * 80}")
    print("STELLAR REDSHIFT PREDICTION - ADVANCED MODEL TRAINING")
    print(f"{'=' * 80}")

    DATA_PATH = "datasets/ml_preprocessed/stellar_redshift_training.csv"
    OUTPUT_DIR = "datasets/ml_preprocessed/models"

    if not os.path.exists(DATA_PATH):
        print(f" Data file not found: {DATA_PATH}")
        print("Available data files:")
        for f in Path("datasets/catalogs").glob("*.csv"):
            print(f"  - {f}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y, features = load_and_prepare_data(DATA_PATH)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print(f"\n Data split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")

    all_models = {}

    gb_models = train_gradient_boosting_models(X_train, y_train, X_val, y_val)
    all_models.update(gb_models)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_x = RobustScaler()
    scaler_y = StandardScaler()

    nn_model, nn_val_rmse, nn_val_r2 = train_neural_network(
        X_train, y_train, X_val, y_val, scaler_x, scaler_y, device
    )

    all_models["neural_net"] = {
        "model": nn_model,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "device": device,
        "val_rmse": nn_val_rmse,
        "val_r2": nn_val_r2,
    }

    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON")
    print(f"{'=' * 80}")

    for name, info in all_models.items():
        print(
            f"{name:20s}: Val RMSE = {info['val_rmse']:.6f}, R² = {info['val_r2']:.4f}"
        )

    best_model_name = min(all_models.items(), key=lambda x: x[1]["val_rmse"])[0]
    best_model_info = all_models[best_model_name]

    print(f"\n Best Model: {best_model_name}")
    print(f"  Validation RMSE: {best_model_info['val_rmse']:.6f}")
    print(f"  Validation R²: {best_model_info['val_r2']:.4f}")

    print(f"\n{'=' * 80}")
    print("SAVING MODELS")
    print(f"{'=' * 80}")

    if best_model_name == "neural_net":
        torch.save(
            best_model_info["model"].state_dict(),
            os.path.join(OUTPUT_DIR, "best_model_nn.pt"),
        )
        torch.save(
            best_model_info["scaler_x"], os.path.join(OUTPUT_DIR, "scaler_x_best.pth")
        )
        torch.save(
            best_model_info["scaler_y"], os.path.join(OUTPUT_DIR, "scaler_y_best.pth")
        )
        print(f" Saved neural network model")
    else:
        with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "wb") as f:
            pickle.dump(best_model_info["model"], f)

        scaler_x_simple = StandardScaler()
        scaler_y_simple = StandardScaler()
        scaler_x_simple.fit(X_train)
        scaler_y_simple.fit(y_train.reshape(-1, 1))

        with open(os.path.join(OUTPUT_DIR, "scaler_x_best.pkl"), "wb") as f:
            pickle.dump(scaler_x_simple, f)
        with open(os.path.join(OUTPUT_DIR, "scaler_y_best.pkl"), "wb") as f:
            pickle.dump(scaler_y_simple, f)

        print(f" Saved {best_model_name} model")

    metadata = {
        "best_model": best_model_name,
        "features": features,
        "val_rmse": best_model_info["val_rmse"],
        "val_r2": best_model_info["val_r2"],
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
    }

    with open(os.path.join(OUTPUT_DIR, "model_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f" Saved model metadata")

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"\nBest model: {best_model_name}")
    print(f"Files saved in: {OUTPUT_DIR}")
    print(f"\nTo use this model, update inference_rf.py to load from:")
    print(f"  - best_model.pkl (or best_model_nn.pt)")
    print(f"  - scaler_x_best.pkl")
    print(f"  - scaler_y_best.pkl")


if __name__ == "__main__":
    main()
