
import sys
import os
import pickle
import numpy as np


def load_production_model(model_dir="datasets/ml_preprocessed/models"):

    model_path = os.path.join(model_dir, "production_model.pkl")
    scaler_x_path = os.path.join(model_dir, "production_scaler_x.pkl")
    scaler_y_path = os.path.join(model_dir, "production_scaler_y.pkl")
    metadata_path = os.path.join(model_dir, "production_metadata.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Production model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    scaler_x = None
    scaler_y = None
    if os.path.exists(scaler_x_path):
        with open(scaler_x_path, "rb") as f:
            scaler_x = pickle.load(f)
    if os.path.exists(scaler_y_path):
        with open(scaler_y_path, "rb") as f:
            scaler_y = pickle.load(f)

    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

    print(
        f"Loaded production model: {metadata.get('model_type', 'unknown') if metadata else 'unknown'}",
        file=sys.stderr,
    )
    if metadata:
        print(
            f"Model performance: R² = {metadata.get('val_r2', 0):.4f}", file=sys.stderr
        )

    return {
        "model": model,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "metadata": metadata,
    }


def predict(model_info, features):

    x = np.array([features], dtype=np.float32)

    y_pred = model_info["model"].predict(x)[0]

    return y_pred


def main():
    if len(sys.argv) != 8:
        print(
            "Usage: python inference_production.py ra dec teff logg fe_h snr parallax",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        ra = float(sys.argv[1])
        dec = float(sys.argv[2])
        teff = float(sys.argv[3])
        logg = float(sys.argv[4])
        fe_h = float(sys.argv[5])
        snr = float(sys.argv[6])
        parallax = float(sys.argv[7])
    except ValueError as e:
        print(f"Error parsing arguments: {e}", file=sys.stderr)
        sys.exit(1)

    features = [ra, dec, teff, logg, fe_h, snr, parallax]

    try:
        model_info = load_production_model()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        z_value = predict(model_info, features)
    except Exception as e:
        print(f"Error making prediction: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"{z_value:.6f}")


if __name__ == "__main__":
    main()
