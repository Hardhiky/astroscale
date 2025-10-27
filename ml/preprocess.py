import os
from glob import glob

import numpy as np
import pandas as pd

INPUT_DIR = "datasets/catalogs"
OUT_PATH = "datasets/ml_preprocessed/all_sources.csv"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def load_catalog(path, rename_map):
    try:
        df = pd.read_csv(path)
        for k, v in rename_map.items():
            if k in df.columns:
                df.rename(columns={k: v}, inplace=True)
        keep = [
            c
            for c in df.columns
            if c in ["ra", "dec", "teff", "logg", "fe_h", "snr", "z", "parallax"]
        ]
        for col in ["ra", "dec"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["ra", "dec"], inplace=True)
        return df[keep]
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return pd.DataFrame()


csvs = glob(os.path.join(INPUT_DIR, "**", "*.csv"), recursive=True)

print(f"Found {len(csvs)} catalogs")

frames = []

for csv_file in csvs:
    name = os.path.basename(csv_file)
    if "gaia" in name:
        frames.append(
            load_catalog(csv_file, {"ra": "ra", "dec": "dec", "parallax": "parallax"})
        )
    elif "sdss" in name:
        frames.append(load_catalog(csv_file, {"RA": "ra", "DEC": "dec", "z": "z"}))
    elif "apogee" in name:
        frames.append(
            load_catalog(
                csv_file,
                {
                    "RA": "ra",
                    "DEC": "dec",
                    "FE_H": "fe_h",
                    "TEFF": "teff",
                    "LOGG": "logg",
                },
            )
        )
    elif "rave" in name:
        frames.append(
            load_catalog(
                csv_file,
                {
                    "RAdeg": "ra",
                    "DEdeg": "dec",
                    "FeH": "fe_h",
                    "Teff": "teff",
                    "logg": "logg",
                },
            )
        )
    elif "hlf" in name or "hubble" in name:
        frames.append(load_catalog(csv_file, {"RA": "ra", "DEC": "dec"}))
    elif "lamost" in name:
        frames.append(
            load_catalog(
                csv_file,
                {
                    "ra": "ra",
                    "dec": "dec",
                    "teff": "teff",
                    "logg": "logg",
                    "fe_h": "fe_h",
                    "snr": "snr",
                },
            )
        )

if len(frames) == 0:
    raise SystemExit("No catalogs found for preprocessing")

merged = pd.concat(frames, ignore_index=True)

if "fe_h" in merged:
    merged["fe_h"] = merged["fe_h"].clip(-3, 1)
if "snr" in merged:
    merged["snr"] = np.log1p(merged["snr"])
if "teff" in merged:
    merged["teff"] = (merged["teff"] - 3500) / (9000 - 3500)
if "logg" in merged:
    merged["logg"] = (merged["logg"] - 0.5) / (5.0 - 0.5)
if "ra" in merged:
    merged["ra_rad"] = np.deg2rad(merged["ra"])
if "dec" in merged:
    merged["dec_rad"] = np.deg2rad(merged["dec"])

merged.drop_duplicates(subset=["ra", "dec"], inplace=True)

merged.to_csv(OUT_PATH, index=False)
print(f"Saved unified catalog → {OUT_PATH}")
print(f"Total entries: {len(merged):,}")
