import pandas as pd
import numpy as np
import torch
import os

IN_PATH = "datasets/ml_preprocessed/all_sources.csv"
OUT_PATH = "datasets/ml_preprocessed/tensors"

os.makedirs(OUT_PATH, exist_ok=True)

df = pd.read_csv(IN_PATH)

cols = ["ra_rad", "dec_rad", "teff", "logg", "fe_h", "snr", "z", "parallax"]
df = df[cols]

arr = df.to_numpy(np.float32)
mask = ~np.isnan(arr)

tensor = torch.tensor(np.nan_to_num(arr, nan=0.0), dtype=torch.float32)
mask_tensor = torch.tensor(mask, dtype=torch.bool)

torch.save(tensor, os.path.join(OUT_PATH, "features.pt"))
torch.save(mask_tensor, os.path.join(OUT_PATH, "mask.pt"))

print("Saved feature tensor →", os.path.join(OUT_PATH, "features.pt"))
print("Saved mask tensor →", os.path.join(OUT_PATH, "mask.pt"))
print("Shape:", tensor.shape)
