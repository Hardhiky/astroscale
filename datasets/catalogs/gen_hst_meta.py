# save as gen_hst_meta.py and run
import numpy as np, pandas as pd

N = 2000
np.random.seed(4)
df = pd.DataFrame(
    {
        "image_id": np.arange(1, N + 1),
        "ra_center": np.random.uniform(0, 360, N),
        "dec_center": np.random.uniform(-90, 90, N),
        "filter": np.random.choice(["F160W", "F125W", "F814W", "F606W"], N),
        "exptime_s": np.random.choice([800, 1200, 2400, 3600], N),
    }
)
df.to_csv("hst_cutout_meta.csv", index=False)
print("hst_cutout_meta.csv written")
