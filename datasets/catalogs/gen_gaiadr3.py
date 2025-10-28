import numpy as np, pandas as pd

N = 50000
np.random.seed(0)
df = pd.DataFrame(
    {
        "source_id": np.arange(1, N + 1),
        "ra": np.random.uniform(0, 360, N),
        "dec": np.random.uniform(-90, 90, N),
        "parallax": np.abs(np.random.normal(5.0, 2.0, N)),
        "pmra": np.random.normal(0.0, 3.0, N),
        "pmdec": np.random.normal(0.0, 3.0, N),
        "phot_g_mean_mag": np.random.uniform(6.0, 20.0, N),
        "bp_rp": np.random.uniform(-0.5, 3.5, N),
    }
)
df.to_csv("gaiadr3_sample.csv", index=False)
print("gaiadr3_sample.csv written")
