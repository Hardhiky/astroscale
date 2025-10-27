# save as gen_sdss_galaxy.py and run
import numpy as np, pandas as pd

N = 8000
np.random.seed(1)
df = pd.DataFrame(
    {
        "objID": np.arange(1000001, 1000001 + N),
        "ra": np.random.uniform(0, 360, N),
        "dec": np.random.uniform(-90, 90, N),
        "z": np.abs(np.random.normal(0.05, 0.03, N)),  # redshift
        "mag_g": np.random.uniform(14, 24, N),
        "mag_r": np.random.uniform(13.5, 23.5, N),
        "petroRad_r": np.random.exponential(3, N),
    }
)
df.to_csv("sdss_galaxy_sample.csv", index=False)
print("sdss_galaxy_sample.csv written")
