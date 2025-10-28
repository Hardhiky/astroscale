import numpy as np
import pandas as pd

datasets = {
    "datasets/catalogs/rave_dr6_sample.csv": 5000,
    "datasets/catalogs/apogee_dr17_sample.csv": 6000,
    "datasets/catalogs/sdss_dr17_sample.csv": 8000,
    "datasets/catalogs/gaia_dr3/gaiadr3_sample.csv": 7000,
}

for path, n in datasets.items():
    df = pd.DataFrame(
        {
            "source_id": np.arange(1, n + 1),
            "ra": np.random.uniform(0, 360, n),
            "dec": np.random.uniform(-90, 90, n),
            "teff": np.random.uniform(3000, 9000, n),
            "logg": np.random.uniform(0.1, 5.0, n),
            "fe_h": np.random.uniform(-3, 0.5, n),
            "snr": np.random.uniform(10, 200, n),
        }
    )
    df.to_csv(path, index=False)
    print(f"Generated {path} ({n} rows)")
