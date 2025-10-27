import os

import numpy as np
import pandas as pd

OUT = "amost_dr9_mock.csv"


n = 5000
df = pd.DataFrame(
    {
        "obs_id": np.arange(1, n + 1),
        "ra": np.random.uniform(0, 360, n),
        "dec": np.random.uniform(-90, 90, n),
        "teff": np.random.uniform(3500, 9000, n),
        "logg": np.random.uniform(0.5, 5.0, n),
        "fe_h": np.random.uniform(-2.5, 0.5, n),
        "snr": np.random.uniform(20, 200, n),
    }
)
df.to_csv(OUT, index=False)
