import numpy as np, pandas as pd

N = 7000
np.random.seed(3)
df = pd.DataFrame(
    {
        "raveid": np.arange(3000001, 3000001 + N),
        "ra": np.random.uniform(0, 360, N),
        "dec": np.random.uniform(-90, 90, N),
        "rv": np.random.normal(0, 40, N),
        "snr": np.random.uniform(10, 200, N),
        "teff": np.random.uniform(3800, 8000, N),
        "logg": np.random.uniform(0.0, 5.0, N),
        "fe_h": np.random.uniform(-2.5, 0.5, N),
    }
)
df.to_csv("rave_dr6_mock.csv", index=False)
print("rave_dr6_mock.csv written")
