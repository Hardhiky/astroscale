import numpy as np, pandas as pd

N = 6000
np.random.seed(2)
df = pd.DataFrame(
    {
        "star_id": np.arange(2000001, 2000001 + N),
        "ra": np.random.uniform(0, 360, N),
        "dec": np.random.uniform(-90, 90, N),
        "teff": np.random.uniform(3500, 7000, N),
        "logg": np.random.uniform(0, 5, N),
        "fe_h": np.random.uniform(-2.5, 0.5, N),
        "alpha_fe": np.random.uniform(-0.2, 0.5, N),
    }
)
df.to_csv("apogee_dr17_mock.csv", index=False)
print("apogee_dr17_mock.csv written")
