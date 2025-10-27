import pandas as pd
from astroquery.vizier import Vizier

Vizier.ROW_LIMIT = 1000
catalog = "I/355/gaiadr3"

result = Vizier.get_catalogs(catalog)
gaiadr3_sample = result[0].to_pandas()


gaiadr3_sample.to_csv("gaiadr3_sample.csv", index=False)
print("Downloaded Gaia DR3 sample successfully!")
