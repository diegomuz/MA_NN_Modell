import pandas as pd
import numpy as np
import xarray as xr

ds = xr.open_dataset('ugz_ogd_meteo_h1_2021.nc')

df = ds.to_dataframe()

print(df)