import pandas as pd
import numpy as np
import xarray as xr
import subprocess as sb
from Air_Data_gather1 import air_main
from Meteo_Data_gather1 import meteo_main

YEAR = 2021

# Air_Dataset vorbereiten:

air_main(YEAR,100)

# Meteo_Dataset vorbereiten:

meteo_main(YEAR,150)

# Datensätze öffnen:

d_air = xr.open_dataset(f"Data_Preparation/clean_Air_Datasets/Gereinigte Luft-Daten {YEAR}.nc")
d_air  = d_air.to_dataframe()

d_meteo = xr.open_dataset(f"Data_Preparation/clean_Meteo_Datasets/Gereinigte Meteo-Daten {YEAR}.nc")
d_meteo = d_meteo.to_dataframe()
d_meteo.drop('Datum', axis = 1, inplace=True)



print(d_air)
print(d_meteo)

# Trainingsdaten zusammenfügen:

d_training = pd.concat([d_air,d_meteo],axis = 1)

print(d_training)
print(d_training.iloc[0][1:])



dxr_training = xr.Dataset.from_dataframe(d_training)

print(f"{dxr_training}\n")



# In neues file abspeichern:

dxr_training.to_netcdf(f"Data_Preparation/Training_Datasets/Trainingsdaten_{YEAR}.nc", format='NETCDF4', mode='w')

print(f"\nEverything worked well. Trainingsdaten_{YEAR}.nc can now be found in the folder Training_Datasets\n\n ")


