import pandas as pd
import numpy as np
import xarray as xr
from Air_Data_gather_unused import air_main
from Meteo_Data_gather_unused import meteo_main

YEAR = 2021



# Datensätze öffnen:

d_air = air_main(YEAR,100)

d_meteo = meteo_main(YEAR,150)

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


