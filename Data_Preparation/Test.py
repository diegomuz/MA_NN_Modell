import pandas as pd
import numpy as np
import xarray as xr

year = 2021
df = pd.read_csv(f"Data_Preparation/Meteo_Datasets/ugz_ogd_meteo_h1_{year}.csv")

print(df)

dxr = xr.Dataset.from_dataframe(df)

# Notiz:
# Lieber mit pdDataframe arbeiten, funktioniert schneller, resultat dann aber als net cdf file speicher und wieder als pd öffnen

''''
print(dxr['Datum'][0])

dlr = xr.Dataset()

dlr['Wert'] = df['Wert'] 


print(dlr)
print(dxr)
'''

print(dxr)
dxr.to_netcdf(f"Data_Preparation/ugz_ogd_meteo_h1_{year}.nc",format = 'NETCDF4',mode = 'w')


#print(All_data[2])

#Werte1 = All_data["Wert"]

#print(Werte1)

#Werte2 = All_datacsv["Wert"]

#print(Werte2)

#All_data.to_netcdf("All_Data.nc")

#haha = xr.open_dataset("All_Data.nc")


