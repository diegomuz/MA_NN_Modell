import pandas as pd
import numpy as np
import xarray as xr

year = 2023
All_data = pd.read_csv(f"Data_Preparation/Meteo_Datasets/ugz_ogd_meteo_h1_{year}.csv")


Wetterwerte = All_data["Wert"]

c = 0
for i in range (len(Wetterwerte)):
    if np.isnan(Wetterwerte[i]):
        c+= 1

print(f"Number of NaN values = {c}")


# Es hat 22 Werte pro Stunde

Stampfenbachstrasse = []

idx = 0
for i in range(int(len(Wetterwerte)/22)):
    for d in range(8):
        Stampfenbachstrasse.append(Wetterwerte[idx])
        idx += 1
    idx += 14


# Wieder testen, wie oft Daten fehlen und dann diese ersetzten mit dem Durchschnitt derselben Daten von einer Stunde vorher und einer Stunde nachher

# Testen
c = 0
for i in range (len(Stampfenbachstrasse)):
    if np.isnan(Stampfenbachstrasse[i]):
       # print(i)
        c+= 1
print(f"Number of Nan Values in Stampfenbachstrasse: {c}")


# NA-Werte ersetzen:


reserve = 250

for i in range(len(Stampfenbachstrasse)):
    if np.isnan(Stampfenbachstrasse[i]):
        a = None
        b = None
        for l in range(reserve):

            if a != None and b != None:
                break
            
            if not np.isnan(Stampfenbachstrasse[i-(l+1)*8]):
                a = Stampfenbachstrasse[i-(l+1)*8]
            
            if not np.isnan(Stampfenbachstrasse[i+(l+1)*8]):
                b = Stampfenbachstrasse[i+(l+1)*8]

        Stampfenbachstrasse[i] = (a+b)/2
        if i < len(Stampfenbachstrasse)/2:
             tl = l
             ti = i

# Testen:

c = 0
for i in range(len(Stampfenbachstrasse)):
    if np.isnan(Stampfenbachstrasse[i]):
        c+= 1

print(f"Nan values after clean-up:{c}")

print(Stampfenbachstrasse[ti])

print(f"The above value should be the average of {Stampfenbachstrasse[ti-8*tl]} and {Stampfenbachstrasse[ti+8*tl]}")


# Resultate in Neuen File Speichern

res_file = open(f"Data_Preparation/Gereinigte Wetter-Daten SBS {year}.csv",'w')
res_file.writelines([f"{line}\n" for line in Stampfenbachstrasse])


