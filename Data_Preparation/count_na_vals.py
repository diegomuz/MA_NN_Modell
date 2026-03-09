import pandas as pd
import numpy as np

year = 2024

All_data = pd.read_csv(f"Data_Preparation/Meteo_Datasets/ugz_ogd_meteo_h1_{year}.csv")

missing_stampfenbachstrasse = 0
missing_rosengartenstrasse = 0
missing_schimmelstrasse = 0

for line in All_data:
    print(line)


for line in range(len(All_data['Datum'])):
    if All_data['Standort'][line] == 'Zch_Stampfenbachstrasse':
        if np.isnan(All_data['Wert'][line]):
            missing_stampfenbachstrasse += 1
    elif All_data['Standort'][line] == 'Zch_Rosengartenstrasse':
        if np.isnan(All_data['Wert'][line]):
            missing_rosengartenstrasse += 1
    elif All_data['Standort'][line] == 'Zch_Schimmelstrasse':
        if np.isnan(All_data['Wert'][line]):
            missing_schimmelstrasse += 1

print(f'NA Werte Stampfenbachstrasse = {missing_stampfenbachstrasse}')
print(f'NA Werte Rosengartenstrasse = {missing_rosengartenstrasse}')
print(f'NA Werte Schimmelstrasse = {missing_schimmelstrasse}')
      
      

