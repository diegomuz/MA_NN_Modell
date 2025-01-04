import pandas as pd
import numpy as np
import xarray as xr


year = 2023


All_data = pd.read_csv(f"Data_Preparation/Air_Datasets/ugz_ogd_air_h1_{year}.csv")

#All_data = xr.Dataset.from_dataframe(All_data)


Luftwerte = All_data["Wert"]

print(len(Luftwerte))

#print(Luftwerte)

c = 0
for i in range (len(Luftwerte)):
    if np.isnan(Luftwerte[i]):
        #print(i)
        c+= 1

print(f"Number of NaN values = {c}")



c=0
for i in range(len(All_data)):
    if np.isnan(All_data["Wert"][i]):
        #print(i)
        
        c+=1


# neues file kreieren mit den richtigen Daten

# Es hat 24 Werte pro Stunde
Ausgewählte_Daten = []

idx = 0
for i in range(int(len(Luftwerte)/24)):
    for d in range(8):
        Ausgewählte_Daten.append(Luftwerte[idx]) # Nur werte von der Stampfenbachstrasse
        idx += 1

    idx += 16






#print(Ausgewählte_Daten[0])

# Wieder testen, wie oft Daten fehlen und dann diese ersetzten mit dem Durchschnitt derselben Daten von einer Stunde vorher und einer Stunde nachher

# Testen
c = 0
for i in range (len(Ausgewählte_Daten)):
    if np.isnan(Ausgewählte_Daten[i]):
       # print(i)
        c+= 1
print(f"Number of Nan Values in Ausgewählte_Daten: {c}")




# Ersetzen


reserve = 20

for i in range(len(Ausgewählte_Daten)):
    if np.isnan(Ausgewählte_Daten[i]):
        a = None
        b = None

        for l in range(reserve):

            if a != None and b != None:
                break
            
            #if 8 < i < (len(Ausgewählte_Daten)-8):
            if not np.isnan(Ausgewählte_Daten[i+(l+1)*8]):
                    b = Ausgewählte_Daten[i+(l+1)*8]
                    

            if  not np.isnan(Ausgewählte_Daten[i-(l+1)*8]):
                    a = Ausgewählte_Daten[i-(l+1)*8]


        if i < len(Ausgewählte_Daten)/3:
            ti = i
            tl = l 

        Ausgewählte_Daten[i] = (a+b)/2
                
                


# Testen

c = 0
for i in range (len(Ausgewählte_Daten)):
    if np.isnan(Ausgewählte_Daten[i]):
        c+= 1
print(f"Nan values after clean-up:{c}")

print(Ausgewählte_Daten[ti])


print(f"The above value should be the average of {Ausgewählte_Daten[ti+tl*8]} and {Ausgewählte_Daten[ti-tl*8]}")



# Resultate in Neuen File Speichern

res_file = open(f"Data_Preparation/Gereinigte Luft-Daten SBS {year}.csv",'w')
res_file.writelines('"Wert"\n')
res_file.writelines([f"{line}\n" for line in Ausgewählte_Daten])


# Eventuell noch Daten Scalen