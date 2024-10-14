import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler




def air_main(year,reserve):

    All_data = pd.read_csv(f"Data_Preparation/Air_Datasets/ugz_ogd_air_h1_{year}.csv")

    # Plan: Resultat soll wieder ein pandas dataframe sein, das Aussihet wie 'All_data' aber keine NA werte mehr hat und wo es nur noch hat 
    # Stampfenbachstrasse und Rosengarten/Schimmelstrasse zusammen. Die beiden letzteren sollen kontext geben dem Modell
    # Es werden nur Jahre verwendet, wo Stampfenbachstrasse deutlich am wenigsten fehlende Messwerte hat

    print(All_data)

    print(All_data.iloc[0].values[0])

    Luftwerte = All_data["Wert"]

    print(len(Luftwerte))


    c = 0
    for i in range (len(Luftwerte)):
        if np.isnan(Luftwerte[i]):
            #print(i)
            c+= 1

    print(f"Number of NaN values = {c}")

    # neues file kreieren mit den richtigen Daten

    # Es hat 24 Werte pro Stunde
    Stampfenbachstrasse = []

    idx = 0
    for i in range(int(len(Luftwerte)/24)):
        for d in range(8):
            Stampfenbachstrasse.append(Luftwerte[idx]) # Nur werte von der Stampfenbachstrasse
            idx += 1

        idx += 16



    Schimmel_Rosengartenstrasse = []

    s = [Luftwerte[i] for i in range(len(Luftwerte)) if All_data['Standort'][i] == "Zch_Schimmelstrasse" or All_data['Standort'][i] == "Zch_Rosengartenstrasse"]


    c = 0
    for i in range(len(s)):
        if np.isnan(s[i]):
            c+= 1

    print(f"Nan-vals in s: {c}")

    test_l = []


    for i in range(int(len(s)/12)):
        for d in range(6):
            test_l = [] # Liste mit nicht NA-Werten, aus dessen Einträgen soll der Durchschnittswert berechnet werden
            if not np.isnan(s[12*i + d]):
                test_l.append(s[12*i + d])
            if not np.isnan(s[12*i + d + 6]):
                test_l.append(s[12*i + d + 6])

            if len(test_l) == 0:
                Schimmel_Rosengartenstrasse.append(s[12*i + d])
            else:
                Schimmel_Rosengartenstrasse.append(sum(test_l)/len(test_l))



    c = 0
    for i in range(len(Stampfenbachstrasse)):
        if np.isnan(Stampfenbachstrasse[i]):
            c+= 1

    print(f"Nan-vals in Stampfenbachstrasse: {c}")


    c = 0
    for i in range(len(Schimmel_Rosengartenstrasse)):
        if np.isnan(Schimmel_Rosengartenstrasse[i]):
            c+= 1

    print(f"Nan-vals in Schimmel_Rosengartenstrasse: {c}")

    #print(f"{Schimmel_Rosengartenstrasse[7]} should be the average of {s[13]} and {s[19]}")


    # Fehlende Daten ersetzen: 

    # Zuerst für die Stampfenbachstrasse: 
    

    for i in range(len(Stampfenbachstrasse)):
        if np.isnan(Stampfenbachstrasse[i]):
            a = None
            b = None

            for l in range(reserve):
                
                if a != None and b != None:
                    break

                if not np.isnan(Stampfenbachstrasse[i + (l+1)*8]) and b == None:
                    b = Stampfenbachstrasse[i + (l+1)*8]
                
                if not np.isnan(Stampfenbachstrasse[i - (l+1)*8]) and a == None:
                    a = Stampfenbachstrasse[i - (l+1)*8]
        
            Stampfenbachstrasse[i] = (a+b)/2

    # Dann für die Schimmel_Rosengartenstrasse:

    

    for i in range(len(Schimmel_Rosengartenstrasse)):
        if np.isnan(Schimmel_Rosengartenstrasse[i]):
            a = None
            b = None

            for l in range(reserve): 
                
                if a != None and b != None:
                    break

                if not np.isnan(Schimmel_Rosengartenstrasse[i + (l+1)*6]) and b == None:
                    b = Schimmel_Rosengartenstrasse[i + (l+1)*6]

                if not np.isnan(Schimmel_Rosengartenstrasse[i - (l+1)*6]) and a == None:
                    a = Schimmel_Rosengartenstrasse[i - (l+1)*6]
            
            Schimmel_Rosengartenstrasse[i] = (a+b)/2

            ti = i
            tl = l

    # Nach NaN-Werten überprüfen:

    # Stampfenbachstrasse:

    c = 0 
    for i in range(len(Stampfenbachstrasse)):
        if np.isnan(Stampfenbachstrasse[i]):
            c += 1
    print(f"Nan values in Stampfenbachstrasse after clean-up: {c}")

    # Schimmel_Rosengartenstrasse:

    c = 0
    for i in range(len(Schimmel_Rosengartenstrasse)):
        if np.isnan(Schimmel_Rosengartenstrasse[i]):
            c =+ 1
    print(f"Nan values in Schimmel_Rosengartenstrasse after clean-up: {c}")


    # Daten zusammenfügen zu neuem Dataframe



    Data = {
        'Datum':[All_data['Datum'][24*i] for i in range(int(len(All_data)/24))],

        'CO' : [Stampfenbachstrasse[8*i] for i in range(int(len(Stampfenbachstrasse)/8))],
        'SO2' : [Stampfenbachstrasse[8*i + 1] for i in range(int(len(Stampfenbachstrasse)/8))],
        'NOx' : [Stampfenbachstrasse[8*i + 2] for i in range(int(len(Stampfenbachstrasse)/8))],
        'NO' : [Stampfenbachstrasse[8*i + 3] for i in range(int(len(Stampfenbachstrasse)/8))],
        'NO2' : [Stampfenbachstrasse[8*i + 4] for i in range(int(len(Stampfenbachstrasse)/8))],
        'O3' : [Stampfenbachstrasse[8*i + 5] for i in range(int(len(Stampfenbachstrasse)/8))],
        'PM10' : [Stampfenbachstrasse[8*i + 6] for i in range(int(len(Stampfenbachstrasse)/8))],
        'PM2.5': [Stampfenbachstrasse[8*i + 7] for i in range(int(len(Stampfenbachstrasse)/8))],

        # Conetxt values:

        'Cont_NOx' : [Schimmel_Rosengartenstrasse[6*i] for i in range(int(len(Schimmel_Rosengartenstrasse)/6))],
        'Cont_NO' : [Schimmel_Rosengartenstrasse[6*i + 1] for i in range(int(len(Schimmel_Rosengartenstrasse)/6))],
        'Cont_NO2' : [Schimmel_Rosengartenstrasse[6*i + 2] for i in range(int(len(Schimmel_Rosengartenstrasse)/6))],
        'Cont_O3' : [Schimmel_Rosengartenstrasse[6*i + 3] for i in range(int(len(Schimmel_Rosengartenstrasse)/6))],
        'Cont_PM10' : [Schimmel_Rosengartenstrasse[6*i + 4] for i in range(int(len(Schimmel_Rosengartenstrasse)/6))],
        'Cont_PM2.5' : [Schimmel_Rosengartenstrasse[6*i + 5] for i in range(int(len(Schimmel_Rosengartenstrasse)/6))]



        
    }



    Clean_air_df = pd.DataFrame.from_dict(Data)


    print(Clean_air_df.iloc[1][1:].values)


                
    # Daten in netcdf file speichern: 

    Clean_dxr = xr.Dataset.from_dataframe(Clean_air_df)

    Clean_dxr.to_netcdf(f"Data_Preparation/clean_Air_Datasets/Gereinigte Luft-Daten {year}.nc", format= 'NETCDF4', mode = 'w' )




             
