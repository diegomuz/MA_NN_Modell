import pandas as pd
import numpy as np
import xarray as xr



def meteo_main(year,reserve):



    All_data = pd.read_csv(f"Data_Preparation/Meteo_Datasets/ugz_ogd_meteo_h1_{year}.csv")

    # Plan: Resultat soll wieder ein pandas dataframe sein, das Aussihet wie 'All_data' aber keine NA werte mehr hat und wo es nur noch hat 
    # Stampfenbachstrasse und Rosengarten/Schimmelstrasse zusammen. Die beiden letzteren sollen kontext geben dem Modell
    # Es werden nur Jahre verwendet, wo Stampfenbachstrasse deutlich am wenigsten fehlende Messwerte h


    print(All_data)

    print(All_data.iloc[0].values[0])

    Meteowerte = All_data["Wert"]

    print(len(Meteowerte))

    c = 0
    for i in range (len(Meteowerte)):
        if np.isnan(Meteowerte[i]):
            #print(i)
            c+= 1

    print(f"Number of NaN values = {c}")

    # neues file kreieren mit den richtigen Daten

    # Es hat 24 Werte pro Stunde


    Stampfenbachstrasse = [Meteowerte[i] for i in range(len(Meteowerte)) if All_data['Standort'][i] == "Zch_Stampfenbachstrasse"]

    Schimmel_Rosengartenstrasse = []

    s = [Meteowerte[i] for i in range(len(Meteowerte)) if All_data['Standort'][i] == "Zch_Schimmelstrasse" or All_data['Standort'][i] == "Zch_Rosengartenstrasse"]

    c = 0
    for i in range(len(s)):
        if np.isnan(s[i]):
            c+= 1

    print(f"Nan-vals in s: {c}")

    test_l = []

    for i in range(int(len(s)/14)):
        for d in range(7):
            test_l = [] # Liste mit nicht NA-Werten, aus dessen Einträgen soll der Durchschnittswert berechnet werden
            if not np.isnan(s[14*i + d]):
                test_l.append(s[14*i + d])
            if not np.isnan(s[14*i + d + 7]):
                test_l.append(s[14*i + d + 7])

            if len(test_l) == 0:
                Schimmel_Rosengartenstrasse.append(s[14*i + d])
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

                if not np.isnan(Schimmel_Rosengartenstrasse[i + (l+1)*7]) and b == None:
                    b = Schimmel_Rosengartenstrasse[i + (l+1)*7]

                if not np.isnan(Schimmel_Rosengartenstrasse[i - (l+1)*7]) and a == None:
                    a = Schimmel_Rosengartenstrasse[i - (l+1)*7]
            
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
        'Datum' : [All_data['Datum'][22*i] for i in range(int(len(All_data)/22))],

        'T' : [Stampfenbachstrasse[8*i] for i in range(int(len(Stampfenbachstrasse)/8))],
        'Hr' : [Stampfenbachstrasse[8*i + 1] for i in range(int(len(Stampfenbachstrasse)/8))],
        'p' : [Stampfenbachstrasse[8*i + 2] for i in range(int(len(Stampfenbachstrasse)/8))],
        'RainDur' : [Stampfenbachstrasse[8*i + 3] for i in range(int(len(Stampfenbachstrasse)/8))],
        'StrGlo' : [Stampfenbachstrasse[8*i + 4] for i in range(int(len(Stampfenbachstrasse)/8))],
        'WD' : [Stampfenbachstrasse[8*i + 5] for i in range(int(len(Stampfenbachstrasse)/8))],
        'WVv' : [Stampfenbachstrasse[8*i + 6] for i in range(int(len(Stampfenbachstrasse)/8))],
        'WVs': [Stampfenbachstrasse[8*i + 7] for i in range(int(len(Stampfenbachstrasse)/8))],

        # Context values:

        'Cont_T' : [Schimmel_Rosengartenstrasse[7*i] for i in range(int(len(Schimmel_Rosengartenstrasse)/7))],
        'Cont_Hr' : [Schimmel_Rosengartenstrasse[7*i + 1] for i in range(int(len(Schimmel_Rosengartenstrasse)/7))],
        'Cont_p' : [Schimmel_Rosengartenstrasse[7*i + 2] for i in range(int(len(Schimmel_Rosengartenstrasse)/7))],
        'Cont_RainDur' : [Schimmel_Rosengartenstrasse[7*i + 3] for i in range(int(len(Schimmel_Rosengartenstrasse)/7))],
        'Cont_WD' : [Schimmel_Rosengartenstrasse[7*i + 4] for i in range(int(len(Schimmel_Rosengartenstrasse)/7))],
        'Cont_WVv' : [Schimmel_Rosengartenstrasse[7*i + 5] for i in range(int(len(Schimmel_Rosengartenstrasse)/7))],
        'Cont_WVs' : [Schimmel_Rosengartenstrasse[7*i + 6] for i in range(int(len(Schimmel_Rosengartenstrasse)/7))]




    }


    Clean_df = pd.DataFrame(Data)


    print(Clean_df)

    print(Clean_df.iloc[0][1:].values)

    # Daten in netcdf file speichern: 

    Clean_dxr = xr.Dataset.from_dataframe(Clean_df)

    #print(Clean_dxr.dims)
    #print(Clean_dxr.variables)

    Clean_dxr.to_netcdf(f"Data_Preparation/clean_Meteo_Datasets/Gereinigte Meteo-Daten {year}.nc", format='NETCDF4', mode = 'w' )



