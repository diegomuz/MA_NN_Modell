import pandas as pd
import numpy as np
import xarray as xr





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

    # Determine order of gases in Datafile (Different for year < 2019 and for year > 2019)

    """""
    # Im moment ignorieren

    Stampfenbach_gas_order = ['CO','SO2','NOx','NO','NO2','O3','PM10','PM2.5']
    Stampfenbach_indexes = [0,0,0,0,0,0,0,0]
    idx  = 0
    for i in range(24):
        if All_data['Standort'][i] == "Zch_Stampfenbachstrasse":
            Stampfenbach_indexes[Stampfenbach_gas_order.index(All_data['Parameter'][i])] = idx
            idx += 1

    Schimmel_Rosengarten_gas_order = ['NOx','NO','NO2','O3','PM10','PM2.5']
    Schimmel_Rosengarten_indexes = [0,0,0,0,0,0]
    idx = 0
    for i in range(24):
        if All_data ['Standort'][i] == "Zch_Schimmelstrasse":
            Schimmel_Rosengarten_indexes[Schimmel_Rosengarten_gas_order.index(All_data['Parameter'][i])] = idx
            idx += 1


    """



    Stampfenbachstrasse = [Luftwerte[i] for i in range(len(Luftwerte)) if All_data['Standort'][i] == "Zch_Stampfenbachstrasse" ]

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

    print('\n Processing...')

    # identify outliers and replace them: 
    
    min_occurence = 6

    
    for cols in Clean_air_df.columns[1:]:
        df = Clean_air_df[cols]
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        IQR = q3-q1
        upper_fence = q3 + 1.7*IQR
        lower_fence = q3 -1.7*IQR
        df = list(df)

        for i in range (len(df)):
        
            if not np.isnan(df[i]):

                if i < reserve + 1 or i > len(df) -reserve -1: 

                     if df.count(df[i]) <= min_occurence and not lower_fence <= df[i] <= upper_fence:
                        if df[i] <= lower_fence:
                             df[i] = lower_fence
                        elif df[i] > upper_fence:
                            df[i] = upper_fence

                        

                else: 
                    if df.count(df[i]) <= min_occurence and not lower_fence <= df[i] <= upper_fence:
                        a = None
                        b = None

                        for l in range(reserve):

                            if a != None and b != None:
                                break

                            if  not np.isnan(df[i + (l+1)]):
                                if lower_fence <= df[i + (l+1)] <= upper_fence:
                                    b = df[i + (l+1)]

                            if  not np.isnan(df[i - (l+1)]):
                                if lower_fence <= df[i - (l+1)] <= upper_fence:
                                    a = df[i - (l+1)]
                        if a == None or b == None:
                            if df[i] <= lower_fence:
                                df[i] = lower_fence

                            elif df[i] > upper_fence:
                                df[i] = upper_fence

                        else:
                            df[i] = (a+b)/2
                
        Clean_air_df[cols] = df
                            


            

                
    # Daten in netcdf file speichern: 

    Clean_dxr = xr.Dataset.from_dataframe(Clean_air_df)

    Clean_dxr.to_netcdf(f"Data_Preparation/clean_Air_Datasets/Gereinigte Luft-Daten {year}.nc", format= 'NETCDF4', mode = 'w' )

    return(Clean_air_df)



