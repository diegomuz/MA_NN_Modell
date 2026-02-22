import pandas as pd
import numpy as np
import xarray as xr



def meteo_main(year,reserve):



    All_data = pd.read_csv(f"Data_Preparation/Meteo_Datasets/ugz_ogd_meteo_h1_{year}.csv")

    # Plan: Resultat soll wieder ein pandas dataframe sein, das Aussihet wie 'All_data' aber keine NA werte mehr hat und wo es nur noch hat 
    # Stampfenbachstrasse und Rosengarten/Schimmelstrasse zusammen. Die beiden letzteren sollen kontext geben dem Modell
    # Es werden nur Jahre verwendet, wo Stampfenbachstrasse deutlich am wenigsten fehlende Messwerte h

    # adjust the csv structure for year 2024 to the one from years 2020-2023
    if year == 2024:
        # the code inside ths if statement was written with the help of ai, the rest is all self written
        df_2024 = All_data
        df_2023 = pd.read_csv("Data_Preparation/Meteo_Datasets/ugz_ogd_meteo_h1_2023.csv")

        # Remove Heubeeribüel
        df_2024 = df_2024[df_2024["Standort"] != "Zch_Heubeeribüel"]

        # Extract correct order from 2023
        station_order = df_2023["Standort"].unique()
        parameter_order = (
            df_2023[df_2023["Standort"] == station_order[0]]["Parameter"].unique()
        )

        # Apply ordering to 2024
        df_2024["Standort"] = pd.Categorical(
            df_2024["Standort"],
            categories=station_order,
            ordered=True
        )

        df_2024["Parameter"] = pd.Categorical(
            df_2024["Parameter"],
            categories=parameter_order,
            ordered=True
        )

        df_2024 = df_2024.sort_values(
            ["Datum", "Standort", "Parameter"]
        ).reset_index(drop=True)

        All_data = df_2024

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


    Clean_meteo_df = pd.DataFrame(Data)


    print(Clean_meteo_df)

    print('\n Processing...')

    outlier_count = 0

    # identify outliers and replace them: 
    
    min_occurence = 6

    
    for cols in Clean_meteo_df.columns[1:]:
        if cols != 'RainDur' and cols != 'Cont_RainDur':


            df = Clean_meteo_df[cols]
            q1 = df.quantile(0.25)
            q3 = df.quantile(0.75)
            IQR = q3-q1
            upper_fence = q3 + 1.7*IQR
            lower_fence = q3 -1.7*IQR
            df = list(df)

            for i in range (len(df)):
            
                if not np.isnan(df[i]):

                    if i < reserve + 1 or i > len(df) -reserve -1: 

                        if df.count(df[i]) <= min_occurence and not (lower_fence <= df[i] <= upper_fence):
                            outlier_count += 1
                            if df[i] <= lower_fence:
                                df[i] = lower_fence
                            elif df[i] > upper_fence:
                                df[i] = upper_fence

                            

                    else: 
                        if df.count(df[i]) <= min_occurence and not (lower_fence <= df[i] <= upper_fence):
                            outlier_count += 1
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
                    
            Clean_meteo_df[cols] = df
                                

    print(f'Removed {outlier_count} outliers\n That is {100*outlier_count/(len(Clean_meteo_df)*(len(Clean_meteo_df.columns)-1))}%') 

    """"
    # Daten in netcdf file speichern: 

    Clean_dxr = xr.Dataset.from_dataframe(Clean_meteo_df)

    #print(Clean_dxr.dims)
    #print(Clean_dxr.variables)

    Clean_dxr.to_netcdf(f"Data_Preparation/clean_Meteo_Datasets/Gereinigte Meteo-Daten {year}.nc", format='NETCDF4', mode = 'w' )
    """
    return(Clean_meteo_df)
    



