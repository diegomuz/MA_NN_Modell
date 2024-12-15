'''
Quellen/Tutorials:

https://engineeringfordatascience.com/posts/matplotlib_subplots/

'''



import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import plotly

import plotly.express as px 
import plotly.io as pio

print(pio.renderers)






YEAR = 2021

df = xr.open_dataset(f"Data_Preparation/Training_Datasets/Trainingsdaten_{YEAR}.nc")
df = df.to_dataframe()

print(df)

print('\n')

feature_keys = ['CO', 'SO2', 'NO2','O3', 'PM10', 'PM2.5']
feature_units = ['(mg/m3)','(µg/m3)','(µg/m3)','(µg/m3)','(µg/m3)','(µg/m3)']


colors = ["blue", "orange","green", "red","purple","brown"]




def visualize_traing_data(data):
    time_data = data['Datum']
    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(10, 10), dpi=80, facecolor="w", edgecolor="k"
    )
    print(axes[0,0])
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i]
        t_data = data[key]
        t_data.index = time_data
        #print(t_data.head())
        t_data.plot(ax = axes[i//2, i % 2],color = c, title = f"{key} - {feature_units[i]}", rot = 20)
    plt.tight_layout()
    plt.show()


def histogram_boxplot(data, key, unit):
    t_data = data[key]
    fig = px.histogram(t_data, marginal='box', title= f"{key} - ({unit})")

    
    #plotly.offline.plot(fig,image = 'svg',filename=f"{key}_Boxplot_Histogram.svg")
   
    fig.show()
    
visualize_traing_data(df)
#histogram_boxplot(df, 'NO2', 'mg/m3')




# scale outliers to normal values: 



