import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


year_list = [2020,2021,2022,2023]



features = ['O3']
to_predict_feature = 'O3'

def prepare_data():
    #prepare data

    training_df = pd.DataFrame()

    for year in year_list: 
        df = xr.open_dataset(f"Data_Preparation/Training_Datasets/Trainingsdaten_{year}.nc")
        df = df.to_dataframe()
        training_df = pd.concat([training_df, df], axis=0, ignore_index=True)




    # remove features that won't be used:

    for feature in training_df.columns:
        if feature not in features:
            training_df.drop(feature, axis = 1, inplace=True)

    return training_df


def calculate_roc(df, to_predict_feature):
    diff = []
    roc_l = []
    for i in range(len(df[to_predict_feature])-1):
        diff.append(df[to_predict_feature][i+1]-df[to_predict_feature][i])
        if df[to_predict_feature][i] != 0:
            perc = 100*(df[to_predict_feature][i+1]-df[to_predict_feature][i])/df[to_predict_feature][i]
            roc_l.append(perc)

    

    return diff, roc_l


diff, roc_l = calculate_roc(prepare_data(), to_predict_feature)

mean_diff = np.mean(diff)
mean_roc = np.round(np.mean(roc_l),2)

#print(roc_l)

plt.plot(roc_l, color = 'red')
plt.title('Änderungsrate der Ozonkonzentration')
plt.ylabel('Änderungsrate - %')
plt.xlabel('Stunden')
plt.yscale('linear')


plt.text(0.95, 0.95, f'mittlere Änderngsrate = {mean_roc}%', 
         transform=plt.gca().transAxes, 
         fontsize=8, 
         verticalalignment='top', 
         horizontalalignment='right',
         bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

plt.show()
    