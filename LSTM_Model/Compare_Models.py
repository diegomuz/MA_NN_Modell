
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import json 




models = [ {'name':'O3-History(dim-32_range-1_forward-12_batch-32_lookback-12_features-1).json',
            'feature':'O3',
           'Type':1,
           'Model':'LSTM',
           'LSTM_Dimension':30,
           'forward':12,
           'timesteps':12,
           'epochs':30,
           'Features':1},
           {'name':'O3-History(dim-32_range-1_forward-12_batch-32_lookback-12_features-7).json',
            'feature':'O3',
           'Type': 1,
           'Model':'LSTM',
           'LSTM_Dimension':30,
           'timesteps':24,
           'forward':12,
           'epochs':20,
           'Features':7},
           {'name':'O3-History(dim-32_range-1_forward-12_batch-32_lookback-12_features-16).json',
            'feature':'O3',
           'Type': 1,
           'Model':'LSTM',
           'LSTM_Dimension':30,
           'timesteps':12,
           'epochs':20,
           'Features':16},
                      {'name':'O3-History(dim-32_range-1_forward-12_batch-32_lookback-12_features-24).json',
            'feature':'O3',
           'Type': 1,
           'Model':'LSTM',
           'LSTM_Dimension':30,
           'timesteps':12,
           'epochs':20,
           'Features':24},
                      {'name':'O3-History(dim-32_range-1_forward-12_batch-32_lookback-12_features-35).json',
            'feature':'O3',
           'Type': 1,
           'Model':'LSTM',
           'LSTM_Dimension':30,
           'timesteps':12,
           'epochs':20,
           'Features':35},

          
        ]       


def load_metrics(models):
    LSTM_histories = []
    XGB_histories = []
    for model in models:
        if model['Model'] == 'LSTM':

            with open(f'LSTM_Model/Histories/{model['name']}', 'r') as f:
                  LSTM_histories.append(json.load(f))
                
        
        elif model['Model'] == 'XGBOOST':
            with open(f'XGboost_Regression/Histories/{model['name']}','r') as f:
                XGB_histories.append(json.load(f))


    LSTM_loss = []
    LSTM_val_loss = []

    for history in LSTM_histories:
        LSTM_loss.append(np.average(history['loss']))
        LSTM_val_loss.append(np.average(history['val_loss']))

    XGB_loss = []
    XGB_val_loss = []

    for history in XGB_histories:
        c = 0
        for loss in history['validation_0']['rmse']:
            c += loss**2
        XGB_loss.append(c)

        c = 0
        for loss in history['validation_1']['rmse']:
            c += loss**2
        XGB_val_loss.append(c)

    Train_loss = LSTM_loss + XGB_loss

    Val_loss = LSTM_val_loss + XGB_val_loss

    return(Train_loss, Val_loss)




Train_loss, Val_loss = load_metrics(models)



parameter_to_change = 'Features'

model_names = [f'Anzahl {parameter_to_change}: {models[i][parameter_to_change]}' if models[i]['Model'] == 'LSTM' else f'XGBOOST' for i in range(len(models))]
#model_names = ['nur O3-Konzetration', 'O3-Konzentration  + zeitliche Features','nur Hauptfeatures mit hoher Korrelation + zeiliche Features', 'Haupt- und Kontextfeatures mit hoher Korrelation + zeitliche Features', 'alle Features']
#model_names = ['a','b','c','d','e']
print(model_names)



# make bar plot:


# Bar width and positions
bar_width = 0.2
positions = np.arange(len(model_names))

# Plot
plt.figure(figsize=(10, 6))

plt.bar(positions - 1.5 * bar_width, Train_loss, width=bar_width, label='Trainings-MSE', color='skyblue')
plt.bar(positions - 0.5 * bar_width, Val_loss, width=bar_width, label='Test-MSE', color='navy')



#Labels and formatting
plt.xticks(positions - bar_width, model_names)
plt.ylabel('MSE')
#plt.xlabel('Modelle')
plt.title('Trainings- und Test-MSE')
plt.legend()
#plt.tight_layout()
plt.show()




