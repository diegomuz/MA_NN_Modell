import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor

import json

year_list = [2020,2021,2022,2023]

training_df = pd.DataFrame()

for year in year_list: 
    df = xr.open_dataset(f"Data_Preparation/Training_Datasets/Trainingsdaten_{year}.nc")
    df = df.to_dataframe()
    training_df = pd.concat([training_df, df], axis=0, ignore_index=True)



# define what Features should be used for the model training

features = ['Datum', 'CO', 'SO2', 'NOx', 'NO', 'NO2', 'O3', 'PM10', 'PM2.5',
       'Cont_NOx', 'Cont_NO', 'Cont_NO2', 'Cont_O3', 'Cont_PM10', 'Cont_PM2.5',
       'T', 'Hr', 'p', 'RainDur', 'StrGlo', 'WD', 'WVv', 'WVs', 'Cont_T',
       'Cont_Hr', 'Cont_p', 'Cont_RainDur', 'Cont_WD', 'Cont_WVv', 'Cont_WVs']

# remove features that won't be used:

for feature in training_df.columns:
    if feature not in features:
        training_df.drop(feature, axis = 1, inplace=True)



num_of_feautures = 0

# encode month&day in the year and hour of the day:

Datum = pd.to_datetime(training_df['Datum'], format='%Y-%m-%dT%H:%M%z')

day_of_year = Datum.dt.day_of_year
hour_of_day = Datum.dt.hour
month_of_year = Datum.dt.month

# hour encoding:

sin_h = [np.sin(2*np.pi*h/24) for h in hour_of_day]
cos_h = [np.cos(2*np.pi*h/24) for h in hour_of_day]

#day encoding 

sin_d = [np.sin(2*np.pi*d/24) for d in day_of_year]
cos_d = [np.cos(2*np.pi*d/24) for d in day_of_year]

# month encoding:


sin_m = [np.sin(2*np.pi*m/12) for m in month_of_year]
cos_m = [np.cos(2*np.pi*m/12) for m in month_of_year]

# add embeddings to training_df:


training_df['sin_h'] = sin_h
training_df['cos_h'] = cos_h
training_df['sin_d'] = sin_d
training_df['cos_d'] = cos_d
training_df['sin_m'] = sin_m
training_df['cos_m'] = cos_m


print(training_df)



# split, scale data

scaler = MinMaxScaler()

def split_scale_data(df,split_percentage):

    scaled_df = df.drop(['Datum'], axis = 1)
    scaled_df = pd.DataFrame(scaler.fit_transform(scaled_df), columns = scaled_df.columns)
    split_point = round(len(scaled_df)*split_percentage)
    train_data = scaled_df.iloc[:split_point]
    test_data = scaled_df.iloc[split_point:].reset_index(drop=True)

    global num_of_feautures
    num_of_feautures = len(train_data.columns)


    return train_data, test_data



def create_training_data(df,split_percentage, to_predict_feature, timesteps, y_range):
    train_data , test_data = split_scale_data(df,split_percentage)

    print(train_data)
    print(test_data)

    # number of features is number of cols in train_data
    X_tr, Y_tr = [],[]



    for i in range(len(train_data)- timesteps - y_range + 1 -y_forward):
        s = []
        for d in range(timesteps):
            lis = train_data.iloc[i+d]
            
            for element in lis:
                s.append(element)
            
        X_tr.append(s)

        s = []
        for d in range(y_range):
            lis = train_data[to_predict_feature][i+timesteps+d -1 +y_forward]

            s.append(lis)
        
        Y_tr.append(s)

    X_tr = np.array(X_tr)
    Y_tr = np.array(Y_tr)


    X_te, Y_te = [],[]

    for i in range(len(test_data) - timesteps - y_range +1 -y_forward):
        s = []
        for d in range(timesteps):
            lis = test_data.iloc[i+d]

            for element in lis:
                s.append(element)

        X_te.append(s)

        s = []
        for d in range(y_range):
            lis = test_data[to_predict_feature][i+timesteps+d -1 +y_forward]
            s.append(lis)

      
        Y_te.append(s)

    X_te = np.array(X_te)
    Y_te = np.array(Y_te)

    return X_tr, Y_tr, X_te, Y_te


y_forward = 24

def run():
    look_back = 36
    y_range = 1
    to_predict_feature = 'O3'

    X_train, Y_train, X_test, Y_test = create_training_data(training_df, 0.8, to_predict_feature, look_back, y_range)

    print(X_train.shape)

    # build the model:
    model = XGBRegressor(
    n_estimators= 45,   
    learning_rate=0.1,   
    max_depth=6,         
    objective="reg:squarederror"  
)

    eval_set = [(X_train, Y_train), (X_test, Y_test)]

  
    # add early stopping
    #model.fit(X_train, Y_train, eval_set=eval_set, verbose=True)

    model.fit(
        X_train,
        Y_train,
        eval_set=eval_set,
        verbose=True
    )

    print('Model was trained')

   

    # evaluation results
    results = model.evals_result()

    # save the model:
    model.save_model(f'XGboost_Regression/Models/{to_predict_feature}-XGBOOST_range-{y_range}_forward-{y_forward}_lookback-{look_back}_features-{num_of_feautures}.json')

    with open(f'XGboost_Regression/Histories/{to_predict_feature}-History_range-{y_range}_forward-{y_forward}_lookback-{look_back}_features-{num_of_feautures}.json', 'w') as f:
        json.dump(results, f)

run()
