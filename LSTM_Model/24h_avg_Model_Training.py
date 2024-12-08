import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


import tensorflow as tf


from tf_keras.models import Sequential
from tf_keras.layers import  Dense
from tf_keras.layers import LSTM

from tf_keras.callbacks import TensorBoard
import datetime

import json


from sklearn.preprocessing import MinMaxScaler



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

#features = ['Datum','PM10']

# remove features that won't be used:

for feature in training_df.columns:
    if feature not in features:
        training_df.drop(feature, axis = 1, inplace=True)

print(training_df)

# split, scale data

scaler = MinMaxScaler()

def split_scale_data(df,split_percentage):

    scaled_df = df.drop(['Datum'], axis = 1)
    scaled_df = pd.DataFrame(scaler.fit_transform(scaled_df), columns = scaled_df.columns)
    split_point = round(len(scaled_df)*split_percentage)
    train_data = scaled_df.iloc[:split_point]
    test_data = scaled_df.iloc[split_point:].reset_index(drop=True)

    


    return train_data, test_data


# put data into right format: resulting numpy array for x _train called X_tr should have this shape: (samples, timesteps, features)
# Y_tr should have this shape (samples, y_range)
# y_range: how many future hours we predict
# x_timesteps, how many hours back it looks at once to predict the next values

# Predictions will be made for an average value of the next 24 hours

def create_training_data(df,split_percentage, to_predict_feature, timesteps, y_range):
    train_data , test_data = split_scale_data(df,split_percentage)

    print(train_data)
    print(test_data)

    # number of features is number of cols in train_data
    X_tr, Y_tr = [],[]



    for i in range(len(train_data)- timesteps - y_range + 1):
        X_tr.append(train_data.iloc[i: i+timesteps])
        s = train_data[to_predict_feature][i+timesteps : i+timesteps+24*y_range]
        l = []
        for d in range(y_range):
            l.append(sum(s[d*24:d*24+24])/24)
        Y_tr.append(l)
        

    X_tr = np.array(X_tr)
    Y_tr = np.array(Y_tr)


    X_te, Y_te = [],[]

    for i in range(len(test_data) - timesteps - y_range +1):
        X_te.append(test_data.iloc[i: i+timesteps])
        s = test_data[to_predict_feature][i+timesteps : i+timesteps+24*y_range]
        l = []
        for d in range(y_range):
            l.append(sum(s[d*24:d*24+24])/24)
        Y_te.append(l)

    X_te = np.array(X_te)
    Y_te = np.array(Y_te)

    return X_tr, Y_tr, X_te, Y_te





def run():

    # create Trraining and Test Datasets

    model_type = 2

    look_back = 24
    y_range = 1

    LSTM_l1_dimension = 15
    LSTM_l2_dimension = 15

    batchsize = 32
    epochs = 20

    to_predict_feature = 'PM10'

    X_train,Y_train,X_test,Y_test = create_training_data(training_df, 0.8, to_predict_feature, look_back, y_range)

    print(X_train.shape)

    # Build the model:

    model = Sequential()
    if model_type == 1:
        model.add(LSTM(LSTM_l1_dimension, input_shape = (look_back, len(features)-1), return_sequences=False))
    else:
        model.add(LSTM(LSTM_l1_dimension, input_shape = (look_back, len(features)-1), return_sequences=True))

    if model_type == 2:
        model.add(LSTM(LSTM_l2_dimension, return_sequences=False))
        
    model.add(Dense(y_range))

    model.compile(loss = 'MSE', optimizer='adam')

    print(model.summary())


    # Prepare Training visualization:

        # Tensorboard set up: Code from: https://www.tensorflow.org/tensorboard/get_started

    

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)




    # fit the model, inspired by this study:

    history = model.fit(X_train, 
            Y_train, 
            epochs=epochs,
            batch_size = batchsize, 
            validation_data=(X_test, Y_test), 
            callbacks=[tensorboard_callback])



    # save the model

    if model_type == 1:
        
        with open(f'LSTM_Model/24h_avg_Histories/{to_predict_feature}-History(dim-{LSTM_l1_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{len(features)-1}).json', 'w') as f:
            json.dump(history.history, f)

        model.save(f'LSTM_Model/24h_avg_Models/{to_predict_feature}-Model(dim-{LSTM_l1_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{len(features)-1}).keras')

    if model_type == 2:

        with open(f'LSTM_Model/24h_avg_Histories/{to_predict_feature}-History-Type{model_type}(dim-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{len(features)-1}).json', 'w') as f:
            json.dump(history.history, f)

        model.save(f'LSTM_Model/24h_avg_Models/{to_predict_feature}-Model_Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{len(features)-1}).keras')


run()












