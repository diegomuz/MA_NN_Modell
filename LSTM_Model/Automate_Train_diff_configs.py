# import what is needed

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


import tensorflow as tf


from tf_keras.models import Sequential
from tf_keras.layers import  Dense
from tf_keras.layers import LSTM


from tf_keras.callbacks import TensorBoard

from tf_keras.callbacks import EarlyStopping
from tf_keras.callbacks import ModelCheckpoint
import datetime

import json


from sklearn.preprocessing import MinMaxScaler

from Model_Training import split_scale_data

from Model_Training import create_training_data




model_configs = [{'Type':'1','to_predict_feature':'O3', 'LSTM_l1_dimension':10, 'look_back':12, 'features': ['Datum','O3'], 'batchsize':32,  }
                 
                 
                 
                 
                 
                 
                 ]



year_list = [2020,2021,2022,2023]





# create Trraining and Test Datasets

def prepare_data(features):
    
    training_df = pd.DataFrame()

    for year in year_list: 
        df = xr.open_dataset(f"Data_Preparation/Training_Datasets/Trainingsdaten_{year}.nc")
        df = df.to_dataframe()
        training_df = pd.concat([training_df, df], axis=0, ignore_index=True)


    # remove features that won't be used:

    for feature in training_df.columns:
        if feature not in features:
            training_df.drop(feature, axis = 1, inplace=True)

    print(training_df)

    return(training_df)



def run(model_type, look_back, y_range, LSTM_l1_dimension,LSTM_l2_dimension,batchsize,epochs, to_predict_feature, features)
    model_type = model_type

    look_back = look_back
    y_range = y_range

    LSTM_l1_dimension = LSTM_l1_dimension
    LSTM_l2_dimension = LSTM_l2_dimension

    batchsize = batchsize
    epochs = epochs

    to_predict_feature = to_predict_feature

    X_train,Y_train,X_test,Y_test = create_training_data(prepare_data(features), 0.8, to_predict_feature, look_back, y_range)

    print(X_train.shape)

    # Build the model:

    model = Sequential()
    if model_type == 1:
        model.add(LSTM(LSTM_l1_dimension, input_shape = (look_back, len(features)-1), return_sequences=False))

    if model_type == 2:
        model.add(LSTM(LSTM_l1_dimension, input_shape = (look_back, len(features)-1), return_sequences=True))
        model.add(LSTM(LSTM_l2_dimension, return_sequences=False))
        
    model.add(Dense(y_range))

    model.compile(loss = 'MSE', optimizer='adam')

    print(model.summary())


    # Prepare Training visualization:

        # Tensorboard set up: Code from: https://www.tensorflow.org/tensorboard/get_started

    

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)



    # add callbacks and early stopping:

    if model_type == 1:

        mc = ModelCheckpoint(f'LSTM_Model/Models/{to_predict_feature}-Model(dim-{LSTM_l1_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_epochs-{epochs}_features-{len(features)-1}).keras', 
                             monitor='val_loss', mode = 'min', verbose = 1,  save_best_only=True)

    if model_type == 2:

        mc = ModelCheckpoint(f'LSTM_Model/Models/{to_predict_feature}-Model_Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_epochs-{epochs}_features-{len(features)-1}).keras',
                             monitor = 'val_loss', mode = 'min', verbose=1, save_best_only=True)

    es = EarlyStopping(monitor = 'val_loss',patience='3', mode = 'min', verbose = 1)

    # fit the model, inspired by this study:

    history = model.fit(X_train, 
            Y_train, 
            epochs=epochs,
            batch_size = batchsize, 
            validation_data=(X_test, Y_test), 
            callbacks=[tensorboard_callback, es, mc])



    # save the model

    if model_type == 1:
        
        with open(f'LSTM_Model/Histories/{to_predict_feature}-History(dim-{LSTM_l1_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_epochs-{epochs}_features-{len(features)-1}).json', 'w') as f:
            json.dump(history.history, f)

        #model.save(f'LSTM_Model/Models/{to_predict_feature}-Model(dim-{LSTM_l1_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_epochs-{epochs}_features-{len(features)-1}).keras')

    if model_type == 2:

        with open(f'LSTM_Model/Histories/{to_predict_feature}-History-Type{model_type}(dim-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_epochs-{epochs}_features-{len(features)-1}).json', 'w') as f:
            json.dump(history.history, f)

        #model.save(f'LSTM_Model/Models/{to_predict_feature}-Model_Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_epochs-{epochs}_features-{len(features)-1}).keras')

    print('Model was trained')



