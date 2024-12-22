import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


import tensorflow as tf


from tf_keras.models import Sequential
from tf_keras.layers import  Dense
from tf_keras.layers import LSTM
from tf_keras.layers import Dropout

from tf_keras.callbacks import TensorBoard
from tf_keras.callbacks import EarlyStopping
from tf_keras.callbacks import ModelCheckpoint


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

#features = ['Datum','O3']

"""""

features = ['Datum','O3','T', 'Hr', 'p', 'RainDur', 'StrGlo', 'WD', 'WVv', 'WVs', 'Cont_T',
        'Cont_Hr', 'Cont_p', 'Cont_RainDur', 'Cont_WD', 'Cont_WVv', 'Cont_WVs']

"""

num_of_feautures = 0

# remove features that won't be used:

for feature in training_df.columns:
    if feature not in features:
        training_df.drop(feature, axis = 1, inplace=True)

print(training_df)

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


# put data into right format: resulting numpy array for x _train called X_tr should have this shape: (samples, timesteps, features)
# Y_tr should have this shape (samples, y_range)
# y_range: how many future hours we predict
# x_timesteps, how many hours back it looks at once to predict the next values


def create_training_data(df,split_percentage, to_predict_feature, timesteps, y_range):
    train_data , test_data = split_scale_data(df,split_percentage)

    print(train_data)
    print(test_data)

    # number of features is number of cols in train_data
    X_tr, Y_tr = [],[]



    for i in range(len(train_data)- timesteps - y_range + 1 -y_forward):
        X_tr.append(train_data.iloc[i: i+timesteps])
        Y_tr.append(train_data[to_predict_feature][i+timesteps +y_forward -1 : i+timesteps+y_range + y_forward -1])




    X_tr = np.array(X_tr)
    Y_tr = np.array(Y_tr)

    #alle + 12 wegmachen nachher

    X_te, Y_te = [],[]

    for i in range(len(test_data) - timesteps - y_range +1 -y_forward):
        X_te.append(test_data.iloc[i: i+timesteps])
        Y_te.append(test_data[to_predict_feature][i+timesteps + y_forward -1 : i+timesteps+y_range + y_forward -1])

    X_te = np.array(X_te)
    Y_te = np.array(Y_te)

    print(X_te[:14])
    print(Y_te[:14])

    return X_tr, Y_tr, X_te, Y_te


y_forward = 24


def run():

    # create Trraining and Test Datasets

    model_type = 3


    look_back = 12
    y_range = 1

    LSTM_l1_dimension = 32
    LSTM_l2_dimension = 32
    LSTM_l3_dimension = 32

    batchsize = 32
    epochs = 50

    to_predict_feature = 'O3'

    X_train,Y_train,X_test,Y_test = create_training_data(training_df, 0.8, to_predict_feature, look_back, y_range)

    print(X_train.shape)

    # Build the model:

    model = Sequential()
    if model_type == 1:
        model.add(LSTM(LSTM_l1_dimension, input_shape = (look_back, num_of_feautures), return_sequences=False) )

    if model_type == 2:
        model.add(LSTM(LSTM_l1_dimension, input_shape = (look_back, num_of_feautures), return_sequences=True))
        model.add(LSTM(LSTM_l2_dimension, return_sequences=False))

    if model_type == 3:
        model.add(LSTM(LSTM_l1_dimension, input_shape = (look_back, num_of_feautures), return_sequences=True))
        #model.add(Dropout(0.2))
        model.add(LSTM(LSTM_l2_dimension, return_sequences=True))
        #model.add(Dropout(0.2))
        model.add(LSTM(LSTM_l2_dimension, return_sequences=False))
        #model.add(Dropout(0.2))
        
    model.add(Dense(y_range))

    model.compile(loss = 'MSE', optimizer='adam')

    print(model.summary())


    # Prepare Training visualization:

        # Tensorboard set up: Code from: https://www.tensorflow.org/tensorboard/get_started

    

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


    # add callbacks and early stopping:

    if model_type == 1:

        mc = ModelCheckpoint(f'LSTM_Model/Models/{to_predict_feature}-Model(dim-{LSTM_l1_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).keras', 
                             monitor='val_loss', mode = 'min', verbose = 1,  save_best_only=True)

    if model_type == 2:

        mc = ModelCheckpoint(f'LSTM_Model/Models/{to_predict_feature}-Model_Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).keras',
                             monitor = 'val_loss', mode = 'min', verbose=1, save_best_only=True)
        
    if model_type == 3:

        mc = ModelCheckpoint(f'LSTM_Model/Models/{to_predict_feature}-Model_Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_dim3-{LSTM_l3_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).keras',
                             monitor = 'val_loss', mode = 'min', verbose=1, save_best_only=True)

    es = EarlyStopping(monitor = 'val_loss', patience = 4,mode = 'min', verbose = 1)

    


    # fit the model, inspired by this study:

    history = model.fit(X_train, 
            Y_train, 
            epochs=epochs,
            batch_size = batchsize, 
            validation_data=(X_test, Y_test), 
            callbacks=[tensorboard_callback, es, mc])


    # save the model

    if model_type == 1:
        
        with open(f'LSTM_Model/Histories/{to_predict_feature}-History(dim-{LSTM_l1_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).json', 'w') as f:
            json.dump(history.history, f)

        #model.save(f'LSTM_Model/Models/{to_predict_feature}-Model(dim-{LSTM_l1_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_epochs-{epochs}_features-{len(features)-1}).keras')

    if model_type == 2:

        with open(f'LSTM_Model/Histories/{to_predict_feature}-History-Type{model_type}(dim-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).json', 'w') as f:
            json.dump(history.history, f)

        #model.save(f'LSTM_Model/Models/{to_predict_feature}-Model_Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_epochs-{epochs}_features-{len(features)-1}).keras')


    if model_type == 3:

        with open(f'LSTM_Model/Histories/{to_predict_feature}-History-Type{model_type}(dim-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_dim3-{LSTM_l3_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).json', 'w') as f:
            json.dump(history.history, f)


    print('Model was trained')

#X_train,Y_train,X_test,Y_test = create_training_data(training_df, 0.8, 'O3', 1, 1)

run()









