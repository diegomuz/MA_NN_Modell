#import tensorflow as tf
import tf_keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from statistics import mean

from sklearn.preprocessing import MinMaxScaler




year_list = [2020,2021,2022,2023]

# define what Features should be used for the model training

features = ['Datum', 'CO', 'SO2', 'NOx', 'NO', 'NO2', 'O3', 'PM10', 'PM2.5',
        'Cont_NOx', 'Cont_NO', 'Cont_NO2', 'Cont_O3', 'Cont_PM10', 'Cont_PM2.5',
        'T', 'Hr', 'p', 'RainDur', 'StrGlo', 'WD', 'WVv', 'WVs', 'Cont_T',
        'Cont_Hr', 'Cont_p', 'Cont_RainDur', 'Cont_WD', 'Cont_WVv', 'Cont_WVs']

features = ['Datum', 'O3']

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


#    print(training_df['O3'][28075:])
    return(training_df)

scaler = MinMaxScaler()

def split_scale_data(df,split_percentage):

    scaled_df = df.drop(['Datum'], axis = 1)
    scaled_df = pd.DataFrame(scaler.fit_transform(scaled_df), columns = scaled_df.columns)
    split_point = round(len(scaled_df)*split_percentage)
    train_data = scaled_df.iloc[:split_point]
    test_data = scaled_df.iloc[split_point:].reset_index(drop=True)

    
#    print(test_data['O3'][24:])

    return train_data, test_data

def create_training_data(df,split_percentage, to_predict_feature, timesteps, y_range):
    train_data , test_data = split_scale_data(df,split_percentage)

    print(train_data)
    print(test_data)

    # number of features is number of cols in train_data
    X_tr, Y_tr = [],[]

    for i in range(len(train_data)- timesteps - y_range + 1):
        X_tr.append(train_data.iloc[i: i+timesteps])
        Y_tr.append(train_data[to_predict_feature][i+timesteps : i+timesteps+y_range])

    X_tr = np.array(X_tr)
    Y_tr = np.array(Y_tr)


    X_te, Y_te = [],[]

    for i in range(len(test_data) - timesteps - y_range +1):
        X_te.append(test_data.iloc[i: i+timesteps])
        Y_te.append(test_data[to_predict_feature][i+timesteps : i+timesteps+y_range])

    X_te = np.array(X_te)
    Y_te = np.array(Y_te)

    return X_tr, Y_tr, X_te, Y_te

model_type = 1

look_back = 24
y_range = 1
LSTM_l1_dimension = 30

LSTM_l2_dimension = 15

batchsize = 32
epochs = 30

to_predict_feature = 'O3'

predict_range = 72

training_df = prepare_data()



X_train,Y_train,X_test,Y_test = create_training_data(training_df, 0.8, to_predict_feature, look_back, y_range)

# load the model:

if model_type == 1:

    model = tf_keras.models.load_model(f'LSTM_Model/Models/{to_predict_feature}-Model(dim-{LSTM_l1_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{len(features)-1}).keras')
else:
    model = tf_keras.models.load_model(f'LSTM_Model/Models/{to_predict_feature}-Model_Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{len(features)-1}).keras')


model.summary()






# compare model predictions with actual data:

def inverse_scale(array):
    
    my_df = training_df.drop(['Datum'], axis = 1)
    d = {}
    for column in my_df.columns:
        
        col_val = 0
        d[column] = [col_val for i in range(len(array))]

    d[to_predict_feature] = array

    
    df = pd.DataFrame(d)

    

    df = pd.DataFrame(scaler.inverse_transform(df),columns=df.columns)

    #print(df)

    actual_values = np.array(df[to_predict_feature])

    

    return(actual_values)


def calculate_rmse_accuracy(y_actual, y_predicted):
    rmse = np.sqrt(np.mean((y_actual-y_predicted)**2))
    return rmse




actual_vals = []

# change the following code, so that it makes sense for y_range > 1

delta = 100

shift = int((3 - look_back/12)*12) + delta

for i in range(int(predict_range/y_range)):
    for item in Y_test[:predict_range + shift][(i*y_range)+int(shift/y_range)]:
        actual_vals.append(item)

""""
for sub_list in Y_test[:predict_range]:
    for item in sub_list:
        actual_vals.append(item)

"""

actual_vals = np.array(actual_vals)
#actual_vals = actual_vals.reshape(-1,1)

#actual_vals = scaler.inverse_transform(actual_vals)

print(actual_vals)

actual_vals = inverse_scale(actual_vals)

#print(actual_vals)

print(actual_vals)




predicted_vals = []

Model_prediction = model.predict(X_test[:predict_range+y_range+shift])

for i in range(int(predict_range/y_range)):
    for item in Model_prediction[(i+1)*y_range + int(shift/y_range)]:
        predicted_vals.append(item)

""""
for sub_list in model.predict(X_test[:predict_range]):
    for item in sub_list:
        predicted_vals.append(item)
"""
predicted_vals = np.array(predicted_vals)

predicted_vals = inverse_scale(predicted_vals)

print(predicted_vals)





plt.plot(actual_vals, label = 'Actual Value', color = 'green')


plt.plot(predicted_vals, label = 'Prediction', color = 'blue' )

rmse = calculate_rmse_accuracy(actual_vals,predicted_vals)

# Add RMSE as a note to the top right of the plot
plt.text(0.95, 0.95, f"RMSE = {rmse:.2f}", 
         transform=plt.gca().transAxes, 
         fontsize=10, 
         verticalalignment='top', 
         horizontalalignment='right',
         bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

plt.title('Predicted vs Actual Values')

plt.legend(loc="upper left")


if model_type == 1:

    plt.savefig(f'Graphics/{to_predict_feature}-Prediction_Fig(dim-{LSTM_l1_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{len(features)-1}).pdf')
if model_type == 2:
    plt.savefig(f'Graphics/{to_predict_feature}-Prediction_Fig-Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{len(features)-1}).pdf')





plt.show()

