#import tensorflow as tf
import tf_keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from statistics import mean

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error





year_list = [2020,2021,2022,2023]

# define what Features should be used for the model training



features = ['Datum', 'CO', 'SO2', 'NOx', 'NO', 'NO2', 'O3', 'PM10', 'PM2.5',
        'Cont_NOx', 'Cont_NO', 'Cont_NO2', 'Cont_O3', 'Cont_PM10', 'Cont_PM2.5',
        'T', 'Hr', 'p', 'RainDur', 'StrGlo', 'WD', 'WVv', 'WVs', 'Cont_T',
        'Cont_Hr', 'Cont_p', 'Cont_RainDur', 'Cont_WD', 'Cont_WVv', 'Cont_WVs']

""""
features = ['Datum', 'CO', 'NOx', 'NO', 'NO2', 'O3',
       'Cont_NOx', 'Cont_NO', 'Cont_NO2', 'Cont_O3',
       'T', 'Hr',  'StrGlo',  'WVv', 'WVs', 'Cont_T',
       'Cont_Hr', 'Cont_WVv', 'Cont_WVs']

#features = ['Datum', 'CO', 'NOx', 'NO', 'NO2', 'O3',
 #      'T', 'Hr',  'StrGlo',  'WVv', 'WVs']
"""
#features = ['Datum', 'O3']

num_of_feautures = 0


def prepare_data():
    #prepare data

    training_df = pd.DataFrame()

   

    for year in year_list: 
        df = xr.open_dataset(f"Data_Preparation/Training_Datasets/Trainingsdaten_{year}.nc")
        df = df.to_dataframe()
        training_df = pd.concat([training_df, df], axis=0, ignore_index=True)

    print('Datum am Ende:')
    print(training_df['Datum'][int(0.8*len(training_df))])


    # remove features that won't be used:

    for feature in training_df.columns:
        if feature not in features:
            training_df.drop(feature, axis = 1, inplace=True)

    # encode month&day in the year and hour of the day:

    Datum = pd.to_datetime(training_df['Datum'], format='%Y-%m-%dT%H:%M%z')

    day_of_year = Datum.dt.day_of_year
    hour_of_day = Datum.dt.hour
    month_of_year = Datum.dt.month

    # hour encoding:

    sin_h = [np.sin(2*np.pi*h/24) for h in hour_of_day]
    cos_h = [np.cos(2*np.pi*h/24) for h in hour_of_day]

    #day encoding 

    sin_d = [np.sin(2*np.pi*d/365) for d in day_of_year]
    cos_d = [np.cos(2*np.pi*d/365) for d in day_of_year]

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

    print(sin_h[:24])

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
        X_tr.append(train_data.iloc[i: i+timesteps])
        Y_tr.append(train_data[to_predict_feature][i+timesteps +y_forward: i+timesteps+y_range + y_forward])

    X_tr = np.array(X_tr)
    Y_tr = np.array(Y_tr)


    X_te, Y_te = [],[]

    for i in range(len(test_data) - timesteps - y_range +1 -y_forward):
        X_te.append(test_data.iloc[i: i+timesteps])
        Y_te.append(test_data[to_predict_feature][i+timesteps + y_forward: i+timesteps + y_range + y_forward])

    X_te = np.array(X_te)
    Y_te = np.array(Y_te)

    #print('Xtest and Ytest')
    #print(X_te[0])
    #print(Y_te[0])

    return X_tr, Y_tr, X_te, Y_te

model_type = 1


look_back = 12
y_range = 1
y_forward = 24
LSTM_l1_dimension = 32
LSTM_l2_dimension = 32
LSTM_l3_dimension = 32
LSTM_l4_dimension = 64

batchsize = 32
epochs = 30

to_predict_feature = 'O3'

predict_range = 168
delta = 3500

training_df = prepare_data()



X_train,Y_train,X_test,Y_test = create_training_data(training_df, 0.8, to_predict_feature, look_back, y_range)

# load the model:

if model_type == 1:

    model = tf_keras.models.load_model(f'LSTM_Model/Models/{to_predict_feature}-Model(dim-{LSTM_l1_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).keras')
if model_type == 2:
    model = tf_keras.models.load_model(f'LSTM_Model/Models/{to_predict_feature}-Model_Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).keras')
if model_type == 3:
    model = tf_keras.models.load_model(f'LSTM_Model/Models/{to_predict_feature}-Model_Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_dim3-{LSTM_l3_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).keras')
if model_type == 4:
    model = tf_keras.models.load_model(f'LSTM_Model/Models/{to_predict_feature}-Model_Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_dim3-{LSTM_l3_dimension}_dim4-{LSTM_l4_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).keras')

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






actual_vals = []

# change the following code, so that it makes sense for y_range > 1



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

#print(actual_vals)

actual_vals = inverse_scale(actual_vals)

#print(actual_vals)

#print(actual_vals)




predicted_vals = []

Model_prediction = model.predict(X_test[:predict_range+y_range+shift])




for i in range(int(predict_range/y_range)):
    for item in Model_prediction[(i+0)*y_range + int(shift/y_range)]:
        predicted_vals.append(item)


predicted_vals = np.array(predicted_vals)

predicted_vals = inverse_scale(predicted_vals)


print(actual_vals)
print(predicted_vals)








plt.plot(actual_vals, label = 'Echte Werte', color = 'green')

#plt.plot(actual_vals,  color = 'red')

plt.plot(predicted_vals, label = 'Vorhersagen', color = 'blue' )

#x_vals = [x[0][0] for x in X_test]
#x_vals = inverse_scale(np.array(x_vals[y_range+int(shift/y_range):y_range+int(shift/y_range) + predict_range]))

#plt.plot(x_vals, label = 'x_vals', color = 'red')




rmse = np.sqrt(mean_squared_error(actual_vals,predicted_vals))
mae = mean_absolute_error(actual_vals,predicted_vals)
correlation = np.corrcoef(actual_vals,predicted_vals)[0,1]

metrics_text = f"RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nKorrelation = {correlation:.2f}"

print(metrics_text)

# Add RMSE as a note to the top right of the plot



plt.text(0.95, 0.95, metrics_text, 
         transform=plt.gca().transAxes, 
         fontsize=8, 
         verticalalignment='top', 
         horizontalalignment='right',
         bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))




#plt.yticks(range(10,100,10))



plt.legend(loc="upper left")


plt.title(f'{to_predict_feature} - Konzentration')

plt.ylabel('Konzentration in µg/m3')
plt.xlabel('Stunden')




if model_type == 1:

    plt.savefig(f'Graphics/{to_predict_feature}-Prediction_Fig(dim-{LSTM_l1_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).pdf')
if model_type == 2:
    plt.savefig(f'Graphics/{to_predict_feature}-Prediction_Fig-Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).pdf')


if model_type == 3:
    plt.savefig(f'Graphics/{to_predict_feature}-Prediction_Fig-Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_dim3-{LSTM_l3_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).pdf')

if model_type == 4:
    plt.savefig(f'Graphics/{to_predict_feature}-Prediction_Fig-Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_dim3-{LSTM_l3_dimension}_dim4-{LSTM_l4_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).pdf')



plt.show()


