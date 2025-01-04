import pandas as pd
import numpy as np
import xarray as xr
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






a,b,c,d = create_training_data(training_df, 0.8, 'O3', 24, 5)

print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)

