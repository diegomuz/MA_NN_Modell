import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


from sklearn.preprocessing import MinMaxScaler


from xgboost import XGBRegressor



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

scaler = MinMaxScaler()

def split_scale_data(df,split_percentage):

    scaled_df = df.drop(['Datum'], axis = 1)
    scaled_df = pd.DataFrame(scaler.fit_transform(scaled_df), columns = scaled_df.columns)
    split_point = round(len(scaled_df)*split_percentage)
    train_data = scaled_df.iloc[:split_point]
    test_data = scaled_df.iloc[split_point:].reset_index(drop=True)

    


    return train_data, test_data



def create_training_data(df,split_percentage, to_predict_feature, timesteps, y_range):
    train_data , test_data = split_scale_data(df,split_percentage)

    print(train_data)
    print(test_data)

    # number of features is number of cols in train_data
    X_tr, Y_tr = [],[]



    for i in range(len(train_data)- timesteps - y_range + 1):
        s = []
        for d in range(timesteps):
            lis = train_data.iloc[i+d]
            
            for element in lis:
                s.append(element)
            
        X_tr.append(s)

        s = []
        for d in range(y_range):
            lis = train_data[to_predict_feature][i+timesteps+d+y_range-1]

            s.append(lis)
        
        Y_tr.append(s)

    X_tr = np.array(X_tr)
    Y_tr = np.array(Y_tr)


    X_te, Y_te = [],[]

    for i in range(len(test_data) - timesteps - y_range +1):
        s = []
        for d in range(timesteps):
            lis = test_data.iloc[i+d]

            for element in lis:
                s.append(element)

        X_te.append(s)

        s = []
        for d in range(y_range):
            lis = test_data[to_predict_feature][i+timesteps+d+y_range-1]
            s.append(lis)

      
        Y_te.append(s)

    X_te = np.array(X_te)
    Y_te = np.array(Y_te)

    return X_tr, Y_tr, X_te, Y_te


def inverse_scale(array):
    
    my_df = training_df.drop(['Datum'], axis = 1)
    d = {}
    for column in my_df.columns:
        
        col_val = 0
        d[column] = [col_val for i in range(len(array))]

    d[to_predict_feature] = array

    
    df = pd.DataFrame(d)

    

    df = pd.DataFrame(scaler.inverse_transform(df),columns=df.columns)

    print(df)

    actual_values = np.array(df[to_predict_feature])

    

    return(actual_values)





look_back = 24
y_range = 1



to_predict_feature = 'PM10'

X_train,Y_train,X_test,Y_test = create_training_data(training_df, 0.8, to_predict_feature, look_back, y_range)

model = XGBRegressor()
model.load_model(f'XGboost_Regression/Models/{to_predict_feature}-XGBOOST_range-{y_range}_lookback-{look_back}_features-{len(features)-1}.json')



predict_range = 48

actual_vals = []



for i in range(int(len(Y_test[:predict_range])/y_range)):
    for item in Y_test[:predict_range][(i)*y_range]:
        actual_vals.append(item)

actual_vals = np.array(actual_vals)

actual_vals = inverse_scale(actual_vals)

predicted_vals = []

Model_prediction = model.predict(X_test[:predict_range+y_range])

for i in range(int(len(Model_prediction)/y_range)-y_range):
    for item in [Model_prediction[(i+1)*y_range]]:
        predicted_vals.append(item)

predicted_vals = np.array(predicted_vals)

predicted_vals = inverse_scale(predicted_vals)

plt.plot(actual_vals, label = 'Actual Value', color = 'green')


plt.plot(predicted_vals, label = 'Prediction', color = 'blue' )

plt.title('Predicted vs Actual Values')

plt.legend(loc="upper left")

plt.savefig(f'Graphics/{to_predict_feature}-XGBOOST_Prediction_Fig_range-{y_range}_lookback-{look_back}_features-{len(features)-1}.pdf')  


plt.show()
