

import tensorflow
import tf_keras
from tf_keras.models import load_model
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

"""
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


def create_training_data(df, split_percentage: list, to_predict_feature, timesteps, y_range):
    train_data, test_data = split_scale_data(df, split_percentage[0])
    
    # Further split test_data into validation and test
    val_split = (split_percentage[1]-split_percentage[0])/(1-split_percentage[0])
    split_point = round(len(test_data) * val_split)
    val_data = test_data.iloc[:split_point].reset_index(drop=True)
    test_data = test_data.iloc[split_point:].reset_index(drop=True)

    print(train_data)
    print(val_data)
    print(test_data)

    # number of features is number of cols in train_data
    X_tr, Y_tr = [],[]

    for i in range(len(train_data) - timesteps - y_range + 1 - y_forward):
        X_tr.append(train_data.iloc[i: i+timesteps])
        Y_tr.append(train_data[to_predict_feature][i+timesteps + y_forward - 1 : i+timesteps+y_range + y_forward - 1])

    X_tr = np.array(X_tr)
    Y_tr = np.array(Y_tr)

    # Validation data
    X_val, Y_val = [], []

    for i in range(len(val_data) - timesteps - y_range + 1 - y_forward):
        X_val.append(val_data.iloc[i: i+timesteps])
        Y_val.append(val_data[to_predict_feature][i+timesteps + y_forward - 1 : i+timesteps+y_range + y_forward - 1])

    X_val = np.array(X_val)
    Y_val = np.array(Y_val)

    # Test data
    X_te, Y_te = [], []

    for i in range(len(test_data) - timesteps - y_range + 1 - y_forward):
        X_te.append(test_data.iloc[i: i+timesteps])
        Y_te.append(test_data[to_predict_feature][i+timesteps + y_forward - 1 : i+timesteps+y_range + y_forward - 1])

    X_te = np.array(X_te)
    Y_te = np.array(Y_te)

    #print(X_val[:14])
    #print(Y_val[:14])
    #print(X_te[:14])
    #print(Y_te[:14])

    return X_tr, Y_tr, X_te, Y_te, X_val, Y_val


model_type = 1


look_back = 36
y_range = 1
y_forward = 1
LSTM_l1_dimension = 32
LSTM_l2_dimension = 128
LSTM_l3_dimension = 128
LSTM_l4_dimension = 64

batchsize = 32
epochs = 30

to_predict_feature = 'O3'



delta = 0

use_baseline = True


training_df = prepare_data()


X_train,Y_train,X_test,Y_test, X_val, Y_val = create_training_data(training_df, [0.7,0.9], to_predict_feature, look_back, y_range)


predict_range = 168


# load the model:

if model_type == 1:

    model = load_model(f'LSTM_Model/Models/{to_predict_feature}-Model(dim-{LSTM_l1_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).keras', compile = False)
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

"""
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


baseline_vals = None
baseline_metrics_text = ''
if use_baseline:
    # Baseline prediction: use previous hour ozone concentration from the last timestep of each input window.
    # This produces a "naive" baseline that does not use the LSTM model.
    o3_col_index = list(training_df.drop(['Datum'], axis=1).columns).index(to_predict_feature)

    baseline_scaled = X_test[shift: shift + int(predict_range/y_range), -1, o3_col_index]
    baseline_vals = inverse_scale(baseline_scaled)

    baseline_rmse = np.sqrt(mean_squared_error(actual_vals, baseline_vals))
    baseline_mae = mean_absolute_error(actual_vals, baseline_vals)
    baseline_corr = np.corrcoef(actual_vals, baseline_vals)[0,1]
    baseline_metrics_text = f"Baseline (prev hour): RMSE = {baseline_rmse:.2f}, MAE = {baseline_mae:.2f}, Corr = {baseline_corr:.2f}"


# do bootstrapping to evaluate range of error metrics:
# 24 hour samples are gonna be evaluated and then the values at 2.5% and 97.5% of the st disrtibution are gonna be taken



def block_eval_surety_metrics(y_true, y_pred, block_size = 24, reps = 4000):
    n = len(y_true)
    n_blocks = int(np.ceil(n / block_size))

    rmse_list = []
    mae_list = []
    corr_list = []

    for i in range(reps):

        sampled_indices = []

        for l in range(n_blocks):
            start = np.random.randint(0, n-block_size)

            block_idx = list(range(start, start + block_size))

            sampled_indices.extend(block_idx)
        
        sampled_indices = sampled_indices[:n]

        y_t = y_true[sampled_indices]
        y_p = y_pred[sampled_indices]

        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        corr = np.corrcoef(y_t, y_p)[0,1]

        rmse_list.append(rmse)
        mae_list.append(mae)
        corr_list.append(corr)

    rmse_arr = np.array(rmse_list)
    mae_arr = np.array(mae_list)
    corr_arr = np.array(corr_list)

    results = {
        "rmse_mean": rmse_arr.mean(),
        "rmse_std": rmse_arr.std(),
        "rmse_ci_low": np.percentile(rmse_arr, 2.5),
        "rmse_ci_high": np.percentile(rmse_arr, 97.5),

        "mae_mean": mae_arr.mean(),
        "mae_std": mae_arr.std(),

        "corr_mean": corr_arr.mean(),
        "corr_std": corr_arr.std()
    }

    return results

lstm_bootstrap = block_eval_surety_metrics(actual_vals, predicted_vals)

#print(results)





plt.figure(figsize=(10,5))

plt.plot(actual_vals, label='Echte Werte', color='green')
plt.plot(predicted_vals, label='LSTM Vorhersage', color='blue')

if use_baseline:
    plt.plot(baseline_vals, label='Baseline (Vorherige Stunde)', color='orange', linestyle='--')
    baseline_bootstrap = block_eval_surety_metrics(actual_vals, baseline_vals)


# ---- Metrics for LSTM ----
rmse = np.sqrt(mean_squared_error(actual_vals,predicted_vals))
mae = mean_absolute_error(actual_vals,predicted_vals)
correlation = np.corrcoef(actual_vals,predicted_vals)[0,1]

metrics_text = (
    "LSTM Modell\n"
    f"RMSE: {rmse:.2f}\n"
    f"MAE: {mae:.2f}\n"
    f"Korrelation: {correlation:.2f}"
)

# ---- Metrics for Baseline ----
if use_baseline:
    baseline_rmse = np.sqrt(mean_squared_error(actual_vals, baseline_vals))
    baseline_mae = mean_absolute_error(actual_vals, baseline_vals)
    baseline_corr = np.corrcoef(actual_vals, baseline_vals)[0,1]

    baseline_text = (
        "\n\nBaseline (Vorherige Stunde)\n"
        f"RMSE: {baseline_rmse:.2f}\n"
        f"MAE: {baseline_mae:.2f}\n"
        f"Korrelation: {baseline_corr:.2f}"
    )

    metrics_text += baseline_text


# ---- Metrics Box ----
plt.text(
    0.98, 0.75,
    metrics_text,
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9)
)

plt.legend(loc="upper left")

plt.title(f'{to_predict_feature} - Konzentration')
plt.ylabel('Konzentration in µg/m³')
plt.xlabel('Stunden')

plt.tight_layout()




if model_type == 1:

    plt.savefig(f'Graphics/{to_predict_feature}-Prediction_Fig(dim-{LSTM_l1_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}_with_baseline).pdf')
if model_type == 2:
    plt.savefig(f'Graphics/{to_predict_feature}-Prediction_Fig-Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}_with_baseline).pdf')


if model_type == 3:
    plt.savefig(f'Graphics/{to_predict_feature}-Prediction_Fig-Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_dim3-{LSTM_l3_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}_with_baseline).pdf')

if model_type == 4:
    plt.savefig(f'Graphics/{to_predict_feature}-Prediction_Fig-Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_dim3-{LSTM_l3_dimension}_dim4-{LSTM_l4_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}_with_baseline).pdf')

print('created graph')

plt.show()

print("LSTM Bootstrap Metrics:")
print(f"RMSE: {lstm_bootstrap['rmse_mean']:.2f} ± {lstm_bootstrap['rmse_std']:.2f}")
print(f"MAE: {lstm_bootstrap['mae_mean']:.2f} ± {lstm_bootstrap['mae_std']:.2f}")
print(f"Corr: {lstm_bootstrap['corr_mean']:.2f} ± {lstm_bootstrap['corr_std']:.2f}")
print(f"RMSE 95% CI: [{lstm_bootstrap['rmse_ci_low']:.2f}, {lstm_bootstrap['rmse_ci_high']:.2f}]")

if use_baseline:
    print("\nBaseline Bootstrap Metrics:")
    print(f"RMSE: {baseline_bootstrap['rmse_mean']:.2f} ± {baseline_bootstrap['rmse_std']:.2f}")
    print(f"MAE: {baseline_bootstrap['mae_mean']:.2f} ± {baseline_bootstrap['mae_std']:.2f}")
    print(f"Corr: {baseline_bootstrap['corr_mean']:.2f} ± {baseline_bootstrap['corr_std']:.2f}")
    print(f"RMSE 95% CI: [{baseline_bootstrap['rmse_ci_low']:.2f}, {baseline_bootstrap['rmse_ci_high']:.2f}]")