import json
import matplotlib.pyplot as plt

features = ['Datum', 'CO', 'SO2', 'NOx', 'NO', 'NO2', 'O3', 'PM10', 'PM2.5',
        'Cont_NOx', 'Cont_NO', 'Cont_NO2', 'Cont_O3', 'Cont_PM10', 'Cont_PM2.5',
        'T', 'Hr', 'p', 'RainDur', 'StrGlo', 'WD', 'WVv', 'WVs', 'Cont_T',
        'Cont_Hr', 'Cont_p', 'Cont_RainDur', 'Cont_WD', 'Cont_WVv', 'Cont_WVs']

#features = ['Datum','O3']

num_of_feautures = 35
model_type = 3

look_back = 12
y_range = 1
y_forward = 1
LSTM_l1_dimension = 64

LSTM_l2_dimension = 64

LSTM_l3_dimension = 64

batchsize = 32
epochs = 20

to_predict_feature = 'O3'

if model_type == 1:


    with open(f'LSTM_Model/Histories/{to_predict_feature}-History(dim-{LSTM_l1_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).json', 'r') as f:
        loaded_history = json.load(f)

if model_type == 2:
    with open(f'LSTM_Model/Histories/{to_predict_feature}-History-Type{model_type}(dim-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).json', 'r') as f:
        loaded_history = json.load(f)

if model_type == 3:
    with open(f'LSTM_Model/Histories/{to_predict_feature}-History-Type{model_type}(dim-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_dim3-{LSTM_l3_dimension}_range-{y_range}_forward-{y_forward}_batch-{batchsize}_lookback-{look_back}_features-{num_of_feautures}).json', 'r') as f:
        loaded_history = json.load(f)


plt.plot(loaded_history['loss'], label='Training Loss')
plt.plot(loaded_history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('linear')
plt.legend()

if model_type == 1:

    plt.savefig(f'Graphics/{to_predict_feature}-Trainprogress_Fig(dim-{LSTM_l1_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{len(features)-1}).pdf')
if model_type == 2:
    plt.savefig(f'Graphics/{to_predict_feature}-Trainprogress_Fig-Type-{model_type}(dim1-{LSTM_l1_dimension}_dim2-{LSTM_l2_dimension}_range-{y_range}_batch-{batchsize}_lookback-{look_back}_features-{len(features)-1}).pdf')




plt.show()

