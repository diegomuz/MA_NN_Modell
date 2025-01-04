import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

ds = xr.open_dataset('Data_Preparation\Training_Datasets\Trainingsdaten_2021.nc')

d_training = ds.to_dataframe()

print(d_training)

numeric_d_training = d_training.drop('Datum', axis = 1)

print(numeric_d_training)

# scale the data

scaler = MinMaxScaler()

numeric_d_training = pd.DataFrame(scaler.fit_transform(numeric_d_training), columns = numeric_d_training.columns)

# perform KNN-Imputation

imputer = KNNImputer(n_neighbors=5)
numeric_d_training = pd.DataFrame(imputer.fit_transform(numeric_d_training),columns = numeric_d_training.columns)



# scale the data back


numeric_d_training = pd.DataFrame(scaler.inverse_transform(numeric_d_training), columns=numeric_d_training.columns)





# add the 'Datum' columns back

d_training = pd.concat([d_training['Datum'], numeric_d_training], axis = 1)

print(d_training)

print(f"Numer of NaN values per Column after cleanup:\n {d_training.isna().sum()}")