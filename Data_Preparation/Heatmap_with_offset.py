"""
Benutzte Quellen/Tutorials: 

https://www.geeksforgeeks.org/how-to-create-a-seaborn-correlation-heatmap-in-python/
https://blog.quantinsti.com/creating-heatmap-using-python-seaborn/

"""




import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
year = 2023
offset = 3

d1 = pd.read_csv(f"Data_Preparation/Air_Datasets/ugz_ogd_air_h1_{year}.csv")
d2 = pd.read_csv(f"Data_Preparation/Meteo_Datasets/ugz_ogd_meteo_h1_{year}.csv")
d3 = pd.read_csv(f"Data_Preparation/Traffic Datasets/ugz_ogd_traffic_h1_2023.csv")


# fit d3 to the other data:

d3.drop(['Datum','Richtung','Spur','Klasse.ID','Intervall','Status'], axis = 1, inplace=True)

s= d3['Standort'] + " " + d3['Klasse.Text']
d3['Standort_Parameter'] = s

d3.drop(['Standort','Klasse.Text'], axis = 1, inplace=True)
d3.rename(columns = {'Anzahl':'Wert'}, inplace = True)
""""
s = []
for i in range(len(d3)):
    if np.isnan(d3['Wert'][i]):
        s.append(0)
    else:
        s.append(d3['Wert'][i])

d3['Wert'] = s
"""
print(d3)

s = []
b = []
idx = 0
for i in range(int(len(d3)/6)):
    for q in range(3):
        b = []
        if np.isnan(d3['Wert'][idx]) == False:
            b.append(d3['Wert'][idx])
        if np.isnan(d3['Wert'][idx +3]) == False:
            b.append(d3['Wert'][idx +3])
        
        if len(b) == 0:
            s.append(d3['Wert'][idx])
        else:
            s.append(sum(b))
                   
        idx += 1
    idx += 3

l = d3['Standort_Parameter'][:int(len(d3)/2)]

k = pd.DataFrame()
k['Wert'] = s
k['Standort_Parameter'] = l

d3 = k

print(d3)





# concat d1,d2
data = pd.concat([d1,d2], ignore_index= True)



# remove unnecessary columns

data.drop(['Datum', 'Einheit', 'Intervall', 'Status'], axis = 1, inplace= True)



s = data['Standort'] + " " + data['Parameter']

data['Standort_Parameter'] = s

data.drop(['Standort','Parameter'], axis = 1, inplace = True)



# concat d3, data
data = pd.concat([data,d3], ignore_index=True)


# switch order of cols:

l = pd.DataFrame()
l['Standort_Parameter'] = data['Standort_Parameter']

l['Wert'] = data['Wert']

data = l



# replace NaN values with 0

print(data)
""""
s = []
for i in range(len(data)):
    if np.isnan(data["Wert"][i]):
        s.append(0)
    else:
        s.append(data['Wert'][i])

data['Wert'] = s

"""

c = 0

for i in range(len(data)):
    if np.isnan(data['Wert'][i]):
        c+= 1

print(c)

# Den Code gerade unterhalb (groupby) habe ich nicht alleine geschrieben. Ich habe nachgeschaut wie man das macht
grouped_data = data.groupby('Standort_Parameter')['Wert'].apply(list).reset_index()

#grouped_data = grouped_data.explode('Wert').reset_index(drop=True)




print(grouped_data)

selected_rows = [36]
selected_cols = [30,34,6,19, 18, 35,7,20,41, 37, 9, 22, 38, 10, 23, 40, 43, 31,42, 44, 45, 48,  47, 32, 39]

fin_data = {}

for i in range(len(grouped_data)):

    if i in selected_rows:
        fin_data[f"{grouped_data['Standort_Parameter'][i]}"] = grouped_data['Wert'][i][offset:]
    else: 
         fin_data[f"{grouped_data['Standort_Parameter'][i]}"] = grouped_data['Wert'][i][:len(grouped_data['Wert'][i])-offset]





data = pd.DataFrame(fin_data)

#print(s)

print("\n")



# creating the heatmap

correlation_matrix = data.corr(numeric_only=True, method='pearson')



#selected_rows = [30,33,34,35,36,37,38]

#selected_cols = [4,8,11,12,13,14,16,17,24,25,27,28,29,30,33,34,35,36,37,38,40,31,42,43,44,45,46,48,39,32,47]



# Select the features between which the correlation should be calculated.
# grouped data contains the indexes of each feature

correlation_matrix = correlation_matrix.iloc[selected_cols, selected_rows]

#print(correlation_matrix)

plt.figure(figsize=(10, 7))


heatmap = sns.heatmap(correlation_matrix, vmin = -1, vmax = 1, annot = True, fmt=".2f", xticklabels = data.columns[selected_rows])

plt.xticks(rotation = 20, fontsize = 8)
plt.yticks(fontsize = 8)

#heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)

plt.show()
