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

fin_data = {
    f"{grouped_data['Standort_Parameter'][0]}" : grouped_data['Wert'][0],
    f"{grouped_data['Standort_Parameter'][1]}" : grouped_data['Wert'][1],
    f"{grouped_data['Standort_Parameter'][2]}" : grouped_data['Wert'][2],
    f"{grouped_data['Standort_Parameter'][3]}" : grouped_data['Wert'][3],
    f"{grouped_data['Standort_Parameter'][4]}" : grouped_data['Wert'][4],
    f"{grouped_data['Standort_Parameter'][5]}" : grouped_data['Wert'][5],
    f"{grouped_data['Standort_Parameter'][6]}" : grouped_data['Wert'][6],
    f"{grouped_data['Standort_Parameter'][7]}" : grouped_data['Wert'][7],
    f"{grouped_data['Standort_Parameter'][8]}" : grouped_data['Wert'][8],
    f"{grouped_data['Standort_Parameter'][9]}" : grouped_data['Wert'][9],
    f"{grouped_data['Standort_Parameter'][10]}" : grouped_data['Wert'][10],
    f"{grouped_data['Standort_Parameter'][11]}" : grouped_data['Wert'][11],
    f"{grouped_data['Standort_Parameter'][12]}" : grouped_data['Wert'][12],
    f"{grouped_data['Standort_Parameter'][13]}" : grouped_data['Wert'][13],
    f"{grouped_data['Standort_Parameter'][14]}" : grouped_data['Wert'][14],
    f"{grouped_data['Standort_Parameter'][15]}" : grouped_data['Wert'][15],
    f"{grouped_data['Standort_Parameter'][16]}" : grouped_data['Wert'][16],
    f"{grouped_data['Standort_Parameter'][17]}" : grouped_data['Wert'][17],
    f"{grouped_data['Standort_Parameter'][18]}" : grouped_data['Wert'][18],
    f"{grouped_data['Standort_Parameter'][19]}" : grouped_data['Wert'][19],
    f"{grouped_data['Standort_Parameter'][20]}" : grouped_data['Wert'][20],
    f"{grouped_data['Standort_Parameter'][21]}" : grouped_data['Wert'][21],
    f"{grouped_data['Standort_Parameter'][22]}" : grouped_data['Wert'][22],
    f"{grouped_data['Standort_Parameter'][23]}" : grouped_data['Wert'][23],
    f"{grouped_data['Standort_Parameter'][24]}" : grouped_data['Wert'][24],
    f"{grouped_data['Standort_Parameter'][25]}" : grouped_data['Wert'][25],
    f"{grouped_data['Standort_Parameter'][26]}" : grouped_data['Wert'][26],
    f"{grouped_data['Standort_Parameter'][27]}" : grouped_data['Wert'][27],
    f"{grouped_data['Standort_Parameter'][28]}" : grouped_data['Wert'][28],
    f"{grouped_data['Standort_Parameter'][29]}" : grouped_data['Wert'][29],
    f"{grouped_data['Standort_Parameter'][30]}" : grouped_data['Wert'][30],
    f"{grouped_data['Standort_Parameter'][31]}" : grouped_data['Wert'][31],
    f"{grouped_data['Standort_Parameter'][32]}" : grouped_data['Wert'][32],
    f"{grouped_data['Standort_Parameter'][33]}" : grouped_data['Wert'][33],
    f"{grouped_data['Standort_Parameter'][34]}" : grouped_data['Wert'][34],
    f"{grouped_data['Standort_Parameter'][35]}" : grouped_data['Wert'][35],
    f"{grouped_data['Standort_Parameter'][36]}" : grouped_data['Wert'][36],
    f"{grouped_data['Standort_Parameter'][37]}" : grouped_data['Wert'][37],
    f"{grouped_data['Standort_Parameter'][38]}" : grouped_data['Wert'][38],
    f"{grouped_data['Standort_Parameter'][39]}" : grouped_data['Wert'][39],
    f"{grouped_data['Standort_Parameter'][40]}" : grouped_data['Wert'][40],
    f"{grouped_data['Standort_Parameter'][41]}" : grouped_data['Wert'][41],
    f"{grouped_data['Standort_Parameter'][42]}" : grouped_data['Wert'][42],
    f"{grouped_data['Standort_Parameter'][43]}" : grouped_data['Wert'][43],
    f"{grouped_data['Standort_Parameter'][44]}" : grouped_data['Wert'][44],
    f"{grouped_data['Standort_Parameter'][45]}" : grouped_data['Wert'][45],
    f"{grouped_data['Standort_Parameter'][46]}" : grouped_data['Wert'][46],
    f"{grouped_data['Standort_Parameter'][47]}" : grouped_data['Wert'][47],
    f"{grouped_data['Standort_Parameter'][48]}" : grouped_data['Wert'][48]
    
    

}

data = pd.DataFrame(fin_data)

#print(s)

print("\n")

print(data.iloc[:,4:])

# creating the heatmap

correlation_matrix = data.corr(numeric_only=True, method='pearson')

selected_rows = [30,33,34,35,36,37,38]

selected_cols = [4,8,11,12,13,14,16,17,24,25,27,28,29,30,33,34,35,36,37,38,40,31,42,43,44,45,46,48,39,32,47]

correlation_matrix = correlation_matrix.iloc[selected_cols, selected_rows]

#print(correlation_matrix)

plt.figure(figsize=(15, 15))


heatmap = sns.heatmap(correlation_matrix, vmin = -1, vmax = 1, annot = True, fmt=".2f", xticklabels = data.columns[selected_rows])

plt.xticks(rotation = 20, fontsize = 8)
plt.yticks(fontsize = 8)

#heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)

plt.show()
