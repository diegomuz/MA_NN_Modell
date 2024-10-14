import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

All_data = pd.read_csv("Data_Preparation/Air_Datasets/ugz_ogd_air_h1_2023.csv")

Luftwerte = All_data["Wert"]

# Dieser Code soll Luftmessungen von verschiedenen Stationen vergleichen, zum schauen, ob ich Durschnittwerte von den Stationen
#benutzen kann, falls sie nicht zu fest abweichen, oder mich auf eine Station fokussiere

Stampfenbachstrasse = []

idx = 0
for i in range(int(len(Luftwerte)/24)):
    for i in range(6):
        Stampfenbachstrasse.append(Luftwerte[idx+2]) # Nur werte von der Stampfenbachstrasse
        idx += 1

    idx += 18


Schimmelstrasse = []

idx = 0
for i in range(int(len(Luftwerte)/24)):
    for i in range(6):
        Schimmelstrasse.append(Luftwerte[idx+8]) # Nur werte von der Schimmelstrasse
        idx += 1

    idx += 18


Rosengartenstrasse = []

idx = 0
for i in range(int(len(Luftwerte)/24)):
    for i in range(6):
        Rosengartenstrasse.append(Luftwerte[idx+18]) # Nur werte von der Rosengartenstrasse
        idx += 1

    idx += 18

X = np.array( [i for i in range(len(Rosengartenstrasse))])

plt.plot(X[200:250],Rosengartenstrasse[200:250],color = 'r', label = 'Rosengartenstrasse')
plt.plot(X[200:250],Stampfenbachstrasse[200:250],color = 'b', label = 'Stampfenbachstrasse')
plt.plot(X[200:250],Schimmelstrasse[200:250],color = 'g', label = 'Schimmelstrasse')

plt.ylabel('Luftwerte')

#plt.xticks(X)

plt.legend()
plt.show()

