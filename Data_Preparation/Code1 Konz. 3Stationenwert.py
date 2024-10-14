import pandas as pd
import numpy as np

All_data = pd.read_csv("Data_Preparation/Air_Datasats/ugz_ogd_air_h1_2022.csv")

Luftwerte = All_data["Wert"]

print(len(Luftwerte))

#print(Luftwerte)

c = 0
for i in range (len(Luftwerte)):
    if np.isnan(Luftwerte[i]):
        #print(i)
        c+= 1

print(f"Number of NaN values = {c}")



c=0
for i in range(len(All_data)):
    if np.isnan(All_data["Wert"][i]):
        #print(i)
        
        c+=1

#print(c)
#print(f"All_data[150517] = {All_data["Wert"][150517]}")

# neues file kreieren mit den richtigen Daten

Ausgewählte_Daten = []

idx = 0
for i in range(int(len(Luftwerte)/24)):
    for i in range(2):
        Ausgewählte_Daten.append(Luftwerte[idx])
        idx += 1
    for i in range(6):
        test_l = [] # Liste mit nicht NA-Werten, aus dessen Einträgen soll der Durchschnittswert berechnet werden
       
        if np.isnan(Luftwerte[idx]) == False:
            test_l.append(Luftwerte[idx])
        if np.isnan(Luftwerte[idx+6]) == False:
            test_l.append(Luftwerte[idx+6])
        if np.isnan(Luftwerte[idx+16]) == False:
            test_l.append(Luftwerte[idx+16])
        
        c = 0
        for nums in test_l:
            c += nums/len(test_l)
        

        Ausgewählte_Daten.append(c) #Durchschnitt von Stampfenbachstrasse,Rosengartenstrasse, Schimmelstrasse
        idx += 1 
    idx += 16




print(len(Ausgewählte_Daten))
print(len(All_data))

#print(Ausgewählte_Daten[0])

# Wieder testen, wie oft Daten fehlen und dann diese ersetzten mit dem Durchschnitt derselben Daten von einer Stunde vorher und einer Stunde nachher

# Testen
c = 0
for i in range (len(Ausgewählte_Daten)):
    if np.isnan(Ausgewählte_Daten[i]):
       # print(i)
        c+= 1
print(f"Number of Nan Values in Ausgewählte_Daten: {c}")

#print(Ausgewählte_Daten[68277])
#print(Ausgewählte_Daten[68275])




# Ersetzen

# Current problem: np.isnan doesnt work somehow, now I need another function to determine if a value is a number or not
reserve = 100

for i in range(len(Ausgewählte_Daten)):
    if np.isnan(Ausgewählte_Daten[i]):
        a = 0
        b = 0

        for l in range(reserve):

            if a != 0 and b != 0:
                break
            
            if 8 < i < (len(Ausgewählte_Daten)-8):
                if  np.isnan(Ausgewählte_Daten[i+(l+1)*8]) == False and b == 0:
                    b = Ausgewählte_Daten[i+(l+1)*8]
                    

                if  np.isnan(Ausgewählte_Daten[i-(l+1)*8]) == False and a == 0:
                    a = Ausgewählte_Daten[i-(l+1)*8]

            #else:
                #if  np.isnan(Ausgewählte_Daten[i+(l+1)*8]) == False and b == "nothing":
                   # b = Ausgewählte_Daten[i+(l+1)*8]
                   # a = b
       # print(a)
       # print(b)
       # print(l+1)
       # print(i)
        if i < len(Ausgewählte_Daten)/2:
            ti = i
            tl = l

        Ausgewählte_Daten[i] = (a+b)/2
                
                


# Testen

c = 0
for i in range (len(Ausgewählte_Daten)):
    if np.isnan(Ausgewählte_Daten[i]):
        c+= 1
print(f"Nan values after clean-up:{c}")

print(Ausgewählte_Daten[ti])

print(f"The above value should be the average of {Ausgewählte_Daten[ti+tl*8]} and {Ausgewählte_Daten[ti-tl*8]}")





#print(All_data["Wert"][150524])


#print(np.isnan(All_data["Wert"][150524]))

