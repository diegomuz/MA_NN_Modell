Deutsch:
Die Programme sind in fünf Ordner aufgeteilt: 
Data_Preparation, LSTM_Model, XGboost_Regression, Graphics und logs. 

Für die Arbeit wurden jedoch nur die Ordner Data_Preparation, LSTM_Model und Graphics verwendet. Die anderen beiden Ordner enthalten trainierte XGboost Modell und Sammlungen der Fehlermetriken für die LSTM-Modelle. Die Programme in diesem Foldern wurden jedoch nicht für die schrifliche Arbeit verwendet.

Es folgt nun eine Erläuterung über die drei Ordner Data_Preparation, LSTM_Model und Graphics.
Programme, welche in diesen Ordnern nicht verwendet wurden haben das Suffix "_unused" im Dateinamen.

Data_Preparation:
Es enthält folgende Programme:

Download_Raw_Data.py: Lädt die rohen Datensätze von der Stadt Zürich aus der Website https://data.stadt-zuerich.ch/dataset herunter und speichert sie in den entsprechenden Ordnern "Air_Datasets", "Meteo_Datasets" und "Traffic_Datasets" ab.
Heatmap.py: Erstellt eine Heatmap mit der Korrelation zwischen der Ozonkonzentration und den anderen Features im Datensatz.
Air_Data_gather.py: Kreiert Haupt- und Kontextfeatures der Luftqualitätsdaten und ersetzt Ausreisserwerte.
Meteo_Data_gather.py: Kreiert Haupt- und Kontextfeatures der Meteodaten und ersetzt Ausreisserwerte.
Create_Training_Dataset.py: Nimmt die von Ausreissern bereinigten Dataframes aus Air_Data_gather.py und Meteo_Data_gather.py und ersetzt NaN-Werte durch KNN-Imputation. Speichert die bereinigten Trainingsdatensätze im Ordner "Training_Datasets" ab.


LSTM_Model:
