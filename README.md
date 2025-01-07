#LSTM-Modelle für die Vorhersage der Ozonkonzentration

Die Programme sind in fünf Ordner aufgeteilt: 
- Data_Preparation 
- LSTM_Model
- XGboost_Regression
- Graphics
- logs. 

Für die Arbeit wurden jedoch nur die Ordner Data_Preparation, LSTM_Model und Graphics verwendet. Die anderen beiden Ordner enthalten trainierte XGboost Modell und Sammlungen der Fehlermetriken für die LSTM-Modelle. Die Programme in diesem Foldern wurden jedoch nicht für die schrifliche Arbeit verwendet.


Es folgt nun eine Erläuterung über die drei Ordner Data_Preparation, LSTM_Model und Graphics.
Programme, welche in diesen Ordnern nicht verwendet wurden haben das Suffix "_unused" im Dateinamen.


##Data_Preparation:

Enthält folgende Programme:

- Download_Raw_Data.py: 

Lädt die rohen Datensätze von der Stadt Zürich aus der Website https://data.stadt-zuerich.ch/dataset herunter und speichert sie in den entsprechenden Ordnern "Air_Datasets", "Meteo_Datasets" und "Traffic_Datasets" ab.

- Heatmap.py: 

Erstellt eine Heatmap mit der Korrelation zwischen der Ozonkonzentration und den anderen Features im Datensatz. Es kann spezifiziert werden zwischen welchen Features in den Datensätzen die Korrelation bestimmt wird.

- Air_Data_gather.py: 

Kreiert Haupt- und Kontextfeatures der Luftqualitätsdaten und ersetzt Ausreisserwerte.

- Meteo_Data_gather.py: 

Kreiert Haupt- und Kontextfeatures der Meteodaten und ersetzt Ausreisserwerte.

- Create_Training_Dataset.py: 

Nimmt die von Ausreissern bereinigten Dataframes aus Air_Data_gather.py und Meteo_Data_gather.py und ersetzt NaN-Werte durch KNN-Imputation. Speichert die bereinigten Trainingsdatensätze im Ordner "Training_Datasets" ab.
Das Jahr, von welchem der Trainingsdatensatz kreiert wird, kann vom User angegeben werden.

##LSTM_Model:

Hier werden die verschiedenen LSTM-Modelle trainiert und evaluiert. 
Der Ordner enthält folgende Programme:

- Model_Training.py: 

Hier werden die LSTM-Modelle trainiert. Es kann spezifiziert werden von welchem Feature die Vorhersagen gemacht werden. Ebenfalls kann bestimmt werden mit welchen Inputfeatures das Modell trainiert wird. 


