# LSTM-Modelle für die Vorhersage der Ozonkonzentration

Die Programme sind in fünf Ordner aufgeteilt: 
- Data_Preparation 
- LSTM_Model
- XGboost_Regression
- Graphics
- logs

Für die Arbeit wurden jedoch nur die Ordner Data_Preparation, LSTM_Model und Graphics verwendet. Die anderen beiden Ordner enthalten trainierte XGboost Modelle und Sammlungen der Fehlermetriken für die LSTM-Modelle. Die Programme in diesen Ordnern wurden jedoch nicht für die schriftliche Arbeit verwendet.

Es folgt nun eine Erläuterung über die zwei Ordner Data_Preparation und LSTM_Model.
Programme, welche in diesen Ordnern nicht verwendet wurden, haben das Suffix "_unused" im Dateinamen.

## Data_Preparation:

Enthält folgende Programme:

1. Download_Raw_Data.py: 

Lädt die rohen Datensätze von der Stadt Zürich aus der Website https://data.stadt-zuerich.ch/dataset herunter und speichert sie in den entsprechenden Ordnern "Air_Datasets", "Meteo_Datasets" und "Traffic_Datasets" ab.

2. Heatmap.py: 

Erstellt eine Heatmap mit der Korrelation zwischen der Ozonkonzentration und den anderen Features im Datensatz. Es kann spezifiziert werden, zwischen welchen Features in den Datensätzen die Korrelation bestimmt wird.

3. Air_Data_gather.py: 

Kreiert Haupt- und Kontextfeatures der Luftqualitätsdaten und ersetzt Ausreisserwerte.

4. Meteo_Data_gather.py: 

Kreiert Haupt- und Kontextfeatures der Meteodaten und ersetzt Ausreisserwerte.

5. Create_Training_Dataset.py: 

Nimmt die von Ausreissern bereinigten Dataframes aus Air_Data_gather.py und Meteo_Data_gather.py und ersetzt NaN-Werte durch KNN-Imputation. Speichert die bereinigten Trainingsdatensätze im Ordner "Training_Datasets" ab.
Das Jahr, von welchem der Trainingsdatensatz kreiert wird, kann vom User angegeben werden.

## LSTM_Model:

Hier werden die verschiedenen LSTM-Modelle trainiert und evaluiert. 
Der Ordner enthält folgende Programme:

1. Model_Training.py: 

  Hier werden die LSTM-Modelle trainiert. Es kann spezifiziert werden, von welchem Feature die Vorhersagen gemacht werden. Ebenfalls kann bestimmt werden, mit welchen Inputfeatures das Modell trainiert wird. 

Folgende Parameter können angepasst werden:

- Anzahl LSTM-Layer: Wird angegeben durch *model_type*.
- Outputdimension der LSTM-Layer: Wird angegeben durch *LSTM_l_dimension*.
- t, Anzahl der vorhergehenden Stundenmessungen, anhand derer das Modell die Vorhersagen macht: Wird angegeben durch *look_back*.
- ∆t, Vorhersagehorizont, also wie viele Stunden in die Zukunft sich die vorhergesagte Stunde befindet: Wird angegeben durch *y_forward*.
  
Die Modelle werden im Ordner *Models* abgespeichert, die Test- und Trainings-MSE-Werte werden im Ordner *Histories* abgespeichert.

2. Automate_Train_diff_configs.py:

  Funktioniert wie *Model_Training.py*. Hier können mehrere verschiedene Modelle auf einmal trainiert werden, wobei bei jedem die veränderbaren Parameter der Modellarchitektur und des Modelltrainings angegeben werden können.

3. Evaluate_Model.py:

  Hier werden die Modelle evaluiert. Man spezifiziert das Modell, welches untersucht wird. Es kann mit *predict_range* angegeben werden, für wie viele Stunden die Vorhersage ist. Ebenfalls kann mit *delta* angegeben werden, von der wievielten Stunde im Testdatensatz aus die Vorhersagen gemacht werden. Die Vorhersagen werden dann aufgezeichnet und mit den echten Werten verglichen. Es werden die Fehlermetriken *RMSE*, *MAE* und *Pearson-Koeffizient* berechnet. Die entstehenden Diagramme werden dann im Ordner *Graphics* abgespeichert.

4. Plot_Training_Progress.py:

   Hier werden das Trainings- und Test-MSE während des Modelltrainings aufgezeichnet. Man kann so sehen, wie die Modelle lernen. Die Diagramme werden im Ordner *Graphics* abgespeichert.

5. Compare_Models.py:

Hier können die Trainings- und Test-MSE-Werte mehrerer Modelle mit einem Balkendiagramm verglichen werden. Die Modelle, die verglichen werden, können im Code spezifiziert werden.




