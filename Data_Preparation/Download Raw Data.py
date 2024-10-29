import requests


# Hier werden die Datensätze runtergeladen von des website der Stadt Zürich:
# mit diesem Tutorial: https://www.tutorialspoint.com/downloading-files-from-web-using-python

n = 1

for i in range(n):

    year = 2024  - i

    # Luftdatensätze:

    url = f"https://data.stadt-zuerich.ch/dataset/ugz_luftschadstoffmessung_stundenwerte/download/ugz_ogd_air_h1_{year}.csv"
    r = requests.get(url, allow_redirects=True)

    open(f"Data_Preparation/Air_Datasets/ugz_ogd_air_h1_{year}.csv", 'wb').write(r.content)

    # Meteodatensätze:

    url = f"https://data.stadt-zuerich.ch/dataset/ugz_meteodaten_stundenmittelwerte/download/ugz_ogd_meteo_h1_{year}.csv"
    r = requests.get(url, allow_redirects=True)

    open(f"Data_Preparation/Meteo_Datasets/ugz_ogd_meteo_h1_{year}.csv", 'wb').write(r.content)