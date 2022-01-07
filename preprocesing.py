import pandas as pd
import numpy as np
import random
import datetime as dt

df=pd.read_csv('timeseries.csv')

# remplacement des valeurs manquantes des features
df[["High", "Low","Open","Close","Volume","Adj Close"]]
df.fillna(df.mean(), inplace=True)
df["company_name"]
df.fillna(df.mode(), inplace=True)

# transformation de la valeur catégorielle en numérique
encode_nums={"company_name":{'amazon': 1,'ame_airl': 2,'apple': 3,'marriott': 4,'netflix': 5,'walmart': 6}}
df = df.apply(lambda x: x.astype(str).str.lower())
df=df.replace(encode_nums)

# Sauvegarder le fichier transformer sous format csv
df.to_csv('timeseries_trans.csv')