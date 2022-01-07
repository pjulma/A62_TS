%matplotlib inline

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Pour lire les données boursières de Yahoo
from pandas_datareader.data import DataReader

# Pour les horodatages
from datetime import datetime

# Les actions que nous utiliserons pour cette analyse
tech_list = ['AAPL', 'AMZN', 'WMT', 'NFLX','MAR','AAL']

# Heures de fin et de début de la capture de données
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)


# Boucle For pour saisir les données Yahoo Finance et les définir en tant que dataframe
for stock in tech_list:   
    # Définir DataFrame comme symbole boursier
    globals()[stock] = DataReader(stock, 'yahoo', start, end)
    
# 
company_list = [AAPL, AMZN, WMT, NFLX, MAR, AAL]
company_name = ["APPLE", "AMAZON", "WALMART", "NETFLIX","MARRIOTT","AME_AIRL"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
df = pd.concat(company_list, axis=0)

# Sauvegarder les données collectées sur yahoo dans notre repertoire
df.to_csv('timeseries.csv')