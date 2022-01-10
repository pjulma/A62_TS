import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Pour lire les données boursières de Yahoo
from pandas_datareader.data import DataReader

# Pour les horodatages
from datetime import datetime, date



# Les actions que nous utiliserons pour cette analyse
tech_list = ['AAPL', 'AMZN', 'WMT', 'NFLX','MAR','AAL']

# Heures de fin et de début de la capture de données
end = datetime.now()
start = datetime(end.year - 5, end.month, end.day)


# Boucle For pour saisir les données Yahoo Finance et les définir en tant que dataframe
for stock in tech_list:   
    #print("stock ---------- ",stock)
    # Définir DataFrame comme symbole boursier
    globals()[stock] = DataReader(stock, 'yahoo', start, end)
    
 #
 # create dataframe
company_list = [AAPL, AMZN, WMT, NFLX, MAR, AAL]
company_name = ["APPLE", "AMAZON", "WALMART", "NETFLIX","MARRIOTT","AME_AIRL"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
df = pd.concat(company_list, axis=0)




##### Identifier les données manquantes 
missing_data = df.isnull()
missing_data.head(5)
missing_data.shape
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    #print("") 
df.isnull().value_counts()


##### Ajuster les données manquantes 

