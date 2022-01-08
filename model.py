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

#

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
#df.tail(10)

#
# Let's see a historical view of the closing price
plt.figure(figsize=(18, 15))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(3, 2, i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"{tech_list[i - 1]}")
    plt.savefig('historicalClosingPrince.png',dpi=120)

#
#  ARIMA
stock_data = AAPL
#plot close price
plt.figure(figsize=(15,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Close Prices')
plt.plot(stock_data['Close'])
plt.title('APPLE closing price')
plt.show()
plt.savefig('appleClosePrice.png',dpi=12)

#
#Test for staionarity
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(20).mean()
    rolstd = timeseries.rolling(20).std()
    #Plot rolling statistics:
    plt.figure(figsize=(15,6))
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
test_stationarity(df_close)
plt.savefig('test_stationarity.png',dpi=120)

#
#split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data');
plt.plot(test_data, 'blue', label='Test data');
plt.legend();
plt.savefig('trainTest.png',dpi=120)

#
# Build Model
from statsmodels.tsa.arima_model import ARIMA
# Modeling
# Build Model
model = ARIMA(train_data, order=(1,1,0))  
fitted = model.fit(disp=-1)  
#print(fitted.summary())

#
#Let’s now begin forecasting stock prices on the test dataset with a 95% confidence level.
# Forecast
fc, se, conf = fitted.forecast(126, alpha=0.05)  # 95% conf


#
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# report performance
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))

#
# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("MSE:  {0:2.5f} \n".format(mse))
        outfile.write("MAE: {0:2.5f}\n".format(mae))
        outfile.write("RMSE: {0:2.5f}\n".format(rmse))
        outfile.write("MAPE: {0:2.5f}\n".format(mape))

# save the model here

    #Sauvegarder le modele
import pickle #une autre strategie
pickle.dump(fitted,open("modele_LR.pkl","wb"))