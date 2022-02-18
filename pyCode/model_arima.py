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

from getparameters import parametres

variables = parametres()
start = variables.getStartDate()
end = variables.getEndDate()
stock = variables.getActionName()

# Get the stock quote
df = DataReader(stock, data_source='yahoo', start=start, end=end)

#Distribution of the dataset
df_close = df['Adj Close']

df_log = np.log(df_close)

#split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]

from statsmodels.tsa.arima_model import ARIMA,ARIMAResults
# Build Model
model = ARIMA(train_data, order=(1,1,0))  
fitted = model.fit(disp=-1)

#Let’s now begin forecasting stock prices on the test dataset with a 95% confidence level.
# Forecast
lenp=int(len(df_log))-int(len(df_log)*0.9)
fc, se, conf = fitted.forecast(lenp, alpha=0.05)  # 95% conf

#
# Make as pandas series
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
# Plot
plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data, label='training data')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title(stock+' Price Prediction')
plt.xlabel('Time')
plt.ylabel(stock+' Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.savefig('priceprediction.png',dpi=120)


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

##
# Write scores to a file
with open("metrics_arima.txt", 'w') as outfile:
        outfile.write("MSE:  {0:2.5f} \n".format(mse))
        outfile.write("MAE: {0:2.5f}\n".format(mae))
        outfile.write("RMSE: {0:2.5f}\n".format(rmse))
        outfile.write("MAPE: {0:2.5f}\n".format(mape))

# save the model here

fitted.save('modele_LR.pkl')
# load model
loaded = ARIMAResults.load('modele_LR.pkl')
