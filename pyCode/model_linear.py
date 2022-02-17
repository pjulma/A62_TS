#%matplotlib inline

import pandas as pd
import numpy as np

from pandas_datareader.data import DataReader
from datetime import datetime
import matplotlib.pyplot as plt

from getparameters import parametres

variables = parametres()
start = variables.getStartDate()
end = variables.getEndDate()
stock = variables.getActionName()

# Get the stock quote
df = DataReader(stock, data_source='yahoo', start=start, end=end)


#end = datetime.now()
#start = datetime(end.year - 10, end.month, end.day)

# Get the stock quote
#stock='AAPL'
#df = DataReader(stock, data_source='yahoo', start=start, end=end)

#Show teh data
df

# 
#Create a new dataframe with only the "Adj Close" column
data = df.filter(['Adj Close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .8 ))

#Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# number of days used for the prediction of a day.
ndays = 60

#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_train data sets
x_train = []; y_train = []

for i in range(ndays, len(train_data)):
    x_train.append(train_data[i-ndays:i, 0])
    y_train.append(train_data[i, 0])

    
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)


#Create the testing data set
test_data = scaled_data[training_data_len - ndays: , :]

#Create the data sets x_test and y_test
x_test = []; y_test = dataset[training_data_len:, :]

for i in range(ndays, len(test_data)):
    x_test.append(test_data[i-ndays:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# train an test with linear regression
# Build the Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Train
# Get the models predicted price values 
predictions_train = model.predict(x_train)
#predictions_train = scaler.inverse_transform([predictions_train]).T
# Get the root mean squared error (RMSE)
rmse_train = np.sqrt(np.mean(((predictions_train - y_train) ** 2)))
print("rmse_train = ",rmse_train)
mae_train = mean_absolute_error(y_train, predictions_train)
print('MAE _train: '+str(mae_train))
mse_train = mean_squared_error(y_train, predictions_train)
print('MSE _train: '+str(mse_train))
mape_train = np.mean(np.abs(predictions_train - y_train)/np.abs(y_train))
print('MAPE _train: '+str(mape_train))


# Test
# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform([predictions]).T

print('-------------------------------------')
# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print("rmse = ",rmse)
mae = mean_absolute_error(y_test, predictions)
print('MAE: '+str(mae))
mse = mean_squared_error(y_test, predictions)
print('MSE: '+str(mse))
mape = np.mean(np.abs(predictions - y_test)/np.abs(y_test))
print('MAPE: '+str(mape))

# Write scores to a file
with open("metrics_linear.txt", 'w') as outfile:
        outfile.write("MSE Train:  {0:2.5f} \n".format(mse_train))
        outfile.write("MAE Train: {0:2.5f}\n".format(mae_train))
        outfile.write("RMSE Train: {0:2.5f}\n".format(rmse_train))
        outfile.write("MAPE Train: {0:2.5f}\n".format(mape_train))
        # test
        outfile.write("MSE Test:  {0:2.5f} \n".format(mse))
        outfile.write("MAE Test: {0:2.5f}\n".format(mae))
        outfile.write("RMSE Test: {0:2.5f}\n".format(rmse))
        outfile.write("MAPE Test: {0:2.5f}\n".format(mape))

        
# 
# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
valid['Erreur'] = predictions - data[training_data_len:]

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model - Linear Regression de '+stock, fontsize=28)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
plt.show()
plt.savefig('LinearPrediction.png',dpi=120)


# save the model to disk
import pickle
filename = 'modele_Linear.pkl'
pickle.dump(model, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
