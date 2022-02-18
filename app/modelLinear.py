from math import sqrt
import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plotly import graph_objs as go
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader as web
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pickle

st.title("Predicion de l'action: ")
input_Date = st.date_input('start date', date(2012, 1, 1))
input_end = st.date_input('end date', date.today())

if len(str(input_Date)) > 1:
    StartDate = input_Date
    # '2012-01-01'
    EndDate = date.today().strftime('%Y-%m-%d')
else:
    StartDate = '2012-01-01'
    EndDate = date.today().strftime('%Y-%m-%d')

if len(str(input_end)) > 1:
    EndDate = input_end
else:
    EndDate = input_end
    
stocks = ('AAL', 'AAPL', 'AMZN', 'MAR', 'NFLX', 'WMT')
select_stock = st.selectbox('selection du dataset pour la prediction', stocks)

n_years = st.slider('Année de la prediction :', 1, 6)
period = n_years * 365


@st.cache(allow_output_mutation=True)
def load_data(ticker):
    dfl = web(ticker, 'yahoo', StartDate, EndDate)
    dfl.reset_index(inplace=True)
    return dfl


df_load = st.text("Chargement des données ...")
df = load_data(select_stock)
df_load.text("Chargement des données terminées")
#
st.markdown("Données brutes de " + select_stock+" du "+str(StartDate)+" au "+str(EndDate))
st.write(df.tail())


# plot raw data
def plot_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Open"], name="open price"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Adj Close"], name="close price"))
    fig.layout.update(title_text="Series de données temporelles de " + select_stock, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_data()

data = df.filter(['Adj Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil(len(dataset) * .8))

# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# number of days used for the prediction of a day.
ndays = 60

# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
# Split the data into x_train and y_train data sets
x_train = [];
y_train = []

for i in range(ndays, len(train_data)):
    x_train.append(train_data[i - ndays:i, 0])
    y_train.append(train_data[i, 0])

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Create the testing data set
test_data = scaled_data[training_data_len - ndays:, :]

# Create the data sets x_test and y_test
x_test = [];
y_test = dataset[training_data_len:, :]

for i in range(ndays, len(test_data)):
    x_test.append(test_data[i - ndays:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# number of days used for the prediction of a day.
ndays = 60

# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
# Split the data into x_train and y_train data sets
x_train = [];
y_train = []

for i in range(ndays, len(train_data)):
    x_train.append(train_data[i - ndays:i, 0])
    y_train.append(train_data[i, 0])

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Create the testing data set
test_data = scaled_data[training_data_len - ndays:, :]

# Create the data sets x_test and y_test
x_test = [];
y_test = dataset[training_data_len:, :]

for i in range(ndays, len(test_data)):
    x_test.append(test_data[i - ndays:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# train an test with linear regression
# Build the Linear Regression model

model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Train
# Get the models predicted price values
predictions_train = model.predict(x_train)
# predictions_train = scaler.inverse_transform([predictions_train]).T
# Get the root mean squared error (RMSE)
rmse_train = np.sqrt(np.mean(((predictions_train - y_train) ** 2)))
print("rmse_train = ", rmse_train)
mae_train = mean_absolute_error(y_train, predictions_train)
print('MAE _train: ' + str(mae_train))
mse_train = mean_squared_error(y_train, predictions_train)
print('MSE _train: ' + str(mse_train))
mape_train = np.mean(np.abs(predictions_train - y_train) / np.abs(y_train))
print('MAPE _train: ' + str(mape_train))

# Test
# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform([predictions]).T

print('-------------------------------------')
# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print("rmse = ", rmse)
mae = mean_absolute_error(y_test, predictions)
print('MAE: ' + str(mae))
mse = mean_squared_error(y_test, predictions)
print('MSE: ' + str(mse))
mape = np.mean(np.abs(predictions - y_test) / np.abs(y_test))
print('MAPE: ' + str(mape))

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
    # st.metric(label, value, delta=None, delta_color="normal")

#
# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
valid['Erreur'] = predictions - data[training_data_len:]

# Visualize the data
figl = plt.figure(figsize=(16, 8))
plt.title('Model - Linear Regression de '+select_stock, fontsize=28)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
st.pyplot(figl)

#st.metric(label="Train", value="{0:2.5f}\n".format(mse_train), delta="mse")

# Metriques de performance des données et de test
st.markdown('Les métriques de performance du modèle de ' + select_stock)
 # Train
coltr1, coltr2, coltr3, coltr4 = st.columns(4)
coltr1.metric("mse", "{0:2.5f}\n".format(mse_train), "train")
coltr2.metric("rmse", "{0:2.5f}\n".format(rmse_train), "train")
coltr3.metric("mae", "{0:2.5f}\n".format(mae_train), "tain")
coltr4.metric("mape", "{0:2.5f}\n".format(mape_train), "train")
 # Test
colte1, colte2, colte3, colte4 = st.columns(4)
colte1.metric("mse", "{0:2.5f}\n".format(mse), "test")
colte2.metric("rmse", "{0:2.5f}\n".format(rmse), "test")
colte3.metric("mae", "{0:2.5f}\n".format(mae), "test")
colte4.metric("mape", "{0:2.5f}\n".format(mape), "test")