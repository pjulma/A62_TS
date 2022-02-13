from math import sqrt
import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plotly import graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#import statsmodels.api as sm
from statsmodels import api as sm
from sklearn.model_selection import train_test_split as split
from pandas_datareader.data import DataReader as web

try:
    from scipy.signal._signaltools import _centered as trim_centered
except ImportError:
    # Must be using SciPy <1.8.0 where this function was moved (it's not a
    # public SciPy function, but we need it here)
    from scipy.signal.signaltools import _centered as trim_centered

# set_input_date=date.today().strftime('%Y-%m-%d')
# input_Date= st.text_input("Postcode : ",set_input_date)

st.title("Predicion de l'action: ")
input_Date = st.date_input('start date')

if len(str(input_Date)) > 1:
    StartDate = '2012-01-01'
    # input_Date
    EndDate = date.today().strftime('%Y-%m-%d')
else:
    StartDate = '2012-01-01'
    EndDate = date.today().strftime('%Y-%m-%d')

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
st.subheader("Données brutes de " + select_stock)
st.write(df.tail())


# plot raw data
def plot_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Open"], name="open price"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="close price"))
    fig.layout.update(title_text="Series de données temporelles de " + select_stock, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_data()

dfsar=df
dfsar['Date'] = pd.to_datetime(dfsar['Date'], format='%Y-%m-%d')
# Making Date as Index
dfsar.set_index('Date', inplace=True)
dfsar['Date'] = dfsar.index

# Data after feature selection
data_feature_selected = dfsar.drop(axis=1, labels=['Open', 'High', 'Low', 'Adj Close', 'Volume'])
col_order = ['Date', 'Close']
data_feature_selected = data_feature_selected.reindex(columns=col_order)

# Resample Data to Monthly instead of Daily by Aggregating Using Mean
monthly_mean = data_feature_selected['Close'].resample('M').mean()
monthly_data = monthly_mean.to_frame()

monthly_data['Year'] = monthly_data.index.year
monthly_data['Month'] = monthly_data.index.strftime('%B')
monthly_data['dayofweek'] = monthly_data.index.strftime('%A')
monthly_data['quarter'] = monthly_data.index.quarter


# Stock Prices Year & Month Wis
group_by_yr = []
list_years = monthly_data['Year'].unique()
dict_IQR = {}
for yr in list_years:
    group_by_yr.append('df' + str(yr))

for enum, yr in enumerate(list_years):
    group_by_yr[enum] = monthly_data[str(yr)]['Close']
    dict_IQR[str(yr)] = stats.iqr(group_by_yr[enum])
    
# Stock Prices Year & Month Wise
figSM = plt.figure(figsize=(20, 10))
palette = sns.color_palette("mako_r", 4)
a = sns.barplot(x="Year", y="Close", hue='Month', data=monthly_data)
a.set_title("Stock Prices Year & Month Wise", fontsize=15)
plt.legend(loc='upper left')
st.pyplot(figSM)

#
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 12))
acf = plot_acf(monthly_data['Close'], lags=90, ax=ax1)
ax1.set_title('AutoCorrelation Long Term')
acf = plot_acf(monthly_data['Close'], lags=30, ax=ax2)
ax2.set_title('AutoCorrelation Short Term')
ax1.set_ylabel('Correlation')
ax1.set_xlabel('Lags')
ax2.set_ylabel('Correlation')
ax2.set_xlabel('Lags')

plt.show()

# Differencing By 1
monthly_diff = monthly_data['Close'] - monthly_data['Close'].shift(1)
monthly_diff[1:].plot(c='grey')
monthly_diff[1:].rolling(20).mean().plot(label='Rolling Mean', c='orange')
monthly_diff[1:].rolling(20).std().plot(label='Rolling STD', c='yellow')
plt.legend(prop={'size': 12})
plt.plot(monthly_diff)
plt.show()

# split arima modele
modelling_series = monthly_data['Close']
train, test = split(modelling_series, train_size=0.95, shuffle=False)

# forecast arima modele
model = sm.tsa.SARIMAX(train, order=(2, 1, 1), seasonal_order=(0, 2, 1, 12))
results = model.fit()
forecasts_train = results.predict(start='2012-01-31', end='2021-07-31')
forecasts_test = results.predict(start='2021-08-31', end='2022-02-28')

fig, (ax1, ax2) = plt.subplots(2, figsize=(18, 10))

forecasts_train.plot(label='Forecasts', ax=ax1, title='SARIMA Forecasting -Train Data')
train.plot(label='Actual', ax=ax1)
ax1.set_ylabel('Stock Price')

forecasts_test.plot(label='Forecasts', ax=ax2, title='SARIMA Forecasting -Test Data')
test.plot(label='Actual', ax=ax2)
ax2.set_ylabel('Stock Price')


ax1.legend()
ax2.legend()
plt.tight_layout(pad=2)


def plot_forecast_train():
    #mask = (df['Date'] > '2000-6-1') & (df['Date'] <= '2000-6-10')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list_years, y=forecasts_train, name="Forecasts"))
    fig.add_trace(go.Scatter(x=list_years, y=train, name="Actual"))
    fig.layout.update(title_text="SARIMA Forecasting -Train Data' " + select_stock, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_forecast_train()

def plot_forecast_test():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list_years, y=forecasts_test, name="Forecasts"))
    fig.add_trace(go.Scatter(x=list_years, y=test, name="Actual"))
    fig.layout.update(title_text="SARIMA Forecasting -Test Data' " + select_stock, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_forecast_test()

#
# report performance
mse = mean_squared_error(test, forecasts_test)
mae = mean_absolute_error(test, forecasts_test)
rmse = math.sqrt(mean_squared_error(test, forecasts_test))
mape = np.mean(np.abs(forecasts_test - test)/np.abs(forecasts_test))

mse_ = ('MSE: %.3f' % mse)
mae_ = ('MAE: %.3f' % mae)
rmse_ = ('RMSE: %.3f' % rmse)
mape_ = ('MAPE: %.3f' % mape)
st.markdown('Les métriques du modèle de ' + select_stock)
st.markdown(mse_)
st.markdown(mae_)
st.markdown(rmse_)
st.markdown(mape_)