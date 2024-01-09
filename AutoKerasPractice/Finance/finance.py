import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from pprint import pprint
from pylab import plt, mpl
plt.style.use('seaborn')

url = 'http://hilpisch.com/aiif_eikon_id_eur_usd.csv'
symbol = 'EUR_USD'
raw = pd.read_csv(url, index_col=0, parse_dates=True)

def generate_data():
    data = pd.DataFrame(raw['CLOSE'])
    data.columns = [symbol]
    data = data.resample('30min', label='right').last().ffill()
    return data

data = generate_data()
data = (data - data.mean()) / data.std()
p = data[symbol].values
p = p.reshape((len(p), -1))

print("p = {}\n".format(p))

lags = 5
g = TimeseriesGenerator(p, p, length=lags, batch_size=5)

def create_rnn_model(hu=100, lags=lags, layer='SimpleRNN', features=1, algorithm='estimation'):
    model = Sequential()
    if layer == 'SimpleRNN':
        model.add(SimpleRNN(hu, activation='relu', input_shape=(lags, features)))
    else:
        model.add(LSTM(hu, activation='relu', input_shape=(lags, features)))
    if algorithm == 'estimation':
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

model = create_rnn_model()

model.fit_generator(g, epochs=500, steps_per_epoch=10, verbose=False)

y = model.predict(g, verbose=False)
data['pred'] = np.nan
data['pred'].iloc[lags:] = y.flatten()
data[[symbol, 'pred']].plot(figsize=(10,6), style=['b','r-.'], alpha=0.75)
data[[symbol, 'pred']].iloc[50:100].plot(figsize=(10,6), style=['b','r-.'], alpha=0.75)