import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math

from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model

matplotlib.use('TkAgg')

CSV_FILE = "../lstm_data/tsla_clean.csv"
DAYS_BEFORE = 20 # Anzahl der Tage in der Vergangenheit, die betrachtet werden müssen

df = pd.read_csv(CSV_FILE)
df.index = pd.to_datetime(df['date'], format='%Y-%m-%d')

temp = df['close']

#    X           Y
# [[[1 2 3],[2],[3],[4],[5]]] [6]
# [[[2],[3],[4],[5],[6]]] [7]
# [[[3],[4],[5],[6],[7]]] [8]

def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)

WINDOW_SIZE = 5
X, y = df_to_X_y(temp,WINDOW_SIZE)


X_train, y_train = X[:1000], y[:1000]
X_val, y_val = X[1000:1500], y[1000:1500]
X_test, y_test = X[1500:], y[1500:]

model1 = Sequential()
model1.add=(InputLayer((WINDOW_SIZE,1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))


cp = ModelCheckpoint('model1/', save_best_only=True)

model1.compile(loss=MeanSquaredError(),optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

model1.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=1000, callbacks=[cp])

# model1 = load_model('model1/')

train_predictions = model1.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals': y_train})

print(train_results)