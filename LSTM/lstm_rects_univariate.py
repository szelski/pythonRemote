#
# Vorhersage von einem Aktienkurs mit Keras und LSTMs in TensorFlow 2
#

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import sys

from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import RootMeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sqlite3
import configparser


# This line is specifying that the TkAgg (Tkinter Agg) backend should be used.
# Tkinter is the standard Python interface to the Tk GUI toolkit, and 'Agg' stands for Anti-Grain Geometry,
# which is a high-quality rendering engine for C++.
matplotlib.use('TkAgg')

config = configparser.ConfigParser()
config.read('../config.ini')

path_database_file = config['path']['database_file']
conn = sqlite3.connect(path_database_file)

symbol = "NAS100_0_"

query_do_buy = "SELECT POINTS_SPAN FROM " + symbol + "AI_DATA_BUY"
do_buy_df = pd.read_sql_query(query_do_buy, conn)

print(do_buy_df)

# sys.exit()
#
# # Quelle: https://www.macrotrends.net/stocks/charts/TSLA/tesla/stock-price-history
# # Spalten:
# # 'date' 'open' 'high' 'low' 'close' 'volume'
#
# CSV_FILE = "lstm_data/tsla.csv"
DAYS_BEFORE = 20  # Anzahl der Tage in der Vergangenheit, die betrachtet werden müssen
#
# initial_stock_data = np.loadtxt(CSV_FILE,delimiter=",",skiprows=9,usecols=(4),comments="#",dtype=float)

initial_stock_data = do_buy_df

print("original data")
print(initial_stock_data)

# .reshape(-1, 1): This reshapes the array to have one column
# and as many rows as needed. The "-1" in the reshape function
# is a placeholder that means the size of that dimension is inferred
# based on the length of the original array. So, in this case,
# it transforms the array into a column vector.

np.random.seed(17)

initial_stock_data = np.array(initial_stock_data,dtype="float").reshape(-1,1)

print("reshaped to column vector and converted to float")
print(initial_stock_data)

# # # Normalisierung der Werte
min_max_scaler = MinMaxScaler(feature_range=(0,1))
stock_data = min_max_scaler.fit_transform(initial_stock_data)

# stock_data = initial_stock_data

print("scaled between 0 and 1")
print(stock_data)

# Reorganisiert die Daten

def arrange_data(data, days):
    days_before_values = [] # T- days
    days_values = []  # T
    for i in range(len(data) - days -1):
        days_before_values.append(data[i:(i+days)]) # append a range 0 : (0 -19)
        days_values.append(data[i + days]) # append value of 0 + 20
    return np.array(days_before_values),np.array(days_values)


def split_to_percentage(data, percentage):
    return data[0: int(len(data)*percentage)], data[int(len(data)*percentage):]


days_before_values, days_values = arrange_data(stock_data,DAYS_BEFORE)

print("days before values")
print(days_before_values)

print("day values")
print(days_values)


# days_before_values = days_before_values.reshape((days_before_values.shape[0],DAYS_BEFORE,1))



#-----------------------------------

# Wir nehmen nur ein Teil des Datasets, um das Training durchzuführen
# Der Rest (X_test und Y_test) wird für die "virtuelle" Prognose benutzt
# Splitting des Datasets

X_train, X_test = split_to_percentage(days_before_values, 0.8) #  80% Training
Y_train, Y_test = split_to_percentage(days_values, 0.8) # 20% Test

print("X_train")
print(X_train)
# print("Y_train")
# print(Y_train)
# print("X_test")
# print(X_test)
# print("Y_test")
# print(Y_test)
#



# sys.exit()

# Build the LSTM model
stock_model = Sequential()
stock_model.add(LSTM(units=50, return_sequences=True, input_shape=(DAYS_BEFORE,1)))
stock_model.add(LSTM(units=50))
stock_model.add(Dense(units=1))
stock_model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.mse])
# Train the model
stock_model.fit(X_train, Y_train, epochs=10, batch_size=32)




# # # Definition des Keras Modells
# stock_model = Sequential()
# stock_model.add(LSTM(10,input_shape=(DAYS_BEFORE,1),return_sequences=True))
# stock_model.add(LSTM(5,activation="relu"))
# stock_model.add(Dense(1))
# sgd = SGD(lr=0.01)
# stock_model.summary()
# stock_model.compile(loss="mean_squared_error", optimizer=sgd, metrics=[tf.keras.metrics.mse])
# stock_model.fit(X_train, Y_train, epochs=100, verbose=1)


# Das Modell wird gespeichert
stock_model.save("keras_stock.h5")
# stock_model.save("keras_stock.keras")

# Evaluation der Testdaten
score, _ = stock_model.evaluate(X_test,Y_test)
rmse = math.sqrt(score)
print("RMSE {}".format(rmse))

# Vorhersage mit den "unbekannten" Test-Dataset
predictions_on_test = stock_model.predict(X_test)
predictions_on_test = min_max_scaler.inverse_transform(predictions_on_test)

# ... und mit dem Trainings-Dataset
predictions_on_training = stock_model.predict(X_train)

print("predictions_on_training")
print(predictions_on_training)

predictions_on_training = min_max_scaler.inverse_transform(predictions_on_training)

# Wir shiften nach rechts, damit das Testergebnis grafisch direkt nach der Trainingskurve startet.
shift = range(len(predictions_on_training)-1, len(stock_data) - 1 - DAYS_BEFORE - 1)

# Anzeige der Kurven mit matplotlib
plt.plot(initial_stock_data, color="#CFCEC4",label="Kurs")
plt.plot(predictions_on_training, label="Training", color="green")
plt.plot(shift,predictions_on_test, label="Test", color="red", dashes=[6, 2])
plt.legend(loc='upper left')
plt.show()