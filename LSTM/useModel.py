import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import sys

from keras.models import Sequential, load_model
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import RootMeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sqlite3
import configparser


SEQUENCE_LENGTH = 20


config = configparser.ConfigParser()
config.read('../config.ini')


table = "NAS100_CLOSE_PRICE_HOUR"
column = "CLOSE_PRICE"

path_database_file = config['path']['database_file']
conn = sqlite3.connect(path_database_file)
query_do_buy = "SELECT " + column + " FROM " + table + " ORDER BY ROWID DESC LIMIT 20"
do_buy_df = pd.read_sql_query(query_do_buy, conn)
do_buy_np_array = np.array(do_buy_df,dtype="float").reshape(-1,1)
do_buy_np_array = do_buy_np_array[::-1]
scaler = MinMaxScaler(feature_range=(0, 1))
do_buy_np_array_scaled = scaler.fit_transform(do_buy_np_array)
new_sequences = np.reshape(do_buy_np_array_scaled, (1, 20, 1))
lstm_model = load_model('keras_stock.h5')
predictions_on_test = lstm_model.predict(new_sequences)
predicted_values_original_scale = scaler.inverse_transform(predictions_on_test)
scalar_value = float(predicted_values_original_scale[0, 0])
print(scalar_value)
