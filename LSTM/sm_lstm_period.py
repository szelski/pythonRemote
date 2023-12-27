
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import logging
from logging.handlers import RotatingFileHandler
from abc import ABC, abstractmethod
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


import sqlite3
import configparser

config = configparser.ConfigParser()
# Read the INI file
config.read('../config.ini')
modelFile_buy = config['path']['model_file_buy']
modelFile_sell = config['path']['model_file_sell']
path_database_file = config['path']['database_file']


# Create a logger
logger = logging.getLogger('app_main')
logger.setLevel(logging.DEBUG)
# Create a rotating file handler
log_file = 'logs/app_main.log'
max_log_size = 1024 * 1024  # 1 MB
backup_count = 5  # Number of backup log files to keep
file_handler = RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=backup_count)
# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(file_handler)
logger.info('INITIALIZING')

# Define the State interface
class State(ABC):
    @abstractmethod
    def handle(self, context):
        pass

# Concrete implementations of the State interface
class WaitForData():

    def __init__(self):
        logger.info('State: WaitForData')

    def handle(self, context):
        context.data = pd.read_sql_query(context.getDataQuery, context.conn)

        if not context.data.empty:
            logger.info('State: WaitForData: Data received!')
            logger.info(context.data)
            context.cursor.execute(context.deleteDataQuery)
            context.conn.commit()

            context.set_state(WriteResultInDb())

class WriteResultInDb():

    def __init__(self):
        logger.info('State: WriteResultInDb')

    def handle(self, context):
        raw_sequence = pd.read_sql_query(context.query_sequence, context.conn)
        sequence_np = np.array(raw_sequence, dtype="float").reshape(-1, 1)
        sequence_np = sequence_np[::-1]
        sequence_np_scaled = context.scaler.fit_transform(sequence_np)
        sequence_np_scaled_shaped = np.reshape(sequence_np_scaled, (1, 20, 1))
        prediction = context.lstm_model.predict(sequence_np_scaled_shaped)
        predicted_values_original_scale = context.scaler.inverse_transform(prediction)
        scalar_value = float(predicted_values_original_scale[0, 0])

        current_date = datetime.now().strftime('%Y-%m-%d')

        current_time = datetime.now()
        next_hour = current_time + timedelta(hours=1)
        current_time = next_hour.strftime('%H:%M:%S')

        data_to_insert = [
            (current_date, current_time, str(scalar_value)),
        ]

        context.cursor.executemany('INSERT INTO ' + context.resultTable +
                                   ' (DATE, TIME, RESULT) '
                                   'VALUES (?, ?, ?)',
                                   data_to_insert)
        context.cursor.executemany('INSERT INTO ' + context.resultTableTmp +
                                   ' (DATE, TIME, RESULT) '
                                   'VALUES (?, ?, ?)',
                                   data_to_insert)

        context.conn.commit()
        context.resultID = context.resultID + 1

        logger.info("predicted: " + str(scalar_value))

        context.set_state(WaitForData())
# Define the Context interface
class Context(ABC):
    @abstractmethod
    def request(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

# Concrete implementation of the Context interface
class MyContext(Context):

    conn = sqlite3.connect(path_database_file)
    cursor = conn.cursor()
    sequence_length = "20"
    symbol = "NAS100" + "_"
    period = "DAY"

    triggerTable = symbol + "LSTM_TRIGGER_TMP"
    dataTable = symbol + "CLOSE_PRICE_" + period
    dataColumn = "CLOSE_PRICE"
    resultTable = symbol + "LSTM_RESULT"
    resultTableTmp = symbol + "LSTM_RESULT_TMP"

    getDataQuery = "SELECT * FROM " + triggerTable
    deleteDataQuery = "DELETE FROM " + triggerTable
    query_sequence = "SELECT " + dataColumn + " FROM " + dataTable + " ORDER BY ROWID DESC LIMIT " + sequence_length

    data = 0
    data_tailored = 0
    resultID = 0
    scaler = MinMaxScaler(feature_range=(0, 1))
    lstm_model = load_model('keras_stock.h5')

    def __init__(self, state):
        self._state = state

    def request(self):
        return self._state.handle(self)

    def set_state(self, state):
        self._state = state

# Example usage:

ctx = MyContext(WaitForData())

while True:
    ctx.request()
    time.sleep(1)