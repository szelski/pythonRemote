# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
