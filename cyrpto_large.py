import numpy as np
import pandas as pd
import os
import random
from sklearn import preprocessing
from collections import deque
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

DIR = "C:/Users/Jackson/Documents/Data Science/CryptoCurrencies/"

dataset = DIR + "crypto_data/crypto_data.csv"

names = ["slug", "symbol","name","date","ranknow","open", "high", "low", "close", "volume", "market","close_ratio", "spread"]
df = pd.read_csv(dataset, names = names)

df.head()