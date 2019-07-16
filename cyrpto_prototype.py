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

SEQ_LEN = 60 # last n minutes to predict from
FUTURE_PERIOD_PREDICT = 3 # next n minutes to predict
RATIO_TO_PREDICT = "BCH-USD"
EPOCHS = 10
BATCH_SIZE = 64
#NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess(df):
    df = df.drop("future", 1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    
    random.shuffle(sequential_data)       

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))
 
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []
    
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    
    return np.array(X),y    

main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "ETH-USD","BCH-USD"]
for ratio in ratios:
    dataset = DIR + f"crypto_data/{ratio}.csv"
    
    names = ["time", "low", "high", "open", "close", "volume"]
    df = pd.read_csv(dataset, names = names)
    # print(df.head())
    for name in names[1:]:
        df.rename(columns = {name:f"{ratio}_"+name}, inplace = True)

    df.set_index("time", inplace = True)
    
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df["future"] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

main_df["target"] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))

#print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head(10))

times = sorted(main_df.index.values)
last_5pct = times[-int(len(times)*0.05)]
# print(last_5pct)

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess(main_df)
validation_x, validation_y = preprocess(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Don't buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")


dense_layers = [0, 1, 2]
layer_sizes = [64, 128, 256] #for lstm
lstm_layers = [3,4,5]

for dense_layers in dense_layers:
    for layer_size in layer_sizes:
        for lstm_layers in lstm_layers:
            NAME = f"{RATIO_TO_PREDICT}-{lstm_layers}-lstm-{layer_size}-nodes-{dense_layers}-dense-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

            model = Sequential()
            model.add(CuDNNLSTM(layer_size, input_shape = (train_x.shape[1:]), return_sequences = True))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())

            for l in range(lstm_layers - 2):
                model.add(CuDNNLSTM(layer_size,return_sequences = True))
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

            model.add(CuDNNLSTM(layer_size))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())

            for l in range(dense_layers):
                model.add(Dense(32, activation = "relu"))
                model.add(Dropout(0.2))

            model.add(Dense(2, activation = "softmax"))

            opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)

            model.compile(loss = "sparse_categorical_crossentropy",
                        optimizer = opt,
                        metrics = ["accuracy"])

            logdir = os.path.join("logs",NAME,)

            tensorboard = TensorBoard(log_dir = logdir)
            filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}" # unique file name that will include the epoch and validation acc
            #monitor = EarlyStopping(monitor="val_loss", min_delta=1e3, patience = 5, verbose = 1, mode = "auto")
            checkpoint = ModelCheckpoint(os.path.join(DIR,"models/{}.hdf5",).format(filepath, monitor = "val_acc" ,verbose = 1, save_best_only = True, mode = "max"))

            model.fit(
            train_x, train_y,
            batch_size = BATCH_SIZE,
            epochs = EPOCHS,
            validation_data = (validation_x, validation_y),
            callbacks = [tensorboard, checkpoint]
            )






