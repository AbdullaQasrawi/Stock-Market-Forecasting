#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from tensorflow.keras.models import Model

import logging

from Processing import StockProcessor

class LSTM_model:
    def __init__(self, X_train):
        self.model = Sequential()
        self.model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        self.model.add(LSTM(units=64, return_sequences=True))
        self.model.add(LSTM(units=32))
        self.model.add(Dense(units=1))
        adam = Adam(learning_rate=0.001)
        self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, verbose=1):
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        
    def save(self, filepath):
    # Save the model to an h5 file
        self.model.save(filepath)

def run_training():
    processor = StockProcessor()
    
    df = processor.load_yahoo()
    train_data = df[:1000]
    test_data = df[1000:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled_data = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))
    test_scaled_data = scaler.transform(test_data['Close'].values.reshape(-1, 1))

    window_size = 5
    X_train_lstm, y_train_lstm = processor.create_dataset(train_scaled_data, window_size)
    X_test_lstm, y_test_lstm = processor.create_dataset(test_scaled_data, window_size)
    

    lstm_model = LSTM_model(X_train_lstm)
    lstm_model.train(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=1)
    lstm_model.save('model222.h5')

if __name__ == '__main__':
    run_training()
