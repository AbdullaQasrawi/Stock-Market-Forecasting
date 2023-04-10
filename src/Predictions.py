#!/usr/bin/env python
# coding: utf-8

## This class contain two functions first one (predict) used to predict image label,and the second function (pred_sequence) used to predit if sequence of images have the action of fliping

import logging
from Train import LSTM_model, run_training
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from Processing import StockProcessor

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import logging
from Processing import StockProcessor


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

class StockPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.processor = StockProcessor()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_model(self):
        self.model = load_model(self.model_path)

    def make_predictions(self, data):
        # Scale the input data using the same scaler used to train the model
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        # Create input sequences for the LSTM model
        window_size = 5
        X_lstm, y_lstm = self.processor.create_dataset(scaled_data, window_size)

        # Make predictions using the LSTM model
        y_pred_lstm = self.model.predict(X_lstm)
        predictions = self.scaler.inverse_transform(y_pred_lstm)

        return predictions
        
    def plot_predictions(self, data, predictions):
        # Get the actual stock prices
        actual_prices = data['Close'].values[5:]

        # Plot the actual and predicted stock prices
        plt.plot(actual_prices, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.legend()
        plt.show()

    def run_predictions(self):
        # Load saved model
        self.load_model()

        # Load test data
        df = self.processor.load_yahoo()
        test_data = df[1000:]

        # Make predictions
        predictions = self.make_predictions(test_data)
      
      # Plot predicted prices
        self.plot_predictions(test_data, predictions)
        
        
        # Print predicted prices
        for i, pred in enumerate(predictions):
            print(f"Day {i+1}: {pred[0]:.2f}")

if __name__ == '__main__':
    run_predictions()
    
predictor = StockPredictor('../models/LSTM_Model.h5')
predictor.run_predictions()