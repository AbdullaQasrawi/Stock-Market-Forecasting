#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

class StockProcessor:
    def __init__(self):
        self.df = None
    
    def read_data(self, data_name, sheet_num, file_path=None):
        """
        This method reads the data from an Excel file and returns a clean dataframe.
        """
        xls = pd.ExcelFile(file_path)
        df = xls.parse(sheet_num)
        df.drop([df.shape[0]-1], axis=0, inplace=True)
        
        k_data = df[df['Vol.'].astype(str).str.contains('K')]
        df = df[df["Vol."].str.contains("K") == False]
        k_data['Vol.'] = k_data['Vol.'].str.replace('K', '')
        k_data['Vol.'] = k_data['Vol.'].apply(pd.to_numeric)
        df = pd.concat([df, k_data], join="inner")
        df['Vol.'] = df['Vol.'].str.replace('M', '').replace('-', '')
        df[["Date"]] = df[["Date"]].apply(pd.to_datetime)
        df[["Price", "Open", "High", "Low", "Change %", 'Vol.']] = df[["Price", "Open", "High", "Low", "Change %", 'Vol.']].apply(pd.to_numeric)
        df = df.set_index('Date')
        df = df.sort_values(by=['Date'], ascending=True)
        
        self.df = df
        return self.df

    
    def test_stationarity(self, timeseries):
        #Determing rolling statistics
        rolmean = timeseries.rolling(6).mean()
        rolstd = timeseries.rolling(6).std()
        #Plot rolling statistics:
        plt.grid(True)
        plt.plot(timeseries, color='blue',label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean and Standard Deviation')
        plt.show(block=False)
        print("Results of dickey fuller test")
        adft = adfuller(timeseries,autolag='AIC')
        # output for dft will give us without defining what the values are.
        #hence we manually write what values does it explains using a for loop
        output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
        for key,values in adft[4].items():
            output['critical value (%s)'%key] =  values
        print(output)
        
    def moving_average_graph(self, timeseries, rolling_value=5):
        # Moving average method
        moving_avg = timeseries.rolling(rolling_value).mean()
        plt.figure(figsize=(22,10))
        plt.grid(True)
        plt.plot(timeseries, color = "red",label = "Original")
        plt.plot(moving_avg, color='black', label = "rolling_mean")
        plt.title("Mean Prices of Stock")
        plt.xlabel("Date")
        plt.ylabel("Mean prices")
        plt.legend()
        plt.show()
        
        
    def moving_average_embedding(self, df, timeseries, rolling_value=5):
        moving_avg = timeseries.rolling(rolling_value).mean()
        differences = timeseries - moving_avg
        differences.dropna(inplace=True)
        df['difference'] = differences
        df['difference'].fillna(df['difference'].mean(), inplace=True)
        df['Date'] = df.index
        df = df.reset_index(drop=True)
        
        return df
    
    def load_yahoo(self, start_date = '2018-01-01', end_date = '2023-03-26'):
        # Define the ticker symbol
        tickerSymbol = 'MSFT'

        # Get data for the past year
        start_date = '2018-01-01'
        end_date = '2023-03-26'
        tickerData = yf.Ticker(tickerSymbol)
        tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
        # Remove the time component from the Date column
        tickerDf['Date'] = tickerDf.index.date
        tickerDf.set_index(tickerDf['Date'],inplace=True)
        # Print the modified DataFrame
        return tickerDf
        
    def create_dataset(self, data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:(i+window_size), 0])
            y.append(data[i+window_size, 0])
        return np.array(X), np.array(y)



  
