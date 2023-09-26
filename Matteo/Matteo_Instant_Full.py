# Data Collection

## Importing Libraries
from yahoo_fin import stock_info as si
import datetime as dt
import pandas_datareader as pdr
import pandas as pd

ticker = 'TSLA'
start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime(2023, 1, 1)
fred_symbols = ['UNRATE', 'GDP', 'FEDFUNDS', 'CPIAUCNS', 'M2', 'DGS10', 'PCE', 'T10Y2Y', 'USROA', 'USROE', 'WTISPLC', 'HOUST', 'INDPRO', 'PAYEMS', 'BAMLH0A0HYM2', 'GS10', 'BASE', 'RIFSPPFAAD01NB', 'EXUSEU', 'NETEXP']


historical_data = si.get_data(ticker, start_date, end_date, interval='1d')
historical_data = historical_data.drop(columns=['ticker'])

fred_df = pdr.get_data_fred(fred_symbols, start_date, end_date)

df_join = historical_data.join(fred_df, how='left')
dataset = df_join.fillna(method='ffill').fillna(method='bfill')
