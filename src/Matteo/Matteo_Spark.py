# Data Collection

## Importing Libraries
from yahoo_fin import stock_info as si
import datetime as dt
import pandas_datareader as pdr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


ticker = 'TSLA'
start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime(2023, 1, 1)
fred_symbols = ['UNRATE', 'GDP', 'FEDFUNDS', 'CPIAUCNS', 'M2', 'DGS10', 'PCE', 'T10Y2Y', 'USROA', 'USROE', 'WTISPLC', 'HOUST', 'INDPRO', 'PAYEMS', 'BAMLH0A0HYM2', 'GS10', 'BASE', 'RIFSPPFAAD01NB', 'EXUSEU', 'NETEXP']


historical_data = si.get_data(ticker, start_date, end_date, interval='1d')
historical_data = historical_data.drop(columns=['ticker'])
fred_df = pdr.get_data_fred(fred_symbols, start_date, end_date)
historical_data = historical_data.reset_index()
fred_df = fred_df.reset_index()
fred_df = fred_df.fillna(method='ffill').fillna(method='bfill')
historical_data = historical_data.rename(columns={'index': 'DATE'})


# Initialize SparkSession
spark = SparkSession.builder \
    .appName("StockPrediction") \
    .getOrCreate()

historical_data_spark = spark.createDataFrame(historical_data)
fred_df_spark = spark.createDataFrame(fred_df)


# Joining the dataframes
df_join_spark = historical_data_spark.join(fred_df_spark, on="DATE", how="left")

# Drop unnecessary columns
dataset_spark = df_join_spark.drop('adjclose', 'ticker')

# Convert back to Pandas for modeling
dataset = dataset_spark.toPandas()

X = dataset.drop(columns=['close'])
y = dataset['close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Model Evaluation
lr_predictions = lr_model.predict(X_test)

lr_rmse = mean_squared_error(y_test, lr_predictions, squared=False)
lr_r2 = lr_model.score(X_test, y_test)
print(f"RMSE: {lr_rmse}")
print(f"R-squared: {lr_r2}")

plt.figure(figsize=(14,7))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, lr_predictions, label='Predicted', color='red')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()


explainer = LimeTabularExplainer(X_train.values, training_labels=y_train.values, feature_names=X_train.columns, mode='regression')
exp = explainer.explain_instance(X_test.values[0], lr_model.predict)
exp.show_in_notebook()