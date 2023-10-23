# Import necessary libraries
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from yahoo_fin import stock_info as si
import datetime as dt
from pyspark.ml import Pipeline
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, concat, lpad, col, monotonically_increasing_id
import pyspark.sql.functions as F

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("StockPrediction") \
    .config("spark.sql.debug.maxToStringFields", "100") \
    .getOrCreate()

# Define ticker, start date, end date, and FRED symbols
ticker = 'TSLA'
start_date = dt.datetime(2013, 9, 30)
end_date = dt.datetime(2023, 1, 1)
fred_symbols = ['SP500', 'DJIA', 'NASDAQCOM', 'VIXCLS', 'GVZCLS', 'DTWEXBGS', 'IUDSOIA', 'BAMLHE00EHYIEY', 'DFF', 'T10Y2Y', 'DGS10', 'T10YIE', 'T5YIE', 'DTB3']

# Collect historical data and FRED data
historical_data = si.get_data(ticker, start_date, end_date, interval='1d')
fred_df = pdr.get_data_fred(fred_symbols, start_date, end_date)

# Create Spark DataFrames from Pandas DataFrames
historical_data_spark = spark.createDataFrame(historical_data.reset_index())
fred_df_spark = spark.createDataFrame(fred_df.reset_index())

# Rename columns and add 'fulldate' as an integer column
historical_data_spark = historical_data_spark.withColumnRenamed("index", "DATE")
historical_data_spark = historical_data_spark.drop('ticker', 'adjclose')
historical_data_spark = historical_data_spark.withColumn("year", year("DATE"))
historical_data_spark = historical_data_spark.withColumn("month", month("DATE"))
historical_data_spark = historical_data_spark.withColumn("day", dayofmonth("DATE"))
historical_data_spark = historical_data_spark.withColumn('fulldate', concat(historical_data_spark['year'],
                                       lpad(historical_data_spark['month'], 2, '0'),
                                       lpad(historical_data_spark['day'], 2, '0')))
historical_data_spark = historical_data_spark.withColumn('fulldate', historical_data_spark['fulldate'].cast('int'))

# Add an 'Index' column to FRED data
fred_df_spark = fred_df_spark.withColumn("Index", monotonically_increasing_id())

# Join historical and FRED data, and order by date
dataset_spark = historical_data_spark.join(fred_df_spark, on="DATE", how="left")
dataset_spark = dataset_spark.orderBy("DATE")
dataset_spark = dataset_spark.withColumn("Index", monotonically_increasing_id())
dataset_spark = dataset_spark.drop("DATE")

# Drop rows with null values
dataset_spark = dataset_spark.dropna()

# Define feature columns and split data into training and testing sets
feature_columns = [col_name for col_name in dataset_spark.columns if col_name != 'close']
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
split_ratio = 0.8
split_point = int(dataset_spark.count() * split_ratio)
train_data = dataset_spark.limit(split_point)
test_data = dataset_spark.subtract(train_data)
train_data = train_data.orderBy("Index")
test_data = test_data.orderBy("Index")

# Create a Linear Regression model and pipeline
lr = LinearRegression(labelCol='close', featuresCol='features')
pipeline = Pipeline(stages=[vector_assembler, lr])

# Fit the Linear Regression model
lr_model = pipeline.fit(train_data)

# Make predictions and evaluate the model
lr_predictions = lr_model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
lr_rmse = evaluator.evaluate(lr_predictions)

# Convert predictions to Pandas DataFrame for plotting
predictions_pd = lr_predictions.select("Index", "close", "prediction").toPandas()
actual_close = predictions_pd["close"]
dates = predictions_pd["Index"]
predicted_close = predictions_pd["prediction"]

# Create a time series plot
plt.figure(figsize=(12, 6))
plt.plot(dates, actual_close, label="Actual Close", color="b")
plt.plot(dates, predicted_close, label="Predicted Close", color="r")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Actual vs. Predicted Close Prices")
plt.legend()
plt.grid(True)
plt.show()
