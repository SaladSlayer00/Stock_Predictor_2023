import pandas_datareader as pdr
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from yahoo_fin import stock_info as si
import datetime as dt
from pyspark.ml import Pipeline
from pyspark.sql.functions import monotonically_increasing_id

def forwardFillImputer(df, cols=[], partitioner="Index", value='NaN'):
    for c in cols:
        # Define the window specification with the partitioner
        window_spec = Window.orderBy(partitioner)

        # Replace value with NULL
        df = df.withColumn(c, F.when(F.col(c) != value, F.col(c)).otherwise(F.lit(None)))

        # Forward fill using the last non-null value within the partition
        df = df.withColumn(c, F.last(c, True).over(window_spec))

    return df

# Data Collection
ticker = 'TSLA'
start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime(2023, 1, 1)
fred_symbols = ['UNRATE', 'GDP', 'FEDFUNDS', 'CPIAUCNS', 'M2', 'DGS10', 'PCE', 'T10Y2Y', 'USROA', 'USROE', 'WTISPLC',
                'HOUST', 'INDPRO', 'PAYEMS', 'BAMLH0A0HYM2', 'GS10', 'BASE', 'RIFSPPFAAD01NB', 'EXUSEU', 'NETEXP']
historical_data = si.get_data(ticker, start_date, end_date, interval='1d')
fred_df = pdr.get_data_fred(fred_symbols, start_date, end_date)
# Data Pre-Processing with Spark

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("StockPrediction") \
    .config("spark.sql.debug.maxToStringFields", "100") \
    .getOrCreate()

historical_data_spark = spark.createDataFrame(historical_data.reset_index())
fred_df_spark = spark.createDataFrame(fred_df.reset_index())
historical_data_spark = historical_data_spark.withColumnRenamed("index", "DATE")
historical_data_spark = historical_data_spark.drop('ticker', 'adjclose')
historical_data_spark = historical_data_spark.withColumn("year", year("DATE"))
historical_data_spark = historical_data_spark.withColumn("month", month("DATE"))
historical_data_spark = historical_data_spark.withColumn("day", dayofmonth("DATE"))
fred_df_spark = fred_df_spark.withColumn("Index", monotonically_increasing_id())

dataset_spark = historical_data_spark.join(fred_df_spark, on="DATE", how="left")
dataset_spark = dataset_spark.orderBy("DATE")
dataset_spark = dataset_spark.withColumn("Index", monotonically_increasing_id())
dataset_spark = dataset_spark.drop("DATE")
dataset_spark = forwardFillImputer(dataset_spark, cols=[i for i in fred_symbols])
dataset_spark = dataset_spark.dropna()

# Prepare data for MLlib
feature_columns = [col_name for col_name in dataset_spark.columns if col_name != 'close']
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Determine the split point based on the desired ratio
split_ratio = 0.8  # 80% for training, 20% for testing
split_point = int(dataset_spark.count() * split_ratio)

# Split the data into training and testing sets
train_data = dataset_spark.limit(split_point)
test_data = dataset_spark.subtract(train_data)
# Reorder by index
train_data = train_data.orderBy("Index")
test_data = test_data.orderBy("Index")

lr = LinearRegression(labelCol='close', featuresCol='features')
pipeline = Pipeline(stages=[vector_assembler, lr])

# Linear Regression in Spark
lr_model = pipeline.fit(train_data)

# Model Evaluation in Spark
lr_predictions = lr_model.transform(test_data)

evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
lr_rmse = evaluator.evaluate(lr_predictions)
print(f"RMSE: {lr_rmse}")