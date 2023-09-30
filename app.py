####################### Import necessary libraries  ###############################
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
import streamlit as st

stock_ticker = ["AAPL", "TSLA", "GOOG"]  # Add more stock symbols as needed

model_list = ["Linear Regression", "Random Forest"]  # Add more stock symbols as needed

fred_symbols = ['SP500', 'DJIA', 'NASDAQCOM', 'VIXCLS', 'GVZCLS', 'DTWEXBGS',
                'IUDSOIA', 'BAMLHE00EHYIEY', 'DFF', 'T10Y2Y', 'DGS10', 'T10YIE',
                'T5YIE', 'DTB3']

start_date = dt.datetime(2013, 9, 30)
end_date = dt.datetime.now()

def dataCollectionAndPreProcessing(ticker):
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
                                                                                lpad(historical_data_spark['month'], 2,
                                                                                     '0'),
                                                                                lpad(historical_data_spark['day'], 2,
                                                                                     '0')))
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

    return dataset_spark

########################################### ML ###########################################


##########################################################################################



########################################### UI ###########################################
# Create a Streamlit web app
st.title("Stock Price Prediction")



# Initialize SparkSession
spark = SparkSession.builder \
    .appName("StockPrediction") \
    .config("spark.sql.debug.maxToStringFields", "100") \
    .getOrCreate()

# Create a selectbox for choosing a stock
selected_stock = st.selectbox("Select a Stock", stock_ticker)

# Create a selectbox for choosing a machine learning model
selected_model = st.selectbox("Select a Model", model_list)  # Add more models as needed


# Create a button to trigger analysis
if st.button("Perform Analysis"):
    st.write(f"Analyzing {selected_stock} stock...")
    # Perform analysis here and display results
    dataset_spark = dataCollectionAndPreProcessing(selected_stock)
    st.write("Done!")

    # Define feature columns and split data into training and testing sets
    feature_columns = [col_name for col_name in dataset_spark.columns if col_name != 'close']
    vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    split_ratio = 0.8
    split_point = int(dataset_spark.count() * split_ratio)
    train_data = dataset_spark.limit(split_point)
    test_data = dataset_spark.subtract(train_data)
    train_data = train_data.orderBy("Index")
    test_data = test_data.orderBy("Index")

    # Create a machine learning model based on the user's selection
    if selected_model == "Linear Regression":
        model = LinearRegression(labelCol='close', featuresCol='features')
        evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
    elif selected_model == "Random Forest":
        model = RandomForestClassifier(labelCol='close', featuresCol='features')
        evaluator = MulticlassClassificationEvaluator(labelCol="close", predictionCol="prediction",
                                                      metricName="accuracy")
    else:
        st.error("Invalid model selection")

    # Create a pipeline for the selected model
    pipeline = Pipeline(stages=[vector_assembler, model])

    # Fit the model
    model_fit = pipeline.fit(train_data)

    # Make predictions
    predictions = model_fit.transform(test_data)

    # Evaluate the model and display the results
    if selected_model == "Linear Regression":
        rmse = evaluator.evaluate(predictions)
        st.write(f"Root Mean Squared Error (RMSE) for {selected_model}: {rmse}")
    elif selected_model == "Random Forest":
        accuracy = evaluator.evaluate(predictions)
        st.write(f"Accuracy for {selected_model}: {accuracy}")

    # Convert predictions to Pandas DataFrame for plotting
    predictions_pd = predictions.select("Index", "close", "prediction").toPandas()
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

############################################################################################
