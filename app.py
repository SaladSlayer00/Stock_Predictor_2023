####################### Import necessary libraries  ###############################
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from yahoo_fin import stock_info as si
from pyspark.ml.regression import RandomForestRegressor
import datetime as dt
from pyspark.ml import Pipeline
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, concat, lpad, monotonically_increasing_id
import streamlit as st
import pandas as pd

stock_ticker = ["AAPL", "TSLA", "GOOG"]  # Add more stock symbols as needed

model_list = ["Linear Regression", "Random Forest", "LSTM"]  # Add more stock symbols as needed

fred_symbols = ['SP500', 'DJIA', 'NASDAQCOM', 'VIXCLS', 'GVZCLS', 'DTWEXBGS',
                'IUDSOIA', 'BAMLHE00EHYIEY', 'DFF', 'T10Y2Y', 'DGS10', 'T10YIE',
                'T5YIE', 'DTB3']

start_date = dt.datetime(2013, 9, 30)
end_date = dt.datetime.now()

def current_data_preprocessing(ticker):
    # Collect historical data and FRED data
    historical_data = si.get_data(ticker, start_date, end_date, interval='1d')
    fred_df = pdr.get_data_fred(fred_symbols, start_date, end_date)

    # Plot historical data
    st.write("Showing Historical Close Prices until today")
    plot_historical_data(historical_data.index, historical_data["close"])

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


def predict_future_close_prices(model_fit, feature_columns, future_timestamps):
    # Create a DataFrame for future timestamps
    future_data = pd.DataFrame({'fulldate': future_timestamps}).astype({'fulldate': 'int'})

    # Create a Spark DataFrame from the Pandas DataFrame
    future_data_spark = spark.createDataFrame(future_data)

    # Use the vector assembler to assemble features
    future_data_spark = vector_assembler.transform(future_data_spark)

    # Make predictions for future timestamps
    future_predictions = model_fit.transform(future_data_spark)

    # Extract the predicted close prices
    future_predicted_close = future_predictions.select("prediction").rdd.map(lambda row: row[0]).collect()

    return future_predicted_close


def plot_predictions(dates, actual_close, predicted_close):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_close, label="Actual Close", color="b")
    plt.plot(dates, predicted_close, label="Predicted Close", color="r")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Actual vs. Predicted Close Prices")
    plt.legend()
    plt.grid(True)
    st.pyplot()

def plot_historical_data(dates, historical_close):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, historical_close, label="Historical Close", color="b")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Historical Close Prices")
    plt.legend()
    plt.grid(True)
    st.pyplot()


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

    st.header("Data Collection and Preprocessing")
    st.write(f"Analyzing {selected_stock} stock...")
    # Perform analysis here and display results
    dataset_spark = current_data_preprocessing(selected_stock)

    # Define feature columns and split data into training and testing sets
    feature_columns = [col_name for col_name in dataset_spark.columns if col_name != 'close']
    vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    split_ratio = 0.8
    split_point = int(dataset_spark.count() * split_ratio)
    train_data = dataset_spark.limit(split_point)
    test_data = dataset_spark.subtract(train_data)
    train_data = train_data.orderBy("Index")
    test_data = test_data.orderBy("Index")

    st.header("Predicting Close Prices")

    st.write("Training Data to be used for training the model")

    # Create a machine learning model based on the user's selection
    if selected_model == "Linear Regression":
        model = LinearRegression(labelCol='close', featuresCol='features')
        evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
    elif selected_model == "Random Forest":
        # Create a Random Forest Regressor model
        model = RandomForestRegressor(labelCol='close', featuresCol='features', numTrees=10)
        evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
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
    elif selected_model == "Random Forest":
        rmse = evaluator.evaluate(predictions)

    # Convert predictions to Pandas DataFrame for plotting
    predictions_pd = predictions.select("Index", "close", "prediction").toPandas()
    actual_close = predictions_pd["close"]
    dates = predictions_pd["Index"]
    predicted_close = predictions_pd["prediction"]

    st.write("Plotting Actual vs Predicted Close Prices")

    plot_predictions(dates, actual_close, predicted_close)

    #Print in a beautful way the error rmse of the model
    st.write("The RMSE of the model is: ", rmse)


    #Do you want to see the future?
    st.header("Predicting Future Close Prices")
    st.write("Do you want to see the future?")

    if st.button("Show me the future!"):
        st.write(f"Analyzing the future {selected_stock} stock with {selected_model}")

        # Input field for selecting the number of days into the future
        days_into_future = st.number_input("Enter the number of days into the future:", min_value=1, value=10)

        # Button to trigger future predictions
        if st.button("Show me the future!"):
            # Generate future timestamps for prediction from dates array
            last_date = dates.iloc[-1]
            future_timestamps = [last_date + i for i in range(1, days_into_future + 1)]

            # Call the predict_future_close_prices function
            future_predicted_close = predict_future_close_prices(model_fit, feature_columns, future_timestamps)

            # Create a DataFrame for the future predictions
            future_predictions_df = pd.DataFrame(
                {'fulldate': future_timestamps, 'predicted_close': future_predicted_close})

            # Display the future predictions
            st.write("Future Close Price Predictions:")
            st.write(future_predictions_df)

############################################################################################
