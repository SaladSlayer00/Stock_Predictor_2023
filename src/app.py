####################### Import necessary libraries  ###############################
from lime.lime_tabular import LimeTabularExplainer
from pyspark.ml.regression import LinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from yahoo_fin import stock_info as si
import datetime as dt
from pyspark.ml import Pipeline
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, concat, lpad, monotonically_increasing_id
import streamlit as st
import pandas as pd
from io import BytesIO


stock_ticker = ["AAPL", "TSLA", "GOOG", "MSFT", "NVDA", "AMZN"]  # Add more stock symbols as needed

model_list = ["Linear Regression", "Random Forest", "Gradient Boosted Tree", 'LSTM']  # Add more stock symbols as needed

fred_symbols = ['SP500', 'DJIA', 'NASDAQCOM', 'VIXCLS', 'GVZCLS', 'DTWEXBGS',
                'IUDSOIA', 'BAMLHE00EHYIEY', 'DFF', 'T10Y2Y', 'DGS10', 'T10YIE',
                'T5YIE', 'DTB3']

start_date = dt.datetime(2013, 9, 30)
end_date = dt.datetime.now()

def current_data_preprocessing(ticker):
    # Get historical data from Yahoo Finance
    historical_data = si.get_data(ticker, start_date, end_date, interval='1d')
    fred_df = pdr.get_data_fred(fred_symbols, start_date, end_date)

    # Plot historical data
    st.write("Showing Historical Close Prices until today")
    plot_historical_data(historical_data.index, historical_data["close"])

    # Create Spark DataFrames from Pandas DataFrames
    historical_data_spark = spark.createDataFrame(historical_data.reset_index())
    fred_df_spark = spark.createDataFrame(fred_df.reset_index())

    # Data Preprocessing
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
    fred_df_spark = fred_df_spark.withColumn("Index", monotonically_increasing_id())
    dataset_spark = historical_data_spark.join(fred_df_spark, on="DATE", how="left")
    dataset_spark = dataset_spark.orderBy("DATE")
    dataset_spark = dataset_spark.withColumn("Index", monotonically_increasing_id())
    dataset_spark = dataset_spark.dropna()

    # Save into an array the DATE column
    dates = dataset_spark.select("DATE").collect()
    # Dates to convert into pandas array
    dates = [date[0] for date in dates]
    dataset_spark_training = dataset_spark.drop("DATE")
    # Save into an array the close column
    all_closes = dataset_spark.select("close").collect()


    return dataset_spark_training, dates, all_closes

def data_exploration(dataset_spark):
    st.header("Data Exploration")
    st.write(f"Performing exploration of Close Prices of {selected_stock} stock...")
    # %%
    # Use Spark SQL to get some basic statistics and print the results
    dataset_spark.createOrReplaceTempView("dataset_spark")

    # Calculate summary statistics for the "close" column
    summary_query = "SELECT MIN(close) AS min_close, MAX(close) AS max_close, AVG(close) AS mean_close, STDDEV(close) AS stddev_close FROM dataset_spark"
    summary_result = spark.sql(summary_query)

    # Convert Spark DataFrame to Pandas DataFrame
    summary_df = summary_result.toPandas()

    # Extract metric values
    min_close = summary_df['min_close'].iloc[0]
    max_close = summary_df['max_close'].iloc[0]
    mean_close = summary_df['mean_close'].iloc[0]
    stddev_close = summary_df['stddev_close'].iloc[0]

    # Display metrics in Streamlit
    st.metric("Minimum Close", min_close)
    st.metric("Maximum Close", max_close)
    st.metric("Average Close", mean_close)
    st.metric("Standard Deviation of Close", stddev_close)
    # Define the number of bins for the histogram
    num_bins = 20

    # Calculate bin width
    bin_width = (summary_result.collect()[0]["max_close"] - summary_result.collect()[0]["min_close"]) / num_bins

    # Generate the histogram data
    histogram_query = f"SELECT CAST((close - {summary_result.collect()[0]['min_close']}) / {bin_width} AS INT) AS bin, COUNT(*) AS frequency FROM dataset_spark GROUP BY bin ORDER BY bin"
    histogram_data = spark.sql(histogram_query)

    # Collect the histogram data to the driver
    histogram_data_df = histogram_data.toPandas()

    # Plot the histogram using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.bar(histogram_data_df["bin"] * bin_width + summary_result.collect()[0]["min_close"],
            histogram_data_df["frequency"], width=bin_width, edgecolor='k')
    plt.xlabel("Close Price")
    plt.ylabel("Frequency")
    plt.title("Histogram of Close Prices")

    # Display the plot in Streamlit
    st.pyplot(plt)


def plot_predictions(dates, dates_test, all_closes, predicted_close):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(dates, all_closes, color="b")
    ax.plot(dates_test, actual_close, label="Actual Close", color="b")
    ax.plot(dates_test, predicted_close, label="Linear Regression Close", color="r")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.set_title("Actual vs. Predicted Close Prices")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_historical_data(dates, historical_close):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, historical_close, label="Historical Close", color="b")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.set_title("Historical Close Prices")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


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
    dataset_spark, dates, all_closes = current_data_preprocessing(selected_stock)

    data_exploration(dataset_spark)

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

    st.write(f"Predicting {selected_stock} stock...")

    # Create a machine learning model based on the user's selection
    if selected_model == "Linear Regression":
        model = LinearRegression(labelCol='close', featuresCol='features')
        evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
    elif selected_model == "Random Forest":
        # Create a Random Forest Regressor model
        model = RandomForestRegressor(labelCol='close', featuresCol='features', numTrees=10)
        evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
    elif selected_model == "Gradient Boosted Tree":
        model = GBTRegressor(labelCol='close', featuresCol='features', maxIter=10)
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
    rmse = evaluator.evaluate(predictions)


    # Convert predictions to Pandas DataFrame for plotting
    predictions_pd = predictions.select("Index", "close", "prediction").toPandas()

    # Extract the actual "close" values and timestamp
    actual_close = predictions_pd["close"]

    # Extract the predicted values
    predicted_close = predictions_pd["prediction"]

    # Take just last 20% of dates (the testing set)
    dates_test = dates[split_point:]

    st.write("Plotting Actual vs Predicted Close Prices")

    plot_predictions(dates, dates_test, all_closes, predicted_close)

    #Print in a beautful way the error rmse of the model
    st.metric(label="RMSE", value=rmse)


    st.title("Model Explanations")
    st.write(f"Explaining predictions of {selected_stock} stock...")


    if selected_model is not "Linear Regression":
        st.header("Feature Importance")
        importance = model_fit.stages[-1].featureImportances.toArray()

        # Plotting feature importance
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(feature_columns, importance, label="Gradient Boosted Trees", alpha=0.6)
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance")
        ax.legend()
        ax.set_xticklabels(feature_columns, rotation=45)

        # Display the plot in Streamlit
        st.pyplot(fig)

    train_data_pd = train_data.toPandas()
    X_train = train_data_pd.drop(['close'], axis=1)
    y_train = train_data_pd[['close']]
    test_data_pd = test_data.toPandas()
    X_test = test_data_pd.drop(['close'], axis=1)
    y_test = test_data_pd[['close']]

    explainer = LimeTabularExplainer(X_train.values,
                                         mode="regression",
                                         feature_names=X_train.columns.tolist(),
                                         training_labels=y_train,
                                         verbose=True)
    instance = X_test.iloc[0].values

    # For the Linear Regression model
    prediction_func = lambda x: model_fit.transform(
            spark.createDataFrame(pd.DataFrame(x, columns=X_test.columns))).select("prediction").toPandas().values
    exp = explainer.explain_instance(instance, prediction_func)

    # Plot the explanation using LIME's built-in visualizations
    fig = exp.as_pyplot_figure()
    plt.tight_layout()

    # Convert the plot to a PNG image in memory
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Provide the PNG image as a downloadable file in Streamlit
    #st.download_button('Download the explanation', buf.getvalue(), file_name='lime_explanation.png', mime='image/png')
    # Display the PNG image in the Streamlit app
    st.image(buf, caption="LIME Explanation", use_column_width=True)

    st.balloons()
