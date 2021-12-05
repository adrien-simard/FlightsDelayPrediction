# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 22:04:38 2021

@author: adrie
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import dense_rank, desc, avg,col, when
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import argparse
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.sql.functions import udf, monotonically_increasing_id

def data_loader(path):
    df = spark.read.csv(path, header=True)
    return df

def data_cleaner(df):
    """
    Clean the data
    :param df: the spark dataframe we want to clean
    :return: the cleaned spark dataframe
    """
    # dropping the Forbidden variables
    forb_cols = ("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay","WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
    useless_cols = ("Year", "Cancelled", "CancellationCode", "FlightNum", "TailNum")
    maybe =["DepTime", "CRSDepTime", "CRSArrTime"]
    df = df.drop(*forb_cols)
    # remove th cancealed flights 
    df = df.filter(df["Cancelled"] == 0)
    # replace "NA" with None
    df = df.replace('NA', None)
    # drop unnecessary columns 
    df = df.drop(*useless_cols)
    df = df.na.drop()
    # delete the rows where ArrDelay is None since this records are labless
    df = df.filter(df["ArrDelay"].isNotNull())
    #drop duplicates 
    df = df.dropDuplicates()
    #change the datatype
    not_integer_columns = ["UniqueCarrier", "Origin", "Dest", "ArrDelay"]
    numerical_columns = list (set(df.columns).difference(set(not_integer_columns)))
    for column in numerical_columns:
        df = df.withColumn(column, df[column].cast("integer"))
    #label should be double
    df = df.withColumn("ArrDelay", df["ArrDelay"].cast("double"))
    
    data_mean_delay = df.select(*("UniqueCarrier", "DepDelay")).groupby("UniqueCarrier").avg()
    origin_delay = df.select(*("Origin","DepDelay")).groupby('Origin').avg()
    data_mean_delay = data_mean_delay.sort(data_mean_delay["avg(DepDelay)"].desc())

    rank_airline = []
    for i in range(1, data_mean_delay.count()+1):
        rank_airline.append(i)
    data_mean_delay = data_mean_delay.repartition(1).withColumn(
                      "CarrierRank", 
                      udf(lambda id: rank_airline[id])(monotonically_increasing_id()))
    
    df = df.join(data_mean_delay, on=['UniqueCarrier'])
    df = df.drop("avg(DepDelay)")
    
    origin_delay = origin_delay.sort(origin_delay["avg(DepDelay)"].desc())
    rank_origin = []
    for i in range(1, origin_delay.count()+1):
        rank_origin.append(i)
    origin_delay = origin_delay.repartition(1).withColumn(
                      "OriginRank", 
                      udf(lambda id: rank_origin[id])(monotonically_increasing_id()))
    
    df = df.join(origin_delay, on=['Origin'])
    df = df.drop("avg(DepDelay)")
    df = df.drop(*('Origin', 'UniqueCarrier'))
    
    dest_delay = df.select(*("Dest","DepDelay")).groupby('Dest').avg()
    dest_delay = dest_delay.sort(dest_delay["avg(DepDelay)"].desc())
    
    rank_dest = []
    for i in range(1, dest_delay.count()+1):
        rank_dest.append(i)
    dest_delay = dest_delay.repartition(1).withColumn(
                      "DestRank", 
                      udf(lambda id: rank_dest[id])(monotonically_increasing_id()))
    
    df = df.join(dest_delay, on=['Dest'])
    df = df.drop("avg(DepDelay)")
    df = df.drop("Dest")
    
    df = df.withColumn("CarrierRank", df["CarrierRank"].cast("integer"))
    df = df.withColumn("OriginRank", df["OriginRank"].cast("integer"))
    df = df.withColumn("DestRank", df["DestRank"].cast("integer"))
    
    return df




if __name__ == "__main__":
    
    spark = SparkSession \
        .builder \
        .appName("Spark project") \
        .getOrCreate()
    parser = argparse.ArgumentParser()

    parser.add_argument('-T', '--data', type=str, help='Path to the   data', required=True)
    parser.add_argument('-P', '--predict', action='count', default=0, help=' for predicting')


    args = parser.parse_args()
    data_path = args.data  
    # load the data
    flightDelay_data=data_loader(data_path)
    
########################Data cleaning####################
    # pre-pre-processing
    flightDelay_data = data_cleaner(flightDelay_data)
    #vectorize the data
    from pyspark.ml.feature import VectorAssembler
    data = flightDelay_data
    ignore = ['ArrDelay']
    assembler = VectorAssembler(
        inputCols=[x for x in data.columns if x not in ignore],
        outputCol='features')
    
    data = assembler.transform(data)
    data = data.withColumnRenamed("ArrDelay", "label")
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    from pyspark.ml import Pipeline
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    
########################LinearRegression Model training####################
    
    lr = LinearRegression(featuresCol='features',
                          labelCol='label',
                          maxIter=6,
                          solver="l-bfgs")
    
    modelEvaluator = RegressionEvaluator(metricName="r2")
    paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 0.3, 0.5])\
                                  .addGrid(lr.elasticNetParam, [0.3, 0.6, 0.8])\
                                  .build()
    
    crossval = CrossValidator(estimator=lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=modelEvaluator,
                              numFolds=3)
    
    cvModel = crossval.fit(trainingData)
    #best model Summary 
    trainingSummary = cvModel.bestModel.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
########################LinearRegression prediction on test data####################
    eval_rmse = RegressionEvaluator(metricName="rmse")
    eval_rmse.evaluate(cvModel.transform(testData))
    print("rmse", eval_rmse )
    
    eval_r2 = RegressionEvaluator(metricName="r2")
    eval_rmse.evaluate(cvModel.transform(testData))
    print("r2", eval_r2 )
    cvModel.bestModel.save('lr_model')
    
########################DecisionTreeRegressor Model training####################
    from pyspark.ml.regression import DecisionTreeRegressor
    
    dt = DecisionTreeRegressor()
    modelEvaluator_dt = RegressionEvaluator(metricName="r2")
    paramGrid_dt = ParamGridBuilder().addGrid(dt.maxDepth, [5, 8, 10])\
                                     .addGrid(dt.minInstancesPerNode, [1, 5, 10])\
                                     .build()
    
    crossval_dt = CrossValidator(estimator=dt,
                              estimatorParamMaps=paramGrid_dt,
                              evaluator=modelEvaluator_dt,
                              numFolds=3)
    
    cvModel_dt = crossval_dt.fit(trainingData)
    #best model Summary 
    trainingSummary = cvModel_dt.bestModel.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
########################DecisionTreeRegressor prediction on test data####################
    eval_rmse = RegressionEvaluator(metricName="rmse")
    eval_rmse.evaluate(cvModel_dt.transform(testData))
    print("rmse", eval_rmse )
    
    eval_r2 = RegressionEvaluator(metricName="r2")
    eval_rmse.evaluate(cvModel_dt.transform(testData))
    print("r2", eval_r2 )
    cvModel_dt.bestModel.save()

    spark.stop('dt_model')
    
    
    
    
    
    
    
    
    
    
    
    
