import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType,BooleanType,DateType

if __name__== "__main__":
    
    spark = SparkSession \
        .builder \
        .appName("Spark app") \
        .getOrCreate()
    sparkContext=spark.sparkContext
    
    path = "s3://the-museum/2006.csv"
    data = spark.read.csv(path,header=True)
    
    
    #remove some columns
    data = data.drop(*("ArrTime","ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay","LateAircraftDelay","TailNum", "CancellationCode", "FlightNum", "TailNum", "Year"))
    
    
    data = data.withColumn("DepTime",data.DepTime.cast('integer'))
    data = data.withColumn("CRSArrTime",data.CRSArrTime.cast('integer'))
    data = data.withColumn("CRSElapsedTime",data.CRSElapsedTime.cast('integer'))
    data = data.withColumn("ArrDelay",data.ArrDelay.cast('integer'))
    data = data.withColumn("Distance",data.Distance.cast('integer'))
    data = data.withColumn("CRSDepTime",data.CRSDepTime.cast('integer'))
    data = data.withColumn("TaxiOut",data.TaxiOut.cast('integer'))
    data = data.withColumn("DayOfWeek",data.DayOfWeek.cast('integer'))
    data = data.withColumn("DepDelay",data.DepDelay.cast('double'))
    
    from pyspark.sql.functions import isnull, when, count, col
    
    data.select([count(when(col(c).isNull(), c)).alias(c) for c in data.columns])
    
    
    
    data = data.replace('NA', None)
    data = data.na.drop()
    
    #remove cancelled flights
    data = data.filter((data.Cancelled != 1))
    #remove cancelled column
    data = data.drop("Cancelled")
    
    
    data.select([count(when(col(c).isNull(), c)).alias(c) for c in data.columns])
    
    data_num = data.drop("UniqueCarrier", "Origin", "Dest")
    data_num = data_num.withColumn("Month",data_num.Month.cast('integer'))
    data_num = data_num.withColumn("DayofMonth",data_num.DayofMonth.cast('integer'))
    data_num = data_num.withColumn("DepDelay",data_num.DepDelay.cast('integer'))
    
    data_num.printSchema()
    #Compute the correlation beetwen prediction variables and target variable
    for i in data_num.columns:
        print("Corr entre ArrDelay" + " et " + i + ": ", round(data_num.corr("ArrDelay", i, method=None), 5))
    
    data_mean_delay = data.select(*("UniqueCarrier", "DepDelay")).groupby("UniqueCarrier").avg()
    origin_delay = data.select(*("Origin","DepDelay")).groupby('Origin').avg()
    
    #data_mean_delay.show()
    
    from pyspark.sql import Window
    import pyspark.sql.functions as psf
    
    wM = Window.orderBy(psf.desc("avg(DepDelay)"))
    data_mean_delay = data_mean_delay.withColumn(
        "CarrierRank", 
        psf.dense_rank().over(wM)
    )
    
   
    # we are going to compute a ranking for companies and airport based on the average delays
    data_mean_delay = data_mean_delay.drop("avg(DepDelay")
    data = data.join(data_mean_delay, on=['UniqueCarrier'])
    
    
    data = data.drop("avg(DepDelay)")
    
    data.columns
    
    wO = Window.orderBy(psf.desc("avg(DepDelay)"))
    origin_delay = origin_delay.withColumn(
        "OriginRank", 
        psf.dense_rank().over(wO)
    )
    
    
    
    data = data.join(origin_delay, on=['Origin'])
    #drop useless colum
    data = data.drop("avg(DepDelay)")
    
    data.columns
    data = data.drop(*('Origin', 'UniqueCarrier'))
    
    
    dest_delay = data.select(*("Dest","DepDelay")).groupby('Dest').avg()
    
    wD = Window.orderBy(psf.desc("avg(DepDelay)"))
    dest_delay = dest_delay.withColumn(
        "DestRank", 
        psf.dense_rank().over(wD)
    )
    
    data = data.join(dest_delay, on=['Dest'])
    
    data = data.drop("avg(DepDelay)")
    
    data = data.drop("Dest")
    
    data = data.withColumn("DayofMonth",data.DayofMonth.cast('integer'))
    data = data.withColumn("Month",data.Month.cast('integer'))
    
    data.printSchema()
    
    #remove the ouliers in the interval [0.1, 0.9] for the DepDelay. 
    
    quantiles_depDelay = data.approxQuantile("DepDelay", [0.1, 0.9], 0) #Il faut choisir les quantiles ici dans la liste.
    
    quantiles_depDelay
    
    import pyspark.sql.functions as f
    data.select(
        "*",
        *[
            f.when(
                f.col("DepDelay").between(quantiles_depDelay[0], quantiles_depDelay[1]),
                0
            ).otherwise(1).alias("DepDelay_out") 
        ]
    )
    
    #data = data.filter((data.DepDelay_out != 1))
    
    from pyspark.ml.feature import VectorAssembler
    
    ignore = ['ArrDelay']
    assembler = VectorAssembler(
        inputCols=[x for x in data.columns if x not in ignore],
        outputCol='features')
    
    data = assembler.transform(data)
    
    data_ml = data.select(['features', 'ArrDelay'])
    
 
    
    from pyspark.ml.feature import PCA
    pca = PCA(k=2, inputCol="features", outputCol="pca_features")
    
    model = pca.fit(data)
    
    df_pca = model.transform(data)
    
    model.explainedVariance
    
    
    data_stripplot = data.drop(*('Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime',
                                 'ArrDelay', 'Dest', 'Distance', 'TaxiOut', 'DestRank', 'OriginRank'))
    
    data_stripplot.columns
    
    spark.conf.set("spark.sql.execution.arrow.enabled","true")
    data_stripplot_pandas = data_stripplot.toPandas()
    
    #----------- MACHINE LEARNING -----------------
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.evaluation import RegressionEvaluator
    
    #Split the data
    (training_data, test_data) = data_ml.randomSplit([0.8,0.2])
    #LinearRegression
    lr = LinearRegression(featuresCol = 'features', labelCol='ArrDelay', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(training_data)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))
    
    
    
    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
    
    lr_predictions = lr_model.transform(test_data)
    #lr_predictions.select("prediction","ArrDelay","features").show(5)
    
    
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                     labelCol="ArrDelay",metricName="r2")
    print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
    
    lr_evaluator_rmse = RegressionEvaluator(predictionCol="prediction", \
                     labelCol="ArrDelay",metricName="rmse")
    print("RMSE: %f" % lr_evaluator_rmse.evaluate(lr_predictions))
    
    data_ml_pca = df_pca.select(['pca_features', 'ArrDelay'])
    
    (training_data_pca, test_data_pca) = data_ml_pca.randomSplit([0.8,0.2])
    
    lr_pca = LinearRegression(featuresCol = 'pca_features', labelCol='ArrDelay', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model_pca = lr_pca.fit(training_data_pca)
    print("Coefficients: " + str(lr_model_pca.coefficients))
    print("Intercept: " + str(lr_model_pca.intercept))
    
    trainingSummary_pca = lr_model_pca.summary
    print("RMSE: %f" % trainingSummary_pca.rootMeanSquaredError)
    print("r2: %f" % trainingSummary_pca.r2)
    
    lr_predictions_pca = lr_model_pca.transform(test_data_pca)
    #lr_predictions.select("prediction","ArrDelay","features").show(5)
    
    
    lr_evaluator_pca = RegressionEvaluator(predictionCol="prediction", \
                     labelCol="ArrDelay",metricName="r2")
    print("R Squared (R2) on test data = %g" % lr_evaluator_pca.evaluate(lr_predictions_pca))
    
    lr_evaluator_rmse_pca = RegressionEvaluator(predictionCol="prediction", \
                     labelCol="ArrDelay",metricName="rmse")
    print("RMSE: %f" % lr_evaluator_rmse_pca.evaluate(lr_predictions_pca))
    
    assembler = VectorAssembler(
        inputCols=['DepDelay'],
        outputCol='feature')
    
    data2 = assembler.transform(data)
    
    data_ml2 = data2.select(['feature', 'ArrDelay'])
    
    #Split the data
    (training_data, test_data) = data2.randomSplit([0.8,0.2])
    
    lr = LinearRegression(featuresCol = 'feature', labelCol='ArrDelay', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(training_data)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))
    
    
    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
    
    lr_predictions_1 = lr_model.transform(test_data)
    #lr_predictions.select("prediction","ArrDelay","features").show(5)
    
    
    lr_evaluator_1 = RegressionEvaluator(predictionCol="prediction", \
                     labelCol="ArrDelay",metricName="r2")
    print("R Squared (R2) on test data = %g" % lr_evaluator_1.evaluate(lr_predictions_1))
    
    lr_evaluator_rmse_1 = RegressionEvaluator(predictionCol="prediction", \
                     labelCol="ArrDelay",metricName="rmse")
    print("RMSE: %f" % lr_evaluator_rmse_1.evaluate(lr_predictions_pca))
    
    assembler = VectorAssembler(
        inputCols=['DepDelay','TaxiOut'],
        outputCol='feature')
    
    data2 = assembler.transform(data)
    
   
    
    
    data_ml2 = data2.select(['feature', 'ArrDelay'])
    
    #Split the data
    (training_data, test_data) = data_ml2.randomSplit([0.8,0.2])
    
    from pyspark.ml.classification import LogisticRegression
    
    lr2 = LogisticRegression(featuresCol='feature', labelCol='ArrDelay', maxIter=10,regParam=0.0)
    lrModel2=lr2.fit(training_data)
    print("Coefficients: " + str(lrModel2.coefficients))
    print("Intercept: " + str(lrModel2.intercept))
    
    
    
    
    from pyspark.ml.regression import GBTRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
    
    # Train a GBT model.
    gbt = GBTRegressor(featuresCol="features", labelCol='ArrDelay', maxIter=10)
    
    # Train model.  This also runs the indexer.
    model = gbt.fit(training_data)
    
    # Make predictions.
    predictions = model.transform(test_data)
    
    # Select example rows to display.
    # predictions.select("prediction", "ArrDelay", "features")
    
    # Select (prediction, true label) and compute test error
    evaluator_rmse = RegressionEvaluator(
        labelCol="ArrDelay", predictionCol="prediction", metricName="rmse")
    
    evaluator_r2 = RegressionEvaluator(
        labelCol="ArrDelay", predictionCol="prediction", metricName="r2")
    
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    print("R Squared (R2) on test data = %g" % r2)
    spark.stop()





