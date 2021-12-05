
from pyspark.sql.functions import dense_rank, desc, avg,col, when
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler, UnivariateFeatureSelector
from pyspark.ml import Pipeline
import argparse
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
import os

if __name__== "__main__":
    
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
        # remove outliers
        outliers_col = ["CRSElapsedTime", "ArrDelay", "DepDelay", "TaxiOut", "Distance"]
        bounds = {
            c: dict(
                zip(["q1", "q3"], df.approxQuantile(c, [0.25, 0.75], 0))
            )
            for c in outliers_col
        }
    
        for c in bounds:
            iqr = bounds[c]['q3'] - bounds[c]['q1']
            bounds[c]['lower'] = bounds[c]['q1'] - (iqr * 1.5)
            bounds[c]['upper'] = bounds[c]['q3'] + (iqr * 1.5)
        df = df.select("*",*[when(
                        col(c).between(bounds[c]['lower'], bounds[c]['upper']),0)
                                .otherwise(1).alias(c+"_out") 
                        for c in outliers_col])
        for column in outliers_col:
            c=column+"_out"
            df = df.filter((df[c] == 0))
            df =df.drop(c)
    
        return df
    
    
    def category_rank(df, partionBy, output_col):
        """
        transform categorical data into numerical using ranking based on the delay
        :param df: the spark dataframe with the categorical variables to transform
        :param partionBy: string fof the name of the categorical variable to transform
        :param partionBy: string fof the name of the outputted transformed variable
        :return: the input dataframe with the transformed features
        """
        wM = Window.partitionBy(partionBy)
        df = df.withColumn(
            "GroupedAvgDelay",
            avg(col("Depdelay")).over(wM))
        win_rank = Window.orderBy(desc("GroupedAvgDelay")) # partionBy was not used in purpose to use the entire column
        df = df.withColumn(output_col, dense_rank().over(win_rank))
        df =df.drop(*["GroupedAvgDelay",partionBy ])
        return df
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
    
        parser.add_argument('-T', '--data', type=str, help='Path to the   data', required=True)
        parser.add_argument('-P', '--predict', action='count', default=0, help=' for predicting')
    
    
        args = parser.parse_args()
        data_path = args.data
        predict = bool(args.predict)
    ################################################ data preparation########################################
        # load the data
        flightDelay_data=data_loader(data_path)
        # pre-pre-processing
        flightDelay_data = data_cleaner(flightDelay_data)
        # feature engineering
        # category feature encoding
        flightDelay_data = category_rank(flightDelay_data,partionBy= "UniqueCarrier", output_col="CarrierRank")
        flightDelay_data  = category_rank(flightDelay_data,partionBy= "Origin", output_col="OriginRank")
        flightDelay_data  = category_rank(flightDelay_data,partionBy= "Dest", output_col="DestRank")
        flightDelay_data  = flightDelay_data.withColumnRenamed("ArrDelay", "label")
        # split the data( 90% traning, 10% test)
        (training_data, test_data) = flightDelay_data.randomSplit([0.9,0.1], seed=45)
        #vectorize the data
        label = "ArrDelay"
        features = [feature for feature in training_data.columns if feature != label]
        vector_assembler = VectorAssembler(inputCols = features,
                                        outputCol = "features_vector")
        #Normalizing the data 
        standardScaler_feature = StandardScaler(inputCol = "features_vector",
                                        outputCol = "features")
    
        #################################### training the model ####################################
        if not predict :
            lr = LinearRegression(maxIter=10)
            rf = RandomForestRegressor(impurity='variance')
    
            modelEvaluator = RegressionEvaluator()
            stages_lr = [vector_assembler, standardScaler_feature, lr]
            pipeline = Pipeline(stages = stages_lr)
            paramGrid_lr = ParamGridBuilder()\
                    .addGrid(lr.regParam, [0.1,0.3]) \
                    .build()
            crossval_lr = CrossValidator(estimator=pipeline,
                                        estimatorParamMaps=paramGrid_lr,
                                        evaluator=modelEvaluator,
                                        numFolds=3,
                                        seed=45)  
    
            # Run cross-validation, and choose the best set of parameters for LinearRegression
            cvModel_lr = crossval_lr.fit(training_data)
    
            stages_rf = [vector_assembler, standardScaler_feature, rf]
            pipeline = Pipeline(stages = stages_rf)
            paramGrid_rf = ParamGridBuilder()\
                    .addGrid(rf.maxDepth, [3,4,5]) \
                    .addGrid(rf.numTrees,[2,3,4])\
                    .build()
    
    
    
            crossval_rf = CrossValidator(estimator=pipeline,
                                        estimatorParamMaps=paramGrid_rf,
                                        evaluator=modelEvaluator,
                                        numFolds=3,
                                        seed=45)
    
            # Run cross-validation, and choose the best set of parameters for RandomFores
            cvModel_rf = crossval_rf.fit(training_data)
            # Print the coefficients and intercept for linear regression
            print("linear regression parameters:\nn")
            print("Coefficients: %s" % str(cvModel_lr.bestModel.stages[-1].coefficients))
            print("Intercept: %s" % str(cvModel_lr.bestModel.stages[-1].intercept))
    
            # Summarize the model over the training set and print out some metrics
            print("linear regression metrics on training data:\nn")
            trainingSummary_lr = cvModel_lr.bestModel.stages[-1].summary
            print("numIterations: %d" % trainingSummary_lr.totalIterations)
            print("RMSE: %f" % trainingSummary_lr.rootMeanSquaredError)
            print("r2: %f" % trainingSummary_lr.r2)
    
            print("RandomFores metrics on training data :\nn")
            trainingSummary_rf = cvModel_rf.bestModel.stages[-1].summary
            print("numIterations: %d" % trainingSummary_rf.totalIterations)
            print("RMSE: %f" % trainingSummary_rf.rootMeanSquaredError)
            print("r2: %f" % trainingSummary_rf.r2)
    
            ############################## prediction on test data####################
    
            if trainingSummary_lr.r2 > trainingSummary_rf.r2:
                print("The best model is linear regression")
                cvModel_lr.save("./models/model")
                best_model = cvModel_lr.bestModel.stages[-1]
            else:
                print("The best model is RandomFores")
    
                cvModel_rf.save("./models/model")
                best_model = cvModel_rf.bestModel.stages[-1]
    
            prediction = best_model.transform(test_data)
            evaluator = RegressionEvaluator(
                        labelCol="label", predictionCol="prediction", metricName="rmse")
            rmse = evaluator.evaluate(prediction)
            print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
        if predict:
            modelEvaluator = RegressionEvaluator()
            model = CrossValidatorModel.load("./models/model")
            stages= [vector_assembler, standardScaler_feature, model.bestModel.stages[-1]]
            pipeline = Pipeline(stages = stages)
            prediction = pipeline.transform(data_path)
            evaluator = RegressionEvaluator(
                        labelCol="label", predictionCol="prediction", metricName="rmse")
            rmse = evaluator.evaluate(prediction)
            print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    
    spark.stop()





