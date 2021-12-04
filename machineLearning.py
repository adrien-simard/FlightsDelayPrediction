#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, BooleanType, DateType


# In[5]:


spark = SparkSession     .builder     .appName("Spark app")     .getOrCreate()


# In[6]:


path = "s3://the-museum/flights/2006_dataProcess"
data = spark.read.format("csv")    .option("recursiveFileLookup", "true")    .option("header",True)    .load(path)
data.take(10)


# In[7]:


data = data.select("ArrDelay", "DepDelay", "TaxiOut")
data.show()


# In[8]:


for c in data.columns:
    data = data.withColumn(c, data[c].cast('integer'))


# In[9]:


from pyspark.ml.feature import VectorAssembler

ignore = ['ArrDelay']
assembler = VectorAssembler(
    inputCols=[x for x in data.columns if x not in ignore],
    outputCol='features')

data = assembler.transform(data)


# In[10]:


data = data.withColumnRenamed("ArrDelay", "label")


# In[14]:


(trainingData, testData) = data.randomSplit([0.7, 0.3])


# ## GBT

# In[21]:


from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator


# In[22]:


# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a GBT model.
gbt = GBTRegressor(featuresCol="features", maxIter=10, maxDepth=2)


# In[23]:


# Chain indexer and GBT in a Pipeline
pipeline = Pipeline(stages=[gbt])
# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)
# Make predictions.
predictions = model.transform(testData)


# In[24]:


# Select example rows to display.
predictions.select("prediction", "label", "features").show()


# In[25]:


# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="label",
                                predictionCol="prediction",
                                metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="label",
                                predictionCol="prediction",
                                metricName="r2")
r2 = evaluator.evaluate(predictions)
print("R-squared on test data = %g" % r2)


# ## Linear Regression

# In[15]:


from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol='features',
                      labelCol='label',
                      maxIter=10,
                      regParam=0.3,
                      elasticNetParam=0.8)
lr_model = lr.fit(trainingData)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))


# In[16]:


trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# In[ ]:


from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

lr = LinearRegression(featuresCol='features',
                      labelCol='label',
                      maxIter=5,
                      solver="l-bfgs")
modelEvaluator = RegressionEvaluator()
pipeline = Pipeline(stages=[lr])
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01,0.3,0.2]).addGrid(lr.elasticNetParam, [0, 1,0.8,0.6]).addGrid(lr.maxIter, [2,5,10,12]).build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=modelEvaluator,
                          numFolds=3)

cvModel = crossval.fit(trainingData)
trainingSummary = cvModel.bestModel.summary


# ## RandomForestRegressor

# In[31]:


from pyspark.ml.regression import RandomForestRegressor


# In[ ]:





# In[34]:


rf = RandomForestRegressor(featuresCol="features", labelCol="label")

# Chain indexer and forest in a Pipeline
pipeline = Pipeline(stages=[rf])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)


# In[35]:


# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="label",
                                predictionCol="prediction",
                                metricName="rmse")
rmse = evaluator.evaluate(predictions)
evaluator = RegressionEvaluator(labelCol="label",
                                predictionCol="prediction",
                                metricName="r2")
r2 = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
print("R-squared on test data = %g" % r2)


# ## Decision tree

# In[36]:


from pyspark.ml.regression import DecisionTreeRegressor


# In[37]:


# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="features")
# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[dt])


# In[38]:


# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)


# In[41]:


# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)


# In[48]:


# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="label",
                                predictionCol="prediction",
                                metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("rmse on test data = %g" % rmse)
evaluator = RegressionEvaluator(labelCol="label",
                                predictionCol="prediction",
                                metricName="r2")
r2 = evaluator.evaluate(predictions)
print("Rsquared on test data = %g" % r2)


# In[2]:




