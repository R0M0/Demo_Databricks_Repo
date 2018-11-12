# Databricks notebook source
# Please note your SAS_url link will be different
SAS_url = 'https://bloblmlinaday.blob.core.windows.net/churn/Telco_Customer_Churn_Cluster.csv?sp=r&st=2018-09-04T01:11:16Z&se=2028-09-04T09:11:16Z&spr=https&sv=2017-11-09&sig=hiDEdQdQjhSYaqlrgusEz1FZZc3it7vHAz%2B98TgANXg%3D&sr=b'

# COMMAND ----------

import pandas as pd
pandas_df = pd.read_csv(SAS_url)

# COMMAND ----------

df = spark.createDataFrame(pandas_df)

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.groupBy('PaymentMethod').sum('TotalCharges'))

# COMMAND ----------

df.createTempView('sqlView')

# COMMAND ----------

# MAGIC %sql
# MAGIC select PaymentMethod, sum(TotalCharges) from sqlView group by 1 order by 1 asc;

# COMMAND ----------

display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.groupBy('Churn').count())

# COMMAND ----------

# MAGIC %sql
# MAGIC select Assignments, count(*) as Total_Count from sqlView group by 1 order by 2 desc;

# COMMAND ----------

from pyspark.sql import functions as F
df = df.withColumn('Churn', F.when(df['Churn']=='Yes', 1).otherwise(0))

# COMMAND ----------

categorical_variables = [i[0] for i in df.dtypes if i[1] == 'string']

# COMMAND ----------

categorical_variables

# COMMAND ----------

categorical_variables = categorical_variables[1:]
categorical_variables

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
indexers = []
for categoricalCol in categorical_variables:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"_numeric")
    indexers += [stringIndexer]

# COMMAND ----------

models = []
for model in indexers:
    indexer_model = model.fit(df)
    models+=[indexer_model]
    
for i in models:
    df = i.transform(df)

# COMMAND ----------

df.printSchema

# COMMAND ----------

display(df.select('PaymentMethod', 'PaymentMethod_numeric'))

# COMMAND ----------

from pyspark.sql.functions import min, max

minMonthly = df.agg(min(df.MonthlyCharges)).head()[0]
maxMonthly = df.agg(max(df.MonthlyCharges)).head()[0]

minTotal = df.agg(min(df.TotalCharges)).head()[0]
maxTotal = df.agg(max(df.TotalCharges)).head()[0]


df = df.withColumn('MonthlyCharges_Normalized', (df.MonthlyCharges - minMonthly)/(maxMonthly-minMonthly))
df = df.withColumn('TotalCharges_Normalized', (df.TotalCharges - minTotal)/(maxTotal-minTotal))


# COMMAND ----------

display(df.select('MonthlyCharges', 'MonthlyCharges_Normalized'))

# COMMAND ----------

features = [
 'Assignments',
 
 'gender_numeric',
 'Partner_numeric',
 'Dependents_numeric',
 'PhoneService_numeric',
 'MultipleLines_numeric',
 'InternetService_numeric',
 'OnlineSecurity_numeric',
 'OnlineBackup_numeric',
 'DeviceProtection_numeric',
 'TechSupport_numeric',
 'StreamingTV_numeric',
 'StreamingMovies_numeric',
 'Contract_numeric',
 'PaperlessBilling_numeric',
 'PaymentMethod_numeric', 
 
 'MonthlyCharges_Normalized',
 'TotalCharges_Normalized', 
 
 'SeniorCitizen', 
 'tenure'
]

# COMMAND ----------

features

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

feature_vectors = VectorAssembler(
        inputCols = features,
        outputCol = "features")

# COMMAND ----------

df = feature_vectors.transform(df)

# COMMAND ----------

display(df.select('features'))

# COMMAND ----------

display(df.select('Churn', 'features'))

# COMMAND ----------

df = df.withColumnRenamed('Churn', 'label')

# COMMAND ----------

trainDF, testDF = df.randomSplit([0.8, 0.2], seed = 1234)

# COMMAND ----------

print('training data: '+str(trainDF.count()), ', testing data: '+str(testDF.count()))

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
logreg = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
LogisticRegressionModel = logreg.fit(trainDF)

# COMMAND ----------

predictedDF = LogisticRegressionModel.transform(testDF)

# COMMAND ----------

display(predictedDF.select('label', 'probability', 'prediction'))

# COMMAND ----------

display(predictedDF.crosstab('label', 'prediction'))

# COMMAND ----------

from sklearn import metrics

actual = predictedDF.select('label').toPandas()
predicted = predictedDF.select('prediction').toPandas()
print('accuracy score = {}'.format(metrics.accuracy_score(actual, predicted)*100))

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
print('ROC score = {}'.format(round(evaluator.evaluate(predictedDF)*100,2)))

# COMMAND ----------

predictedDF.createOrReplaceTempView('PredictionScoresDF')

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- delete from PredictionScore;
# MAGIC create table PredictionScore as
# MAGIC select * from PredictionScoresDF;

# COMMAND ----------

