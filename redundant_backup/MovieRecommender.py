# Databricks notebook source
# DBTITLE 1,Importing Packages
# Computational and Visualisation Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pyspark Packages
from pyspark.sql import functions as F
from pyspark.sql.functions import col, desc
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

# COMMAND ----------

# DBTITLE 1,Data Ingestion and Pre-processing
user_data = spark.read.option('sep', '\t').csv('/mnt/ml-100k/u.data')
user_data = user_data.select(col('_c0').cast('int').alias('user_id'), col('_c1').cast('int').alias('item_id'), \
                             col('_c2').cast('int').alias('rating'), col('_c3').alias('timestamp').cast('bigint'))
user_data = user_data \
            .withColumn('date_f', F.to_timestamp(F.from_unixtime(col('timestamp'), 'dd-MM-yyyy HH:mm:ss'), 'dd-MM-yyyy HH:mm:ss'))\
            .withColumn('date_s', F.to_date(F.from_unixtime(col('timestamp'), 'yyyyMMdd'), 'yyyyMMdd'))

# Splitting the data into training and testing set
train, test = user_data.randomSplit([.8, .2])

display(user_data.sample(False, 0.1), 100)

# COMMAND ----------

# DBTITLE 1,Summary Statistics on the filtered data
display (user_data.describe())

# COMMAND ----------

# DBTITLE 1,Review count across ratings
display(user_data.groupBy('rating').agg(F.count(F.lit(1)).alias('Total Ratings')))

# COMMAND ----------

# DBTITLE 1,Review count across calendar days
display(user_data.groupBy('date_s').agg(F.count(F.lit(1)).alias('Total Ratings')))

# COMMAND ----------

# DBTITLE 1,Top 40 most rated movies
display(user_data.groupBy('item_id').agg(F.count(F.lit(1)).alias('Count of Recommendation')).sort(desc('Count of Recommendation')).limit(40))

# COMMAND ----------

# DBTITLE 1,Activity count of top 40 users
display(user_data.groupBy('user_id').agg(F.count(F.lit(1)).alias('Count of Recommendation')).sort(desc('Count of Recommendation')).limit(40))

# COMMAND ----------

# DBTITLE 1,Collaborative Filtering
# ALS Model Hyperparameter values were separately computed for the best model
movie_recommender_inst = ALS(maxIter=28, regParam=0.1, userCol="user_id", itemCol="item_id", ratingCol="rating", coldStartStrategy="drop", implicitPrefs=True)
movie_recommender_model = movie_recommender_inst.fit(train)

computed_predictions = movie_recommender_model.transform(test)
reg_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse_model = reg_evaluator.evaluate(computed_predictions)

print ("Computed Root-mean-square error = ", rmse_model)

# COMMAND ----------

# DBTITLE 1,Understanding the Predictions Data
display (computed_predictions, 250)

# COMMAND ----------

display (computed_predictions.describe())

# COMMAND ----------

# DBTITLE 1,Generating top 5 movies to be recommended for each user
computed_user_recommendations = movie_recommender_model.recommendForAllUsers(10)
display (computed_user_recommendations.limit(200))

# COMMAND ----------

# DBTITLE 1,Generating top 15 users for each movie
computed_movie_recommendations = movie_recommender_model.recommendForAllItems(15)
display(computed_movie_recommendations.limit(200))

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC Dataset has been procured from *https://grouplens.org/datasets/movielens/*
# MAGIC 
# MAGIC The ALS Classifier indicated least RMSE for 28 iterations and 0.1 regParam value. Overall the classifier offers excellent approach to evaluate, build, and productionalize production-grade collaborative filtering based recommendation systems. The published notebook is available at - https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3173713035751393/3658598530030623/2308983777460038/latest.html
