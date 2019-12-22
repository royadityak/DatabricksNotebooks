# Databricks notebook source
# Computational and Visualisation Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *

# Pyspark Packages
from pyspark.sql import functions as F
from pyspark.sql.functions import col, desc
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# DBTITLE 1,Loading Data
us_tax = spark.read.csv('/databricks-datasets/data.gov/irs_zip_code_data/data-001/2013_soi_zipcode_agi.csv', header=True)
farmer_market = spark.read.csv('/databricks-datasets/data.gov/farmers_markets_geographic_data/data-001/market_data.csv', header=True)

# COMMAND ----------

# DBTITLE 1,Sample of Tax Data
display (us_tax.sample(False, 0.3), 100)

# COMMAND ----------

# DBTITLE 1,Sample of Farmer market data
display (farmer_market.sample(False, 0.3), 100)

# COMMAND ----------

# DBTITLE 1,Descriptive Statistics on US Tax Data
display (us_tax.describe())

# COMMAND ----------

# DBTITLE 1,Descriptive Statistics on Farmer Market Data
display (farmer_market.describe())

# COMMAND ----------

# DBTITLE 1,Taxable interest for each state
us_tax_filtered = us_tax\
    .withColumn("zipcode", (col('zipcode')/10).cast('int'))\
    .withColumn("single_returns", col('mars1').cast('int'))\
    .withColumn("joint_returns", col('mars2').cast('int'))\
    .withColumn("numdep", col('numdep').cast('int'))\
    .withColumn("total_income_amount", col('A02650').cast('double'))\
    .withColumn("taxable_interest_amount", col('A00300').cast('double'))\
    .withColumn("net_capital_gains", col('a01000').cast('double'))\
    .withColumn("biz_net_income", col('a00900').cast('double'))

us_tax_filtered = us_tax_filtered [['state', 'zipcode', 'single_returns', 'joint_returns', 'numdep', 'total_income_amount', 'taxable_interest_amount', 'net_capital_gains', 'biz_net_income']]
us_tax_filtered.createOrReplaceTempView('us_tax_filtered')

# Average Taxable Interest across the US States
display (us_tax_filtered)

# COMMAND ----------

# DBTITLE 1,Capital Gains for top 40 zip
display(us_tax_filtered.filter('zipcode NOT IN (0000, 9999)').groupby('zipcode').agg(F.sum('net_capital_gains').alias('capital_gains'))\
        .sort(desc('capital_gains')).limit(40))

# COMMAND ----------

# DBTITLE 1,Gains for top 40 zip
display(us_tax_filtered.filter('zipcode NOT IN (0000, 9999)').groupby('zipcode').agg(F.sum('biz_net_income').alias('business_net_income'), \
     F.sum('net_capital_gains').alias('capital_gains'), (F.sum('net_capital_gains') + F.sum('biz_net_income')).alias('capital_and_business_income'))\
      .sort(desc('capital_and_business_income')).limit(40))

# COMMAND ----------

# DBTITLE 1,Market Counts per State
display(farmer_market.groupBy('state').agg(F.count(F.lit(1)).alias('Total Markets')))

# COMMAND ----------

# DBTITLE 1,Data Preparation for Modeling
us_tax_cumm = us_tax_filtered.groupBy('zipcode').sum()
farmer_market_cleaned = farmer_market.withColumn("zipcode", (col("zip")/10)).groupby('zipcode').count()\
                        .select(col('count').cast('double').alias('count'), col('zipcode').alias('zip'))
us_tax_expanded = farmer_market_cleaned.join(us_tax_cumm, farmer_market_cleaned.zip == us_tax_cumm.zipcode, how='outer').na.fill(0)
us_tax_expanded = us_tax_expanded.select(col('count'), col('zip'), col('zipcode'), col('sum(zipcode)').alias('sum_zipcode'), col('sum(single_returns)').alias('sum_single_returns'), col('sum(joint_returns)').alias('sum_joint_returns'), col('sum(numdep)').alias('sum_numdep'), col('sum(total_income_amount)').alias('sum_total_income_amount'), col('sum(taxable_interest_amount)').alias('sum_taxable_interest_amount'), col('sum(net_capital_gains)').alias('sum_net_capital_gains'), col('sum(biz_net_income)').alias('sum_biz_net_income'))

# Bringing featured columns as a single columns
feature_columns = ['sum_zipcode', 'sum_single_returns', 'sum_joint_returns', 'sum_numdep', 'sum_total_income_amount', 'sum_taxable_interest_amount', 'sum_net_capital_gains', 'sum_biz_net_income']
us_tax_assembler_model = VectorAssembler(inputCols=feature_columns, outputCol='features')
us_tax_expanded_prepared = us_tax_assembler_model.transform(us_tax_expanded)

# Splitting the data into training and testing set
train, test = us_tax_expanded_prepared.randomSplit([.65, .25])

# COMMAND ----------

# DBTITLE 1,Linear Regression Model
lrReg = LinearRegression (maxIter=50, regParam=0.3, labelCol='count', elasticNetParam=0.4)
lrModel = lrReg.fit (train)
lrModelSummary = lrModel.summary
print ("Computed Coefficients = ", lrModel.coefficients)
print ("Computed Intercepts = ", lrModel.intercept)
print ("Objective History = ", lrModelSummary.objectiveHistory)
print ("Mean Absolute Error, MAE = ", lrModelSummary.meanAbsoluteError)
print ("RMSE = {0}".format(lrModelSummary.rootMeanSquaredError))
print ("R^2 = {0}".format(lrModelSummary.r2))

# COMMAND ----------

print (lrModel.explainParams())

# COMMAND ----------

# DBTITLE 1,Plot of Fitted v/s Residual
display (lrModel, test, "fittedVsResiduals")

# COMMAND ----------

# DBTITLE 1,Linear Regression Residuals Plot
display (lrModelSummary.residuals)

# COMMAND ----------

# DBTITLE 1,Testing the Linear Regression Model on the hold-out sample
lrPredictions = lrModel.transform(test)
lrEvaluator = RegressionEvaluator(labelCol="count", predictionCol="prediction", metricName="rmse")
rmse = lrEvaluator.evaluate(lrPredictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# COMMAND ----------

display(lrPredictions, 1000)

# COMMAND ----------

display(lrPredictions)

# COMMAND ----------

# DBTITLE 1,Generalized Linear Regression
# MAGIC %md
# MAGIC Contrasted with linear regression where the output is assumed to follow a Gaussian distribution, generalized linear models (GLMs) are specifications of linear models where the response variable Yi follows some distribution from the exponential family of distributions. Spark’s GeneralizedLinearRegression interface allows for flexible specification of GLMs which can be used for various types of prediction problems including linear regression, Poisson regression, logistic regression, and others. 
# MAGIC Reference: https://spark.apache.org/docs/2.2.0/ml-classification-regression.html#generalized-linear-regression

# COMMAND ----------

# DBTITLE 1,Generalized Linear Regression
glrReg = GeneralizedLinearRegression (family='gaussian', link='identity', maxIter=50, regParam=0.3, labelCol='count')
glrModel = glrReg.fit (train)
glrModelSummary = glrModel.summary
print ("GLR Computed Coefficients = ", glrModel.coefficients)
print ("GLR Computed Intercept = ", glrModel.intercept)
print ("T Values = ", glrModelSummary.tValues)
print ("P Values = ", glrModelSummary.pValues)
print ("GLR Coefficient of Standard Errors = ", glrModelSummary.coefficientStandardErrors)
print ("Dispersion = ", glrModelSummary.dispersion)
print ("Deviance = ", glrModelSummary.deviance)
print ("Null Deviance = ", glrModelSummary.nullDeviance)
print ("Residual Degree of Freedom Null = ", glrModelSummary.residualDegreeOfFreedomNull)
print ("AIC = ", glrModelSummary.aic)

# COMMAND ----------

# DBTITLE 1,Plot of Residuals
display (glrModelSummary.residuals())

# COMMAND ----------

# DBTITLE 1,Prediction on the holdout sample
glrPredictions = glrModel.transform(test)
glrEvaluator = RegressionEvaluator(labelCol="count", predictionCol="prediction", metricName="rmse")
rmse_glr = glrEvaluator.evaluate(glrPredictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse_glr)

# COMMAND ----------

display (glrPredictions, 1000)

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC 
# MAGIC We have used the publicly available dataset from the US Government department, and we build our ***Linear Regression (LR)*** and ***Generalized Linear Regression (GLR)*** model on top of it, which we have then used to make predictions on our hold-out or test sample. The RMSE scores were improved from LR to GLR. In the later versions, I plan to include Random Forest and Gradient Boosted Tree on the same dataset for more advanced Data Science use-case. We also realized that we need to evaluate hyper-parameters to achieve the best model. 
# MAGIC The published notebook is available at - *https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3173713035751393/175225107118479/2308983777460038/latest.html*
