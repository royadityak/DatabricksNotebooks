# Databricks notebook source
# Computational and Visualisation Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *

# Pyspark Packages
from pyspark.sql import functions as F
from pyspark.sql.functions import col, desc, trim
from pyspark.sql.types import *
from pyspark.ml import Pipeline

# COMMAND ----------

# DBTITLE 1,Auxiliary Functions
def return_institution_type(institutional_sector): return 'Private' if (institutional_sector.startswith('Private')) else 'Public'
def return_profit_type(institutional_sector): return "Not for Profit" if ('not-for-profit' in institutional_sector) else 'For Profit'

udf_institutional_type = udf(return_institution_type, StringType())
udf_profit_type = udf(return_profit_type, StringType())

# COMMAND ----------

univ_data = spark.sql("SELECT * FROM ipeds_data_raw")
postal_code = spark.sql("SELECT * FROM us_postal_codes")
univ_abbreviations = spark.sql("SELECT * FROM university_abbreviations")
pre_final_col_list = univ_data.columns + ['postal']
univ_data_expanded_pre = univ_data.join(postal_code, univ_data.fips_state_code==postal_code.state,how='leftouter')[pre_final_col_list]

# Adding columns for institutional_type and profit_type
univ_data_expanded_pre = univ_data_expanded_pre.withColumn('institutional_type', udf_institutional_type(col('institutional_sector')))
univ_data_expanded_pre = univ_data_expanded_pre.withColumn('profit_type', udf_profit_type(col('institutional_sector')))

# Adding abreviations (if available for each university)
final_col_list = univ_data_expanded_pre.columns + ['university_abbreviations']
univ_data_expanded = univ_data_expanded_pre.join(univ_abbreviations, univ_data_expanded_pre.name == univ_abbreviations.university_name, how='leftouter')[final_col_list]
univ_data_expanded.createOrReplaceTempView('univ_data_expanded')
display (univ_data_expanded, 100)

# COMMAND ----------

display (univ_data_expanded.describe())

# COMMAND ----------

# MAGIC %sql
# MAGIC --Universities by State
# MAGIC SELECT fips_state_code, count(*) AS count_of_universities
# MAGIC FROM univ_data_expanded
# MAGIC GROUP BY fips_state_code
# MAGIC ORDER BY count_of_universities DESC, fips_state_code DESC

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Most tuition rates across the States
# MAGIC SELECT postal, SUM(tuition_2010_11)  FROM univ_data_expanded
# MAGIC GROUP BY postal

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Colleges by sector
# MAGIC with filtered_query as (
# MAGIC   SELECT 
# MAGIC   fips_state_code, 
# MAGIC   CASE WHEN institutional_sector LIKE 'Private%' THEN 'Private' ELSE 'Public' END AS institutional_type
# MAGIC   FROM univ_data_expanded
# MAGIC )
# MAGIC SELECT fips_state_code AS States, institutional_type, count(*) AS count_total
# MAGIC FROM filtered_query
# MAGIC GROUP BY fips_state_code, institutional_type
# MAGIC ORDER BY count_total DESC, fips_state_code DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Colleges by geographical region
# MAGIC SELECT geographic_region, SUM(endowment_assets_gasb) AS total_endowment_gasb, SUM(endowment_assets_fasb) AS total_endowment_fasb
# MAGIC FROM univ_data_expanded
# MAGIC GROUP BY geographic_region
# MAGIC HAVING (total_endowment_gasb > 0 and total_endowment_fasb > 0)
# MAGIC ORDER BY total_endowment_gasb DESC, total_endowment_fasb DESC

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Identifying top historically black colleges on account of enrollment by regions
# MAGIC SELECT fips_state_code AS States, geographic_region AS Regions, MAX(enrollment_total) AS enrollment_total
# MAGIC FROM univ_data_expanded
# MAGIC WHERE historical_black_institution == 'Yes'
# MAGIC GROUP BY geographic_region, fips_state_code
# MAGIC HAVING enrollment_total > 2000

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Colleges by Urbanization Degree
# MAGIC SELECT urban_centric_locale AS Urbanization_Type, institutional_type, count(*) AS count_institutions
# MAGIC FROM univ_data_expanded
# MAGIC GROUP BY urban_centric_locale, institutional_type
# MAGIC ORDER BY count_institutions DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Colleges by Carnegie Classification
# MAGIC SELECT carnegie_classification_2010 AS Carnegie_Classification, profit_type, count(*) AS count_institutions
# MAGIC FROM univ_data_expanded
# MAGIC GROUP BY carnegie_classification_2010, profit_type
# MAGIC ORDER BY count_institutions DESC

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Full-time vs Part-time enrollment for top full-time enrollment institutions
# MAGIC with ft_inst as (
# MAGIC   SELECT name AS institution_name, zip, SUM(fulltime_estimated_enrollment) AS fulltime_enrollment
# MAGIC   FROM univ_data_expanded
# MAGIC   GROUP BY name, zip
# MAGIC   ORDER BY fulltime_enrollment DESC
# MAGIC   LIMIT 30
# MAGIC )
# MAGIC SELECT institution_name, fulltime_enrollment, parttime_estimated_enrollment AS parttime_enrollment
# MAGIC FROM ft_inst
# MAGIC INNER JOIN univ_data_expanded
# MAGIC ON (ft_inst.institution_name=univ_data_expanded.name and ft_inst.zip=univ_data_expanded.zip)
# MAGIC WHERE parttime_estimated_enrollment > 0
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Average full-time and part-time enrollment by state
# MAGIC SELECT fips_state_code AS States, AVG(fulltime_estimated_enrollment) as avg_fulltime_enrollment, AVG(parttime_estimated_enrollment) as avg_parttime_enrollment
# MAGIC FROM univ_data_expanded
# MAGIC GROUP BY fips_state_code

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Percent of college enrollment graduates and against race/ethinicity
# MAGIC with races_demographic_view as (
# MAGIC   SELECT postal, fips_state_code, 'American_Indian_Alaska' AS race, SUM(enrollment_graduate*percent_grad_american_indian_alaska) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
# MAGIC   UNION ALL SELECT postal, fips_state_code, 'Asian' AS race, SUM(enrollment_graduate*percent_grad_asian) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
# MAGIC   UNION ALL SELECT postal, fips_state_code, 'African_American' AS race, SUM(enrollment_graduate*percent_grad_african_american) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
# MAGIC   UNION ALL SELECT postal, fips_state_code, 'Hispanic_Latino' AS race, SUM(enrollment_graduate*percent_grad_hispanic_latino) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
# MAGIC   UNION ALL SELECT postal, fips_state_code, 'Hawaiian_Pacific' AS race, SUM(enrollment_graduate*percent_grad_hawaiian_pacific_islander) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
# MAGIC   UNION ALL SELECT postal, fips_state_code, 'White' AS race, SUM(enrollment_graduate*percent_grad_white) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
# MAGIC   UNION ALL SELECT postal, fips_state_code, 'Multi_Racial' AS race, SUM(enrollment_graduate*percent_grad_multi_racial) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
# MAGIC   UNION ALL SELECT postal, fips_state_code, 'Unknown' AS race, SUM(enrollment_graduate*percent_grad_nonresident_alien) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
# MAGIC )
# MAGIC SELECT postal, race, SUM(total_grad_enrollment)
# MAGIC FROM races_demographic_view
# MAGIC GROUP BY postal, race

# COMMAND ----------

# MAGIC %sql
# MAGIC -- College count by religious affiliations
# MAGIC SELECT religion, count(*) AS total_institutions
# MAGIC FROM univ_data_expanded
# MAGIC WHERE religion != 'Not applicable'
# MAGIC GROUP BY religion

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Plot of 4 years tuition fees by sector
# MAGIC with tuition_rates_view AS (
# MAGIC   SELECT '2010-11' AS year, AVG(tuition_2010_11) AS avg_tuition FROM univ_data_expanded
# MAGIC   UNION ALL SELECT '2011-12' AS year, AVG(tuition_2011_12) AS avg_tuition FROM univ_data_expanded
# MAGIC   UNION ALL SELECT '2012-13' AS year, AVG(tuition_2012_13) AS avg_tuition FROM univ_data_expanded
# MAGIC   UNION ALL SELECT '2013-14' AS year, AVG(tuition_2013_14) AS avg_tuition FROM univ_data_expanded
# MAGIC )
# MAGIC SELECT year, avg_tuition
# MAGIC FROM tuition_rates_view

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Top 30 Universities with higher difference between in-state and out-state tuition for the year 2013-14
# MAGIC with view_1 AS (
# MAGIC   SELECT name, university_abbreviations FROM univ_data_expanded
# MAGIC ),
# MAGIC view_2 AS (
# MAGIC   SELECT 
# MAGIC     name, 
# MAGIC     SUM(tuition_outstate_2013_14) AS outstate_tuition, 
# MAGIC     SUM(tuition_instate_2013_14) AS instate_tuition, 
# MAGIC     SUM(tuition_outstate_2013_14 - tuition_instate_2013_14) AS tuition_difference
# MAGIC   FROM univ_data_expanded
# MAGIC   GROUP BY name, zip
# MAGIC   HAVING tuition_difference > 0
# MAGIC   ORDER BY tuition_difference DESC
# MAGIC   LIMIT 30
# MAGIC )
# MAGIC SELECT CASE WHEN university_abbreviations IS NOT NULL THEN university_abbreviations ELSE view_2.name END AS name, outstate_tuition, instate_tuition, tuition_difference
# MAGIC FROM view_2 LEFT OUTER JOIN view_1
# MAGIC ON (view_2.name = view_1.name)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Top 30 Universities with lower difference between in-state and out-state tuition for the year 2013-14
# MAGIC with view_1 AS (
# MAGIC   SELECT name, university_abbreviations FROM univ_data_expanded WHERE university_abbreviations IS NOT NULL
# MAGIC ),
# MAGIC view_2 AS (
# MAGIC   SELECT 
# MAGIC     name, 
# MAGIC     SUM(tuition_outstate_2013_14) AS outstate_tuition, 
# MAGIC     SUM(tuition_instate_2013_14) AS instate_tuition, 
# MAGIC     SUM(tuition_outstate_2013_14 - tuition_instate_2013_14) AS tuition_difference
# MAGIC   FROM univ_data_expanded
# MAGIC   GROUP BY name, zip
# MAGIC   HAVING tuition_difference > 0
# MAGIC   ORDER BY tuition_difference ASC
# MAGIC   LIMIT 30
# MAGIC )
# MAGIC SELECT CASE WHEN university_abbreviations IS NOT NULL THEN university_abbreviations ELSE view_2.name END AS name, outstate_tuition, instate_tuition, tuition_difference
# MAGIC FROM view_2 LEFT OUTER JOIN view_1
# MAGIC ON (view_2.name = view_1.name)

# COMMAND ----------

# MAGIC %sql 
# MAGIC --Top 25 university based on sum of endowment assets
# MAGIC WITH uni_abbreviation_view AS (
# MAGIC   SELECT name, university_abbreviations FROM univ_data_expanded WHERE university_abbreviations IS NOT NULL
# MAGIC ),
# MAGIC endowment_view AS (
# MAGIC   SELECT name, 
# MAGIC   CASE WHEN SUM(endowment_assets_gasb) IS NULL THEN SUM(endowment_assets_fasb) ELSE SUM(endowment_assets_gasb) END AS total_endowments
# MAGIC   FROM univ_data_expanded
# MAGIC   GROUP BY name, zip
# MAGIC   HAVING total_endowments > 0
# MAGIC )
# MAGIC SELECT 
# MAGIC   CASE WHEN university_abbreviations IS NOT NULL THEN university_abbreviations ELSE endowment_view.name END AS name, 
# MAGIC   total_endowments
# MAGIC FROM endowment_view LEFT OUTER JOIN uni_abbreviation_view
# MAGIC ON (endowment_view.name = uni_abbreviation_view.name)
# MAGIC ORDER BY total_endowments DESC
# MAGIC LIMIT 25

# COMMAND ----------

# MAGIC %sql 
# MAGIC --Bottom 25 university based on sum of endowment assets
# MAGIC WITH uni_abbreviation_view AS (
# MAGIC   SELECT name, university_abbreviations FROM univ_data_expanded WHERE university_abbreviations IS NOT NULL
# MAGIC ),
# MAGIC endowment_view AS (
# MAGIC   SELECT name, 
# MAGIC   CASE WHEN SUM(endowment_assets_gasb) IS NULL THEN SUM(endowment_assets_fasb) ELSE SUM(endowment_assets_gasb) END AS total_endowments
# MAGIC   FROM univ_data_expanded
# MAGIC   GROUP BY name, zip
# MAGIC   HAVING total_endowments > 0
# MAGIC )
# MAGIC SELECT 
# MAGIC   CASE WHEN university_abbreviations IS NOT NULL THEN university_abbreviations ELSE endowment_view.name END AS name, 
# MAGIC   total_endowments
# MAGIC FROM endowment_view LEFT OUTER JOIN uni_abbreviation_view
# MAGIC ON (endowment_view.name = uni_abbreviation_view.name)
# MAGIC ORDER BY total_endowments ASC
# MAGIC LIMIT 30
