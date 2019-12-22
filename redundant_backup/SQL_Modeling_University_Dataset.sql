-- Databricks notebook source
-- DBTITLE 1,Introduction
-- MAGIC %md
-- MAGIC <b> <i> Data Sources - </b> </i> 
-- MAGIC <ul>
-- MAGIC   <li> <i> IPEDS Dataset - </i> https://nces.ed.gov/ipeds/use-the-data </li>
-- MAGIC   <li> <i> Postal Codes - </i> http://www.geonames.org/postal-codes/postal-codes-us.html </li>
-- MAGIC   <li> <i> US University Abbreviations - </i> https://en.wikipedia.org/wiki/List_of_colloquial_names_for_universities_and_colleges_in_the_United_States </li>
-- MAGIC </ul>
-- MAGIC <hr/>
-- MAGIC Implemented and Explored SQL Aggregations on the dataset.

-- COMMAND ----------

-- DBTITLE 1,Required Packages
-- MAGIC %py
-- MAGIC from pyspark.sql.types import StringType
-- MAGIC from pyspark.sql.functions import col

-- COMMAND ----------

-- DBTITLE 1,Auxiliary Functions
-- MAGIC %py
-- MAGIC def return_institution_type(institutional_sector): return 'Private' if (institutional_sector.startswith('Private')) else 'Public'
-- MAGIC def return_profit_type(institutional_sector): return "Not for Profit" if ('not-for-profit' in institutional_sector) else 'For Profit'
-- MAGIC 
-- MAGIC udf_institutional_type = udf(return_institution_type, StringType())
-- MAGIC udf_profit_type = udf(return_profit_type, StringType())

-- COMMAND ----------

-- DBTITLE 1,Data Import and Preprocessing
-- MAGIC %py
-- MAGIC univ_data = spark.sql("SELECT * FROM ipeds_data_raw")
-- MAGIC postal_code = spark.sql("SELECT * FROM us_postal_codes")
-- MAGIC univ_abbreviations = spark.sql("SELECT * FROM university_abbreviations")
-- MAGIC pre_final_col_list = univ_data.columns + ['postal']
-- MAGIC univ_data_expanded_pre = univ_data.join(postal_code, univ_data.fips_state_code==postal_code.state,how='leftouter')[pre_final_col_list]
-- MAGIC 
-- MAGIC # Adding columns for institutional_type and profit_type
-- MAGIC univ_data_expanded_pre = univ_data_expanded_pre.withColumn('institutional_type', udf_institutional_type(col('institutional_sector')))
-- MAGIC univ_data_expanded_pre = univ_data_expanded_pre.withColumn('profit_type', udf_profit_type(col('institutional_sector')))
-- MAGIC 
-- MAGIC # Adding abreviations (if available for each university)
-- MAGIC final_col_list = univ_data_expanded_pre.columns + ['university_abbreviations']
-- MAGIC univ_data_expanded = univ_data_expanded_pre.join(univ_abbreviations, univ_data_expanded_pre.name == univ_abbreviations.university_name, how='leftouter')[final_col_list]
-- MAGIC univ_data_expanded.createOrReplaceTempView('univ_data_expanded')
-- MAGIC display (univ_data_expanded, 100)

-- COMMAND ----------

-- DBTITLE 1,Summary Statistics on the data
-- MAGIC %py
-- MAGIC display (univ_data_expanded.describe())

-- COMMAND ----------

-- DBTITLE 1,Universities by State
SELECT fips_state_code, count(*) AS count_of_universities
FROM univ_data_expanded
GROUP BY fips_state_code
ORDER BY count_of_universities DESC, fips_state_code DESC

-- COMMAND ----------

-- DBTITLE 1,Most tuition rates across the States
SELECT postal, SUM(tuition_2010_11)  FROM univ_data_expanded
GROUP BY postal

-- COMMAND ----------

-- DBTITLE 1,Colleges by sector
with filtered_query as (
  SELECT 
  fips_state_code, 
  CASE WHEN institutional_sector LIKE 'Private%' THEN 'Private' ELSE 'Public' END AS institutional_type
  FROM univ_data_expanded
)
SELECT fips_state_code AS States, institutional_type, count(*) AS count_total
FROM filtered_query
GROUP BY fips_state_code, institutional_type
ORDER BY count_total DESC, fips_state_code DESC

-- COMMAND ----------

-- DBTITLE 1,Colleges by geographical region
SELECT geographic_region, SUM(endowment_assets_gasb) AS total_endowment_gasb, SUM(endowment_assets_fasb) AS total_endowment_fasb
FROM univ_data_expanded
GROUP BY geographic_region
HAVING (total_endowment_gasb > 0 and total_endowment_fasb > 0)
ORDER BY total_endowment_gasb DESC, total_endowment_fasb DESC

-- COMMAND ----------

-- DBTITLE 1,Identifying top historically black colleges on account of enrollment by regions
SELECT fips_state_code AS States, geographic_region AS Regions, MAX(enrollment_total) AS enrollment_total
FROM univ_data_expanded
WHERE historical_black_institution == 'Yes'
GROUP BY geographic_region, fips_state_code
HAVING enrollment_total > 2000

-- COMMAND ----------

-- DBTITLE 1,Colleges by Urbanization degree
SELECT urban_centric_locale AS Urbanization_Type, institutional_type, count(*) AS count_institutions
FROM univ_data_expanded
GROUP BY urban_centric_locale, institutional_type
ORDER BY count_institutions DESC

-- COMMAND ----------

-- DBTITLE 1,Colleges by Carnegie classification
SELECT carnegie_classification_2010 AS Carnegie_Classification, profit_type, count(*) AS count_institutions
FROM univ_data_expanded
GROUP BY carnegie_classification_2010, profit_type
ORDER BY count_institutions DESC

-- COMMAND ----------

-- DBTITLE 1,Full-time vs Part-time enrollment for top full-time enrollment institutions
with ft_inst as (
  SELECT name AS institution_name, zip, SUM(fulltime_estimated_enrollment) AS fulltime_enrollment
  FROM univ_data_expanded
  GROUP BY name, zip
  ORDER BY fulltime_enrollment DESC
  LIMIT 30
),
uni_abbreviation_view AS (
  SELECT name, university_abbreviations FROM univ_data_expanded WHERE university_abbreviations IS NOT NULL
),
ft_inst_expanded_view AS (
  SELECT 
    institution_name,
    zip,
    fulltime_enrollment,
    CASE WHEN name IS NOT NULL THEN name ELSE institution_name END AS institution_abbreviation
  FROM ft_inst LEFT OUTER JOIN uni_abbreviation_view
  ON (ft_inst.institution_name=uni_abbreviation_view.name)
)
SELECT institution_abbreviation, fulltime_enrollment, parttime_estimated_enrollment AS parttime_enrollment
FROM ft_inst_expanded_view
INNER JOIN univ_data_expanded
ON (ft_inst_expanded_view.institution_name=univ_data_expanded.name and ft_inst_expanded_view.zip=univ_data_expanded.zip)
WHERE parttime_estimated_enrollment > 0
LIMIT 12

-- COMMAND ----------

-- DBTITLE 1,Average full-time and part-time enrollment by state
SELECT fips_state_code AS States, AVG(fulltime_estimated_enrollment) as avg_fulltime_enrollment, AVG(parttime_estimated_enrollment) as avg_parttime_enrollment
FROM univ_data_expanded
GROUP BY fips_state_code

-- COMMAND ----------

-- DBTITLE 1,Percent of college enrollment graduates and against race/ethnicity
with races_demographic_view as (
  SELECT postal, fips_state_code, 'American_Indian_Alaska' AS race, SUM(enrollment_graduate*percent_grad_american_indian_alaska) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
  UNION ALL SELECT postal, fips_state_code, 'Asian' AS race, SUM(enrollment_graduate*percent_grad_asian) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
  UNION ALL SELECT postal, fips_state_code, 'African_American' AS race, SUM(enrollment_graduate*percent_grad_african_american) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
  UNION ALL SELECT postal, fips_state_code, 'Hispanic_Latino' AS race, SUM(enrollment_graduate*percent_grad_hispanic_latino) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
  UNION ALL SELECT postal, fips_state_code, 'Hawaiian_Pacific' AS race, SUM(enrollment_graduate*percent_grad_hawaiian_pacific_islander) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
  UNION ALL SELECT postal, fips_state_code, 'White' AS race, SUM(enrollment_graduate*percent_grad_white) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
  UNION ALL SELECT postal, fips_state_code, 'Multi_Racial' AS race, SUM(enrollment_graduate*percent_grad_multi_racial) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
  UNION ALL SELECT postal, fips_state_code, 'Unknown' AS race, SUM(enrollment_graduate*percent_grad_nonresident_alien) AS total_grad_enrollment FROM univ_data_expanded GROUP BY postal, fips_state_code
)
SELECT postal, race, SUM(total_grad_enrollment)
FROM races_demographic_view
GROUP BY postal, race

-- COMMAND ----------

-- DBTITLE 1,College count by religious affiliations
SELECT religion, count(*) AS total_institutions
FROM univ_data_expanded
WHERE religion != 'Not applicable'
GROUP BY religion

-- COMMAND ----------

-- DBTITLE 1,Plot of 4 years tuition fees by sector
with tuition_rates_view AS (
  SELECT '2010-11' AS year, AVG(tuition_2010_11) AS avg_tuition FROM univ_data_expanded
  UNION ALL SELECT '2011-12' AS year, AVG(tuition_2011_12) AS avg_tuition FROM univ_data_expanded
  UNION ALL SELECT '2012-13' AS year, AVG(tuition_2012_13) AS avg_tuition FROM univ_data_expanded
  UNION ALL SELECT '2013-14' AS year, AVG(tuition_2013_14) AS avg_tuition FROM univ_data_expanded
)
SELECT year, avg_tuition
FROM tuition_rates_view

-- COMMAND ----------

-- DBTITLE 1,Top 30 Universities with higher difference between in-state and out-state tuition for the year 2013-14
with view_1 AS (
  SELECT name, university_abbreviations FROM univ_data_expanded
),
view_2 AS (
  SELECT 
    name, 
    SUM(tuition_outstate_2013_14) AS outstate_tuition, 
    SUM(tuition_instate_2013_14) AS instate_tuition, 
    SUM(tuition_outstate_2013_14 - tuition_instate_2013_14) AS tuition_difference
  FROM univ_data_expanded
  GROUP BY name, zip
  HAVING tuition_difference > 0
  ORDER BY tuition_difference DESC
  LIMIT 30
)
SELECT CASE WHEN university_abbreviations IS NOT NULL THEN university_abbreviations ELSE view_2.name END AS name, outstate_tuition, instate_tuition, tuition_difference
FROM view_2 LEFT OUTER JOIN view_1
ON (view_2.name = view_1.name)

-- COMMAND ----------

-- DBTITLE 1,Top 30 Universities with lower difference between in-state and out-state tuition for the year 2013-14
with view_1 AS (
  SELECT name, university_abbreviations FROM univ_data_expanded WHERE university_abbreviations IS NOT NULL
),
view_2 AS (
  SELECT 
    name, 
    SUM(tuition_outstate_2013_14) AS outstate_tuition, 
    SUM(tuition_instate_2013_14) AS instate_tuition, 
    SUM(tuition_outstate_2013_14 - tuition_instate_2013_14) AS tuition_difference
  FROM univ_data_expanded
  GROUP BY name, zip
  HAVING tuition_difference > 0
  ORDER BY tuition_difference ASC
  LIMIT 30
)
SELECT CASE WHEN university_abbreviations IS NOT NULL THEN university_abbreviations ELSE view_2.name END AS name, outstate_tuition, instate_tuition, tuition_difference
FROM view_2 LEFT OUTER JOIN view_1
ON (view_2.name = view_1.name)

-- COMMAND ----------

-- DBTITLE 1,Top 25 Universities based on the endowment assets (FASB, GASB)
WITH uni_abbreviation_view AS (
  SELECT name, university_abbreviations FROM univ_data_expanded WHERE university_abbreviations IS NOT NULL
),
endowment_view AS (
  SELECT name, 
  CASE WHEN SUM(endowment_assets_gasb) IS NULL THEN SUM(endowment_assets_fasb) ELSE SUM(endowment_assets_gasb) END AS total_endowments
  FROM univ_data_expanded
  GROUP BY name, zip
  HAVING total_endowments > 0
)
SELECT 
  CASE WHEN university_abbreviations IS NOT NULL THEN university_abbreviations ELSE endowment_view.name END AS name, 
  total_endowments
FROM endowment_view LEFT OUTER JOIN uni_abbreviation_view
ON (endowment_view.name = uni_abbreviation_view.name)
ORDER BY total_endowments DESC
LIMIT 25

-- COMMAND ----------

-- DBTITLE 1,Bottom 25 Universities based on the endowment assets (FASB, GASB)
WITH uni_abbreviation_view AS (
  SELECT name, university_abbreviations FROM univ_data_expanded WHERE university_abbreviations IS NOT NULL
),
endowment_view AS (
  SELECT name, 
  CASE WHEN SUM(endowment_assets_gasb) IS NULL THEN SUM(endowment_assets_fasb) ELSE SUM(endowment_assets_gasb) END AS total_endowments
  FROM univ_data_expanded
  GROUP BY name, zip
  HAVING total_endowments > 0
)
SELECT 
  CASE WHEN university_abbreviations IS NOT NULL THEN university_abbreviations ELSE endowment_view.name END AS name, 
  total_endowments
FROM endowment_view LEFT OUTER JOIN uni_abbreviation_view
ON (endowment_view.name = uni_abbreviation_view.name)
ORDER BY total_endowments ASC
LIMIT 30

-- COMMAND ----------

-- DBTITLE 1,Conclusion
-- MAGIC %md
-- MAGIC <i> The published version of the notebook is available at - </i> <br/>
-- MAGIC https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3173713035751393/3391157573338707/2308983777460038/latest.html
