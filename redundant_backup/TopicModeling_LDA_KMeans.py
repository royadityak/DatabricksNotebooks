# Databricks notebook source
# DBTITLE 1,Loading the required packages
# Computational and Visualisation Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string, re
from collections import OrderedDict
from wordcloud import WordCloud
from itertools import chain, groupby

# Pyspark Packages
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, NGram
from pyspark.ml.clustering import LDA
from pyspark.ml import Pipeline

# NLTK Packages
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer

# COMMAND ----------

# DBTITLE 1,Auxiliary Modules
# MAGIC %md ## Helper, Utility and Wrapper Modules to assist in the later part of the notebook

# COMMAND ----------

lemmatizer = WordNetLemmatizer()
def lemmatize(token):
  """ Function to return lemmatized token"""
  return lemmatizer.lemmatize(token)

# COMMAND ----------

punctuation = set(string.punctuation) 
default_stopwords = stopwords.words('english')
custom_stopwords = ['charliecuntskies ', 'bts', 'imisscath ', 'imisscath', 'mcflyforgermany', 'charliecuntskies', 'delongeday', 'rlly', 'barakatday', 'blah']
stopwords_f = set(default_stopwords + custom_stopwords)
pattern_repl_list = [".", ",", "(", ")", "!", "@", "#"]

replacement_patterns = [
(r"won't", "will not"),
(r"haven't", "have not"),
(r"can't", "cannot"),
(r"i'm", "i am"),
(r"ain't", "am not"),
(r"(\w+)'ll", "$1 will"),
(r"(\w+)n't", "$1 not"),
(r"(\w+)'ve", "$1 have"),
(r"(\w+)'s", "$1 is"),
(r"(\w+)'re", "$1 are"),
(r"(\w+)'d", "$1 would"),
]

def preprocessed_chats (text):
  """ Function to extract processed column over Spark Dataframe """
  text_token = text.lower().split()
  
  text_token_filtered = []
  for token in text_token:
    for pattern in pattern_repl_list: token = token.replace(pattern, '') # Limit patterns from pattern_repl_list
    for pattern in replacement_patterns: token = token.replace(pattern[0], pattern[1]) # Limit patterns from replacement_patterns list
    text_token_filtered.append(token)
  
  text_token_filtered_pre = [word for word in text_token_filtered if len(word) > 4]
  text_sw_removed = [word for word in text_token_filtered if word not in stopwords_f]
  text_punc_removed = [word for word in text_sw_removed if word not in punctuation]
  text_lemmatized = [lemmatize(word) for word in text_punc_removed]
  return text_lemmatized

preprocessing_udf = udf(preprocessed_chats, ArrayType(StringType()))

# COMMAND ----------

def plotter(x_vals, y_vals, xlabel, ylabel, title, color=None):
  """ Custom Pyplot Plotter"""
  if color is None:
    color='red'
  fig, ax = plt.subplots()
  ax.plot(x_vals, y_vals, color=color)
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  display (fig)

# COMMAND ----------

def expanded_frame(summary_df):
  """ Function to return words and termWeights each for a particular cluster"""
  df_list = []
  for index, row in summary_df.toPandas().iterrows():
    words_list = row['words']
    words_weight = row['termWeights']

    for itr in range(len(words_list)):
      expanded_dict = OrderedDict()
      expanded_dict['cluster'] = row['cluster']
      expanded_dict['words'] = words_list[itr]
      expanded_dict['termWeights'] = words_weight[itr]
      df_list.append(expanded_dict)

  new_df = spark.createDataFrame(df_list)
  return (new_df)

# COMMAND ----------

def return_cluster_summary(lda_model):
  """ Function to retrieve formatted cluster summary for analysis"""
  cluster_summary_df_list = []
  final_df_column_list = ['cluster', 'words', 'termWeights']
  
  vocab = lda_model.stages[0].vocabulary 
  topics = lda_model.stages[1].describeTopics()
  topics_words = topics.rdd.map(lambda row: row['termIndices']) .map(lambda idx_list: [vocab[idx] for idx in idx_list]).collect()
  
  for idx, topic in enumerate(topics_words):
    cluster_summary_dict = OrderedDict()
    cluster_summary_dict['cluster'] = idx + 1
    cluster_summary_dict['words'] = [word for word in topic]
    cluster_summary_df_list.append(cluster_summary_dict)

  summary_pre_df = spark.createDataFrame(cluster_summary_df_list)
  summary_df = summary_pre_df.join(topics, summary_pre_df.cluster == topics.topic)[final_df_column_list]
  #new_summary_df = expanded_frame(summary_df)[final_df_column_list]
  return summary_df

# COMMAND ----------

def get_optimal_topics(topic_range, maxIteration, input_df):
  """ Function to compute optimal topics within a custom input range """
  ll_vals, lp_vals, cluster_no = [], [], []
  cv = CountVectorizer(inputCol='tokenized_sentiments', outputCol='features')
  lda = LDA ()
  lda_pipeline = Pipeline(stages=[cv, lda])
  
  for topic_no in topic_range:
    paramap = {lda.k: topic_no, lda.maxIter:maxIteration, cv.vocabSize:100000, cv.minDF:5.0}
    lda_i_model = lda_pipeline.fit (input_df, paramap)
    lda_i_results = lda_i_model.transform (input_df)
    
    ll_vals.append(lda_i_model.stages[1].logLikelihood(lda_i_results))
    lp_vals.append(lda_i_model.stages[1].logPerplexity(lda_i_results))
    cluster_no.append(topic_no)
  
  return ll_vals, lp_vals, cluster_no

# COMMAND ----------

def concat(type):
  """ Function to concat multiple Array columns in a dataframe into a single column"""
  def concat_(*args):
      return list(chain(*args))
  return udf(concat_, ArrayType(type))

#PySpark UDF
concat_string_arrays = concat(StringType())

# COMMAND ----------

def get_top_ngrams(input_df, cap):
  """ Function to return top ngrams grouped by cluster"""
  top_ngrams_perCluster = {}
  
  for itr in range(0, 12, 1):
    input_df_sub = input_df.filter("cluster_predictions == %d" % itr)
    
    voc_ngram = list(chain(*input_df_sub[['tokenized_sentiments']].toPandas()['tokenized_sentiments'].tolist()))
    voc_ngram_freq = {key:len(list(group)) for key, group in groupby(voc_ngram)}
    voc_ngram_freq_sorted = OrderedDict(sorted(voc_ngram_freq.items(), key=lambda it: it[1], reverse=True))
    top_ngrams_i =  ' '.join(list(set([key.encode('utf-8') for key,value in voc_ngram_freq_sorted.items()[:3000]]))[:cap])
    
    top_ngrams_perCluster[itr] = top_ngrams_i
  
  return top_ngrams_perCluster

# COMMAND ----------

# DBTITLE 1,Data Load and Basic Manipulation
twitter_sentiments = spark.read.csv('/mnt/lda/twitter_sentiment.csv', header=True) # Only taking a limited set
twitter_sentiments = twitter_sentiments[['SentimentText']]
twitter_sentiments = twitter_sentiments.withColumn("tokenized_sentiments", preprocessing_udf("SentimentText"))

# COMMAND ----------

# DBTITLE 1,LDA Modelling on the Data : 8 Clusters
twitter_sentiments_cv = CountVectorizer(inputCol='tokenized_sentiments', outputCol='features') # Creating featues column
twitter_sentiments_lda = LDA () # Training a LDA Model
Lda_Pipeline = Pipeline(stages=[twitter_sentiments_cv, twitter_sentiments_lda])

# Parameterized values
paramap = {twitter_sentiments_lda.k: 9, twitter_sentiments_lda.maxIter:50, twitter_sentiments_cv.vocabSize:100000, twitter_sentiments_cv.minDF:5.0}

LdaModel = Lda_Pipeline.fit(twitter_sentiments, paramap)
LdaResults = LdaModel.transform(twitter_sentiments)

cluster_summary_df = return_cluster_summary(LdaModel)
display (cluster_summary_df)

# COMMAND ----------

fig, ((t1,t2), (t3,t4), (t5,t6), (t7,t8)) = plt.subplots(4, 2)
wordcloud = WordCloud(background_color='white')

t1.imshow(wordcloud.generate(' '.join(cluster_summary_df.filter('cluster == 1')[['words']].collect()[0].words).encode('utf-8')))
t2.imshow(wordcloud.generate(' '.join(cluster_summary_df.filter('cluster == 2')[['words']].collect()[0].words).encode('utf-8')))
t3.imshow(wordcloud.generate(' '.join(cluster_summary_df.filter('cluster == 3')[['words']].collect()[0].words).encode('utf-8')))
t4.imshow(wordcloud.generate(' '.join(cluster_summary_df.filter('cluster == 4')[['words']].collect()[0].words).encode('utf-8')))
t5.imshow(wordcloud.generate(' '.join(cluster_summary_df.filter('cluster == 5')[['words']].collect()[0].words).encode('utf-8')))
t6.imshow(wordcloud.generate(' '.join(cluster_summary_df.filter('cluster == 6')[['words']].collect()[0].words).encode('utf-8')))
t7.imshow(wordcloud.generate(' '.join(cluster_summary_df.filter('cluster == 7')[['words']].collect()[0].words).encode('utf-8')))
t8.imshow(wordcloud.generate(' '.join(cluster_summary_df.filter('cluster == 8')[['words']].collect()[0].words).encode('utf-8')))

t1.set_title('Cluster 1')
t2.set_title('Cluster 2')
t3.set_title('Cluster 3')
t4.set_title('Cluster 4')
t5.set_title('Cluster 5')
t6.set_title('Cluster 6')
t7.set_title('Cluster 7')
t8.set_title('Cluster 8')

t1.figure.set_size_inches(20, 20)
t2.figure.set_size_inches(20, 20)
t3.figure.set_size_inches(20, 20)
t4.figure.set_size_inches(20, 20)
t5.figure.set_size_inches(20, 20)
t6.figure.set_size_inches(20, 20)
t7.figure.set_size_inches(20, 20)
t8.figure.set_size_inches(20, 20)

t1.set_axis_off()
t2.set_axis_off()
t3.set_axis_off()
t4.set_axis_off()
t5.set_axis_off()
t6.set_axis_off()
t7.set_axis_off()
t8.set_axis_off()
display (fig)

# COMMAND ----------

# DBTITLE 1,Computation for Optimal Number of topics
topic_range = range(6, 15, 2)
ll_vals, lp_vals, cluster_no = get_optimal_topics(topic_range, 50, twitter_sentiments.sample(False, 0.4))

# COMMAND ----------

# DBTITLE 1,Evaluating Log-likelihood Score
plotter(cluster_no, ll_vals, xlabel='Clusters', ylabel='LogLikelihood Scores', title='Plot of LogLikelihood against number of clusters', color='green')

# COMMAND ----------

# DBTITLE 1,Evaluating Log-Perplexity Score
plotter(cluster_no, lp_vals, xlabel='Clusters', ylabel='LogPerplexity Scores', title='Plot of LogPerplexity against number of clusters', color='blue')

# COMMAND ----------

# DBTITLE 1,LDA Modelling with Optimal Cluster - 12 Clusters
# Parameterized values
paramap = {twitter_sentiments_lda.k: 13, twitter_sentiments_lda.maxIter:50, twitter_sentiments_cv.vocabSize:100000, twitter_sentiments_cv.minDF:6.0}

LdaModel = Lda_Pipeline.fit(twitter_sentiments, paramap)
LdaResults_Best = LdaModel.transform(twitter_sentiments)

cluster_summary_df_Best = return_cluster_summary(LdaModel)
display (cluster_summary_df_Best)

# COMMAND ----------

fig, ((t1,t2), (t3,t4), (t5,t6), (t7,t8), (t9, t10), (t11, t12)) = plt.subplots(6, 2)
wordcloud = WordCloud(background_color='white')

t1.imshow(wordcloud.generate(' '.join(cluster_summary_df_Best.filter('cluster == 1')[['words']].collect()[0].words).encode('utf-8')))
t2.imshow(wordcloud.generate(' '.join(cluster_summary_df_Best.filter('cluster == 2')[['words']].collect()[0].words).encode('utf-8')))
t3.imshow(wordcloud.generate(' '.join(cluster_summary_df_Best.filter('cluster == 3')[['words']].collect()[0].words).encode('utf-8')))
t4.imshow(wordcloud.generate(' '.join(cluster_summary_df_Best.filter('cluster == 4')[['words']].collect()[0].words).encode('utf-8')))
t5.imshow(wordcloud.generate(' '.join(cluster_summary_df_Best.filter('cluster == 5')[['words']].collect()[0].words).encode('utf-8')))
t6.imshow(wordcloud.generate(' '.join(cluster_summary_df_Best.filter('cluster == 6')[['words']].collect()[0].words).encode('utf-8')))
t7.imshow(wordcloud.generate(' '.join(cluster_summary_df_Best.filter('cluster == 7')[['words']].collect()[0].words).encode('utf-8')))
t8.imshow(wordcloud.generate(' '.join(cluster_summary_df_Best.filter('cluster == 8')[['words']].collect()[0].words).encode('utf-8')))
t9.imshow(wordcloud.generate(' '.join(cluster_summary_df_Best.filter('cluster == 9')[['words']].collect()[0].words).encode('utf-8')))
t10.imshow(wordcloud.generate(' '.join(cluster_summary_df_Best.filter('cluster == 10')[['words']].collect()[0].words).encode('utf-8')))
t11.imshow(wordcloud.generate(' '.join(cluster_summary_df_Best.filter('cluster == 11')[['words']].collect()[0].words).encode('utf-8')))
t12.imshow(wordcloud.generate(' '.join(cluster_summary_df_Best.filter('cluster == 12')[['words']].collect()[0].words).encode('utf-8')))

t1.set_title('Cluster 1')
t2.set_title('Cluster 2')
t3.set_title('Cluster 3')
t4.set_title('Cluster 4')
t5.set_title('Cluster 5')
t6.set_title('Cluster 6')
t7.set_title('Cluster 7')
t8.set_title('Cluster 8')
t9.set_title('Cluster 9')
t10.set_title('Cluster 10')
t11.set_title('Cluster 11')
t12.set_title('Cluster 12')

t1.figure.set_size_inches(20, 20)
t2.figure.set_size_inches(20, 20)
t3.figure.set_size_inches(20, 20)
t4.figure.set_size_inches(20, 20)
t5.figure.set_size_inches(20, 20)
t6.figure.set_size_inches(20, 20)
t7.figure.set_size_inches(20, 20)
t8.figure.set_size_inches(20, 20)
t9.figure.set_size_inches(20, 20)
t10.figure.set_size_inches(20, 20)
t11.figure.set_size_inches(20, 20)
t12.figure.set_size_inches(20, 20)

t1.set_axis_off()
t2.set_axis_off()
t3.set_axis_off()
t4.set_axis_off()
t5.set_axis_off()
t6.set_axis_off()
t7.set_axis_off()
t8.set_axis_off()
t9.set_axis_off()
t10.set_axis_off()
t11.set_axis_off()
t12.set_axis_off()
display (fig)

# COMMAND ----------

# DBTITLE 1,NGrams Modeling on the best cluster
ngram_range = range(2, 5, 1)
ngrams = [NGram(n=ngram, inputCol="tokenized_sentiments", outputCol="{0}_grams".format(ngram)) for ngram in ngram_range]
ngrams_pipeline = Pipeline(stages=ngrams)
ngram_df = ngrams_pipeline.fit(LdaResults).transform(LdaResults)

# Merge Ngrams into a single column
ngram_df = ngram_df.select(col('SentimentText'), col('tokenized_sentiments'), col('features'), col('topicDistribution'), concat_string_arrays(col("2_grams"), col("3_grams"), col("4_grams")).alias('ngrams'))

# COMMAND ----------

# DBTITLE 1,Visualising Top 30 Ngrams
voc_ngram = list(chain(*ngram_df[['ngrams']].toPandas()['ngrams'].tolist()))
voc_ngram_freq = {key:len(list(group)) for key, group in groupby(voc_ngram)}
voc_ngram_freq_sorted = OrderedDict(sorted(voc_ngram_freq.items(), key=lambda it: it[1], reverse=True))
top_30_ngrams =  ' '.join(list(set([key.encode('utf-8') for key,value in voc_ngram_freq_sorted.items()[:3000]]))[:30])

fig, ax = plt.subplots()
wordcloud = WordCloud(background_color='black')
ax.imshow(wordcloud.generate(top_30_ngrams))
ax.set_title('Top 30 NGrams')
ax.figure.set_size_inches(10, 10)
ax.set_axis_off()
display(fig)

# COMMAND ----------

# DBTITLE 1,Bisecting KMeans Clustering
from pyspark.ml.clustering import BisectingKMeans

kmeans_inst = BisectingKMeans(featuresCol='features', predictionCol='cluster_predictions', maxIter=50, k=12, minDivisibleClusterSize=5.0, seed=42)
kmeans_mod = kmeans_inst.fit(LdaResults)
LdaResults_kmeans = kmeans_mod.transform(LdaResults)
cluster_centers = kmeans_mod.clusterCenters()

display (LdaResults_kmeans, 100)

# COMMAND ----------

# DBTITLE 1,Finding Optimal Number of Clusters Using KMeans
costs = []
clusters = []

for cluster_i in range(6, 15, 2):
    kmeans_i = BisectingKMeans(featuresCol='features', predictionCol='cluster_predictions', maxIter=30, k=cluster_i, minDivisibleClusterSize=5.0, seed=43)
    model_i = kmeans_i.fit(LdaResults_kmeans.sample(False, 0.2, seed=42))
    
    # Updating the list
    costs.append(model_i.computeCost(LdaResults_kmeans))
    clusters.append(cluster_i)

# plotter(x_vals, y_vals, xlabel, ylabel, title, color=None)
plotter (clusters, costs, 'Clusters', 'Cost', 'Detecting Optimal Number of clusters using KMeans', color='blue')

# COMMAND ----------

# DBTITLE 1,Visualizing Top NGrams per Cluster
top30_ngrams_perCluster = get_top_ngrams(LdaResults_kmeans, 30)
fig, ((t1,t2), (t3,t4), (t5,t6), (t7,t8), (t9, t10), (t11, t12)) = plt.subplots(6, 2)
wordcloud = WordCloud(background_color='white')

t1.imshow(wordcloud.generate(top30_ngrams_perCluster.get(0)))
t2.imshow(wordcloud.generate(top30_ngrams_perCluster.get(1)))
t3.imshow(wordcloud.generate(top30_ngrams_perCluster.get(2)))
t4.imshow(wordcloud.generate(top30_ngrams_perCluster.get(3)))
t5.imshow(wordcloud.generate(top30_ngrams_perCluster.get(4)))
t6.imshow(wordcloud.generate(top30_ngrams_perCluster.get(5)))
t7.imshow(wordcloud.generate(top30_ngrams_perCluster.get(6)))
t8.imshow(wordcloud.generate(top30_ngrams_perCluster.get(7)))
t9.imshow(wordcloud.generate(top30_ngrams_perCluster.get(8)))
t10.imshow(wordcloud.generate(top30_ngrams_perCluster.get(9)))
t11.imshow(wordcloud.generate(top30_ngrams_perCluster.get(10)))
t12.imshow(wordcloud.generate(top30_ngrams_perCluster.get(11)))

t1.set_title('Cluster 1')
t2.set_title('Cluster 2')
t3.set_title('Cluster 3')
t4.set_title('Cluster 4')
t5.set_title('Cluster 5')
t6.set_title('Cluster 6')
t7.set_title('Cluster 7')
t8.set_title('Cluster 8')
t9.set_title('Cluster 9')
t10.set_title('Cluster 10')
t11.set_title('Cluster 11')
t12.set_title('Cluster 12')

t1.figure.set_size_inches(20, 20)
t2.figure.set_size_inches(20, 20)
t3.figure.set_size_inches(20, 20)
t4.figure.set_size_inches(20, 20)
t5.figure.set_size_inches(20, 20)
t6.figure.set_size_inches(20, 20)
t7.figure.set_size_inches(20, 20)
t8.figure.set_size_inches(20, 20)
t9.figure.set_size_inches(20, 20)
t10.figure.set_size_inches(20, 20)
t11.figure.set_size_inches(20, 20)
t12.figure.set_size_inches(20, 20)

t1.set_axis_off()
t2.set_axis_off()
t3.set_axis_off()
t4.set_axis_off()
t5.set_axis_off()
t6.set_axis_off()
t7.set_axis_off()
t8.set_axis_off()
t9.set_axis_off()
t10.set_axis_off()
t11.set_axis_off()
t12.set_axis_off()
display (fig)

# COMMAND ----------

# DBTITLE 1,Summary Statistics on KMeans Processed Result
LdaResults_kmeans.createOrReplaceTempView('LdaResults_kmeans')
LdaResults_kmeans_summary = spark.sql('''with tbl1 as (select cluster_predictions AS summ_cluster_predictions, count(*) count_total from LdaResults_kmeans group by cluster_predictions) 
                             select tbl1.*, row_number() over (order by count_total desc) as cluster_prediction from tbl1 order by count_total desc''')
LDA_Results_kmeans_summary_expanded = LdaResults_kmeans_summary.join(LdaResults_kmeans, LdaResults_kmeans_summary.summ_cluster_predictions == LdaResults_kmeans.cluster_predictions, how='inner')[['SentimentText', 'tokenized_sentiments', 'count_total', 'cluster_predictions', 'topicDistribution']]
display (LDA_Results_kmeans_summary_expanded)
spark.catalog.dropTempView('LdaResults_kmeans')

# COMMAND ----------

# DBTITLE 1,Which cluster forms the largest part of the dataset?
display (LDA_Results_kmeans_summary_expanded)

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md 
# MAGIC The dataset we used in this notebook came from a specific user's Facebook feed, so, isn't a very great one to start with. However, both, LDA and Bisecting KMeans implementation demonstrated that 12 topics is an optimal number for topic modeling. 
# MAGIC The published notebook is available at - 
# MAGIC https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3173713035751393/2689111109152757/2308983777460038/latest.html
