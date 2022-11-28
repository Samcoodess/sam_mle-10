# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from delta import *

from pyspark.sql import SparkSession

from pyspark.ml.feature import OneHotEncoder , StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

# MAGIC %%sh
# MAGIC #wget = downloads url(mostly interneT) gets content from the url
# MAGIC wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip

# COMMAND ----------

ls

# COMMAND ----------

# MAGIC %%sh
# MAGIC unzip bank.zip

# COMMAND ----------

!ls -lh
#lh - more info on permissions
#drwxr..... this is permission example 


# COMMAND ----------

# MAGIC %%sh
# MAGIC wc -l bank.csv
# MAGIC #wc = word count ....linux command 

# COMMAND ----------

# MAGIC %%sh
# MAGIC head bank.csv
# MAGIC 
# MAGIC #head will read first 10 lines
# MAGIC #delimeter is how you separet it like : , ;

# COMMAND ----------

!ls

# COMMAND ----------

#spark is framework which can help with distributed contents
#datawarehouse vs datalake
#deltalake is intermediate between ^ these two

import pyspark
from delta import *

builder = pyspark.sql.SparkSession.builder.appName("ml-bank") \
  .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension") \
  .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# COMMAND ----------

!mkdir delta

# COMMAND ----------

# Define the input and output formats and paths and the table name.
write_format = 'delta'
load_path = "file:/databricks/driver/bank.csv"
table_name = 'default.bank4k'
save_path = "file:/databricks/driver/tmp/delta/bank-4k"

# Load the data from its source into a dataframe.
df=spark.read.csv(load_path,sep=';',header=True,inferSchema=True)
# Create table with path using DataFrame's schema and write data to it
# Note if you are overwriting to specificy overwrite option
# df.write.format(write_format).mode('overwrite').save(save_path)
# df.write.format("delta").mode('overwrite').saveAsTable(save_path)
df.printSchema()

# COMMAND ----------

df.write.format(write_format).mode('overwrite').save(save_path)


# COMMAND ----------

df.show()

# COMMAND ----------

df.head(2)


# COMMAND ----------

ls -lh /databricks/driver/tmp/delta/bank-4k/

# COMMAND ----------

import shutil
new_path = '/databricks/driver/tmp/delta/bank-4k'
shutil.rmtree(new_path)

# COMMAND ----------

partition_by = 'job'

# Write the data to its target.
# [YOUR CODE HERE]

# COMMAND ----------

ls -lh /databricks/driver/tmp/delta/bank-4k/

# COMMAND ----------

read_format = 'delta'
load_path = 'file:/databricks/driver/tmp/delta/bank-4k/'

df = spark.read.format(read_format).load(load_path) 

df.printSchema()


# COMMAND ----------

read_format='delta'
load_path = '/content/spark-warehouse/delta_save_path_4k/'

# COMMAND ----------

table_name = "psdf"
path = spark.sql(f"describe detail {table_name}").select("location").collect()[0][0].replace('dbfs:', '')

# COMMAND ----------

df = spark.read.format(read_format).load(load_path) 

df.printSchema()

# COMMAND ----------

df.show()

# COMMAND ----------

import pandas as pd
df.limit(5).toPandas()

# COMMAND ----------

import pyspark.pandas as ps
psdf = df.toPandas()
psdf.head(5)


# COMMAND ----------

psdf.count()
#count gives count of cols

# COMMAND ----------

psdf.describe()

# COMMAND ----------

col_names = [name for name in psdf.dtypes.index]
dtypes = [dtype for dtype in psdf.dtypes.tolist()]

numeric_features = [name for name, dtype in zip(col_names, dtypes) if dtype == 'int32']
nums_psdf = psdf.drop(columns = [c for c in col_names if c not in numeric_features])
nums_psdf.head()

# COMMAND ----------

# Convert Pandas on Spark DataFrame to Spark DataFrame
numeric_data = psdf[numeric_features]

axs = pd.plotting.scatter_matrix(numeric_data, figsize=(8, 8));
n = len(numeric_data.columns)

for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n - 1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

# COMMAND ----------

sdf = psdf.drop(columns=['month', 'day'])
sdf.dtypes

# COMMAND ----------


from pyspark.ml.feature import OneHotEncoder , StringIndexer, VectorAssembler

categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []

for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol = 'y', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

#Pipeline
from pyspark.ml import Pipeline
sdf= spark.createDataFrame(sdf)
# sdf = sdf.to_spark()
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(sdf)

# COMMAND ----------

transformed_df = pipelineModel.transform(sdf)
transformed_df.printSchema()

# COMMAND ----------

selectedCols = ['label', 'features'] + sdf.columns
sdf_bkp = sdf
sdf = transformed_df.select(selectedCols)
sdf.printSchema()

# COMMAND ----------

sdf.show(5)
sdf.take(5)

# COMMAND ----------

train, test = sdf.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
logreg = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
logreg_model = logreg.fit(sdf)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
lrModel = logreg_model
beta = np.sort(lrModel.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()

# COMMAND ----------

predictions = logreg_model.transform(sdf)

# COMMAND ----------

predictions.limit(10).toPandas()
predictions.sample(fraction=0.05).limit(10).toPandas()

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------


