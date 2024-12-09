import argparse
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Wine Quality Prediction Model Evaluation')
parser.add_argument('input_file', type=str, help='Path to the input CSV dataset file')
args = parser.parse_args()

spark_session = SparkSession.builder.appName("WineQualityModelEvaluation").getOrCreate()
spark_ctx = spark_session.sparkContext
spark_ctx.setLogLevel('ERROR')
spark_ctx._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

input_file = args.input_file
model_directory = "s3a://wineapplication/model"

dataframe = (spark_session.read
                .format("csv")
                .option('header', 'true')
                .option("sep", ";")
                .option("inferschema", 'true')
                .load(input_file))

dataframe = dataframe.select(*(col(c).cast("double").alias(c.strip("\"")) for c in dataframe.columns))
trained_model = PipelineModel.load(model_directory)
prediction_results = trained_model.transform(dataframe)
prediction_data = prediction_results.select(['prediction', 'label'])

accuracy_calculator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy_value = accuracy_calculator.evaluate(prediction_results)
print(f'Accuracy on Test Data: {round(accuracy_value, 2)}')

evaluation_metrics = MulticlassMetrics(prediction_data.rdd.map(tuple))
f1_score = evaluation_metrics.weightedFMeasure()
print(f'F1 Score of Wine Quality Prediction: {f1_score}')

spark_session.stop()
