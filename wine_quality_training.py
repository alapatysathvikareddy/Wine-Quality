from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')
spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

training_file = "s3a://wineapplication/datasets/TrainingDataset.csv"
validation_file = "s3a://wineapplication/datasets/ValidationDataset.csv"
output_model_path = "s3a://wineapplication/model/"

train_df = spark.read.csv(training_file, header=True, sep=";", inferSchema=True)
train_df = train_df.select([col(c).cast("double").alias(c.strip("\"")) for c in train_df.columns])

valid_df = spark.read.csv(validation_file, header=True, sep=";", inferSchema=True)
valid_df = valid_df.select([col(c).cast("double").alias(c.strip("\"")) for c in valid_df.columns])

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol', 'quality']

assembler = VectorAssembler(inputCols=features, outputCol="features")
label_indexer = StringIndexer(inputCol="quality", outputCol="label")

train_df.cache()
valid_df.cache()

rf_classifier = RandomForestClassifier(labelCol="label", featuresCol="features", 
                                      numTrees=150, maxDepth=15, seed=150, impurity="gini")

pipeline = Pipeline(stages=[assembler, label_indexer, rf_classifier])
print("Training")
rf_model = pipeline.fit(train_df)
print("Validating")
predictions = rf_model.transform(valid_df)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy}")

predicted_labels = predictions.select("prediction", "label")
metrics = MulticlassMetrics(predicted_labels.rdd.map(tuple))
print(f"Weighted F1 Score: {metrics.weightedFMeasure()}")

param_grid = ParamGridBuilder() \
    .addGrid(rf_classifier.maxDepth, [6, 9]) \
    .addGrid(rf_classifier.numTrees, [50, 150]) \
    .addGrid(rf_classifier.minInstancesPerNode, [6]) \
    .addGrid(rf_classifier.seed, [100, 200]) \
    .addGrid(rf_classifier.impurity, ["entropy", "gini"]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=2)

cv_model = crossval.fit(train_df)

best_model = cv_model.bestModel

predictions = best_model.transform(valid_df)

accuracy = evaluator.evaluate(predictions)
metrics = MulticlassMetrics(predictions.select("prediction", "label").rdd.map(tuple))
print(f"Weighted F1 Score after CrossValidation: {metrics.weightedFMeasure()}")

print("Saving model to the s3 bucket")
best_model.write().overwrite().save(output_model_path)

spark.stop()
