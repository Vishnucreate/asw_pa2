from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Load Training and Validation Data
train_data = spark.read.csv("/home/hadoop/data/TrainingDataset.csv", header=True, inferSchema=True)
val_data = spark.read.csv("/home/hadoop/data/ValidationDataset.csv", header=True, inferSchema=True)

# Prepare feature vector
assembler = VectorAssembler(inputCols=[col for col in train_data.columns if col != 'label'], outputCol="features")
train_data = assembler.transform(train_data).select("features", "label")
val_data = assembler.transform(val_data).select("features", "label")

# Train a Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")
model = lr.fit(train_data)

# Validate the Model
predictions = model.transform(val_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"Validation F1 Score: {f1_score}")

# Save the Model
model.save("/home/hadoop/wine_quality_model")

# Stop Spark session
spark.stop()
