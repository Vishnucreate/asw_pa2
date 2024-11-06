from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Load the trained model from the current directory
model = LogisticRegressionModel.load("wine_quality_model")

# Load and Prepare Test Data (ValidationDataset.csv used as a test dataset)
test_data = spark.read.csv("ValidationDataset.csv", header=True, inferSchema=True)
assembler = VectorAssembler(inputCols=[col for col in test_data.columns if col != 'label'], outputCol="features")
test_data = assembler.transform(test_data).select("features", "label")

# Make predictions
predictions = model.transform(test_data)

# Evaluate Predictions
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"Test F1 Score: {f1_score}")

# Stop Spark session
spark.stop()
