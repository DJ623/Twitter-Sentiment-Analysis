from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Create Spark session
spark = SparkSession.builder.appName("Text Classification with PySpark").getOrCreate()

# Load training and validation data
training_data = spark.read.csv("hdfs:///data/twitter_training.csv", header=False, inferSchema=True)
validation_data = spark.read.csv("hdfs:///data/twitter_validation.csv", header=False, inferSchema=True)

# Rename columns for clarity
columns = ['id', 'Company', 'Label', 'Text']
for i, col_name in enumerate(columns):
    training_data = training_data.withColumnRenamed(f"_c{i}", col_name)
    validation_data = validation_data.withColumnRenamed(f"_c{i}", col_name)

# Remove rows with missing text
training_data = training_data.dropna(subset=['Text'])
validation_data = validation_data.dropna(subset=['Text'])

# Convert string labels to numeric indices
label_indexer = StringIndexer(inputCol="Label", outputCol="Label2")
label_indexer_model = label_indexer.fit(training_data)
training_data = label_indexer_model.transform(training_data)
validation_data = label_indexer_model.transform(validation_data)
label_mapping = label_indexer_model.labels
print("Label Mapping:")
for index, label in enumerate(label_mapping):
    print(f"Index {index} --> Label '{label}'")

# Clean text: remove URLs, hashtags, mentions, non-letters, and convert to lowercase
def clean_text(df, inputCol="Text", outputCol="cleaned_text"):
    df = df.withColumn(outputCol, regexp_replace(df[inputCol], r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', ''))
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'(@|#)\w+', ''))
    df = df.withColumn(outputCol, lower(df[outputCol]))
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'[^a-zA-Z\s]', ''))
    return df

# Clean the text in both datasets (overwriting "Text" with cleaned version)
cleaned_training = clean_text(training_data, inputCol="Text", outputCol="Text")
cleaned_validation = clean_text(validation_data, inputCol="Text", outputCol="Text")

# Build the processing pipeline
tokenizer = Tokenizer(inputCol="Text", outputCol="tokens")
stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="features", vocabSize=10000, minDF=5)
lr = LogisticRegression(maxIter=10, labelCol="Label2", featuresCol="features")
pipeline = Pipeline(stages=[tokenizer, stopwords_remover, count_vectorizer, lr])

# Train the model
model = pipeline.fit(cleaned_training)
# Save the model to HDFS at /models (the directory will be created if it doesn't exist)
model.write().overwrite().save("hdfs://localhost:9000/models/model.pkl")

# Load the model from HDFS
loaded_model = PipelineModel.load("hdfs://localhost:9000/models/model.pkl")

# Display a sample of validation data
cleaned_validation.show(10)

# Generate predictions on validation data
processed_validation = loaded_model.transform(cleaned_validation)
selected_data = processed_validation.select("id", "Text", "prediction", "Label2")
selected_data.show()

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="Label2", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(processed_validation)
print("Accuracy:", accuracy)

# Stop the Spark session
spark.stop()
