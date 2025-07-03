from pyspark.ml import PipelineModel
import re
import nltk
from kafka import KafkaConsumer
from json import loads
from pymongo import MongoClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Conneting to mongodb
client = MongoClient('localhost', 27017)
db = client['bigdata_project'] 
collection = db['tweets'] 

# Cration of a spark session
spark = SparkSession.builder \
    .appName("classify tweets") \
    .getOrCreate()

# loading the model from haddop 
pipeline = PipelineModel.load("hdfs://localhost:9000/models/model.pkl")

# text claening / Preprocessing of Data
def clean_text(text):
    if text is not None:
        # Remove links starting with https://, http://, www., or containing .com
        text = re.sub(r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', '', text)
        
        # Remove words starting with # or @
        text = re.sub(r'(@|#)\w+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return ''
    
class_index_mapping = { 0: "Negative", 1: "Positive", 2: "Neutral", 3: "Irrelevant" }

# Set up the Kafka consumer to subscribe to the 'twitter_topic'
consumer = KafkaConsumer(
    'twitter_topic',  # Topic name where tweets are published
    bootstrap_servers=['localhost:9092'],  # Connect to the  Kafka 
    auto_offset_reset='earliest',  # Start reading message
    enable_auto_commit=True, 
    group_id='my-group',
    value_deserializer=lambda x: loads(x.decode('utf-8'))  # Convert JSON type to Python obj
)

# Process each message as it is received from Kafka
for message in consumer:
    # get the tweet text from the message 
    tweet = message.value[-1]
    
    # cleaning the tweet data i.e text, by removing links, mentions, and other unwanted characters
    preprocessed_tweet = clean_text(tweet)
    
    # Storing the cleaned data in a spark dataframe for further processing
    tweet_df = spark.createDataFrame([(preprocessed_tweet,)], ["Text"])
    
    # applying the spark model
    processed_validation = pipeline.transform(tweet_df)
    prediction = processed_validation.collect()[0][6]  # Get the prediction result from the Spark model output
    
    # Printing out the tweet classification information
    print("-> Original Tweet:", tweet)
    print("-> Cleaned Tweet:", preprocessed_tweet)
    print("-> Sentiment Prediction (numeric):", prediction)
    print("-> Sentiment Label:", class_index_mapping[int(prediction)])
    
    # creatin a document like object to insert into mongodb
    tweet_document = {
        "tweet": tweet,
        "prediction": class_index_mapping[int(prediction)]
    }
    
    # Insert the tweet document into the MongoDB 
    collection.insert_one(tweet_document)
    
    # for formatting 
    print("-" * 50)
