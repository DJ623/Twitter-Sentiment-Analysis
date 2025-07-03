from time import sleep
import csv 
from kafka import KafkaProducer
import json

# The kafka producer is fetching the data from twitter_validation.csv which acts like a database or simulate 
# requests coming from the twitter api , as the original api only allows for 100 twwets of request per month in the free
# acess and doesntnot allow to use fileter which was limiting factor. So this dataset simulates those tweets by producing it after a delay of 3 seconds for real-life simulation.
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: 
                         json.dumps(x).encode('utf-8'))

with open('twitter_validation.csv') as file_obj:
    reader_obj = csv.reader(file_obj)
    for data in reader_obj: 
        # print(data)
        producer.send('twitter_topic', value=data)
        sleep(3)