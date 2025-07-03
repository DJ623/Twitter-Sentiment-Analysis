import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Automatically refresh the dashboard every 10 seconds (10000 milliseconds)
st_autorefresh(interval=10000, key="data_refresh")

from pymongo import MongoClient
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
nltk.download('punkt')
nltk.download('stopwords')


# Loading the spark model
spark = SparkSession.builder.appName("TweetClassification").getOrCreate()
pipeline = PipelineModel.load("hdfs://localhost:9000/models/model.pkl")

# mapping of outputs 
class_index_mapping = {0: "Negative", 1: "Positive", 2: "Neutral", 3: "Irrelevant"}
class_list = ['Negative', 'Positive', 'Neutral', 'Irrelevant']

# Cleaning the text for classification requests from the frontend
def classification_clean_text(text):
    if text is not None:
        # Remove links and hashtags/user mentions
        text = re.sub(r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', '', text)
        text = re.sub(r'(@|#)\w+', '', text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return ''

# For plotting word frequencies 
def plotting_clean_text(text):
    text = re.sub(
        r'http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F]{2}))+',
        '',
        text
    )
    text = re.sub(r'\b@\w+\b', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

def preprocess_text(text):
    text = plotting_clean_text(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens


# For classifing the tweets on the classification screen
def classify_text(text):
    cleaned = classification_clean_text(text)
    
    data = [(cleaned,)]
    df = spark.createDataFrame(data, ["Text"])
    processed = pipeline.transform(df)
    prediction = processed.collect()[0][6]
    return class_index_mapping[int(prediction)]

# plotting  function for word frequencies
def plot_word_frequencies_per_class(data):
   # storing all words for each sentiment class
    all_words = {label: [] for label in class_list}
    
    for entry in data:
        tweet = entry.get('tweet', '')
        prediction = entry.get('prediction', '')
        words = preprocess_text(tweet)
        if prediction in all_words:
            all_words[prediction].extend(words)
    
    # use of matplotlib for plotting chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_labels = []
    frequencies = []
    colors = []
    color_map = {'Negative': '#1f77b4' , 'Positive': '#ff7f0e', 'Neutral': '#2ca02c', 'Irrelevant': '#d62728'}
    
    for label in class_list:
        word_freq = Counter(all_words[label])
        
        # getting top 5 words for each class
        top_words = word_freq.most_common(5)
        for word, freq in top_words:
            x_labels.append(f"{word} ({label})")
            frequencies.append(freq)
            colors.append(color_map[label])
    
    ax.bar(x_labels, frequencies, color=colors, alpha=0.7)
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")
    ax.set_title("Top Word Frequencies for Each Class")
    plt.xticks(rotation=70)
    plt.tight_layout()
    return fig


# connecting to backend storage of mongodb
client = MongoClient('mongodb://localhost:27017/')
db = client['bigdata_project']
tweets_collection = db['tweets']

# basic navigation bar/ as streamlit soent have routing for urls
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Dashboard", "Classify"))


# home page
if page == "Dashboard":
    st.title("Tweet Dashboard")
    
    
    # Getting data from mongodb
    data = list(tweets_collection.find())
    
    if not data:
        st.write("No tweets available in the database.")
    else:
        total = len(data)
        st.write(f"Total tweets: {total}")
        
        # calculating the sentiment counts per class
        sentiment_counts = {label: 0 for label in class_list}
        for entry in data:
            pred = entry.get('prediction', '')
            if pred in sentiment_counts:
                sentiment_counts[pred] += 1
        
        st.subheader("Sentiment Summary")

        # Displaying the data in cloms format for better experience 
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Negative", sentiment_counts.get("Negative", 0))
        col2.metric("Positive", sentiment_counts.get("Positive", 0))
        col3.metric("Neutral", sentiment_counts.get("Neutral", 0))
        col4.metric("Irrelevant", sentiment_counts.get("Irrelevant", 0))

        st.write("")
        st.write("")
        
        # Create a pie chart for sentiment distribution
        fig1, ax1 = plt.subplots()
        ax1.pie(
            list(sentiment_counts.values()),
            labels=list(sentiment_counts.keys()),
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.axis('equal')  
        st.pyplot(fig1)
        
        st.subheader("Word Frequency Visualization")
        fig = plot_word_frequencies_per_class(data)
        st.pyplot(fig)
        
        st.subheader("Tweet Data")
        df = pd.DataFrame(data)
        if '_id' in df.columns:
             df = df.drop(columns=['_id'])
        st.dataframe(df)

# classification page
elif page == "Classify":
    st.title("Tweet Classification")
    tweet_text = st.text_area("Enter tweet text to classify:")
    if st.button("Classify Tweet"):
        if tweet_text.strip() == "":
            st.error("Please enter a tweet.")
        else:
            prediction = classify_text(tweet_text)
            st.write("Predicted Sentiment:", prediction)
