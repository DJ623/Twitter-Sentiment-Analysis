# Twitter-Sentiment-Analysis
A mini project for analyzing, classifying, and visualizing the sentiments of tweets

## **Overview :** 

This project features a real-time Twitter (now X) sentiment analysis model designed for scalability and large-scale data processing using technologies like Apache Spark for model training, Hadoop Distributed File System (HDFS) for storing the model to ensure it is available to different services at the same time, Kafka for fetching the tweets data, MongoDB to store the processed data, Streamlit to design and deploy the dashboard to view the analysis, and Docker to containerize all the services to ensure loose coupling of the system for scalability, availability, and fault tolerance of the system.

## **Prerequisites**

1. **Hadoop installed** (version 3.4.0 on M1 Mac)  
2. **Spark installed** (version 3.5.1 )  
3. **Python 3.9.6** (PySpark requires Python)  
4. **Java 8 or later** (required by Spark)  
5. **Hadoop configurations** should be correctly set up for HDFS and YARN.  
6. **Before starting** the dashboard, make sure to upload the model.pkl file from /model directory to hdfs at /models path  
7. **Install** requirements.txt

## **Environment Details :** 

1. Machine: Macbook Air M1  
2. Architecture : ARM-64  
3. Operating System : macOS (Unix Based)

## **Deployment :**

1. **Install requirements.txt:** 

	Make sure you are in the project’s root directory  
    In terminal:  
    `pip install -r requirements.txt`  
	

2. **Start the MongoDB server (ignore this setup if already installed):**  
   1. If not downloaded , download from [https://www.mongodb.com/try/download/community](https://www.mongodb.com/try/download/community).   
   2. Note : The following setup and installation is **only for Mac ARM-based architecture.**  
   3. Set up and add the mongodb/bin path to global environment of the shell profile for better accessibility and make sure to give proper permissions using sudo.  
   4. Create a data directory in HOME\_DIRECTORY and inside data, create a directory named db.  
   5. In terminal : `sudo mongod –dbpath=HOME_DIRECTORY/path/to/data`

 


3. **Start Hadoop:**  
   In terminal: 
   `start-all.sh`

4. **Check if all services of Hadoop are started (Namenode, Datanode, ResourceManager, SecondaryNameNode, NodeManager, Jps):**   
   In terminal: `jps`

    This shows that Hadoop has started successfully, and it is in pseudo-distributed mode.

5. **Start and run Docker Compose for Kafka and Zookeeper:**  
   **Make sure docker is up and running**  
   In terminal (root directory of project):   
   `docker compose up -d`  
   `docker ps`

6. **Create a topic name twitter in Kafka (run commands in the Kafka shell):**  
   1. Make use of the Docker GUI to access the terminal of Kafka or use the following command: `docker exec -it <kafka-container-id\> /bin/bash`  
   2. In Kafka shell, run the following:  
        
      `kafka-topics --create --topic twitter --bootstrap-server localhost:9092`  
         
      `kafka-topics --describe --topic twitter --bootstrap-server localhost:9092`

7. **Create a new directory named “models” on HDFS:**   
   In terminal: 
    `hdfs dfs -mkdir -p /models`

    You can check the directory using the web interface by visiting [http://localhost:9870](http://localhost:9870) (default port for hadoop configured in core-site.xml)

8. **Upload the model.pkl file located in the model directory of the project to the hdfs directory created in the previous step:**  
   In terminal:   
   `cd model`  
    `hdfs dfs -put /path/to/local/file.pkl  /path/of/hdfs/directory`
  

    Example :  
    `hdfs dfs -put model.pkl /models`

9. **Run the producer.py and consumer.py programs located in the kafka directory :**   
   In terminal :   
   `cd kafka`   
   `python3 producer.py`  
     
   In a new terminal window :   
   `python3 consumer.py`

10. **Run the dashboard.py located in the dashboard directory using streamlit command :**  
    **This will start the dashboard in your browser.**   
    In terminal :   
    `cd dashboard`   
    `streamlit run dashboard.py`   
      
    **The dashboard is refreshed automatically every 10 seconds to update itself based on new incoming data.**
   
***  

## **For training the model and running everything from scratch :** 

NOTE : We have already provided the trained model file as model.pkl file in /model directory. To skip the training part, upload this model. pkl file to hdfs (Done in Previous Section)

1. **Download the dataset from Kaggle: [https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis?select=twitter\_training.csv](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis?select=twitter_training.csv)**  
     
2. **The dataset contains two files named twitter\_training.csv and twitter\_validation.csv. Both the files need to be uploaded to HDFS before training the model (twitter\_classification.py).**  
   1. **Make sure Hadoop is up and running : `jps`**   
   2. **In terminal (where both the files of the dataset are located), run the following :**

   `hdfs dfs -mkdir -p /data` 

   `hdfs dfs -put twitter_training.csv /data`

   `hdfs dfs -put twitter_validation.csv /data`

   

3. **Run the twitter\_classification.py file to train the model:**  
   1. Change the directory to model in the projects’ root directory.  
   2. In terminal : `python3 text_classification.py`  
   3. The model will be trained, and a new directory named /models on HDFS will be created, which will include the trained model.pkl file.

		

4. **Follow steps 5 to 10 in the previous section of Deployment.**  
   **Happy running\!**


