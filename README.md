# Disaster-Response-Pipeline
This repository contains solution for  Udacity Data scientist project - Disaster Response Pipeline

As a part of this project, 
step 1 - we have created a ETL pipeline to clean, process and save data to run a supervised model. 
step 2 - Use the processed data in step 1 to create a ML pipeline and save the resulting model.

There is a web app created in flask that will allow a Emergency worker to key in the message and identify the relavent categories/departments to which the message can be shared with. The model we have saved earlier is used for this functionality. 

Below is a list of instructions to run the ETL, ML pipelines and fire up the flask application

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
