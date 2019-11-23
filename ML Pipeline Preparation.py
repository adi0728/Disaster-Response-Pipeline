#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
import pickle
nltk.download('punkt')
nltk.download('wordnet')


# In[2]:


# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table("InsertTableName", con=engine)


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


X = df["message"]
y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)


# ### 2. Write a tokenization function to process your text data

# In[6]:


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[7]:


pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


# In[8]:


pipeline.get_params()


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X,y)


# In[10]:


# train classifier
pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[11]:


def perf_report(model, X_test, y_test):
    '''
    Function to generate classification report on the model
    Input: Model, test set ie X_test & y_test
    Output: Prints the Classification report
    '''
    y_pred = model.predict(X_test)
    for i, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, i]))


# In[12]:


perf_report(pipeline,X_test,y_test)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[13]:


parameters =  {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100], 
              'clf__estimator__min_samples_split': [2, 4]} 

cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1,verbose=1)


# In[14]:


cv


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[15]:


cv.fit(X_train, y_train)


# In[16]:


perf_report(cv, X_test, y_test)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[17]:


pipeline2 = Pipeline([
    ('vect', CountVectorizer()),
    ('best', TruncatedSVD()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
])


# In[18]:



pipeline2.get_params()


# In[19]:


pipeline2.fit(X_train, y_train)
perf_report(pipeline2, X_test, y_test)


# In[41]:


parameters2 = { 
              'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100],
              'clf__estimator__learning_rate': [1,2] }


# In[42]:


cv2 = GridSearchCV(pipeline2, param_grid=parameters2)
cv2


# In[43]:


cv2.fit(X_train, y_train)
perf_report(cv2, X_test, y_test)


# ### 9. Export your model as a pickle file

# Pipeline has given better results vs. the Grid search and hence saving the model with original pipeline.

# In[20]:



with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




