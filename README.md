# Sentiment-Analysis-Mental-Health
Sentiment analysis system to categorize mental health states using Natural Language Processing (NLP) techniques
The analysis is performed using both a Logistic Regression model and a Neural Network (LSTM) model to evaluate their performance in classifying statements related to mental health.
**Categories**
The model categorizes mental health statements into the following categories:
Normal,
Depression,
Suicidal,
Anxiety,
Stress, 
Bi-Polar and 
Personality Disorder.
**DataSet**
https://drive.google.com/file/d/1WL20v_GayxyhlCm64kv1tu2rQoUus3__/view?usp=sharing
**Implementation**
**Data Preprocessing**
Data Cleaning: Handles missing values and unnecessary columns. Text data is cleaned using regular expressions, tokenization, stopwords removal, and lemmatization.
Feature Extraction: Utilizes TF-IDF Vectorizer for text feature extraction.
Label Encoding: Transforms categorical labels into numeric form.

**Model Training and Evaluation**
Logistic Regression:
Trained using TF-IDF vectorized features.
Evaluated using confusion matrix and classification report.
Neural Network (LSTM):
Tokenizes and pads sequences for input into an LSTM network.
Trained with an embedding layer followed by LSTM layers and dropout for regularization.
Evaluated based on accuracy metrics.
**Implemented Algorithms**
Logistic Regression: A traditional classification algorithm to set a baseline for comparison.
Neural Network (LSTM): A deep learning model designed to handle sequential data and capture more complex patterns in text.
**Results**
Logistic Regression
Accuracy: 75.5%
Confusion Matrix and Classification Report: Provides insights into model performance across different mental health categories.
Neural Network (LSTM)
Test Accuracy: Achieved an accuracy of approximately 76.5% on the test set.
**Comparison**
The Neural Network generally provides better performance than Logistic Regression due to its ability to capture complex patterns in the data.

**Future Work**
Algorithm Improvement, Hyperparameter Tuning, Advanced Feature Engineering and Scalability.

**How to Run: **
Clone the repository. Install Python and required libraries. Download the Kaggle dataset and place it in the project directory. Run the Python script (MainProject.py) to execute the code.
