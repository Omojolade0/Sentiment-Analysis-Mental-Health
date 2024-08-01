
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

# NlTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Dataset 
df = pd.read_csv('FilePath')

# Data Cleaning 
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
df = df.dropna(subset=['statement'])

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[@*]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['Cleaned Statement'] = df['statement'].apply(clean_text)

# Encode Labels 
le = LabelEncoder()
y = le.fit_transform(df['status'])

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Cleaned Statement'])

# Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

print("Test set performance:")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, target_names=le.classes_))


# Data Preparation for Neural Network
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['Cleaned Statement'])
X_seq = tokenizer.texts_to_sequences(df['Cleaned Statement'])
X_pad = pad_sequences(X_seq, maxlen=100)

# Train-Test Split 
X_train_pad, X_test_pad, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Neural Network
nn_model = Sequential()
nn_model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
nn_model.add(LSTM(64, return_sequences=True))
nn_model.add(Dropout(0.5))
nn_model.add(LSTM(32))
nn_model.add(Dropout(0.5))
nn_model.add(Dense(len(le.classes_), activation='softmax'))

nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.1)
loss, accuracy = nn_model.evaluate(X_test_pad, y_test)
print(f'Test accuracy: {accuracy}')

# Prediction Neural Network
def preprocess_text_for_keras(text):
    text_seq = tokenizer.texts_to_sequences([clean_text(text)])
    text_pad = pad_sequences(text_seq, maxlen=100)
    return text_pad

def predict_with_keras(text):
    text_pad = preprocess_text_for_keras(text)
    prediction = nn_model.predict(text_pad)
    predicted_class = prediction.argmax(axis=-1)
    return le.classes_[predicted_class][0]

# Example usage
new_statement = "The past year has been bad for me. I called the NHS mutliple times i would just stay in bed not do anything just cry and i was wondering if i should just take a knie maybe it would be easier if i died  "
print("Predicted status:", predict_with_keras(new_statement))
