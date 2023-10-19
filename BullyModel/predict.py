import pickle
import joblib
import re
import string
import emoji
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained models and vectorizers
tfIdfVectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('modell.joblib')

# Load the necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing functions
def strip_emoji(tweet):
    return emoji.replace_emoji(tweet, replace="")

def strip_all_entities(tweet): 
    tweet = tweet.replace('\r', '').replace('\n', ' ').lower()
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
    tweet = re.sub(r'[^\x00-\x7f]', r'', tweet)
    tweet = re.sub(r'(.)1+', r'1', tweet)
    tweet = re.sub('[0-9]+', '', tweet)
    stopchars = string.punctuation
    table = str.maketrans('', '', stopchars)
    tweet = tweet.translate(table)
    tweet = [word for word in tweet.split() if word not in stop_words]
    tweet = ' '.join(tweet)
    return tweet

def decontract(tweet):
    tweet = re.sub(r"can\'t", "can not", tweet)
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"\'m", " am", tweet)
    return tweet

def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet))
    return new_tweet2

def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) or ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

def remove_mult_spaces(text):
    return re.sub("\s\s+", " ", text)

def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(words) for words in tokenized])

def preprocess(tweet):
    tweet = strip_emoji(tweet)
    tweet = decontract(tweet)
    tweet = strip_all_entities(tweet)
    tweet = clean_hashtags(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_mult_spaces(tweet)
    tweet = lemmatize(tweet)
    return tweet

# Function to predict cyberbullying
def predict_cyberbullying(sentence):
    cleaned_sentence = preprocess(sentence)
    tfidf_sentence = tfIdfVectorizer.transform([cleaned_sentence])
    prediction = model.predict(tfidf_sentence)[0]
    return prediction

# Test the function
input_sentence = input("Enter a sentence: ")
prediction = predict_cyberbullying(input_sentence)
if prediction == 1:
    print("The input sentence is classified as cyberbullying.")
else:
    print("The input sentence is not classified as cyberbullying.")
