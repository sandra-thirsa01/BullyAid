import pickle
import joblib
from joblib import dump,load
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
import string
import emoji
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
#Loading Data
df=pd.read_csv('new_dataset.csv')
df.info()
df.head()
#Checking Missing Values
df.isnull().sum()
df.label.value_counts()[1]
df.label.value_counts()[0]
df
df.drop(['extras'],axis = 1,inplace = True)
df
df.drop(['notes'],axis = 1,inplace = True)
df
df = df.rename(columns={'content': 'tweet'})
df.head
df.shape 
df['label'].value_counts().sort_index().plot.bar()
print("Non-Cyber trolling: ", df.label.value_counts()[0]/len(df.label)*100,"%")
print("Cyber trolling: ", df.label.value_counts()[1]/len(df.label)*100,"%")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
#Preprocessing of Text

#Function to Remove Emojis
def strip_emoji(tweet):
    return emoji.replace_emoji(tweet,replace="")
#Fucntion to Convert text to lowercase, remove (/r, /n characters), URLs, non-utf characters, Numbers, punctuations,stopwords
def strip_all_entities(tweet): 
    tweet = tweet.replace('\r', '').replace('\n', ' ').lower()
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
    tweet = re.sub(r'[^\x00-\x7f]',r'', tweet)
    tweet = re.sub(r'(.)1+', r'1', tweet)
    tweet = re.sub('[0-9]+', '', tweet)
    stopchars= string.punctuation
    table = str.maketrans('', '', stopchars)
    tweet = tweet.translate(table)
    tweet = [word for word in tweet.split() if word not in stop_words]
    tweet = ' '.join(tweet)
    return tweet
#Function to remove contractions
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
#Function to Clean Hashtags
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet))
    return new_tweet2
#Function to Filter Special Characters such as $, &
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)
#Function to remove mutiple sequence spaces
def remove_mult_spaces(text):
    return re.sub("\s\s+" , " ", text)
#Function to apply lemmatization to words
def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    lm = WordNetLemmatizer()
    return ' '.join([lm.lemmatize(words) for words in tokenized])
#Function to Preprocess the text by applying all above functions
def preprocess(tweet):
    tweet = strip_emoji(tweet)
    tweet = decontract(tweet)
    tweet = strip_all_entities(tweet)
    tweet = clean_hashtags(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_mult_spaces(tweet)
    #tweet = stemmer(tweet)
    tweet = lemmatize(tweet)
    return tweet
df['cleaned_tweet'] = df['tweet'].apply(preprocess)
df.head()
#Cleaned text added


#Dealing with Duplicates
df["cleaned_tweet"].duplicated().sum()
df.drop_duplicates("cleaned_tweet", inplace=True)
df.head()
#Duplicates removed


#Tokenization
df['tweet_list'] = df['cleaned_tweet'].apply(word_tokenize)
df.head()
#Checking length of various tweet texts

text_len = []
for tweet in df.tweet_list:
    tweet_len = len(tweet)
    text_len.append(tweet_len)
df['text_len'] = text_len
plt.figure(figsize=(15,8))
ax = sns.countplot(x='text_len', data=df, palette='mako')
plt.title('Count of words in tweets', fontsize=20)
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()
df = df[df['text_len']!=0]
df.shape
df.head()
tfIdfVectorizer=TfidfVectorizer(use_idf=True, sublinear_tf=True)
tfIdf = tfIdfVectorizer.fit_transform(df.cleaned_tweet	.tolist())
dump(tfIdfVectorizer,'vectorizer.joblib')
print(tfIdf)
X=tfIdf.toarray()
y = np.array(df.label.tolist())
#Spltting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#Training data biasness
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))
#Test Data
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))
rfc = RandomForestClassifier(verbose=True) #uses randomized decision trees
epochs = 5
for i in range(epochs):
    # Fit the model to the training data
    rfc.fit(X_train, y_train)
    # Evaluate the model on the test data
    score = rfc.score(X_test, y_test)
    # Print the current epoch and test score
    print(f'Epoch {i+1}/{epochs} - Accuracy: {score:.2f}')

gnb = GaussianNB()
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_over, y_over = ros.fit_resample(X_train, y_train)

gnbmodel = gnb.fit(X_over, y_over.astype('int'))
y_pred = gnbmodel.predict(X_test)
print ("Score:", gnbmodel.score(X_test, y_test))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
lgr = LogisticRegression()
lgr.fit(X_over, y_over)
y_pred = lgr.predict(X_test)
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

dtc = DecisionTreeClassifier()
dtc.fit(X_over, y_over)
y_pred = dtc.predict(X_test)
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
#Ensemble methods from here 
abc = AdaBoostClassifier() 
abc.fit(X_over, y_over)
y_pred = abc.predict(X_test)
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
rfc = RandomForestClassifier(verbose=True) #uses randomized decision trees
rfcmodel = rfc.fit(X_over, y_over)
y_pred = rfc.predict(X_test)
print ("Accuracy:", rfcmodel.score(X_test, y_test))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
pickle.dump(tfIdfVectorizer,open('vectorizers.pkl','wb'))
pickle.dump(rfc,open('models.pkl','wb'))
joblib.dump(rfc,'modell.joblib')