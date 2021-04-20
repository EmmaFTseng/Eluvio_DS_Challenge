import pandas as pd
import string

import re
import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

sentiments= ['anger', 'anticipation', 'disgust', 'fear', 'joy',
             'sadness', 'surprise', 'trust', 'negative', 'positive']

# Number of times we do prediction on NRC emotion words
# We get an average of all predictions to minimize errors.
pred_num = 100
print('number of classifications = ', pred_num)

#
# Load in data for prediction
#
data_set_for_prediction = pd.read_csv('NRC_Emotion.csv', encoding='ISO-8859-1')

sentiment_dict = {} #for example, key = 'nagative', value = ['crook', 'crude', 'cursed']
NRC_dict = {}

# Initialize sentiment_dict. After initialization,
for i in range(len(sentiments)):
    sentiment_dict[sentiments[i]] = []
    NRC_dict[sentiments[i]] = []

# Store emotion lexicons into respective category
for i in range(data_set_for_prediction.shape[0]):
    word = data_set_for_prediction.iloc[i,0]
    sentiment = data_set_for_prediction.iloc[i,1]
    value = data_set_for_prediction.iloc[i,2]

    if (value == 1):
        sentiment_dict[sentiment].append(word)

for w in sentiment_dict:
    print(w, ' = ', len(sentiment_dict[w]))

#
# NRC lexicons are concatenated to become strings. Each string consists of 50 lexicons.
# From one string to next string, 5 words are shifted out and filled with 5 new words in
# the tail. Then, prediction is performed on these strings.
#
from nltk.util import ngrams
import random

for i in range(len(sentiments)):
    random.shuffle(sentiment_dict[sentiments[i]])

    word_list = []
    count = 0
    for ng in ngrams(sentiment_dict[sentiments[i]], 15):
        if (count == 0):
            word_list.append(' '.join(str(k) for k in ng))
            count = 4 
        else:
            count -= 1

    for w in word_list:
        NRC_dict[sentiments[i]].append(w)

#######
####### Start Classifications
#######
        
classifier_names = ["Naive Bayes", "Random Forest", "Linear SVM"]

# Store the total of 0-100 upvotes predictions
# e.g. MNB_count_r1['Anger'] = 100
MNB_count_r1 = {}
RF_count_r1 = {}
SVM_count_r1 = {}

# Store the total of 100-1000 upvotes predictions
# e.g. MNB_count_r2['Negative'] = 70
MNB_count_r2 = {}
RF_count_r2 = {}
SVM_count_r2 = {}

# Store the total of 1000+ upvotes predictions
# e.g. MNB_count_r3['Negative'] = 70
MNB_count_r3 = {}
RF_count_r3 = {}
SVM_count_r3 = {}

# Initialization 
for i in range(len(sentiments)):
    MNB_count_r1[sentiments[i]] = 0
    RF_count_r1[sentiments[i]] = 0
    SVM_count_r1[sentiments[i]] = 0
    
    MNB_count_r2[sentiments[i]] = 0
    RF_count_r2[sentiments[i]] = 0
    SVM_count_r2[sentiments[i]] = 0    

    MNB_count_r3[sentiments[i]] = 0
    RF_count_r3[sentiments[i]] = 0
    SVM_count_r3[sentiments[i]] = 0    

# Iterate through the prediction results to count
# how many are each upvotes range (0-100, 100-1000, 1000+) 
def count(prediction, sens, clf):
    r1_headlines = 0
    r2_headlines = 0
    r3_headlines = 0

    for i in range (len(prediction)):
          if (y_pred[i] == '0-100'):
              r1_headlines = r1_headlines + 1
          elif (y_pred[i] == '100-1000'):
              r2_headlines = r2_headlines + 1
          else:
              r3_headlines = r3_headlines + 1

    # Accumulate all predictions. In the end, calculate average.
    if (clf == classifier_names[0]):
        MNB_count_r1[sens] += r1_headlines
        MNB_count_r2[sens] += r2_headlines       
        MNB_count_r3[sens] += r3_headlines       
    elif (clf == classifier_names[1]):
        RF_count_r1[sens] += r1_headlines
        RF_count_r2[sens] += r2_headlines       
        RF_count_r3[sens] += r3_headlines       
    else:
        SVM_count_r1[sens] += r1_headlines
        SVM_count_r2[sens] += r2_headlines              
        SVM_count_r3[sens] += r3_headlines       

# Clean up input text     
def preprocessing(dataset):
    data = []

    for i in range(dataset.shape[0]):

        sms = dataset.iloc[i, 1]
        
        # convert to lower case
        sms = sms.lower()

        sms = re.sub(r'@', 'mmmnnn-', sms)
        sms = re.sub(r'#', 'hhhggg-', sms)

        # tokenize
        tokenized_sms = word_tokenize(sms)
 
        # remove stop words and lemmatizing, no need of stemming because 'rigged' could
        # be an adjective and we want to retain it. But stemming will change 'rigged' to
        # 'rig'.
        sms_processed = []
        for word in tokenized_sms:
            if word in string.punctuation:
                continue

            if word in set(english_stopwords):
                continue
          
            word = re.sub('mmmnnn-', '@', word)
            word = re.sub('hhhggg-', '#', word)
                   
            sms_processed.append(wnl.lemmatize(word))

        sms_text = " ".join(sms_processed)
        
        data.append(sms_text)
    
    return data
        
data_set = pd.read_csv('news_headlines.csv', encoding='ISO-8859-1')

processed_data = preprocessing(data_set)

# Creating the feature matrix
from sklearn.feature_extraction.text import TfidfVectorizer
matrix = TfidfVectorizer(max_features=1000)

X = matrix.fit_transform(processed_data).toarray()

feature_names = matrix.get_feature_names()

# Retrieve column 0
y = data_set.iloc[:,0]

# Split train and test data
from sklearn.model_selection import StratifiedKFold

X_train, X_test, y_train, y_test = [], [], [], [] 
skfold = StratifiedKFold(n_splits=2)
for train_idx, test_idx in skfold.split(X,y):
	X_train, X_test = X[train_idx], X[test_idx]
	y_train, y_test = y[train_idx], y[test_idx]

# NB
from sklearn.naive_bayes import MultinomialNB

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# SVM
from sklearn.svm import SVC

# Put classifiers together to run as a pipeline
classifiers = [MultinomialNB(),
    RandomForestClassifier(n_estimators=1000, max_depth=100),
    SVC(kernel="linear")]

#
# Start train and predict
#
# Confusion matrix
#from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import accuracy_score  

for clf, name in zip(classifiers, classifier_names):

    #
    # First, train the classifier.
    #
    clf.fit(X_train, y_train)

    # Second, make prediction on X_test.
    y_pred = clf.predict(X_test)

    # Third, match the prediction against actual class to
    # obtain the accuracy of prediction.
    accuracy = accuracy_score(y_test, y_pred)
    print('\n\n-------------------')
    print(name, ', accuracy = {0:0.4f}'.format(accuracy))
    
    ##### Prediction Class on NRC  #####

    for K in range(pred_num):
        for i in range(len(sentiments)):
            #
            # Use same feature names as those used to train the classifiers
            #
            matrix_use_feature_names = TfidfVectorizer(vocabulary=feature_names)

            # NRC_dict[sentiments[i]] is a string with 50 NRC emotion words
            X_sentiments = matrix_use_feature_names.fit_transform(NRC_dict[sentiments[i]]).toarray()
            y_pred = clf.predict(X_sentiments)

            count(y_pred, sentiments[i], name)

#
# All predictions are done. Calculate the average of prediction results.
#
for i in range(len(sentiments)):
    print('---------------------')
    print(sentiments[i])
    print('---MNB---')
    print(int(MNB_count_r1[sentiments[i]]/pred_num))
    print(int(MNB_count_r2[sentiments[i]]/pred_num))
    print(int(MNB_count_r3[sentiments[i]]/pred_num))
    print('---RF---')
    print(int(RF_count_r1[sentiments[i]]/pred_num))
    print(int(RF_count_r2[sentiments[i]]/pred_num))
    print(int(RF_count_r3[sentiments[i]]/pred_num))
    print('---SVM---')
    print(int(SVM_count_r1[sentiments[i]]/pred_num))
    print(int(SVM_count_r2[sentiments[i]]/pred_num))
    print(int(SVM_count_r3[sentiments[i]]/pred_num))
    

