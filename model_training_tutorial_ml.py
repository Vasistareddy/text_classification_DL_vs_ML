from __future__ import print_function

import os, re, sys, string, argparse
import pandas as pd
import numpy as np

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

stemmer = SnowballStemmer('english')
t = str.maketrans(dict.fromkeys(string.punctuation))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', help='path of the filename')
args = parser.parse_args()
file_path = args.dataset_path

def clean_text(text):  
    ## Remove Punctuation
    text = text.translate(t) 
    text = text.split()

    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [stemmer.stem(w) for w in text if not w in stops]
    
    text = " ".join(text)
    text = re.sub(' +',' ', text)
    return text

def dataset_preparation(filepath):
    """ preparing a dataset by:
        stemming and removing stop-words from data-set
        dropping N/A
        dropping duplicates
    """
    if '.csv' in filepath:
        df = pd.read_csv(filepath)
        df['content'] = df['headline'] + ' ' + df['short_description']
        df['label'] = df['category']
        df = df[['content', 'label']]
        df = df.astype('str').applymap(str.lower)
        df = df.applymap(str.strip).replace(r"[^a-z0-9 ]+", '')
        df = df.dropna()
        df['content'] = df['content'].apply(clean_text)
        df = df.dropna()
        df = df.drop_duplicates()
    else:
        raise Exception('dataset file path should be CSV and there must be data exist')
    return df

def report_generation(classifier, train_data, valid_data, train_y, valid_y):
	classifier.fit(train_data, train_y)
	predictions = classifier.predict(valid_data)
	print("Accuracy :", metrics.accuracy_score(predictions, valid_y))
	report = classification_report(valid_y, predictions, output_dict=True, \
		target_names=target_names)
	return report

if __name__ == '__main__':
    df = dataset_preparation(file_path)
    df = df.dropna()

    # split the data into training and validation
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['content'], df['label'])

    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    target_names = list(encoder.classes_) # output labels for report generation

    # count vectorization 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(df['content'])
    xtrain_count = count_vect.transform(train_x)
    xvalid_count = count_vect.transform(valid_x)

    # word level tf-idf vectorization
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(df['content'])
    xtrain_tfidf = tfidf_vect.transform(train_x)
    xvalid_tfidf = tfidf_vect.transform(valid_x)

    # ngram level tf-idf vectorization
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(df['content'])
    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

    # initializing the classifier and fitting the data into the model
    # Naive Bayes
    classifier = naive_bayes.MultinomialNB()
    report = report_generation(classifier, xtrain_count, xvalid_count, train_y, valid_y)
    print("NB Count Vectorizer Report :", report['weighted avg'])

    report = report_generation(classifier, xtrain_tfidf, xvalid_tfidf, train_y, valid_y)
    print("NB TFIDF-Word Report :", report['weighted avg'])

    report = report_generation(classifier, xtrain_tfidf_ngram, xvalid_tfidf_ngram, train_y, valid_y)
    print("NB TFIDF-NGram Report :", report['weighted avg'])

    # Logistic Regression
    classifier = linear_model.LogisticRegression()
    report = report_generation(classifier, xtrain_count, xvalid_count, train_y, valid_y)
    print("LogisticRegression Count Vectorizer Report :", report['weighted avg'])

    report = report_generation(classifier, xtrain_tfidf, xvalid_tfidf, train_y, valid_y)
    print("LogisticRegression TFIDF-Word Report :", report['weighted avg'])

    report = report_generation(classifier, xtrain_tfidf_ngram, xvalid_tfidf_ngram, train_y, valid_y)
    print("LogisticRegression TFIDF-NGram Report :", report['weighted avg'])

    # Support Vector Machines
    classifier = svm.SVC(gamma="scale")
    report = report_generation(classifier, xtrain_count, xvalid_count, train_y, valid_y)
    print("SVM Count Vectorizer Report :", report['weighted avg'])

    report = report_generation(classifier, xtrain_tfidf, xvalid_tfidf, train_y, valid_y)
    print("SVM TFIDF-Word Report :", report['weighted avg'])

    report = report_generation(classifier, xtrain_tfidf_ngram, xvalid_tfidf_ngram, train_y, valid_y)
    print("SVM TFIDF-NGram Report :", report['weighted avg'])