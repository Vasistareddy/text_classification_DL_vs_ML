from __future__ import print_function

import re, os, sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, Dense, Input, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Concatenate
from keras.models import Model
import theano.ifelse
import pickle
import operator
from collections import defaultdict
import pandas as pd
import argparse
from keras import backend as K
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
# import multiprocessing as mp

stemmer = SnowballStemmer('english')
t = str.maketrans(dict.fromkeys(string.punctuation))

# p = mp.Pool(mp.cpu_count())
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1

parser = argparse.ArgumentParser()
parser.add_argument('--filename', help='data filename')
parser.add_argument('--path', help='path where glove and required static data files needed')
parser.add_argument('--label_count', help='output prediction label count')
parser.add_argument('--epochs', help='number of model iterations')
parser.add_argument('--batch_size', help='data batch size in each iteration')
args = parser.parse_args()

filename = args.filename
home_path = args.path
label_count = int(args.label_count)
epochs = int(args.epochs)
batch_size = int(args.batch_size)

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

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dataset_preparation():
    """ preparing a dataset 
        cleaning dataset
        dropping N/A
        dropping duplicates
    """
    if not 'data/{}_cleaned.csv'.format(filename) in os.listdir():
        df = pd.read_csv("data/{}.csv".format(filename), encoding = "ISO-8859-1")
        df['content'] = df['headline'] + ' ' + df['short_description']
        df['label'] = df['category']
        df = df[['content', 'label']]
        df = df.astype('str').applymap(str.lower)
        df = df.applymap(str.strip).replace(r"[^a-z0-9 ]+", '')
        df = df.dropna()
        df['content'] = df['content'].apply(clean_text)
        df = df.dropna()
        df = df.drop_duplicates()
        df.to_csv('data/{}_cleaned.csv'.format(filename), encoding='utf-8', index=False)

    else:
        df = pd.read_csv("data/{}_cleaned.csv".format(filename), encoding = "ISO-8859-1")
        df = df[['content', 'label']]
        df = df.astype('str').applymap(str.lower)
        df = df.dropna()
        df = df.drop_duplicates()
    return df

def loading_embeddings():
    """ loading glove embeddings """
    embeddings_index = {}
    f = open(home_path + 'glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def prepare_embedding_matrix(word_index):
    """ preparing embedding matrix with our data set """

    embeddings_index = loading_embeddings()
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, num_words

def vectorizing_data(df):
    """ vectorizing and splitting the data for training, testing, validating """
    # vectorizing the text samples and labels into a 2D integer tensor
    label_s = df['label'].tolist()
    l = list(set(label_s))
    l.sort()
    labels_index = dict([(j,i) for i, j in enumerate(l)]) 
    labels = [labels_index[i] for i in label_s]

    print('Found %s texts.' % len(df['content']))
    print('labels_index --- ', labels_index)

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(df['content'])
    sequences = tokenizer.texts_to_sequences(df['content'])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    """
    if not home_path + 'word_index_tutorial.pickle' in os.listdir():
        with open(home_path + 'word_index_tutorial.pickle', 'wb') as handle:
            pickle.dump(word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """

    df = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print('Shape of df tensor:', df.shape)
    print('Shape of label tensor:', labels.shape)

    # randomizing and splitting the df into a training set, test set and a validation set
    indices = np.arange(df.shape[0])
    np.random.shuffle(indices)
    df = df[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * df.shape[0])

    x_train = df[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = df[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    x_test = x_train[-num_validation_samples:]
    y_test = y_train[-num_validation_samples:]
    return x_train, y_train, x_test, y_test, x_val, y_val, word_index

def model_generation(embedding_matrix, num_words):
    """ model generation """
    embedding_layer = Embedding(num_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    convs = []
    filter_sizes = [3,4,5]

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    for fsz in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)

    l_merge = Concatenate(axis=1)(convs)
    l_cov1= Conv1D(filters=128, kernel_size=5, activation='relu')(l_merge)
    l_cov1 = Dropout(0.2)(l_cov1)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(filters=128, kernel_size=5, activation='relu')(l_pool1)
    l_cov2 = Dropout(0.2)(l_cov2)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(label_count, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    return model

def training_evaluating_model(model, x_train, y_train, x_test, y_test, x_val, y_val):
    """ training the model with the train and validation data
    and evaluating the model with the test data """
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', f1_m, precision_m, recall_m])

    model.summary()
    # fitting the model
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size)

    """
    model.save_weights(home_path + 'newsfeeds_model_100_32_v2')
    """

    # evaluating the model
    loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy, f1_score, precision, recall

df = dataset_preparation()
print(df.groupby('label').count())

print('vectorizing data')
x_train, y_train, x_test, y_test, x_val, y_val, word_index = vectorizing_data(df)

print('Preparing embedding matrix.')
embedding_matrix, num_words = prepare_embedding_matrix(word_index)

print('model setting up')
model = model_generation(embedding_matrix, num_words)

print('calculating metrics')
loss, accuracy, f1_score, precision, recall = training_evaluating_model(model, x_train, y_train, x_test, y_test, x_val, y_val)

print("loss -- {} \naccuracy -- {} \nf1_score -- {} \nprecision -- {} \nrecall -- {} \n".format(float(format(loss,'.2f')), \
    float(format(accuracy*100,'.2f')), float(format(f1_score*100,'.2f')), float(format(precision*100,'.2f')), float(format(recall*100,'.2f'))))

