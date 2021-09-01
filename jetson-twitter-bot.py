import os
from os import path
import sys
import re
import time
import random
import json
#Data science imports
import pandas as pd
import numpy as np
#AIML imports
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, Dropout, LSTM, Input
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.models import model_from_json
import h5py
#NLP Imports
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
#Module specific imports
import twitter as tw
#import t_keys as keys

#Initialization and globals
PHYSICAL_DEVICES = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)
except:
    pass

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

DIR = os.getcwd()
if not os.path.isdir('./files'):
    os.makedirs('data')
DIR = DIR + '/data'

def tweet_reader(file):
    print("\nParsing tweets from tweets text file...")
    os.chdir(DIR)
    tweet_data = pd.DataFrame(columns=['Tweets', 'LD'])
    with open(file, 'r') as f:
        for line in f:
            hash_removed = re.sub(r'\#\S+', '', str(line))
            mention_removed = re.sub(r'\@\S+', '', str(hash_removed))
            excess_newlines_removed = re.sub(r'\//n\S+', '', str(hash_removed))
            if (mention_removed[0:2] == 'RT'):
                continue #skip retweets
            else:
                filtered = " ".join(mention_removed.split()) #remove extra spaces, tabs, newlines
            LD = lexical_diversity(line)
            tweet_data = tweet_data.append({'Tweets': filtered, 'LD': LD}, ignore_index=True, sort=True)
        tweet_data.drop_duplicates(subset=['Tweets'])
    f.close()
    return tweet_data

def lexical_diversity(text):
    """
    Measurement of variety of words used
    """
    return len(set(str(text))) / len(text)

def sorter(tweet_data):
    inter = []
    words = []
    print("Tokenizing tweets...")
    t = TweetTokenizer()
    for index, r, in tweet_data.iterrows():
        inter.append(r['Tweets'])
        tokens = t.tokenize(r['Tweets'])
        for token in tokens:
            words.append(token)
    return words, inter

def generate(words):
    """
    This kicks off the process of generating text.
    """
    print('Beginning In-Depth Analysis...')
    chars = words
    char_to_num = dict((c, i) for i, c in enumerate(chars))
    input_len = len(words)
    vocab_len = len(chars)
    print(f'Total number of characters: {input_len}')
    print(f'Total vocab: {vocab_len}')
    seq_length = 500
    x_data = []
    y_data = []
    for i in range(0, input_len - seq_length, 1):
        in_seq = words[i:i + seq_length]
        out_seq = words[i + seq_length]
        x_data.append([char_to_num[char] for char in in_seq])
        y_data.append(char_to_num[out_seq])
    n_patterns = len(x_data)
    X = np.reshape(x_data, (n_patterns, seq_length, 1))
    X = X/float(vocab_len)
    y = utils.to_categorical(y_data)

    return X, y, x_data, chars
    #premodeling(X, y)
    #trainer(x_data, CHARS)

def premodeling(X, y):
    """
    This creates the Statistical model for generating Tweets.
    """
    print('Creating the statistical model...')
    filename = 'model.h5'
    global FILENAME
    FILENAME = filename
    if path.exists('model.json'):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('model.h5')
        loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        filepath = 'model.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, \
                                     save_best_only=True, mode='min')
        json_file.close()
        with tf.device('/gpu:0'):
            MODEL.fit(X, y, epochs=7, batch_size=8, callbacks=checkpoint)
            MODEL.load_weights(FILENAME)
            MODEL.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        model_json = MODEL.to_json()

        with open('model.json', 'w') as json_file:
            json_file.write(model_json)
    else:
        modeling(X, y)

def modeling(X, y):
    MODEL.add(LSTM(224, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    MODEL.add(Dropout(0.6))
    MODEL.add(LSTM(224, return_sequences=True))
    MODEL.add(Dropout(0.6))
    MODEL.add(LSTM(224))
    MODEL.add(Dropout(0.4))
    MODEL.add(Dense(y.shape[1], activation='softmax'))
    optimizer = RMSprop(lr=0.001)
    MODEL.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    filepath = 'model.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, \
                                 save_best_only=True, mode='min')
    with tf.device('/gpu:0'):
        MODEL.fit(X, y, epochs=5, batch_size=32, callbacks=checkpoint)
        MODEL.load_weights(FILENAME)
        MODEL.compile(loss='binary_crossentropy', optimizer='rmsprop')
        #MODEL.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model_json = MODEL.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)

def trainer(x_data, chars):
    """
    This trains the model for generating Tweets.
    """
    print('Training the model...')
    global NUM_TO_CHAR
    NUM_TO_CHAR = dict((i, c) for i, c in enumerate(chars))
    start = np.random.randint(0, len(x_data) -1)
    pattern = x_data[start]
    print('Random Seed: Created\n\t')
    #tweet_creator(pattern)

#main
MODEL = Sequential()
words, inter = sorter(tweet_reader('tweets.txt'))
X, y, x_data, chars = generate(words)
premodeling(X, y)
trainer(x_data, chars)