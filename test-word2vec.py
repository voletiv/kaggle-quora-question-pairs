# Pre-requisites
import numpy as np
import pandas as pd
from collections import Counter
import re
import os
from sys import getsizeof
import time
import math
# import cv2

# To clear print buffer
# from IPython.display import clear_output

# tensorflow
import tensorflow as tf
#     with tf.device('/gpu:0'):

# Keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Merge
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

# PARAMETERS
MAX_NB_WORDS = 200000
inputLength = 1014  # input feature length (the paper used 1014)
validationSplit = 0.2
minibatchSize = 100
nEpochs = 100

# # Alphabet
# # Space is the first char because its index has to be 0
# alphabet = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
#             'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6',
#             '7', '8', '9', '!', '^', '+', '-', '=']

# Load alphabet
alphabet = np.load("mathAlphabet.npy")
alphabet = [str(a) for a in alphabet]
print(alphabet)

# Params
inputDim = len(alphabet)  # number of letters (characters) in alphabet

# LOAD TRAINING AND TESTING DATA

# Download train.csv and test.csv from
# https://www.kaggle.com/c/quora-question-pairs/
# TRAIN_DATA_FILE = 'kaggleQuoraTrain.csv'
TEST_DATA_FILE = 'kaggleQuoraTest.csv'
# trainDf = pd.read_csv(TRAIN_DATA_FILE, sep=',')
testDf = pd.read_csv(TEST_DATA_FILE, sep=',')

# Check for any null values
# print(trainDf.isnull().sum())
print(testDf.isnull().sum())

# Add the string 'empty' to empty strings
# trainDf = trainDf.fillna('empty')
testDf = testDf.fillna('empty')

# # Load idx of train and val
# trainIdx = np.load("trainIdx.npy")
# valIdx = np.load("valIdx.npy")


# To clean data
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    # Return a list of words
    return(text)

# Clean text
print("Cleaning text...")

nOfTestQs = len(testDf['question1'])
testQ1s = []
for i, q in enumerate(testDf['question2']):
    print("Cleaning Test Q1s: {0:.2f}".format(float(i) / nOfTestQs), end='\r')
    testQ1s.append(text_to_wordlist(q))

testQ2s = []
for i, q in enumerate(testDf['question2']):
    print("Cleaning Test Q2s: {0:.2f}".format(float(i) / nOfTestQs), end='\r')
    testQ2s.append(text_to_wordlist(q))

print("Cleaned text.")


# Text to sequence of word-idx
testSeq1s = tokenizer.texts_to_sequences(testQ1s)
testSeq2s = tokenizer.texts_to_sequences(testQ2s)

# Padded sequences
testPadSeq1s = pad_sequences(testSeq1s, maxlen=MAX_SEQUENCE_LENGTH)
testPadSeq2s = pad_sequences(testSeq2s, maxlen=MAX_SEQUENCE_LENGTH)

# # Prepare embeddings
# print('Preparing embedding matrix')

# nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

# embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if word in word2vec.vocab:
#         embedding_matrix[i] = word2vec.word_vec(word)

# print('Null word embeddings: %d' %
#       np.sum(np.sum(embedding_matrix, axis=1) == 0))


# To encode questions into wordvecs
def encodeQs(questions, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, embedding_matrix, word_index):
    # Initialize encoded questions array
    encodedQs = np.zeros((len(questions), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), dtype='int32')
    # For each question
    for (q, question) in enumerate(questions):
        print(str(q) + " of " + str(len(questions)) + " = " +
              "{0:.2f}".format(float(q) / len(questions)), end='\r')
        # For each character in question, in reversed order (so latest
        # character is first)
        for (w, word) in enumerate(question[:MAX_SEQUENCE_LENGTH]):
            # print(\"  \"+str(c))
            if word in word_index:
                encodedQs[q][w] = embedding_matrix[word_index[word]]
    print("Done encoding.")
    return encodedQs



# MODEL

# Inputs
input1 = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), dtype='int32')
input2 = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), dtype='int32')

baseNetwork = Sequential()
baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu',  input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM),
                       kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
baseNetwork.add(MaxPooling1D(pool_size=2, strides=2))
baseNetwork.add(Conv1D(256, 5, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
    mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
baseNetwork.add(MaxPooling1D(pool_size=2, strides=2))
baseNetwork.add(Flatten())
baseNetwork.add(Dense(256, activation='relu'))
baseNetwork.add(Dropout(0.9))
baseNetwork.add(BatchNormalization())
baseNetwork.add(Dense(512, activation='relu'))

# because we re-use the same instance `base_network`,
# the weights of the network will be shared across the two branches
processed1 = baseNetwork(input1)
processed2 = baseNetwork(input2)

# Concatenate
conc = Concatenate()([processed1, processed2])
x = Dropout(0.9)(conc)
x = BatchNormalization()(x)

# Dense
x = Dense(512, activation='relu')(x)
x = Dropout(0.9)(x)
x = BatchNormalization()(x)

predictions = Dense(1, activation='sigmoid')(x)

# def euclidean_distance(vects):
#     x, y = vects
#     return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


# def eucl_dist_output_shape(shapes):
#     shape1, shape2 = shapes
#     return (shape1[0], 1)

# distance = Lambda(euclidean_distance,
#                   output_shape=eucl_dist_output_shape)([processedA, processedB])

model = Model([input1, input2], predictions)

print(model.summary())


# Accuracy
def eucAcc(y_true, y_pred):
    thresh = 0.5
    return K.mean(K.equal(y_true, tf.to_float(K.less(thresh, y_pred))), axis=-1)


# Logloss
def eucLL(y_true, y_pred):
    myEps = 1e-15
    probs = K.maximum(K.minimum(y_pred, 1 - myEps), myEps)
    return K.mean(K.binary_crossentropy(probs, y_true), axis=-1)

# Compile
# initLR = 0.001
# momentum = 0.9
# sgd = SGD(lr=initLR, momentum=momentum, decay=0, nesterov=False)
model.compile(loss='binary_crossentropy',
              optimizer='nadam', metrics=[eucAcc, eucLL])
model.load_weights(
    "word2vecCNN-C256-3-P2-C256-5-P2-f256-Do0.5-Bn-f256-conc-Do0.5-Bn-f1024-Do0.5-Bn-f1-nadam-epoch05-tl0.1784-tacc0.9280-tlogl0.1784-vl1.1343-vacc0.6095-vlogl1.1343.hdf5")

preds = model.predict(
    [testPadSeq1s, testPadSeq2s], verbose=1)

yTest = -np.ones((len(testPadSeq1s), 2)).astype(int)
yTest[:, 0] = np.array(list(range(len(testPadSeq1s))))
yTest[:, 1] = np.reshape((preds < 0.15).astype(int), (len(preds),))

np.savetxt("PREDS_word2vecCNN.csv", yTest, fmt='%i',
           delimiter=',', header="test_id,is_duplicate", comments='')
