# https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
# Pre-requisites
import numpy as np
import pandas as pd
import os
import re
import csv
import codecs
from sys import getsizeof
import time
import math
# import cv2
print("Imported pre-reqs")

# To clear print buffer
# from IPython.display import clear_output

# tensorflow

import tensorflow as tf
print("Imported tf")
#     with tf.device('/gpu:0'):

from gensim.models import KeyedVectors
print("Imported gensim")

# Keras
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, Activation
from keras.layers import Conv1D, MaxPooling1D, Merge
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
print("Imported all Keras")

# PARAMETERS
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 300
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

EMBEDDING_FILE = '/home/voletiv/Downloads/GoogleNews-vectors-negative300.bin'

# LOAD TRAINING AND TESTING DATA

# Download train.csv and test.csv from
# https://www.kaggle.com/c/quora-question-pairs/
TRAIN_DATA_FILE = 'kaggleQuoraTrain.csv'
TEST_DATA_FILE = 'kaggleQuoraTest.csv'
trainDf = pd.read_csv(TRAIN_DATA_FILE, sep=',')
# testDf = pd.read_csv(TEST_DATA_FILE, sep=',')

# Check for any null values
print(trainDf.isnull().sum())
# print(testDf.isnull().sum())

# Add the string 'empty' to empty strings
trainDf = trainDf.fillna('empty')
# testDf = testDf.fillna('empty')

# # FIND Q PAIRS WHERE EACH Q OCCURS ONLY ONCE
# q1Ids = trainData[:, 1]
# q2Ids = trainData[:, 2]
# q1IdsBincount = np.bincount(q1Ids.astype('int64'))
# q2IdsBincount = np.bincount(q2Ids.astype('int64'))
# # len(q1IdsBincount) = 537933, len(q2IdsBincount) = 527934
# # To make their lengths equal:
# q1IdsBincount = np.append(q1IdsBincount, 0)
# uniqueQPairs = []
# for i in range(len(q1Ids)):
#     print(str(i) + " of " + str(len(q1Ids)), end='\r')
#     if q1IdsBincount[q1Ids[i]] == 1:
#         if q2IdsBincount[q1Ids[i]] == 0:
#             if q2IdsBincount[q2Ids[i]] == 1:
#                 if q1IdsBincount[q2Ids[i]] == 0:
#                     uniqueQPairs.append(i)

# # Save all unique Q pairs
# np.save("uniqueQPairs", uniqueQPairs)

# # Load all unique Q pairs
# uniqueQPairs = list(np.load("uniqueQPairs.npy"))

# # Make the validation data idx
# valFullIdx = uniqueQPairs
# np.random.shuffle(valFullIdx)
# nOfValQPairs = int(validationSplit * len(trainDf))
# valIdx = valFullIdx[:nOfValQPairs]

# # Make the rest as training data idx
# trainIdx = list(range(len(trainDf)))
# # for i, num in enumerate(valIdx):
# #     print("Removing val idx from trainIdx: " + str(i) + " of " + str(len(valIdx)), end='\r')
# #     trainIdx.remove(num)
# trainIdx = [e for e in trainIdx if e not in valIdx]

# Load idx of train and val
trainIdx = np.load("trainIdx.npy")
valIdx = np.load("valIdx.npy")


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
nOfTrainQs = len(trainDf['question1'])
trainFullQ1s = []
for i, q in enumerate(trainDf['question1']):
    print("Cleaning Train Q1s: {0:.2f}".format(
        float(i) / nOfTrainQs), end='\r')
    trainFullQ1s.append(text_to_wordlist(q))

trainFullQ2s = []
for i, q in enumerate(trainDf['question2']):
    print("Cleaning Train Q2s: {0:.2f}".format(
        float(i) / nOfTrainQs), end='\r')
    trainFullQ2s.append(text_to_wordlist(q))

# nOfTestQs = len(testDf['question1'])
# testQ1s = []
# for i, q in enumerate(testDf['question2']):
#     print("Cleaning Test Q1s: {0:.2f}".format(float(i) / nOfTestQs), end='\r')
#     testQ2s.append(text_to_wordlist(q))

# testQ2s = []
# for i, q in enumerate(testDf['question2']):
#     print("Cleaning Test Q2s: {0:.2f}".format(float(i) / nOfTestQs), end='\r')
#     testQ2s.append(text_to_wordlist(q))

print("Cleaned text.")

# Make train and val data
trainQ1s = [trainFullQ1s[i] for i in trainIdx]
trainQ2s = [trainFullQ2s[i] for i in trainIdx]
valQ1s = [trainFullQ1s[i] for i in valIdx]
valQ2s = [trainFullQ2s[i] for i in valIdx]

# Outputs (whether duplicate or not)
trainData = np.array(trainDf)
trainOutputs = trainData[trainIdx, 5]
valOutputs = trainData[valIdx, 5]

# WORD 2 VEC

# Indexing word vectors
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,
                                             binary=True)

# Tokenizer - index words by numbers
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(trainQ1s + trainQ2s + valQ1s + valQ2s)

# Text to sequence of word-idx
trainSeq1s = tokenizer.texts_to_sequences(trainQ1s)
trainSeq2s = tokenizer.texts_to_sequences(trainQ2s)
valSeq1s = tokenizer.texts_to_sequences(valQ1s)
valSeq2s = tokenizer.texts_to_sequences(valQ2s)
# testSeq1s = tokenizer.texts_to_sequences(testQ1s)
# testSeq2s = tokenizer.texts_to_sequences(testQ2s)

# Word index
word_index = tokenizer.word_index

# Padded sequences
trainPadSeq1s = pad_sequences(trainSeq1s, maxlen=MAX_SEQUENCE_LENGTH)
trainPadSeq2s = pad_sequences(trainSeq2s, maxlen=MAX_SEQUENCE_LENGTH)
valPadSeq1s = pad_sequences(valSeq1s, maxlen=MAX_SEQUENCE_LENGTH)
valPadSeq2s = pad_sequences(valSeq2s, maxlen=MAX_SEQUENCE_LENGTH)
# testPadSeq1s = pad_sequences(testSeq1s, maxlen=MAX_SEQUENCE_LENGTH)
# testPadSeq2s = pad_sequences(testSeq2s, maxlen=MAX_SEQUENCE_LENGTH)

# Final
finalTrainPadSeq1s = np.vstack((trainPadSeq1s, trainPadSeq2s))
finalTrainPadSeq2s = np.vstack((trainPadSeq2s, trainPadSeq1s))
finalTrainOutputs = np.concatenate((trainOutputs, trainOutputs))
finalValPadSeq1s = np.vstack((valPadSeq1s, valPadSeq2s))
finalValPadSeq2s = np.vstack((valPadSeq2s, valPadSeq1s))
finalValOutputs = np.concatenate((valOutputs, valOutputs))

# Prepare embeddings
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)

print('Null word embeddings: %d' %
      np.sum(np.sum(embedding_matrix, axis=1) == 0))

# MODEL

# Inputs
input1 = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), dtype='int32')
input2 = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), dtype='int32')

# Embedding
embeddingLayer = Embedding(nb_words, EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, 
                            trainable=False)

embedded1 = embeddingLayer(input1)
embedded2 = embeddingLayer(input2)

baseNetwork = Sequential()
baseNetwork.add(Conv1D(128, 3, strides=1, padding='valid', activation='relu',  input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM),
                       kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
baseNetwork.add(MaxPooling1D(pool_size=2, strides=2))
baseNetwork.add(Conv1D(128, 5, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
    mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
baseNetwork.add(MaxPooling1D(pool_size=2, strides=2))
baseNetwork.add(Flatten())
baseNetwork.add(Dense(128, activation='relu'))
baseNetwork.add(Dropout(0.9))
baseNetwork.add(BatchNormalization())
baseNetwork.add(Dense(128, activation='relu'))

# because we re-use the same instance `base_network`,
# the weights of the network will be shared across the two branches
processed1 = baseNetwork(embedded1)
processed2 = baseNetwork(embedded2)

# Concatenate
conc = Concatenate()([processed1, processed2])
x = Dropout(0.9)(conc)
x = BatchNormalization()(x)

# Dense
x = Dense(128, activation='relu')(x)
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
              optimizer='nadam', metrics=[eucAcc])


# def contrastive_loss(y_true, y_pred):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 1
#     return K.mean(y_true * K.square(y_pred) +
#                   (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

# rms = RMSprop()
# model.compile(loss=contrastive_loss, optimizer=rms)

# Model Checkpoint
filepath = "word2vecCNN-C128-3-P2-C128-5-P2-f128-Do0.9-Bn-f128-conc-Do0.9-Bn-f128-Do0.9-Bn-f1-nadam-epoch{epoch:02d}-tl{loss:.4f}-tacc{eucAcc:.4f}-vl{val_loss:.4f}-vacc{val_eucAcc:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, verbose=1, save_best_only=False, save_weights_only=True)


# Callbacks
callbacks_list = [checkpoint]

# # SKIP: Load weights
# model.load_weights(
#     "charCNNSmaller-ohE-smAl-val0.2-epoch15-l0.3907-a0.8311-vl0.4616-va0.7891.hdf5")

# Train
history = model.fit([finalTrainPadSeq1s, finalTrainPadSeq2s], finalTrainOutputs,
                    batch_size=minibatchSize, epochs=nEpochs, verbose=1,
                    callbacks=callbacks_list, validation_data=(
                        [finalValPadSeq1s, finalValPadSeq1s], finalValOutputs),
                    initial_epoch=0)
