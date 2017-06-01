# https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
# Pre-requisites
import numpy as np
import pandas as pd
from collections import Counter
import os
import re
import csv
import codecs
from sys import getsizeof
import time
import math
# import cv2

# To clear print buffer
# from IPython.display import clear_output

# tensorflow
# import tensorflow as tf
#     with tf.device('/gpu:0'):

from gensim.models import KeyedVectors

# Keras
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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

EMBEDDING_FILE = '/home/voletiv/Downloads/GoogleNews-vectors-negative300.bin'

# LOAD TRAINING AND TESTING DATA

# Download train.csv and test.csv from
# https://www.kaggle.com/c/quora-question-pairs/
TRAIN_DATA_FILE = 'kaggleQuoraTrain.csv'
TEST_DATA_FILE = 'kaggleQuoraTest.csv'
trainDf = pd.read_csv(TRAIN_DATA_FILE, sep=',')
testDf = pd.read_csv(TEST_DATA_FILE, sep=',')

# Check for any null values
print(trainDf.isnull().sum())
# print(testDf.isnull().sum())

# Add the string 'empty' to empty strings
trainDf = trainDf.fillna('empty')
testDf = testDf.fillna('empty')

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
nOfTestQs = len(testDf['question1'])
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


# To encode questions into char indices
def encodeQs(questions, inputLength, alphabet):
    # Initialize encoded questions array
    encodedQs = np.zeros((len(questions), inputLength), dtype='int32')
    # For each question
    for (q, question) in enumerate(questions):
        print(str(q) + " of " + str(len(questions)) + " = " +
              "{0:.2f}".format(float(q) / len(questions)), end='\r')
        # For each character in question, in reversed order (so latest
        # character is first)
        for (c, char) in enumerate(reversed(question[:inputLength])):
            # print(\"  \"+str(c))
            if char in alphabet:
                encodedQs[q][c] = alphabet.index(char)
            else:
                encodedQs[q][c] = 0
    print("Done encoding.")
    return encodedQs

# Make encoded questions out of training questions 1 and 2
print("encoding train qs - 1 of 2:")
encodedTrainQ1s = encodeQs(trainQ1s, inputLength, alphabet)
print("encoded train q1, encoding train q2:")
encodedTrainQ2s = encodeQs(trainQ2s, inputLength, alphabet)
print("encoded train q1 and q2")
print("encoding val qs - 1 of 2:")
encodedValQ1s = encodeQs(valQ1s, inputLength, alphabet)
print("encoded val q1, encoding val q2:")
encodedValQ2s = encodeQs(valQ2s, inputLength, alphabet)
print("encoded val q1 and q2")


# MODEL

# Inputs
inputA = Input(shape=(inputLength,), dtype='int32')
inputB = Input(shape=(inputLength,), dtype='int32')

# One hot encoding
oheInputA = Lambda(K.one_hot, arguments={
                   'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputA)
oheInputB = Lambda(K.one_hot, arguments={
                   'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputB)


def createBaseNetworkSmall(inputLength, inputDim):
    baseNetwork = Sequential()
    baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu',  input_shape=(inputLength, inputDim),
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Flatten())
    baseNetwork.add(Dense(64, activation='relu'))
    baseNetwork.add(BatchNormalization())
    baseNetwork.add(Dropout(0.5))
    baseNetwork.add(Dense(64, activation='relu'))
    baseNetwork.add(BatchNormalization())
    return baseNetwork

baseNetwork = createBaseNetworkSmall(inputLength, inputDim)

# because we re-use the same instance `base_network`,
# the weights of the network will be shared across the two branches
processedA = baseNetwork(oheInputA)
processedB = baseNetwork(oheInputB)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processedA, processedB])

model = Model([inputA, inputB], distance)

print(model.summary())

# Compile
# initLR = 0.001
# momentum = 0.9
# sgd = SGD(lr=initLR, momentum=momentum, decay=0, nesterov=False)
# model.compile(loss='binary_crossentropy',
#               optimizer='nadam', metrics=['accuracy'])


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)

# Model Checkpoint
filepath = "charCNN-maAl-sepValSplit0.2-C256P3C256P3f64BnDo0.5f64-eucDist-epoch{epoch:02d}-l{loss:.4f}-vl{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, verbose=1, save_best_only=False, save_weights_only=True)


# Callbacks
callbacks_list = [checkpoint]

# # SKIP: Load weights
# model.load_weights(
#     "charCNNSmaller-ohE-smAl-val0.2-epoch15-l0.3907-a0.8311-vl0.4616-va0.7891.hdf5")

# Train
history = model.fit([encodedTrainQ1s, encodedTrainQ2s], trainOutputs,
                    batch_size=minibatchSize, epochs=nEpochs, verbose=1,
                    callbacks=callbacks_list, validation_data=(
                        [encodedValQ1s, encodedValQ2s], valOutputs),
                    initial_epoch=0)
