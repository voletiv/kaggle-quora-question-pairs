# Pre-requisites
import numpy as np
import pandas as pd
from collections import Counter
import os
from sys import getsizeof
import time
import math
# import cv2

# To clear print buffer
# from IPython.display import clear_output

# tensorflow
# import tensorflow as tf
#     with tf.device('/gpu:0'):

# Keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, GRU, Dense, Dropout
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

# # Alphabet
# # Space is the first char because its index has to be 0
# alphabet = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
#             'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6',
#             '7', '8', '9', '!', '?', ':', '$', '%', '^', '*', '+', '-', '=']

# Load alphabet
alphabet = np.load("smallerAlphabet.npy")
alphabet = [str(a) for a in alphabet]
print(alphabet)

# Params
inputDim = len(alphabet)  # number of letters (characters) in alphabet
inputLength = 1014  # input feature length (the paper used 1014)

# DATA

# Load training and test data
# Download train.csv and test.csv from
# https://www.kaggle.com/c/quora-question-pairs/
trainDf = pd.read_csv('kaggleQuoraTrain.csv', sep=',')
# testDf = pd.read_csv('kaggleQuoraTest.csv', sep=',')

# Check for any null values
print(trainDf.isnull().sum())
# print(testDf.isnull().sum())

# Add the string 'empty' to empty strings
trainDf = trainDf.fillna('empty')
# testDf = testDf.fillna('empty')


# To clean text
def cleanText(t):
    # Make lower case
    t = t.str.lower()
    # Remove all characters that are not in the defined alphabet
    # Full alphabet - r"[^a-z0-9?,'.\"-()/:+&’[\]%$^\\={}!“”_*#;|@ ]", ""
    # Final alphabet : [a-z0-9!?:'$%^*+-= ]
    t = t.str.replace(r"[^a-z0-9!?:'$%^&*+-= ]", "")
    # Clean text
    t = t.str.replace(r" & ", " and ")
    t = t.str.replace(r" &", " and ")
    t = t.str.replace(r"& ", " and ")
    t = t.str.replace(r"&", " and ")
    t = t.str.replace(r"what's", "what is")
    t = t.str.replace(r"'s", "")
    t = t.str.replace(r"'ve", " have")
    t = t.str.replace(r"can't", "cannot")
    t = t.str.replace(r"n't", " not")
    t = t.str.replace(r"i'm", "i am")
    t = t.str.replace(r"'re", " are")
    t = t.str.replace(r"'d", " would")
    t = t.str.replace(r"'ll", " will")
    t = t.str.replace(r"'", "")
    t = t.str.replace(r"(\d+)(k)", r"\g<1>000")
    t = t.str.replace(r" e g ", " eg ")
    t = t.str.replace(r" b g ", " bg ")
    t = t.str.replace(r" u s ", " american ")
    t = t.str.replace(r"0s", "0")
    t = t.str.replace(r" 9 11 ", " 911 ")
    t = t.str.replace(r"e - mail", "email")
    t = t.str.replace(r"j k", "jk")
    t = t.str.replace(r"\s{2,}", "")
    return t

# Clean text
trainDf['question1'] = cleanText(trainDf['question1'])
trainDf['question2'] = cleanText(trainDf['question2'])
# testDf['question1'] = cleanText(testDf['question1'])
# testDf['question2'] = cleanText(testDf['question2'])

# Convert into np array
trainData = np.array(trainDf)
# testData = np.array(testDf)

# Inputs
# Get list of questions in Question1 and Question2
trainQs1 = trainData[:, 3]
trainQs2 = trainData[:, 4]
# testQs1 = testData[:, 1]
# testQs2 = testData[:, 2]

# Outputs (whether duplicate or not)
trainOutputs = trainData[:, 5]

# ENCODE DATA


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
print("encoding qs")
encodedTrainQ1s = encodeQs(trainQs1, inputLength, alphabet)
print("encoded q1, encoding q2")
encodedTrainQ2s = encodeQs(trainQs2, inputLength, alphabet)
print("encoded q1 and q2")

# TRAIN

# MODEL

inputA = Input(shape=(inputLength,), dtype='int32')
inputB = Input(shape=(inputLength,), dtype='int32')

# One hot encoding
oheInputA = Lambda(K.one_hot, arguments={
                   'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputA)
oheInputB = Lambda(K.one_hot, arguments={
                   'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputB)

# LSTM
gruLayer = GRU(200, dropout=0.25, recurrent_dropout=0.25)
xA = gruLayer(oheInputA)
xB = gruLayer(oheInputB)

# Concatenate
conc = Concatenate()([xA, xB])

x = Dropout(0.25)(conc)
x = BatchNormalization()(x)

x = Dense(100, activation='relu')(x)
x = Dropout(0.25)(x)
x = BatchNormalization()(x)

predictions = Dense(1, activation='sigmoid')(x)

model = Model([inputA, inputB], predictions)

# Compile
model.compile(loss='binary_crossentropy',
              optimizer='nadam', metrics=['accuracy'])

# Make Val Data
validationSplit = 0.2  # Use this much for validation

# Early Stopping
earlyStopping = EarlyStopping(monitor='loss', patience=3)

# Model Checkpoint
filepath = "GRU300-smAl-val0.2-epoch{epoch:02d}-l{loss:.4f}-a{acc:.4f}-vl{val_loss:.4f}-va{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, verbose=1, save_best_only=False, save_weights_only=True)

# Callbacks
callbacks_list = [checkpoint]

# Hyperparameters
minibatchSize = 80
nEpochs = 500

# # SKIP: Load weights
# model.load_weights(
#     "charCNNSigmoid-SG-BCE-initLR0.01-m0.9-epoch13.hdf5")

# Train
history = model.fit([encodedTrainQ1s, encodedTrainQ2s], trainOutputs,
                    batch_size=minibatchSize, epochs=nEpochs, verbose=1,
                    callbacks=callbacks_list, validation_split=validationSplit,
                    initial_epoch=0)
