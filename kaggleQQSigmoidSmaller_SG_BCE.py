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
from keras.layers import Input, Conv1D, MaxPooling1D
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

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
print("Cleaning text...")
trainDf['question1'] = cleanText(trainDf['question1'])
trainDf['question2'] = cleanText(trainDf['question2'])
# testDf['question1'] = cleanText(testDf['question1'])
# testDf['question2'] = cleanText(testDf['question2'])
print("Cleaned text.")

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
print("encoding qs - 1 of 2:")
encodedTrainQ1s = encodeQs(trainQs1, inputLength, alphabet)
print("encoded q1, encoding q2:")
encodedTrainQ2s = encodeQs(trainQs2, inputLength, alphabet)
print("encoded q1 and q2")


# MODEL

# Inputs
inputA = Input(shape=(inputLength,), dtype='int32')
inputB = Input(shape=(inputLength,), dtype='int32')

# One hot encoding
oheInputA = Lambda(K.one_hot, arguments={
                   'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputA)
oheInputB = Lambda(K.one_hot, arguments={
                   'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputB)


def createBaseNetworkSmaller(inputLength, inputDim):
    baseNetwork = Sequential()
    baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu',  input_shape=(inputLength, inputDim),
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Flatten())
    baseNetwork.add(Dense(32, activation='relu'))
    baseNetwork.add(Dropout(0.5))
    return baseNetwork

baseNetwork = createBaseNetworkSmaller(inputLength, inputDim)

# because we re-use the same instance `base_network`,
# the weights of the network will be shared across the two branches
processedA = baseNetwork(oheInputA)
processedB = baseNetwork(oheInputB)

# Concatenate
conc = Concatenate()([processedA, processedB])
x = BatchNormalization()(conc)

# Dense
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

predictions = Dense(1, activation='sigmoid')(x)

model = Model([inputA, inputB], predictions)

print(model.summary())

# Compile
initLR = 0.001
# momentum = 0.9
# sgd = SGD(lr=initLR, momentum=momentum, decay=0, nesterov=False)
model.compile(loss='binary_crossentropy',
              optimizer='nadam', metrics=['accuracy'])

# MAKE VAL DATA
validationSplit = 0.2  # Use this much for validation


# Learning Rate Schedule
# Halve lr every 3 epochs
def step_decay(epoch):
    initial_lrate = initLR
    drop = 0.5
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop, math.floor(epoch / epochs_drop))
    print("lr dropped to " + str(lrate))
    return lrate

lRate = LearningRateScheduler(step_decay)

# Model Checkpoint
filepath = "charCNN-smAl-C256P3C256P3f32-conc-f64-val0.2-epoch{epoch:02d}-l{loss:.4f}-a{acc:.4f}-vl{val_loss:.4f}-va{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, verbose=1, save_best_only=False, save_weights_only=True)

# Callbacks
callbacks_list = [checkpoint]

# Hyperparameters
minibatchSize = 100
nEpochs = 40

# # SKIP: Load weights
# model.load_weights(
#     "charCNNSmaller-ohE-smAl-val0.2-epoch15-l0.3907-a0.8311-vl0.4616-va0.7891.hdf5")

# Train
history = model.fit([encodedTrainQ1s, encodedTrainQ2s], trainOutputs,
                    batch_size=minibatchSize, epochs=nEpochs, verbose=1,
                    callbacks=callbacks_list, validation_split=validationSplit,
                    initial_epoch=0)
