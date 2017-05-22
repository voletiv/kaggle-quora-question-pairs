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
# with tf.device('/gpu:0'):

# Keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.optimizers import SGD
from keras.initializers import RandomNormal
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

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

# Load alphabet
alphabet = np.load("smallerAlphabet.npy")
alphabet = [str(a) for a in alphabet]
print(alphabet)

# Setting alphabet size
alphabetSize = len(alphabet)

# Params
inputDim = alphabetSize  # number of letters (characters) in alphabet
inputLength = 1014  # input feature length (the paper used 1014)

# To encode questions into char indices


def encodeQs(questions, inputLength, alphabet):
    alphabetSize = len(alphabet)
    # Initialize encoded questions array
    encodedQs = np.zeros((len(questions), inputLength), dtype='int32')
    # For each question
    for (q, question) in enumerate(questions):
        print(str(q) + " of " + str(len(questions)) + " = " +
              "{0:.2f}".format(float(q + 1) / len(questions)), end='\r')
        # For each character in question, in reversed order (so latest
        # character is first)
        for (c, char) in enumerate(reversed(question[:inputLength])):
            # print(\"  \"+str(c))
            if char in alphabet:
                encodedQs[q][c] = alphabet.index(char)
            else:
                encodedQs[q][c] = 0
    return encodedQs

# Make encoded questions out of training questions 1 and 2
print("encoding qs")
encodedTrainQ1s = encodeQs(trainQs1, inputLength, alphabet)
print("encoded q1, encoding q2")
encodedTrainQ2s = encodeQs(trainQs2, inputLength, alphabet)
print("encoded q1 and q2")
#
# np.save("encodedQ1s_70_1014", encodedQ1s)
# np.save("encodedQ2s_70_1014", encodedQ2s)
# print("saved encoded q1 and q2")

# print("Loading encodedQs")
# encodedQ1s = np.load("encodedQ1s_70_1014.npy")
# print("Loaded 1, loading 2")
# encodedQ2s = np.load("encodedQ2s_70_1014.npy")
# print("Loaded encodedQs")

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
    baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu', input_shape=(inputLength, inputDim),
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu',
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu',
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu',
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu',
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu',
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Flatten())
    baseNetwork.add(Dense(1024, activation='relu'))
    baseNetwork.add(Dropout(0.5))
    baseNetwork.add(Dense(1024, activation='relu'))
    baseNetwork.add(Dropout(0.5))
    return baseNetwork


def createBaseNetworkLarge(inputLength, inputDim):
    baseNetwork = Sequential()
    baseNetwork.add(Conv1D(1024, 7, strides=1, padding='valid', activation='relu',  input_shape=(inputLength, inputDim),
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(1024, 7, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(1024, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    baseNetwork.add(Conv1D(1024, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    baseNetwork.add(Conv1D(1024, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    baseNetwork.add(Conv1D(1024, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Flatten())
    baseNetwork.add(Dense(2048, activation='relu'))
    baseNetwork.add(Dropout(0.5))
    baseNetwork.add(Dense(2048, activation='relu'))
    baseNetwork.add(Dropout(0.5))
    return baseNetwork


baseNetwork = createBaseNetworkSmall(inputLength, inputDim)
# baseNetwork = createBaseNetworkLarge(inputDim, inputLength)

# because we re-use the same instance `base_network`,
# the weights of the network will be shared across the two branches
processedA = baseNetwork(oheInputA)
processedB = baseNetwork(oheInputB)

# Concatenate
conc = Concatenate()([processedA, processedB])

predictions = Dense(1, activation='sigmoid')(conc)

model = Model([inputA, inputB], predictions)

# Compile
initLR = 0.01
momentum = 0.9
sgd = SGD(lr=initLR, momentum=momentum, decay=0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# MAKE VAL DATA
validationSplit = 0.2  # Use this much for validation


# Learning Rate Schedule
def step_decay(epoch):
    initial_lrate = initLR
    drop = 0.5
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor(epoch / epochs_drop))
    return lrate

lRate = LearningRateScheduler(step_decay)

# Model Checkpoint
filepath = "charCNNSigmoid-ohE-smAl-val0.2-epoch{epoch:02d}-l{loss:.4f}-a{acc:.4f}-vl{val_loss:.4f}-va{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, verbose=1, save_best_only=False, save_weights_only=True)

# Callbacks
callbacks_list = [lRate, checkpoint]

# Hyperparameters
minibatchSize = 100
nEpochs = 40

# # SKIP: Load weights
# model.load_weights(
#     "charCNNSigmoid-SG-BCE-initLR0.01-m0.9-epoch13.hdf5")

# Train
history = model.fit([encodedTrainQ1s, encodedTrainQ2s], trainOutputs,
                    batch_size=minibatchSize, epochs=nEpochs, verbose=1,
                    callbacks=callbacks_list, validation_split=validationSplit,
                    initial_epoch=0)
