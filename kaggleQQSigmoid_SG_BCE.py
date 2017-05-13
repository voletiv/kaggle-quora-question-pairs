# Pre-requisites
import numpy as np
import pandas as pd
from collections import Counter
import os
from sys import getsizeof
import time
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
from keras.layers import Flatten, Dense, Dropout
from keras.layers.merge import Concatenate
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.initializers import RandomNormal

# Load training and test data
# Download train.csv and test.csv from https://www.kaggle.com/c/quora-question-pairs/
trainDf = pd.read_csv('kaggleQuoraTrain.csv', sep=',')
testDf = pd.read_csv('kaggleQuoraTest.csv', sep=',')

# Check for any null values
print(trainDf.isnull().sum())
print(testDf.isnull().sum())

# Add the string 'empty' to empty strings
trainDf = trainDf.fillna('empty')
testDf = testDf.fillna('empty')

# Convert into np array
trainData = np.array(trainDf)
testData = np.array(testDf)

# Inputs
# Get list of questions in Question1 and Question2
trainQs1 = trainData[:, 3]
trainQs2 = trainData[:, 4]
testQs1 = testData[:, 1]
testQs2 = testData[:, 2]

# Outputs (whether duplicate or not)
outputs = trainData[:, 5]

# Setting alphabet size
alphabetSize = 70

# Params
inputDim = alphabetSize  # number of letters (characters) in alphabet
inputLength = 1014  # input feature length (the paper used 1014)

# # Get list of question IDs and questions in training data
# qsDict = {}
# for data in trainData:
#     qsDict[data[1]] = data[3].lower()
#     qsDict[data[2]] = data[4].lower()
#
# # Save qsDict
# np.save("qsDict", qsDict)
#
# # Extract question IDs and questions
# qIds = list(qsDict.keys())
# questions = list(qsDict.values())
#
# # Append all characters from training data questions into list
# charFullCorpus = []
# for (q, question) in enumerate(questions):
#     # Printing status (makes it slow)
#     # clear_output(); print(str(q)+" of "+str(len(questions)))
#     for char in list(question):
#         charFullCorpus.append(char)
#
# # EXTRACT CHARCTER CORPUS
# charCorpusCountSorted, charCorpusSorted = \
#     map(list, zip(*sorted(zip(Counter(charFullCorpus).values(),
#                               Counter(charFullCorpus).keys()))))
#
# # Assign the most frequent #alphabetSize number of characters as the alphabet for the network
# alphabet = charCorpusSorted[-alphabetSize-1:-1]
#
# # Save alphabet
# np.save("alphabet", alphabet)
#
# # Load alphabet
# alphabet = np.load("alphabet.npy")
# alphabet = [str(a) for a in alphabet]
# print(alphabet)
#
# # To encode questions
# def encodeQs(questions, inputLength, alphabet):
#     alphabetSize = len(alphabet)
#     # Initialize encoded questions array
#     encodedQs = np.zeros((len(questions), inputLength))
#     # For each question
#     for (q, question) in enumerate(questions):
#         # print(q)
#         # For each character in question, in reversed order (so latest character is first)
#         for (c, char) in enumerate(reversed(question[:inputLength])):
#             # print(\"  \"+str(c))
#             if char in alphabet:
#                 encodedQs[q][c] = alphabet.index(char)
#             else:
#                 encodedQs[q][c] = 0
#     return encodedQs
#
# # Make encoded questions out of training questions 1 and 2
# print("encoding qs")
# encodedQ1s = encodeQs(trainQs1, inputLength, alphabet)
# print("encoded q1, encoding q2")
# encodedQ2s = encodeQs(trainQs2, inputLength, alphabet)
# print("encoded q1 and q2")
#
# np.save("encodedQ1s_70_1014", encodedQ1s)
# np.save("encodedQ2s_70_1014", encodedQ2s)
# print("saved encoded q1 and q2")

print("Loading encodedQs")
encodedQ1s = np.load("encodedQ1s_70_1014.npy")
print("Loaded 1, loading 2")
encodedQ2s = np.load("encodedQ2s_70_1014.npy")
print("Loaded encodedQs")


def createBaseNetworkSmall(inputDim, inputLength):
    baseNetwork = Sequential()
    baseNetwork.add(Embedding(input_dim=inputDim,
                              output_dim=inputDim, input_length=inputLength))
    baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Flatten())
    baseNetwork.add(Dense(1024, activation='relu'))
    baseNetwork.add(Dropout(0.5))
    baseNetwork.add(Dense(1024, activation='relu'))
    baseNetwork.add(Dropout(0.5))
    return baseNetwork


def createBaseNetworkLarge(inputDim, inputLength):
    baseNetwork = Sequential()
    baseNetwork.add(Embedding(input_dim=inputDim,
                              output_dim=inputDim, input_length=inputLength))
    baseNetwork.add(Conv1D(1024, 7, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
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


baseNetwork = createBaseNetworkSmall(inputDim, inputLength)
# baseNetwork = createBaseNetworkLarge(inputDim, inputLength)

# Inputs
inputA = Input(shape=(inputLength,))
inputB = Input(shape=(inputLength,))

# because we re-use the same instance `base_network`,
# the weights of the network will be shared across the two branches
processedA = baseNetwork(inputA)
processedB = baseNetwork(inputB)

# Concatenate
conc = Concatenate()([processedA, processedB])

predictions = Dense(1, activation='sigmoid')(conc)

model = Model([inputA, inputB], predictions)

# Compile
initLR = 0.01
momentum = 0.9
sgd = SGD(lr=initLR, momentum=momentum, decay=0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# # Checkpoint
# filepath = "weights-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
#                              save_best_only=True, mode='min')

# # callbacks
# callbacks = [lRate, checkpoint]

# MAKE MINIBATCHES

minibatchSize = 100
nEpochs = 40
nOfQPairs = len(outputs)
validationSplit = 0.0  # Use this much for validation

encodedTrainQ1s = encodedQ1s[:int((1 - validationSplit) * nOfQPairs)]
encodedTrainQ2s = encodedQ2s[:int((1 - validationSplit) * nOfQPairs)]
encodedValQ1s = encodedQ1s[int((1 - validationSplit) * nOfQPairs):]
encodedValQ2s = encodedQ2s[int((1 - validationSplit) * nOfQPairs):]
trainOutputs = outputs[:int((1 - validationSplit) * nOfQPairs)]
valOutputs = outputs[int((1 - validationSplit) * nOfQPairs):]

# Count number of mini-batches
nOfMinibatches = int(len(trainOutputs) / minibatchSize)
print("Size of full trainingData: "+str(len(outputs)))
print("Validation split: "+str(validationSplit))
print("Size of trainingData: "+str(len(trainOutputs)))
print("Size of valData: "+str(len(valOutputs)))
print("nOfMinibatches: "+str(nOfMinibatches))

# Make a list of all the indices
fullIdx = list(range(len(trainOutputs)))

# SKIP: Load weights
model.load_weights("charCNNSigmoid-SG-BCE-initLR0.01-m0.9-epoch11-loss0.1903-acc0.9262.hdf5")

lr = initLR
for n in range(nEpochs):
    print(time.strftime("%c"))
    print("EPOCH " + str(n + 1) + " of " + str(nEpochs))

    # Halve LR every 3rd epoch
    if n != 0 and n % 3 == 0:
        lr /= 2
        print("lr reduced to "+str(lr))
        sgd = SGD(lr=lr, momentum=momentum, nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # SKIP:
    if n < 12:
        continue

    # Shuffle the full index
    np.random.shuffle(fullIdx)
    # print(fullIdx)

    # For each mini-batch
    for m in range(nOfMinibatches):
        print(time.strftime("%c"))
        print("EPOCH " + str(n + 1) + " of " + str(nEpochs))
        print("  minibatch " + str(m + 1) + " of " + str(nOfMinibatches))

        # Compute the starting index of this mini-batch
        startIdx = m * minibatchSize

        # Declare sampled inputs and outputs
        encodedQ1sSample = encodedTrainQ1s[fullIdx[startIdx:startIdx + minibatchSize]]
        encodedQ2sSample = encodedTrainQ2s[fullIdx[startIdx:startIdx + minibatchSize]]
        outputsSample = trainOutputs[fullIdx[startIdx:startIdx + minibatchSize]]

        model.fit([encodedQ1sSample, encodedQ2sSample], outputsSample,
                  batch_size=minibatchSize, epochs=1, verbose=1)

    # Evaluate current model
    print("evaluating current model:")
    if len(valOutputs) > 0:
        loss, acc = model.evaluate([encodedValQ1s, encodedValQ2s], valOutputs)
    else:
        loss, acc = model.evaluate([encodedTrainQ1s, encodedTrainQ2s], trainOutputs)
    print(time.strftime("%c"))
    print("LOSS = "+str(loss)+"; ACC = "+str(acc))
    print("saving current weights.")
    model.save_weights(
        "charCNNSigmoid-SG-BCE-initLR0.01-m0.9-epoch{0:02d}-loss{1:.4f}-acc{2:.4f}.hdf5".format(n, loss, acc))
#
