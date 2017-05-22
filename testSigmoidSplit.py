# Pre-requisites
import numpy as np
print("imported np")
import pandas as pd
print("imported pd")
from collections import Counter
print("imported counter")
import os
print("imported os")
from sys import getsizeof
import time
# import cv2

# tensorflow
import tensorflow as tf
print("imported tf")
# with tf.device('/gpu:0'):

# Keras
from keras import backend as K
print("imported keras backend")
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.initializers import RandomNormal
print("imported all keras")

# Load training and test data
# Download train.csv and test.csv from
# https://www.kaggle.com/c/quora-question-pairs/
trainDf = pd.read_csv('kaggleQuoraTrain.csv', sep=',')
testDf = pd.read_csv('kaggleQuoraTest.csv', sep=',')
print("read Dfs")

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

# Load alphabet
alphabet = np.load("alphabet.npy")
alphabet = [str(a) for a in alphabet]
print(alphabet)


def createSplitBaseNetworkSmall(inputDim, inputLength):
    baseNetwork = Sequential()
    baseNetwork.add(Embedding(input_dim=inputDim,
                              output_dim=inputDim, input_length=inputLength))
    baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    return baseNetwork

baseNetwork = createSplitBaseNetworkSmall(inputDim, inputLength)

# Inputs
inputA = Input(shape=(inputLength,))
inputB = Input(shape=(inputLength,))

# because we re-use the same instance `base_network`,
# the weights of the network will be shared across the two branches
processedA = baseNetwork(inputA)
processedB = baseNetwork(inputB)

# Concatenate
conc = Concatenate()([processedA, processedB])

x = Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
    mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05))(conc)
x = Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
    mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05))(x)
x = Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
    mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05))(x)
x = Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
    mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05))(x)
x = MaxPooling1D(pool_size=3, strides=3)(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(1, activation='sigmoid')(x)

model = Model([inputA, inputB], predictions)

# Compile
initLR = 0.01
momentum = 0.9
sgd = SGD(lr=initLR, momentum=momentum, decay=0, nesterov=False)

# PREDICT

# Load weights
model.load_weights(
    "charCNNSigmoidSplit-SG-BCE-initLR0.01-m0.9-epoch06-loss0.3361-acc0.8588.hdf5")

# Initialize output
yTest = -np.ones((len(testQs1), 2)).astype(int)

# Load current preds
# preds = np.loadtxt("preds_0to2100099.csv",
#                   delimiter=",", skiprows=1).astype(int)
# yTest[0:2100099] = preds[0:2100099]


# To encode questions
def encodeQs(questions, inputLength, alphabet):
    alphabetSize = len(alphabet)
    # Initialize encoded questions array
    encodedQs = np.zeros((len(questions), inputLength))
    # For each question
    for (q, question) in enumerate(questions):
        print(str(q) + " of " + str(len(questions)) + " = " +
              "{0:.2f}".format(float(q) / len(questions)), end='\r')
        # For each character in question
        for (c, char) in enumerate(question[:inputLength]):
            # print("  +str(c))
            if char in alphabet:
                encodedQs[q][c] = alphabet.index(char)
            else:
                encodedQs[q][c] = 0
    print("Done encoding.")
    return encodedQs

# encodedTestQ1s = encodeQs(testQs1, inputLength, alphabet)
# encodedTestQ2s = encodeQs(testQs2, inputLength, alphabet)
# preds = model.predict([encodedTestQ1s, encodedTestQ2s], verbose=1)
# yTest = np.reshape((preds > 0.5).astype(int), (len(preds),))

# Encode testdata and make predictions
nOfQs = len(testQs1)
subsetLength = 1000
nOfSubsets = int(nOfQs / subsetLength)
for subset in range(nOfSubsets):
    #     if subset < 21001:
    #         continue
    #     print(time.strftime("%c"))
    print("Subset " + str(subset + 1) + " of " + str(nOfSubsets) +
          " = {0:.2f}".format(float(subset + 1) / nOfSubsets), end='\r')
    startIdx = subset * subsetLength
    # print("  from " + str(startIdx) + " to " +
    #       str(startIdx + subsetLength - 1))
    # Encode subset qs
    encodedTestQ1s = encodeQs(
        testQs1[startIdx:startIdx + subsetLength], inputLength, alphabet)
    encodedTestQ2s = encodeQs(
        testQs2[startIdx:startIdx + subsetLength], inputLength, alphabet)
    # Make predictions
    preds = model.predict([encodedTestQ1s, encodedTestQ2s])
    yTest[startIdx:startIdx + subsetLength,
          0] = np.array(list(range(startIdx, startIdx + subsetLength)))
    yTest[startIdx:startIdx + subsetLength,
          1] = np.reshape((preds > 0.5).astype(int), (len(preds),))
    # if subset % 1000 == 0:
    #     np.savetxt("preds_0to{0}.csv".format(startIdx + subsetLength - 1), yTest,
    # fmt='%i', delimiter=',', header="test_id,is_duplicate", comments='')

startIdx = nOfSubsets * subsetLength
endIdx = len(testQs1)
encodedTestQ1s = encodeQs(testQs1[startIdx:endIdx], inputLength, alphabet)
encodedTestQ2s = encodeQs(testQs2[startIdx:endIdx], inputLength, alphabet)
preds = model.predict([encodedTestQ1s, encodedTestQ2s])
yTest[startIdx:endIdx, 0] = np.array(list(range(startIdx, endIdx)))
yTest[startIdx:endIdx, 1] = np.reshape(
    (preds > 0.5).astype(int), (len(preds),))

# Save predictions in the format dictated by Kaggle
np.savetxt("PREDS_SigmoidSplit_epoch??.csv".format(endIdx), yTest, fmt='%i',
           delimiter=',', header="test_id,is_duplicate", comments='')

print("Saved predictions.")
