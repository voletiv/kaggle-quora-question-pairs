# Pre-requisites
import numpy as np
import pandas as pd
from collections import Counter
import os
from sys import getsizeof
import time
# import cv2

# To clear print buffer
from IPython.display import clear_output

# tensorflow
import tensorflow as tf
with tf.device('/cpu:0'):

    # Keras
    from keras import backend as K
    from keras.models import Model, Sequential
    from keras.layers import Input, Conv1D, MaxPooling1D
    from keras.layers import Flatten, Dense, Dropout, Lambda
    from keras.layers.merge import Concatenate
    from keras.layers.embeddings import Embedding
    from keras.optimizers import SGD
    from keras.initializers import RandomNormal
    from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
    from keras.utils import np_utils
    from keras.engine.topology import Layer

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

    # Load alphabet
    alphabet = np.load("alphabet.npy")
    alphabet = [str(a) for a in alphabet]
    print(alphabet)


    def createBaseNetworkSmall(inputDim, inputLength):
        baseNetwork = Sequential()
        baseNetwork.add(Embedding(input_dim=inputDim, output_dim=inputDim, input_length=inputLength))
        baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
        baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
        baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
        baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
        baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
        baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
        baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
        baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
        baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
        baseNetwork.add(Flatten())
        baseNetwork.add(Dense(1024, activation='relu'))
        baseNetwork.add(Dropout(0.5))
        baseNetwork.add(Dense(1024, activation='relu'))
        baseNetwork.add(Dropout(0.5))
        return baseNetwork


    def createBaseNetworkLarge(inputDim, inputLength):
        baseNetwork = Sequential()
        baseNetwork.add(Embedding(input_dim=inputDim, output_dim=inputDim, input_length=inputLength))
        baseNetwork.add(Conv1D(1024, 7, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
        baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
        baseNetwork.add(Conv1D(1024, 7, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
        baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
        baseNetwork.add(Conv1D(1024, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
        baseNetwork.add(Conv1D(1024, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
        baseNetwork.add(Conv1D(1024, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
        baseNetwork.add(Conv1D(1024, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=RandomNormal(mean=0.0, stddev=0.02)))
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

    # because we re-use the same instance `baseNetwork`,
    # the weights of the network will be shared across the two branches
    processedA = baseNetwork(inputA)
    processedB = baseNetwork(inputB)

    # Concatenate
    conc = Concatenate()([processedA, processedB])

    # Add more layers
    x = Dense(1024, activation='relu')(conc)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model([inputA, inputB], predictions)

    # Compile
    # model.compile(loss=contrastive_loss, optimizer=sgd, metrics=['accuracy'])
    initLR = 0.01
    momentum = 0.9
    sgd = SGD(lr=initLR, momentum=momentum, decay=1e-5, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # PREDICT

    # Load weights
    model.load_weights("c.hdf5")

    # Initialize output
    yTest = -np.ones((len(testQs1), 2)).astype(int)
    '''
    # Load current preds
    preds = np.loadtxt("preds_0to90899_1000000to2300099.csv", delimiter=",", skiprows=1).astype(int)
    yTest[0:90900] = preds[0:90900]
    yTest[1000000:2300100] = preds[1000000:2300100]
    '''
    # To encode questions
    def encodeQs(questions, inputLength, alphabet):
        alphabetSize = len(alphabet)
        # Initialize encoded questions array
        encodedQs = np.zeros((len(questions), inputLength))
        # For each question
        for (q, question) in enumerate(questions):
            # print(q)
            # For each character in question
            for (c, char) in enumerate(question[:inputLength]):
                # print("  +str(c))
                if char in alphabet:
                    encodedQs[q][c] = alphabet.index(char)
                else:
                    encodedQs[q][c] = 0
        return encodedQs
    '''
    # Encode testdata and make predictions
    nOfQs = len(testQs1)
    subsetLength = 100
    nOfSubsets = int(nOfQs/subsetLength)
    for subset in range(nOfSubsets):
        if subset >= 10000:
            break
        if subset < 8000:
            continue
        print("Subset "+str(subset+1)+" of "+str(nOfSubsets))
        startIdx = subset*subsetLength
        # Encode subset qs
        encodedTestQ1s = encodeQs(testQs1[startIdx:startIdx+subsetLength], inputLength, alphabet)
        encodedTestQ2s = encodeQs(testQs2[startIdx:startIdx+subsetLength], inputLength, alphabet)
        # Make predictions
        preds = model.predict([encodedTestQ1s, encodedTestQ2s])
        yTest[startIdx:startIdx+subsetLength, 0] = np.array(list(range(startIdx, startIdx+subsetLength)))
        yTest[startIdx:startIdx+subsetLength, 1] = np.reshape((preds > 0.5).astype(int), (len(preds),))
        if subset % 1000 == 0:
            np.savetxt("preds_0to90899_800000to{0}_1000000to2300099.csv".format(startIdx+subsetLength-1), yTest, fmt='%i', delimiter=',', header="test_id,is_duplicate", comments='')
    '''
    startIdx = 2300100
    endIdx = len(testQs1)
    encodedTestQ1s = encodeQs(testQs1[startIdx:endIdx], inputLength, alphabet)
    encodedTestQ2s = encodeQs(testQs2[startIdx:endIdx], inputLength, alphabet)
    preds = model.predict([encodedTestQ1s, encodedTestQ2s])
    yTest[startIdx:endIdx, 0] = np.array(list(range(startIdx, endIdx)))
    yTest[startIdx:endIdx, 1] = np.reshape((preds > 0.5).astype(int), (len(preds),))

    # Save predictions in the format dictated by Kaggle
    np.savetxt("preds_{0}to{1}.csv".format(startIdx, endIdx), yTest, fmt='%i', delimiter=',', header="test_id,is_duplicate", comments='')