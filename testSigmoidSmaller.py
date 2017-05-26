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
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
print("imported all keras")

# Load alphabet
alphabet = np.load("smallerAlphabet.npy")
alphabet = [str(a) for a in alphabet]
print(alphabet)

# Params
inputDim = len(alphabet)  # number of letters (characters) in alphabet
inputLength = 1014  # input feature length (the paper used 1014)

# Load training and test data
# Download train.csv and test.csv from
# https://www.kaggle.com/c/quora-question-pairs/
# trainDf = pd.read_csv('kaggleQuoraTrain.csv', sep=',')
testDf = pd.read_csv('kaggleQuoraTest.csv', sep=',')
print("read Dfs")

# Check for any null values
# print(trainDf.isnull().sum())
print(testDf.isnull().sum())

# Add the string 'empty' to empty strings
# trainDf = trainDf.fillna('empty')
testDf = testDf.fillna('empty')


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
# trainDf['question1'] = cleanText(trainDf['question1'])
# trainDf['question2'] = cleanText(trainDf['question2'])
testDf['question1'] = cleanText(testDf['question1'])
testDf['question2'] = cleanText(testDf['question2'])
print("Cleaned text.")

# Convert into np array
# trainData = np.array(trainDf)
testData = np.array(testDf)

# Inputs
# Get list of questions in Question1 and Question2
# trainQs1 = trainData[:, 3]
# trainQs2 = trainData[:, 4]
testQs1 = testData[:, 1]
testQs2 = testData[:, 2]

# Outputs (whether duplicate or not)
# trainOutputs = trainData[:, 5]

# Initialize output
yTest = -np.ones((len(testQs1), 2)).astype(int)
for i in range(len(testQs1)):
    yTest[i, 0] = i

# Load current preds
# preds = np.loadtxt("preds_0to2100099.csv",
#                   delimiter=",", skiprows=1).astype(int)
# yTest[0:2100099] = preds[0:2100099]


# MODEL


def netSigmoid(inputLength, inputDim):
    baseNetwork = Sequential()
    baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu',  input_shape=(inputLength, inputDim),
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
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


def netC256P3C256P3C256P3f128(inputLength, inputDim):
    baseNetwork = Sequential()
    baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu',  input_shape=(inputLength, inputDim),
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Flatten())
    baseNetwork.add(Dense(128, activation='relu'))
    baseNetwork.add(Dropout(0.5))
    return baseNetwork


def netC256P3C256P3f32(inputLength, inputDim):
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


def netC256P3C256P3f64(inputLength, inputDim):
    baseNetwork = Sequential()
    baseNetwork.add(Conv1D(256, 7, strides=1, padding='valid', activation='relu',  input_shape=(inputLength, inputDim),
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Flatten())
    baseNetwork.add(Dense(64, activation='relu'))
    baseNetwork.add(Dropout(0.5))
    return baseNetwork


def modelSigmoid(inputLength, inputDim):
    inputA = Input(shape=(inputLength,), dtype='int32')
    inputB = Input(shape=(inputLength,), dtype='int32')
    # One hot encoding
    oheInputA = Lambda(K.one_hot, arguments={
                       'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputA)
    oheInputB = Lambda(K.one_hot, arguments={
                       'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputB)
    net = netSigmoid(inputLength, inputDim)
    processedA = net(oheInputA)
    processedB = net(oheInputB)
    # Concatenate
    conc = Concatenate()([processedA, processedB])
    x = BatchNormalization()(conc)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model([inputA, inputB], predictions)
    return model


def modelC256P3C256P3C256P3f128_conc_f128(inputLength, inputDim):
    inputA = Input(shape=(inputLength,), dtype='int32')
    inputB = Input(shape=(inputLength,), dtype='int32')
    # One hot encoding
    oheInputA = Lambda(K.one_hot, arguments={
                       'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputA)
    oheInputB = Lambda(K.one_hot, arguments={
                       'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputB)
    net = netC256P3C256P3C256P3f128(inputLength, inputDim)
    processedA = net(oheInputA)
    processedB = net(oheInputB)
    # Concatenate
    conc = Concatenate()([processedA, processedB])
    x = BatchNormalization()(conc)
    # Dense
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model([inputA, inputB], predictions)
    return model


def modelC256P3C256P3C256P3f128_conc(inputLength, inputDim):
    inputA = Input(shape=(inputLength,), dtype='int32')
    inputB = Input(shape=(inputLength,), dtype='int32')
    # One hot encoding
    oheInputA = Lambda(K.one_hot, arguments={
                       'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputA)
    oheInputB = Lambda(K.one_hot, arguments={
                       'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputB)
    net = netC256P3C256P3C256P3f128(inputLength, inputDim)
    processedA = net(oheInputA)
    processedB = net(oheInputB)
    # Concatenate
    conc = Concatenate()([processedA, processedB])
    x = BatchNormalization()(conc)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model([inputA, inputB], predictions)
    return model


def modelC256P3C256P3f32_conc_f64(inputLength, inputDim):
    inputA = Input(shape=(inputLength,), dtype='int32')
    inputB = Input(shape=(inputLength,), dtype='int32')
    # One hot encoding
    oheInputA = Lambda(K.one_hot, arguments={
                       'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputA)
    oheInputB = Lambda(K.one_hot, arguments={
                       'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputB)
    net = netC256P3C256P3f32(inputLength, inputDim)
    processedA = net(oheInputA)
    processedB = net(oheInputB)
    conc = Concatenate()([processedA, processedB])
    x = BatchNormalization()(conc)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model([inputA, inputB], predictions)
    return model


def modelC256P3C256P3f64_conc_f64(inputLength, inputDim):
    inputA = Input(shape=(inputLength,), dtype='int32')
    inputB = Input(shape=(inputLength,), dtype='int32')
    # One hot encoding
    oheInputA = Lambda(K.one_hot, arguments={
                       'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputA)
    oheInputB = Lambda(K.one_hot, arguments={
                       'num_classes': inputDim}, output_shape=(inputLength, inputDim))(inputB)
    net = netC256P3C256P3f64(inputLength, inputDim)
    processedA = net(oheInputA)
    processedB = net(oheInputB)
    conc = Concatenate()([processedA, processedB])
    x = BatchNormalization()(conc)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model([inputA, inputB], predictions)
    return model


# To encode questions into char indices
def encodeQs(questions, inputLength, alphabet):
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
print("encoding qs - 1 of 2:")
encodedTestQ1s = encodeQs(testQs1, inputLength, alphabet)
print("encoded q1, encoding q2:")
encodedTestQ2s = encodeQs(testQs2, inputLength, alphabet)
print("encoded q1 and q2")

# modelC256P3C256P3f32_conc_f64
model = modelC256P3C256P3f32_conc_f64(inputLength, inputDim)
model.load_weights(
    "charCNN-smAl-C256P3C256P3f32-conc-f64-val0.2-epoch07-l0.4063-a0.8243-vl0.4685-va0.7840.hdf5")
preds = model.predict(
    [encodedTestQ1s, encodedTestQ2s], verbose=1)
yTest[:, 1] = np.reshape((preds > 0.5).astype(int), (len(preds),))
np.savetxt("PREDS_modelC256P3C256P3f32_conc_f64-val0.2-epoch07-l0.4063-a0.8243-vl0.4685-va0.7840.csv", yTest, fmt='%i',
           delimiter=',', header="test_id,is_duplicate", comments='')

# modelC256P3C256P3f64_conc_f64
model = modelC256P3C256P3f64_conc_f64(inputLength, inputDim)
model.load_weights(
    "charCNN-smAl-C256P3C256P3f64-conc-f64-val0.2-epoch06-l0.4274-a0.8128-vl0.4768-va0.7742.hdf5")
preds = model.predict(
    [encodedTestQ1s, encodedTestQ2s], verbose=1)
yTest[:, 1] = np.reshape((preds > 0.5).astype(int), (len(preds),))
np.savetxt("PREDS_modelC256P3C256P3f64-conc-f64-val0.2-epoch06-l0.4274-a0.8128-vl0.4768-va0.7742.csv", yTest, fmt='%i',
           delimiter=',', header="test_id,is_duplicate", comments='')

# modelC256P3C256P3C256P3f128_conc
model = modelC256P3C256P3C256P3f128_conc(inputLength, inputDim)
model.load_weights(
    "charCNN-smAl-C256P3C256P3C256P3f128-conc-val0.2-epoch03-l0.4524-a0.7909-vl0.4875-va0.7747.hdf5")
preds = model.predict(
    [encodedTestQ1s, encodedTestQ2s], verbose=1)
yTest[:, 1] = np.reshape((preds > 0.5).astype(int), (len(preds),))
np.savetxt("PREDS_modelC256P3C256P3C256P3f128-conc-val0.2-epoch03-l0.4524-a0.7909-vl0.4875-va0.7747.csv", yTest, fmt='%i',
           delimiter=',', header="test_id,is_duplicate", comments='')

# modelC256P3C256P3C256P3f128_conc_f128
model = modelC256P3C256P3C256P3f128_conc_f128(
    inputLength, inputDim)
model.load_weights(
    "charCNN-smAl-C256P3C256P3C256P3f128-conc-f128-val0.2-epoch15-l0.3907-a0.8311-vl0.4616-va0.7891.hdf5")
preds = model.predict(
    [encodedTestQ1s, encodedTestQ2s], verbose=1)
yTest[:, 1] = np.reshape((preds > 0.5).astype(int), (len(preds),))
np.savetxt("PREDS_modelC256P3C256P3C256P3f128_conc-f128-val0.2-epoch15-l0.3907-a0.8311-vl0.4616-va0.7891.csv", yTest, fmt='%i',
           delimiter=',', header="test_id,is_duplicate", comments='')

# model Sigmoid
model = modelSigmoid(inputLength, inputDim)
model.load_weights(
    "charCNNSigmoid-ohE-smAl-val0.2-epoch04-l0.4438-a0.7960-vl0.4831-va0.7725.hdf5")
preds = model.predict(
    [encodedTestQ1s, encodedTestQ2s], verbose=1)
yTest[:, 1] = np.reshape((preds > 0.5).astype(int), (len(preds),))
np.savetxt("PREDS_modelSigmoid-ohE-smAl-val0.2-epoch04-l0.4438-a0.7960-vl0.4831-va0.7725.csv", yTest, fmt='%i',
           delimiter=',', header="test_id,is_duplicate", comments='')


# # Encode testdata and make predictions
# nOfQs = len(testQs1)
# subsetLength = 100
# nOfSubsets = int(nOfQs / subsetLength)
# for subset in range(nOfSubsets):
#     #     if subset < 21001:
#     #         continue
#     #     print(time.strftime("%c"))
#     print("Subset " + str(subset + 1) + " of " + str(nOfSubsets) +
#           " = {0:.2f}".format(float(subset + 1) / nOfSubsets), end='\r')
#     startIdx = subset * subsetLength
#     # print("  from " + str(startIdx) + " to " +
#     #       str(startIdx + subsetLength - 1))
#     # Encode subset qs
#     encodedTestQ1s = encodeQs(
#         testQs1[startIdx:startIdx + subsetLength], inputLength, alphabet)
#     encodedTestQ2s = encodeQs(
#         testQs2[startIdx:startIdx + subsetLength], inputLength, alphabet)
#     # Make predictions
#     preds = model.predict([encodedTestQ1s, encodedTestQ2s])
#     yTest[startIdx:startIdx + subsetLength,
#           0] = np.array(list(range(startIdx, startIdx + subsetLength)))
#     yTest[startIdx:startIdx + subsetLength,
#           1] = np.reshape((preds > 0.5).astype(int), (len(preds),))
#     # if subset % 1000 == 0:
#     #     np.savetxt("preds_0to{0}.csv".format(startIdx + subsetLength - 1), yTest,
#     # fmt='%i', delimiter=',', header="test_id,is_duplicate", comments='')

# startIdx = nOfSubsets * subsetLength
# endIdx = len(testQs1)
# encodedTestQ1s = encodeQs(testQs1[startIdx:endIdx], inputLength, alphabet)
# encodedTestQ2s = encodeQs(testQs2[startIdx:endIdx], inputLength, alphabet)
# preds = model.predict([encodedTestQ1s, encodedTestQ2s])
# yTest[startIdx:endIdx, 0] = np.array(list(range(startIdx, endIdx)))
# yTest[startIdx:endIdx, 1] = np.reshape(
#     (preds > 0.5).astype(int), (len(preds),))
