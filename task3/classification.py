#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.preprocessing as skpre
import shutil
import os
import csv

from sklearn.decomposition import PCA, RandomizedPCA
from nolearn.dbn import DBN
import tensorflow as tf
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne import nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano
from theano import function, config, shared, tensor
from theano.tensor import *
import pandas as pd


file_train = pd.read_hdf("train.h5", "train")
file_train = np.array(file_train)


Y = np.array([row[0] for row in file_train], dtype = 'int32')
X = np.array([row[1:] for row in file_train])

X = skpre.StandardScaler().fit_transform(X)

num_features = X.shape[1]

layers0 = [('input', InputLayer),
           ('dropout', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           #('dense2', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense3', DenseLayer),
           ('output', DenseLayer)]

params = dict(
              layers=layers0,

              dropout_p=0.05,
              dropout0_p=0.5,
              dropout1_p=0.5,
              
              dense0_num_units=300,
              dense1_num_units=600,
              #dense2_num_units=600,
              dense3_num_units=300,

              

              input_shape=(None, num_features),
              output_num_units=5,
              
              
              #input_nonlinearity = nonlinearities.tanh,
              dense0_nonlinearity = nonlinearities.tanh,
              dense1_nonlinearity = nonlinearities.rectify,
              #dense2_nonlinearity = nonlinearities.tanh,
              dense3_nonlinearity = nonlinearities.sigmoid,
              #dense4_nonlinearity = nonlinearities.sigmoid,
        
              
              output_nonlinearity=softmax,
              
              update=nesterov_momentum,
              update_learning_rate=0.03,
              
              eval_size=0.2,
              verbose=1,
              max_epochs=140,
              regression=False
              )

clf = NeuralNet(**params)

print "PARAMS : "
print params

clf.fit(X, Y)

file_valid = pd.read_hdf("test.h5", "test")
file_valid = np.array(file_valid)

data = np.array([row[0:] for row in file_valid])
data = skpre.StandardScaler().fit_transform(data)


number = range(45324,53461)
    
predict = clf.predict(data)

with open('result.csv', 'wb') as csvfile:
    fieldnames = ['Id', 'y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
    writer.writeheader()
    for x in range(len(number)):
        writer.writerow({'Id': number[x], 'y': predict[x]})

