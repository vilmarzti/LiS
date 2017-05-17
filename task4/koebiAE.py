#
# This file only trains with unlabeled data (unsupervised)
# Soemthing in the realm of an autoencoder is trained to 
# get a decent structure for the network, which should then
# be stored and trained with labeled data
#

import sklearn.preprocessing as skpre
import shutil
import os
import csv

from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import lasagne
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne import nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo
import theano
from theano import function, config, shared, tensor
from theano.tensor import *
import pandas as pd

train_unlabeled = pd.read_hdf("task4/train_unlabeled.h5", "train")

num_features = 128
num_classes = 10

Z = np.array([row[0:] for row in np.array(train_unlabeled)])
Z = skpre.StandardScaler().fit_transform(Z)

layers = [
    ('input', InputLayer), 
    #('dropout0', DropoutLayer),
    ('dense', DenseLayer), 
    ('dropout1', DropoutLayer),
    ('narrow', DenseLayer), 
    ('denseReverse1', DenseLayer), 
    ('dropout2', DropoutLayer),
    ('denseReverse2', DenseLayer), 
    ('output', DenseLayer), 
]

ae = NeuralNet(
    layers=layers,
    max_epochs=100,
    
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,
    
    input_shape=(None, num_features),

    dense_num_units=64,
    narrow_num_units=25,
    denseReverse1_num_units=64,
    denseReverse2_num_units=128,
    output_num_units=128,

    #input_nonlinearity = None, #nonlinearities.sigmoid,
    #dense_nonlinearity = nonlinearities.tanh,
    narrow_nonlinearity = nonlinearities.softplus,
    #denseReverse1_nonlinearity = nonlinearities.tanh,
    denseReverse2_nonlinearity = nonlinearities.softplus,
    output_nonlinearity = nonlinearities.linear, #nonlinearities.softmax,
              
    #dropout0_p=0.1,
    dropout1_p=0.01,
    dropout2_p=0.001,

    regression=True,
    verbose=1
)

ae.initialize()
PrintLayerInfo()(ae)

ae.fit(Z,Z)

learned_parameters = ae.get_all_params_values()
np.save("task4/learned_parameter.npy", learned_parameters)


