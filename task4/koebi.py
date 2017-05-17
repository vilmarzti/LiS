#
# Learning classification with the labeled data and prelaerned structure from a fileall_dynamic
result_file_name = 'AE_init_all_dynamic' 

import sklearn.preprocessing as skpre
import shutil
import os
import csv

from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
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

train_labeled = pd.read_hdf("task4/train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("task4/train_unlabeled.h5", "train")

num_features = 128
num_classes = 10

#Y = np.array([row[0] for row in np.array(train_labeled)], dtype = 'int32')
Y = np.array(train_labeled.get('y'), dtype = 'int32')
#X = np.array([row[1:] for row in np.array(train_labeled)])
X = np.array(train_labeled.drop('y', 1))
X = skpre.StandardScaler().fit_transform(X)

#Z = np.array([row[0:] for row in np.array(train_unlabeled)])
#Z = skpre.StandardScaler().fit_transform(Z)

layers = [
    ('input', InputLayer), 
    ('dense', DenseLayer), 
    ('dropout1', DropoutLayer),
    ('narrow', DenseLayer), 
    ('reverse', DenseLayer), 
    ('coutput', DenseLayer), 
]

nn = NeuralNet(
    layers=layers,
    max_epochs=15,
    
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,
    
    input_shape=(None, num_features),

    dense_num_units=64,
    narrow_num_units=25,
    reverse_num_units=64,
    coutput_num_units=10,

    #input_nonlinearity = None, #nonlinearities.sigmoid,
    #dense_nonlinearity = nonlinearities.tanh,
    narrow_nonlinearity = nonlinearities.softplus,
    reverse_nonlinearity = nonlinearities.sigmoid,
    coutput_nonlinearity = nonlinearities.softmax,
              
    #dropout0_p=0.1,
    dropout1_p=0.01,

    #regression=True,
    regression=False,
    verbose=1
)

nn.initialize()

#nn.load_params_from('task4/koebi_train_history_AE');

PrintLayerInfo()(nn)

nn.fit(X,Y)



test = pd.read_hdf("task4/test.h5", "test")
id_col = test.index
test_data = np.array(test)
test_data = skpre.StandardScaler().fit_transform(test_data)
test_prediction = nn.predict(test_data)

# Write prediction and it's linenumber into a csv file
with open('task4/' + result_file_name + '.csv', 'wb') as csvfile:
  fieldnames = ['Id', 'y']
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  #    print test_prediction
  writer.writeheader()
  for i in range(len(test_prediction)):
      writer.writerow({'Id': id_col[i], 'y': test_prediction[i]})

