#
# Learning classification with the labeled data and prelaerned structure from a file
result_file_name = 'AE48_init_const' 

import sklearn.preprocessing as skpre
import shutil
import os
import csv

import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn import svm
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
test = pd.read_hdf("task4/test.h5", "test")
id_col = test.index
test_data = np.array(test)
test_data = skpre.StandardScaler().fit_transform(test_data)

num_features = 128
num_encoder = 48
num_classes = 10

Y = np.array(train_labeled.get('y'), dtype = 'int32')
X = np.array(train_labeled.drop('y', 1))
X = skpre.StandardScaler().fit_transform(X)

# load old parameters to get encoder
const_layers = [
    ('input', InputLayer), 
    ('dense', DenseLayer), 
    ('dropout1', DropoutLayer),
    ('narrow', DenseLayer), 
]
encoder = NeuralNet(
    layers=const_layers,
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,
    input_shape=(None, num_features),
    dense_num_units=64,
    narrow_num_units=num_encoder,
    narrow_nonlinearity = nonlinearities.softplus,
    regression=True,
)

encoder.initialize()
encoder.load_params_from('task4/koebi_train_history_AE2')

# encode train and test data
x_encoded = encoder.predict(X)
test_encoded = encoder.predict(test_data)
X_plus = np.hstack([X,x_encoded])
test_plus = np.hstack([test_data,test_encoded])

# supervised learning with the encoded data
dynamic_layers = [
    ('input', InputLayer), 
    ('dense', DenseLayer), 
    ('dropout', DropoutLayer), 
    ('dense1', DenseLayer), 
    ('dropout1', DropoutLayer), 
    ('dense2', DenseLayer), 
    ('dropout2', DropoutLayer), 
    ('coutput', DenseLayer), 
]
nn = NeuralNet(
    layers=dynamic_layers,
    max_epochs=100,
    
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,
    
    input_shape=(None, num_encoder+num_features),

    dense_num_units=128,
    dense1_num_units=64,
    dense2_num_units=32,
    coutput_num_units=10,

    dropout_p=0.1,
    dropout1_p=0.025,
    dropout2_p=0.005,

    dense_nonlinearity = nonlinearities.sigmoid,
    dense1_nonlinearity = nonlinearities.sigmoid,
    dense2_nonlinearity = nonlinearities.sigmoid,
    coutput_nonlinearity = nonlinearities.softmax,

    regression=False,
    verbose=1
)

# use neural network for fitting
#nn.initialize()
#PrintLayerInfo()(nn)
#nn.fit(X_plus,Y)
#test_prediction = nn.predict(test_plus))

# use other classifier instead of nn
classifier = ensemble.ExtraTreesClassifier( n_estimators = 1000, max_features = 64, n_jobs=8)
# BAD: classifier = svm.NuSVC()
# Medium: classifier = svm.SVC()
# BAD: classifier = svm.LinearSVC()
classifier = classifier.fit(X_plus,Y)
test_prediction = classifier.predict(test_plus)
scores = cross_val_score(classifier, X_plus, Y, cv=5)
print(scores) 

# Write prediction and it's linenumber into a csv file
with open('task4/' + result_file_name + '.csv', 'wb') as csvfile:
  fieldnames = ['Id', 'y']
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  #    print test_prediction
  writer.writeheader()
  for i in range(len(test_prediction)):
      writer.writerow({'Id': id_col[i], 'y': test_prediction[i]})

