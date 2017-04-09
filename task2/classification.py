#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

import sklearn.linear_model
import sklearn.grid_search
import sklearn.metrics
import sklearn.ensemble
from sklearn.metrics import accuracy_score
import sklearn
from sklearn import naive_bayes
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import neural_network
from sklearn import calibration
from sklearn import tree
from sklearn import linear_model

from sklearn.metrics import mean_squared_error

CVSize = 5
filename = 'train.csv'

def getData(filepath):
    raw    = []
    
    # read from the csv
    with open(filepath, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            raw.append(row)

    # format the data
    raw = raw[1:]
    raw = np.array([[float(val) for val in row] for row in raw])
    return raw


def getClassifier(data, target):
    score = 0
    temp = 0
    
    
    # Classifier to use in BaggingClassifier
    classifier1 = ensemble.ExtraTreesClassifier(min_samples_split = 3, n_estimators = 10, max_features = 4)
    
    # Classifier for GridSearch
    classifier = ensemble.BaggingClassifier(classifier1)
    
    # Params
    param_grid = {'n_estimators' : range(5,25)}
    #param_grid = {'n_estimators' : np.linspace(10,11, num = 2)}
    
    # GridSearch
    grid_search = sklearn.grid_search.GridSearchCV(classifier,param_grid,scoring=sklearn.metrics.make_scorer(accuracy_score),cv=5, n_jobs = 4)
    grid_search.fit(data,target)
    clf = grid_search.best_estimator_
    
    # Print Estimator
    print (clf)
    
    # Print Cross of Validations Scores
    print(cross_val_score(clf, data,target, cv =5, scoring = 'accuracy'))
    
    # Print Mean of Cross Validations Scores
    temp = np.mean(cross_val_score(clf, data,target, cv =5, scoring = 'accuracy'))
    print("Built-in Cross-Validation: {} ".format(temp))
    
    # Martins Version of Cross Validation
    chunk_size = len(data)/CVSize
    for x in range(CVSize):
        
        # These describe where to cut to get our crossdat
        first_step  = x*chunk_size
        second_step = (x+1)*chunk_size
        
        # Get the data parts we train on
        cross_data   = np.vstack((data[:first_step], data[second_step:]))
        cross_target = np.append(target[:first_step], target[second_step:])
        
        # fit and save the coef
        clf.fit(cross_data, cross_target)
        
        # Find mean squared error and print it
        sample_data   = data[first_step:second_step]
        sample_target = target[first_step:second_step]
        
        # Get scores for our model
        pred = clf.predict(sample_data)
        RMSE = accuracy_score(sample_target, pred)
        score += RMSE
    
    score = score/CVSize

    print("Cross-Validation RMSE: {} ".format(score))
    
    # Get global score
    #clf.fit(data, target)
    #pred = clf.predict(data)
    #RMSE = accuracy_score(target, pred)
    #print("RMSE on whole dataset {}".format(RMSE))

    # Return estimator/classifier
    return clf



if __name__ == "__main__":
    # Preprocessing utils
    poly = PolynomialFeatures(degree=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    
    # extract the data and the target values
    raw  = getData(filename)
    target = np.array([row[1] for row in raw])
    data   = np.array([row[2:] for row in raw])
    
    # Uncomment for Normalize/PolyFeatures/Scale/MinMaxScale the data
    #data = preprocessing.normalize(data, norm='l2')
    #data = poly.fit_transform(data)
    #data = preprocessing.scale(data)
    #data = min_max_scaler.fit_transform(data)
    
    #
    clf = getClassifier(data,target)
    
    
    # Get prediction data
    clf.fit(data,target)
    raw = getData('test.csv')
    data   = [row[1:] for row in raw]
    
    #Need to make the same preprocessing here if uncommented above
    #data = min_max_scaler.fit_transform(data)
    
    number = [int(row[0]) for row in raw]
    
    predict = clf.predict(data)
    
    # Write prediction and it's linenumber into a csv file
    with open('result.csv', 'wb') as csvfile:
        fieldnames = ['Id', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for x in range(len(number)):
            writer.writerow({'Id': number[x], 'y': predict[x]})


