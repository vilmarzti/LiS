#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing

import sklearn.linear_model
import sklearn.grid_search
import sklearn.metrics
from sklearn.ensemble import GradientBoostingClassifier

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


def dim2polynom(array):
    #row = np.outer(array, array).flatten()
    #row = np.outer(array, row).flatten()
    return row


def validation(data, target, constant):
    score = 0
    
    
    
    #regressor = svm.NuSVR(kernel = "rbf", cache_size=1500, tol=1e-2)
    regressor = sklearn.linear_model.LassoLarsCV()
    #regressor = GradientBoostingClassifier()
    #param_grid = {'C':np.linspace(25.0 , 20000.0, num = 4), 'gamma':np.linspace(0.01/15.0, 0.5/15.0, 16)}
    #param_grid = {'eps':np.linspace(0.0001, 0.002, 4)}
    param_grid = {}
    grid_search = sklearn.grid_search.GridSearchCV(regressor,param_grid,scoring=sklearn.metrics.make_scorer(sklearn.metrics.mean_squared_error, greater_is_better = False), n_jobs=9)
    grid_search.fit(data,target)
    clf = grid_search.best_estimator_
    print (clf)
    

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
        RMSE = mean_squared_error(sample_target, pred)**0.5
        score += RMSE

    score = score/CVSize

    print("Cross-Validation RMSE: {} ".format(score))

    # Get global score
    clf.fit(data, target)
    pred = clf.predict(data)
    RMSE = mean_squared_error(target, pred)**0.5
    print("RMSE on whole dataset {}".format(RMSE))

    return score



if __name__ == "__main__":
    # plt.plot(goodness, [x for x in range(40, 60)])
    # plt.show()

    # extract the data and the target values
    raw  = getData(filename)
    target = np.array([row[1] for row in raw])
    data   = np.array([row[2:] for row in raw])
    poly = preprocessing.PolynomialFeatures(3)
    #data = poly.fit_transform(data)
    '''
    steps = 300
    start = 10
    goodness = np.zeros(steps)

    for constant in range(start, steps + start):
        print("Constant: {}".format(constant))
        goodness[constant-start] = validation(data, target, constant)

    print(goodness[np.argmin(goodness)])
    '''

    '''
    clf = svm.SVR(kernel="poly", C=29)
    clf.fit(data, target)
    '''

    #clf = svm.NuSVR(kernel = "poly", cache_size=8000, C=32.757936507936506)
    clf = sklearn.linear_model.LassoLarsCV()
    validation(data,target,0)
    clf.fit(data, target)
    # get prediction data
    raw = getData('test.csv')
    data   = [row[1:] for row in raw]
    number = [int(row[0]) for row in raw]
    #data = poly.fit_transform(data)

    predict = clf.predict(data)

    # Write prediction and it's linenumber into a csv file
    with open('result_lass_lars.csv', 'wb') as csvfile:
        fieldnames = ['Id', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for x in range(len(number)):
            writer.writerow({'Id': number[x], 'y': predict[x]})
