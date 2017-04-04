#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

import sklearn.linear_model
import sklearn.grid_search
import sklearn.metrics

from sklearn.metrics import mean_squared_error

CVSize = 5
filename = 'train.csv'
filename = './smooth_training.csv'

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
    row = array
    return row


def validation(data, target, constant):
    score = 0



    regressor = svm.NuSVR(kernel = "poly")

    param_grid = {'C':np.linspace(20.0 , 40.0, 10),
                  'nu':np.linspace(0.0001, 1, 5)}

    grid_search = sklearn.grid_search.GridSearchCV(regressor,
                        param_grid,
                        scoring=sklearn.metrics.make_scorer(sklearn.metrics.mean_squared_error,
                                                            greater_is_better = False),
                        cv=5,
                        n_jobs=-1)
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
    data   = np.array([dim2polynom(row[2:]) for row in raw])

    validation(data, target, 0)
