#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

CVSize = 5
#filename = './smooth_training.csv'
filename = './train.csv'


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


def my_kernel(X, Y):
    print(len(X), len(Y))
    a = np.inner(X, Y)
    m = (1 + a)**2 + np.exp((1.0/20) * np.linalg.norm(X-Y)**2)
    return m


def validation(data, target, constant):
    print("Constant: {}".format(constant))
    score = 0
    clf = svm.SVR(kernel="rbf", C=800, gamma=constant)

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
#
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
    print("\n")

    return score


def write_results(raw_data, clf, scaler):
    data   = [row[1:] for row in raw]
    number = [int(row[0]) for row in raw]

    predict = clf.predict(data)

    # Write prediction and it's linenumber into a csv file
    with open('result.csv', 'wb') as csvfile:
        fieldnames = ['Id', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for x in range(len(number)):
            writer.writerow({'Id': number[x], 'y': predict[x]})


if __name__ == "__main__":
    # extract the data and the target values
    raw  = getData(filename)

    target = np.array([row[1] for row in raw])
    scaler = preprocessing.StandardScaler().fit(target)
    new_target = scaler.transform(target)
    data   = np.array([row[2:] for row in raw])

    # Binary seach
    par_range =  [0.0, 0.5, 1.0]

    for _ in range(100):
        first_half = par_range[0] + (par_range[1] - par_range[0])/2
        second_half = par_range[1] + (par_range[2] - par_range[1])/2
        print(first_half, second_half)

        first_par_range = validation(data, new_target, first_half)
        second_par_range = validation(data, new_target, second_half)

        if first_par_range < second_par_range:
            par_range[2] = par_range[1]
            par_range[1] = first_half
        else:
            par_range[0] = par_range[1]
            par_range[1] = second_half

    clf = svm.SVR(kernel="rbf", C = 800, gamma=par_range[1])
    clf.fit(data, new_target)

    # get prediction data
    raw = getData('test.csv')
    write_results(raw, clf, scaler)
