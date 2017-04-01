#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.metrics import mean_squared_error

CVSize = 5


def getData(filename):
    raw    = []

    # read from the csv
    with open(filename, 'rb') as csvfile:
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
    clf = svm.SVR(kernel="sigmoid", C=constant)

    # extract the data and the target values

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
    raw  = getData('train.csv')
    target = np.array([row[1] for row in raw])
    data   = np.array([dim2polynom(row[2:]) for row in raw])

    steps = 100
    start = 1
    goodness = np.zeros(steps)

    for constant in range(start, steps + start):
        print("Constant: {}".format(constant))
        goodness[constant-start] = validation(data, target, constant)

    print(goodness[np.argmin(goodness)])

    clf = svm.SVR(kernel="poly", C=44)
    clf.fit(data, target)

    # get prediction data
    raw = getData('test.csv')
    data   = [dim2polynom(row[1:]) for row in raw]
    number = [int(row[0]) for row in raw]

    predict = clf.predict(data)

    # Write prediction and it's linenumber into a csv file
    with open('result.csv', 'wb') as csvfile:
        fieldnames = ['Id', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for x in range(len(number)):
            writer.writerow({'Id': number[x], 'y': predict[x]})
