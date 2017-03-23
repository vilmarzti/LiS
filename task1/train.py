#!/usr/bin/env python
import csv
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

CVSize = 10


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
    row = np.append(array, [1.0])
    a = np.outer(row, row).flatten()
    a = np.outer(a, row).flatten()
    return a


if __name__ == "__main__":
    best_coef  = []
    prev_score = 0
    reg = linear_model.LinearRegression(normalize=True, n_jobs=-1)

    # extract the data and the target values
    raw  = getData('train.csv')
    np.random.shuffle(raw)
    target = np.array([row[1] for row in raw])
    data   = np.array([dim2polynom(row[2:]) for row in raw])

    chunk_size = len(data)/CVSize
    print("Crossvalidation with {} samples and sample size {}".format(CVSize,chunk_size))
    for x in range(CVSize):
        # These describe where to cut to get our crossdat
        first_step  = x*chunk_size
        second_step = (x+1)*chunk_size

        # Get the data parts we train on
        cross_data   = np.vstack((data[:first_step], data[second_step:]))
        cross_target = np.append(target[:first_step], target[second_step:])

        # fit and save the coef
        reg.fit(cross_data, cross_target)
        coef = reg.coef_

        # Find mean squared error and print it
        sample_data   = data[first_step:second_step]
        sample_target = target[first_step:second_step]

        # Get scores for our model
        pred = np.dot(sample_data, coef)
        RMSE = mean_squared_error(sample_target, pred)**0.5
        score = reg.score(data, target)

        # Save best score
        if prev_score == 0 or prev_score < score:
            best_coef = coef
            prev_score = score
            best_rmse = RMSE

    print("Best: {} {} ".format(prev_score, best_rmse))

    # Get global score
    pred = np.dot(data, best_coef)
    RMSE = mean_squared_error(target, pred)**0.5
    print("RMSE on whole dataset {}".format(RMSE))

    # get prediction data
    raw = getData('test.csv')
    data   = [dim2polynom(row[1:]) for row in raw]
    number = [int(row[0]) for row in raw]
    predict = np.dot(data, coef)

    # Write prediction and it's linenumber into a csv file
    with open('result.csv', 'wb') as csvfile:
        fieldnames = ['Id', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for x in range(len(number)):
            writer.writerow({'Id': number[x], 'y': predict[x]})