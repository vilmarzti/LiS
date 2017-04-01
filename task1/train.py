#!/usr/bin/env python
import csv
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

CVSize = 15


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
    return a


if __name__ == "__main__":
    best_coef  = []
    prev_score = 0
    reg = linear_model.Lasso(alpha=.1)

    # extract the data and the target values
    raw  = getData('train.csv')
    np.random.shuffle(raw)
    target = np.array([row[1] for row in raw])
    data   = np.array([dim2polynom(row[2:]) for row in raw])

    reg.fit(data, target)
    chunk_size = len(data)/CVSize

    # get prediction data
    raw = getData('test.csv')
    data   = [dim2polynom(row[1:]) for row in raw]
    number = [int(row[0]) for row in raw]
    predict = reg.predict(data)

    # Write prediction and it's linenumber into a csv file
    with open('result.csv', 'wb') as csvfile:
        fieldnames = ['Id', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for x in range(len(number)):
            writer.writerow({'Id': number[x], 'y': predict[x]})
