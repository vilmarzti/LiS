#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

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

# Copmutes all distances between pairs of the given matrix rows
# Diagonals are reaplced wih NaN instead of 0


def dist_matrix(matrix):
    # Compute distances between all points
    distances = np.array([])
    for data_point in matrix:
        new_row = []
        for other_data_point in matrix:
            new_row.append(np.linalg.norm(data_point - other_data_point))

        if len(distances) == 0:
            distances = np.array([new_row])
        else:
            distances = np.vstack((distances, np.array([new_row])))

    # Change diagonal to NaN
    for i in range(len(distances)):
        distances[i][i] = np.NaN
    return distances


def dim2polynom(array):
    row = np.append(array, [1.0])
    a = np.outer(row, row).flatten()
    return a


if __name__ == "__main__":
    best_coef  = []
    prev_score = 0
    reg = linear_model.LinearRegression(n_jobs=-1)

    # extract the data and the target values
    raw  = getData('train.csv')
    target = np.array([row[1] for row in raw])
    data   = np.array([dim2polynom(row[2:]) for row in raw])

    distances   = dist_matrix(raw)
    x_distances = dist_matrix(data)

    # Get Median
    flat_distances = []
    for x in range(len(distances)):
        for y in range(len(distances)):
            dist = distances[x][y]
            if not np.isnan(dist):
                flat_distances.append(dist)

    print("The median is: {}".format(np.median(flat_distances)))

    # Comute nearest neighbour distance for each data point
    nnd   = np.nanmin(distances, axis=0)
    x_nnd = np.nanmin(x_distances, axis=0)

    difference = np.array([abs(val) for val in (nnd - x_nnd)])

    # Get all the indices which are ok
    # acceptabel = [val for val in difference if val < 22]
    # indices = [np(val) for val in acceptabel]
    indices = np.where(difference < 10)

    with open('smooth_training.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
                        'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15'])

        for index in indices[0]:
            writer.writerow(np.asarray(raw[index]))

#    print("indices above cutoff: {}".format(indices))
