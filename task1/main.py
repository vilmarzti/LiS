#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt


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


def get_median(distance_matrix):
    length = len(distance_matrix)
    flat_distances = []

    for x in range(length):
        for y in range(length):
            dist = distance_matrix[x][y]
            if not np.isnan(dist):
                flat_distances.append(dist)

    return np.median(flat_distances)

if __name__ == "__main__":
    best_coef  = []
    prev_score = 0

    # extract the data and the target values
    raw  = getData('train.csv')
    target = np.array([row[1] for row in raw])
    whole  = np.array([row[1:] for row in raw])
    data   = np.array([row[2:] for row in raw])

    distances   = dist_matrix(whole)
    x_distances = dist_matrix(data)

    # Comute nearest neighbour distance for each data point
    nnd   = np.nanmin(distances, axis=0)
    x_nnd = np.nanmin(x_distances, axis=0)
    x_nn    = np.nanargmin(x_distances, axis=1)


    o_difference = []
    for x in range(900):
        nn_index = x_nn[x]
        x_norm = np.linalg.norm(data[x] - data[nn_index])
        distance = abs(target[x] - target[nn_index])/x_norm
        o_difference.append(distance)

    o_difference = np.array(o_difference)
    plt.plot(np.sort(o_difference))
    plt.show()

    difference = np.array([val for val in (nnd - x_nnd)])

    # Get all the indices which are ok
    indices = np.where(o_difference < 25)

    with open('smooth_training.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
                        'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15'])

        for index in indices[0]:
            writer.writerow(np.asarray(raw[index]))

    print("number of indices above cutoff: {}".format(len(indices[0])))
