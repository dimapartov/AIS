import numpy as np


def computeCostByVector(X, Y, theta):
    m = len(Y)
    predictions = np.dot(X, theta)
    errors = predictions - Y
    cost = (1 / (2 * m)) * np.dot(errors, errors)
    return cost


def computeCostByElements(X, Y, theta):
    m = len(Y)
    total_cost = 0

    for i in range(m):
        prediction = X[i].dot(theta)
        error = prediction - Y[i]
        total_cost += error ** 2

    cost = (1 / (2 * m)) * total_cost
    return cost
