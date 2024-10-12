import numpy as np


def computeCostByVector(X, Y, theta):
    m = len(Y)
    predictions = X.dot(theta)
    errors = predictions - Y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost


def computeCostByElements(X, Y, theta):
    m = len(Y)
    total_cost = 0

    for i in range(m):
        prediction = X[i].dot(theta)  # Предсказание для i-й строки
        error = prediction - Y[i]  # Ошибка для i-й строки
        total_cost += error ** 2  # Квадрат ошибки

    cost = (1 / (2 * m)) * total_cost
    return cost
