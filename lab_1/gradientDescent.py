import numpy as np

from computeCost import computeCostByElements
from computeCost import computeCostByVector


def gradientDescentByVector(X, Y, theta, alpha, iterations):
    m = len(Y)
    # cost_history = []

    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - Y
        theta -= (alpha / m) * (X.T.dot(errors))
        # cost_history.append(computeCostByVector(X, Y, theta))

    # return theta, cost_history
    return theta


def gradientDescentByElements(X, Y, theta, alpha, iterations):
    m = len(Y)
    # cost_history = []

    for _ in range(iterations):
        # Поэлементное вычисление предсказаний и ошибок
        predictions = np.zeros(m)
        errors = np.zeros(m)

        for i in range(m):
            predictions[i] = X[i].dot(theta)
            errors[i] = predictions[i] - Y[i]

        # Обновление theta поэлементно
        for j in range(len(theta)):
            gradient = 0
            for i in range(m):
                gradient += errors[i] * X[i][j]
            theta[j] -= (alpha / m) * gradient

        # cost_history.append(computeCostByElements(X, Y, theta))

    # return theta, cost_history
    return theta
