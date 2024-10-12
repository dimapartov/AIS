import numpy as np


def predict_profit(cars, theta):
    X = np.array([1, cars])  # Добавляем единичку для свободного члена
    return X.dot(theta)
