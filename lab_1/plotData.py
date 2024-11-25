import matplotlib.pyplot as plt
import numpy as np


# def plot(X, Y):
#     plt.scatter(X, Y, marker='x', color='r')
#     plt.xlabel('Количество автомобилей')
#     plt.ylabel('Прибыль СТО')
#     plt.show()

def plot(X, Y, theta):
    plt.scatter(X, Y, marker='X', color='green', label='Данные')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль')
    plt.title('График зависимости прибыли от количества автомобилей')

    X_line = np.linspace(min(X), max(X), 100)
    Y_line = theta[0] + theta[1] * X_line

    plt.plot(X_line, Y_line, color='red', label='Линия регрессии')
    plt.legend()
    plt.savefig("plotResult.png")
