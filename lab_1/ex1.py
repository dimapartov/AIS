import numpy as np

import computeCost
import gradientDescent
import plotData
from warmUpExercise import warmUpExercise
from work import predict_profit

# Разминка
warmUpExercise(3)

# Загрузка данных
data = np.loadtxt('train_data.txt', delimiter=',')
X = data[:, 0]
Y = data[:, 1]
m = len(Y)  # количество примеров

# Добавляем столбец единиц для X
X = np.column_stack((np.ones(m), X))

# Инициализация параметров
theta = np.zeros(2)
iterations = 1500
alpha = 0.01  # скорость обучения

# Вычисление стоимости (векторное)
costByVector = computeCost.computeCostByVector(X, Y, theta)
print(f'Значение функции стоимости(векторный способ): {costByVector}')

# Вычисление стоимости (поэлементное)
costByElements = computeCost.computeCostByElements(X, Y, theta)
print(f'Значение функции стоимости(поэлементный способ): {costByElements}')

# Градиентный спуск (векторный)
thetaByVector = gradientDescent.gradientDescentByVector(X, Y, theta, alpha, iterations)
print(f'Градиентный спуск(векторный способ): {thetaByVector}')

# Градиентный спуск (поэлементный)
thetaByElements = gradientDescent.gradientDescentByElements(X, Y, theta, alpha, iterations)
print(f'Градиентный спуск(поэлементный способ): {thetaByElements}')

# Визуализация данных
plotData.plot(X[:, 1], Y, thetaByVector)

cars = float(input("Введите количество автомобилей: "))
profitByVector = predict_profit(cars, thetaByVector)
profitByElements = predict_profit(cars, thetaByElements)
print(f"Прогнозируемая прибыль СТО (векторный способ): {profitByVector}")
print(f"Прогнозируемая прибыль СТО (поэлементный способ): {profitByElements}")
