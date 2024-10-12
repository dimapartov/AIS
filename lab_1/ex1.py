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

# Визуализация данных
plotData.plot(X[:, 1], Y)

# Вычисление стоимости (векторное)
costByVector = computeCost.computeCostByVector(X, Y, theta)
print(f'Initial cost by vector: {costByVector}')

# Вычисление стоимости (поэлементное)
costByElements = computeCost.computeCostByElements(X, Y, theta)
print(f'Initial cost by elements: {costByElements}')

# Градиентный спуск (векторный)
# thetaByVector, cost_history = gradientDescent.gradientDescentByVector(X, Y, theta, alpha, iterations)
thetaByVector = gradientDescent.gradientDescentByVector(X, Y, theta, alpha, iterations)
print(f'Optimized theta by vector: {thetaByVector}')

# Градиентный спуск (поэлементный)
# thetaByElements, cost_history = gradientDescent.gradientDescentByElements(X, Y, theta, alpha, iterations)
thetaByElements = gradientDescent.gradientDescentByElements(X, Y, theta, alpha, iterations)
print(f'Optimized theta by elements: {thetaByElements}')

cars = float(input("Введите количество автомобилей: "))
profitByVector = predict_profit(cars, thetaByVector)
profitByElements = predict_profit(cars, thetaByElements)
print(f"Прогнозируемая прибыль СТО (векторный способ): {profitByVector}")
print(f"Прогнозируемая прибыль СТО (поэлементный способ): {profitByElements}")
