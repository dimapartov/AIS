import numpy as numpy
import matplotlib.pyplot as pyplot

# 1. Чистые данные
x = numpy.arange(1, 21)
y = x  # y_i = x_i

# Значения theta1, которые будем использовать
theta1_values = numpy.linspace(-5, 5, 100)
j_values = []

# 2. Вычисляем функционал ошибки для каждого theta1
for theta1 in theta1_values:
    h_x = theta1 * x  # h(x) = theta1 * x
    j = numpy.sum((h_x - y) ** 2)  # J(theta1)
    j_values.append(j)

# 3. Построение графика зависимости J(theta1) от theta1
pyplot.figure(figsize=(10, 5))
pyplot.plot(theta1_values, j_values, label = 'J(theta1)')
pyplot.title('Зависимость функционала ошибки J от theta1')
pyplot.xlabel('theta1')
pyplot.ylabel('J')
pyplot.yticks(numpy.arange(min(j_values), max(j_values), 10000))
pyplot.xticks(numpy.arange(-5.5, 6, 0.5))  # Устанавливаем шаг 0.5 на оси X
pyplot.grid()
pyplot.legend()
pyplot.tight_layout()  # Автоматическая подгонка
pyplot.show()

# 4. Нахождение theta1min
theta1_min = theta1_values[numpy.argmin(j_values)]
# print(f'theta1min, соответствующее минимуму функционала, равно {theta1_min}')
