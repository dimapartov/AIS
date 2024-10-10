import numpy as numpy
from matplotlib import pyplot as pyplot

x = numpy.arange(1, 21)
y = x  # y_i = x_i

# Значения theta1, которые будем использовать
theta1_values = numpy.linspace(-5, 5, 100)

noise = numpy.random.uniform(-2, 2, size = x.shape)
y_noisy = y + noise  # Добавляем шум

# 2. Вычисляем функционал ошибки для каждого theta1 с зашумленными данными
j_noisy_values = []

for theta1 in theta1_values:
    h_x_noisy = theta1 * x  # h(x) = theta1 * x
    j_noisy = numpy.sum((h_x_noisy - y_noisy) ** 2)  # J(theta1) для зашумленных данных
    j_noisy_values.append(j_noisy)

# 3. Построение графика зависимости J(theta1) от theta1 для зашумленных данных
pyplot.figure(figsize=(10, 5))
pyplot.plot(theta1_values, j_noisy_values, label='J(theta1) с шумом')
pyplot.title('Зависимость функционала ошибки J от theta1 c зашумленными данными)')
pyplot.xlabel('theta1')
pyplot.ylabel('J')
pyplot.yticks(numpy.arange(min(j_noisy_values), max(j_noisy_values), 10000))
pyplot.xticks(numpy.arange(-5.5, 6, 0.5))  # Устанавливаем шаг 0.5 на оси X
pyplot.grid()
pyplot.legend()
pyplot.tight_layout()  # Автоматическая подгонка
pyplot.show()

# 4. Нахождение theta1min для зашумленных данных
theta1_min_noisy = theta1_values[numpy.argmin(j_noisy_values)]
# print(f'theta1min, соответствующее минимуму функционала, с зашумленными данными, равно {theta1_min_noisy}')
