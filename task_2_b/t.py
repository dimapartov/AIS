import numpy as np
import matplotlib.pyplot as plt

# Данные
x = np.arange(1, 21)
y = x  # y_i = x_i для первых данных
y_noisy = x + np.random.uniform(-2, 2, size=x.shape)  # Зашумленные данные

# Значения theta1 для первых данных
theta1_values = [1, 0.8, 0.9, 1.10]

# Найдем оптимальное theta1 для зашумленных данных методом наименьших квадратов
theta1_optimal = 1.02

# Построение графиков
plt.figure(figsize=(10, 6))

# График для значений theta1
for theta1 in theta1_values:
    h_x = theta1 * x  # Аппроксимирующая прямая
    plt.plot(x, h_x, color='blue', alpha=0.2)  # Аппроксимация

# График для оптимального theta1 для зашумленных данных
h_x_noisy_optimal = theta1_optimal * x  # Аппроксимирующая прямая для зашумленных данных
plt.plot(x, h_x_noisy_optimal, color='green', label=f'Approximation (Noisy Data, theta1={theta1_optimal:.2f})', alpha=0.5)

# Отображение исходных данных
plt.scatter(x, y, color='red', label='Experimental Data', marker='o')
plt.scatter(x, y_noisy, color='orange', label='Noisy Data', marker='x')

# Настройки осей
plt.xlim(0, 21)  # Установка границ оси x от 0 до 21
plt.xticks(np.arange(0, 22, 1))  # Установка шагов на оси x (от 0 до 21 с шагом 1)
plt.ylim(0, max(y_noisy) + 5)  # Границы по оси y для лучшей видимости графиков

# Настройки графика
plt.title("Approximating Lines for Experimental and Noisy Data")
plt.xlabel("x")
plt.ylabel("h(x) = theta1 * x")
plt.grid(True)
plt.legend()
plt.show()
