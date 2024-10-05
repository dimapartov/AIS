import numpy as np
import matplotlib.pyplot as plt

# Данные
x = np.arange(1, 21)
y = x + np.random.uniform(-2, 2, size=x.shape)  # Зашумленные данные

# Значения theta1
theta1_values = [0.8, 0.9, 1, 1.10, 1.20]

# Построение графиков для разных значений theta1
plt.figure(figsize=(10, 6))
for theta1 in theta1_values:
    h_x = theta1 * x  # Аппроксимирующая прямая
    plt.plot(x, h_x, color='blue', alpha=0.2)  # Построение графика с небольшим уровнем прозрачности

# Отображение зашумлённых данных
plt.scatter(x, y, color='red', label='Noisy Data')

# Настройки осей
plt.xlim(0, 21)  # Установка границ оси x от 0 до 21
plt.xticks(np.arange(0, 22, 1))  # Установка шагов на оси x (от 0 до 21 с шагом 1)
plt.ylim(0, max(y) * max(theta1_values) + 1)  # Границы по оси y для лучшей видимости графиков

# Настройки графика
plt.title("Approximating Lines for Different Theta1 Values with Noisy Data")
plt.xlabel("x")
plt.ylabel("h(x) = theta1 * x")
plt.grid(True)
plt.legend()
plt.show()
