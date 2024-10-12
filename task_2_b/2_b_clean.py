import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 21)
y = x
theta1_values = np.linspace(-5, 5, 1000)
j_values = []
for theta1 in theta1_values:
    h_x = theta1 * x
    j = np.sum((h_x - y) ** 2)
    j_values.append(j)
noise = np.random.uniform(-2, 2, size=x.shape)
y_noisy = y + noise
j_noisy_values = []
for theta1 in theta1_values:
    h_x_noisy = theta1 * x
    j_noisy = np.sum((h_x_noisy - y_noisy) ** 2)
    j_noisy_values.append(j_noisy)
plt.figure(figsize=(10, 5))
plt.plot(theta1_values, j_values, label='Функционал ошибки J(theta1). Без шума', color='blue')
plt.plot(theta1_values, j_noisy_values, label='Функционал ошибки J(theta1). С шумом', color='red')
plt.title('Зависимость функционала ошибки J от theta1. Данные без шума и с шумом')
plt.xlabel('theta1')
plt.ylabel('J')
plt.grid()
plt.legend()
plt.ylim([0, 200])
plt.xlim([0, 2])
theta1_min = theta1_values[np.argmin(j_values)]
j_min = np.min(j_values)
theta1_min_noisy = theta1_values[np.argmin(j_noisy_values)]
j_min_noisy = np.min(j_noisy_values)
plt.show()
print(f'Минимальное значение функционала ошибки с данными без шума достигнуто при theta1min = {theta1_min}')
print(f'Минимальное значение функционала ошибки с зашумленными данными достигнуто при theta1min = {theta1_min_noisy}')
plt.figure(figsize=(10, 5))
h_x_min = theta1_min * x
plt.plot(x, h_x_min, label=f'Аппрокс. прямая для чистых данных\ntheta1 = {theta1_min:.2f}', color='blue')
plt.scatter(x, y_noisy, label='Зашумленные данные', color='red')
h_x_min_noisy = theta1_min_noisy * x
plt.plot(x, h_x_min_noisy, label=f'Аппрокс. прямая для зашумленных данных\ntheta1 = {theta1_min_noisy:.2f}',
         color='orange')
plt.title('Аппроксимирующие прямые для чистых и зашумленных данных')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.legend()
plt.show()
