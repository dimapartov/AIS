import numpy as np
import matplotlib.pyplot as plt

# 1. Генерация данных (xi = 1, 2, ..., 20, yi = xi)
x = np.arange(1, 21)
y = x  # yi = xi

# Значения theta1, которые будем использовать
theta1_values = np.linspace(-5, 5, 1000)  # Увеличиваем количество точек до 1000 для более гладкого графика

# 2. Вычисляем функционал ошибки J(theta1) для чистых данных
j_values = []
for theta1 in theta1_values:
    h_x = theta1 * x  # h(x) = theta1 * x
    j = np.sum((h_x - y) ** 2)  # J(theta1)
    j_values.append(j)

# 3. Добавляем шум к данным
noise = np.random.uniform(-2, 2, size=x.shape)
y_noisy = y + noise  # Добавляем шум

# 4. Вычисляем функционал ошибки J(theta1) для зашумленных данных
j_noisy_values = []
for theta1 in theta1_values:
    h_x_noisy = theta1 * x  # h(x) = theta1 * x
    j_noisy = np.sum((h_x_noisy - y_noisy) ** 2)  # J(theta1) для зашумленных данных
    j_noisy_values.append(j_noisy)

# 5. Построение графика зависимости J(theta1) для чистых и зашумленных данных
plt.figure(figsize=(10, 5))

# Чистые данные
plt.plot(theta1_values, j_values, label='J(theta1) чистые данные', color='blue')
# Зашумленные данные
plt.plot(theta1_values, j_noisy_values, label='J(theta1) зашумленные данные', color='orange')

plt.title('Зависимость функционала ошибки J(theta1) от theta1 (чистые и зашумленные данные)')
plt.xlabel('theta1')
plt.ylabel('J(theta1)')
plt.grid()
plt.legend()

# Установим лимиты по осям
plt.ylim([0, 200])
plt.xlim([0, 2])

# 6. Нахождение минимума для чистых данных
theta1_min = theta1_values[np.argmin(j_values)]
j_min = np.min(j_values)
# 7. Нахождение минимума для зашумленных данных
theta1_min_noisy = theta1_values[np.argmin(j_noisy_values)]
j_min_noisy = np.min(j_noisy_values)

# 8. Отметка минимума на графике для чистых данных
plt.scatter(theta1_min, j_min, color='red')  # Точка минимума
plt.annotate(f'Minimum чистые\nθ1 = {theta1_min:.2f}\nJ(θ1) = {j_min:.2f}',
             xy=(theta1_min, j_min),
             xytext=(theta1_min + 0.5, j_min + 5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10)

# 9. Отметка минимума на графике для зашумленных данных
plt.scatter(theta1_min_noisy, j_min_noisy, color='red')  # Точка минимума
plt.annotate(f'Minimum зашумлённые\nθ1 = {theta1_min_noisy:.2f}\nJ(θ1) = {j_min_noisy:.2f}',
             xy=(theta1_min_noisy, j_min_noisy),
             xytext=(theta1_min_noisy + 0.5, j_min_noisy + 5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10)

# Показываем график
plt.show()

print(f'Minimum J(theta1) при theta1 (чистые данные) = {theta1_min}')
print(f'Minimum J(theta1) при theta1 (зашумленные данные) = {theta1_min_noisy}')

# --- Второй график: аппроксимирующие прямые по исходным данным ---

plt.figure(figsize=(10, 5))

# Построение чистых данных
plt.scatter(x, y, label='Исходные данные', color='blue')

# Построение аппроксимирующей прямой для чистых данных
h_x_min = theta1_min * x  # Прямая для чистых данных
plt.plot(x, h_x_min, label=f'Аппроксимация (чистые данные)\nθ1 = {theta1_min:.2f}', color='blue')

# Построение зашумленных данных
plt.scatter(x, y_noisy, label='Зашумленные данные', color='orange')

# Построение аппроксимирующей прямой для зашумленных данных
h_x_min_noisy = theta1_min_noisy * x  # Прямая для зашумленных данных
plt.plot(x, h_x_min_noisy, label=f'Аппроксимация (зашумленные данные)\nθ1 = {theta1_min_noisy:.2f}', color='orange')

plt.title('Аппроксимирующие прямые для чистых и зашумлённых данных')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()

# Показываем график
plt.show()
