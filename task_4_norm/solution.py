import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из файла
data = pd.read_csv('ex1data2.txt', header=None)
data.columns = ['Engine_Speed', 'Num_Gears', 'Price']

# Нормировка 1: делим на максимальное значение
# Описание: Каждый элемент делится на максимальное значение в столбце.
# Это приводит к тому, что все значения масштабируются в диапазон [0, 1].
data_normalized_1 = data.copy()
data_normalized_1.iloc[:, 0] = data.iloc[:, 0] / data.iloc[:, 0].max()  # Engine_Speed
data_normalized_1.iloc[:, 1] = data.iloc[:, 1] / data.iloc[:, 1].max()  # Num_Gears
data_normalized_1.iloc[:, 2] = data.iloc[:, 2] / data.iloc[:, 2].max()  # Price

# Нормировка 2: центрируем относительно среднего и делим на диапазон (max - min) значений(масштабируем в диапазоне значений)
# Описание: Масштабирует данные в определённый диапазон (обычно [-1, 1])
# с учётом разности между максимальным и минимальным значением. При этом данные центрируются относительно среднего.
# (x - u) / (max(X) - min(X))
data_normalized_2 = data.copy()
data_normalized_2.iloc[:, 0] = (data.iloc[:, 0] - data.iloc[:, 0].mean()) / (
            data.iloc[:, 0].max() - data.iloc[:, 0].min())
data_normalized_2.iloc[:, 1] = (data.iloc[:, 1] - data.iloc[:, 1].mean()) / (
            data.iloc[:, 1].max() - data.iloc[:, 1].min())
data_normalized_2.iloc[:, 2] = (data.iloc[:, 2] - data.iloc[:, 2].mean()) / (
            data.iloc[:, 2].max() - data.iloc[:, 2].min())

# Нормировка 3: центрируем и делим на стандартное отклонение
# Среднее значение будет равно 0 после нормировки, а стандартное отклонение 1
# Описание: Центрирует данные относительно нуля (среднее значение становится 0)
# и масштабирует их так, чтобы стандартное отклонение стало равным 1. Это называется стандартизацией
# (x - u) / std(X)
#  std = (1/n* sum((x - u)**2))**0.5
data_normalized_3 = data.copy()
data_normalized_3.iloc[:, 0] = (data.iloc[:, 0] - data.iloc[:, 0].mean()) / data.iloc[:, 0].std()
data_normalized_3.iloc[:, 1] = (data.iloc[:, 1] - data.iloc[:, 1].mean()) / data.iloc[:, 1].std()
data_normalized_3.iloc[:, 2] = (data.iloc[:, 2] - data.iloc[:, 2].mean()) / data.iloc[:, 2].std()


# Визуализация исходных и нормализованных данных
plt.figure(figsize=(12, 10))

# Исходные данные (только признаки)
plt.subplot(2, 2, 1)
plt.scatter(data['Engine_Speed'], data['Num_Gears'], color='blue')
plt.title('Исходные Признаки')
plt.xlabel('Скорость Оборота Двигателя')
plt.ylabel('Количество Передач')

# Нормированные данные (первый способ)
plt.subplot(2, 2, 2)
plt.scatter(data_normalized_1['Engine_Speed'], data_normalized_1['Num_Gears'], color='green')
plt.title('Нормировка 1 (max)')
plt.xlabel('Скорость Оборота Двигателя (нормированная)')
plt.ylabel('Количество Передач (нормированная)')

# Нормированные данные (второй способ)
plt.subplot(2, 2, 3)
plt.scatter(data_normalized_2['Engine_Speed'], data_normalized_2['Num_Gears'], color='red')
plt.title('Нормировка 2 (max - min)')
plt.xlabel('Скорость Оборота Двигателя (нормированная)')
plt.ylabel('Количество Передач (нормированная)')

# Нормированные данные (третий способ)
plt.subplot(2, 2, 4)
plt.scatter(data_normalized_3['Engine_Speed'], data_normalized_3['Num_Gears'], color='purple')
plt.title('Нормировка 3 (стандартное отклонение)')
plt.xlabel('Скорость Оборота Двигателя (нормированная)')
plt.ylabel('Количество Передач (нормированная)')

plt.tight_layout()
plt.savefig('normalized_features.png')

# Стандартные функции Python
means_explicit = {}
std_devs_explicit = {}

for col in data.columns:
    means_explicit[col] = sum(data[col]) / len(data[col])
    std_devs_explicit[col] = (sum((data[col] - means_explicit[col]) ** 2) / (len(data[col]))) ** 0.5

print("Средние значения (явное вычисление):\n", means_explicit)
print("Стандартные отклонения (явное вычисление):\n", std_devs_explicit)

# Явно по определению
means_std = data.mean()
std_devs_std = data.std()

print("Средние значения (стандартные функции):\n", means_std)
print("Стандартные отклонения (стандартные функции):\n", std_devs_std)
