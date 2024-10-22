import numpy as np
vector1 = np.array([[1],
                    [2],
                    [3]])  # Вектор 1
vector2 = np.array([[4],
                    [5],
                    [6]])  # Вектор 2

print(f"Размер вектора 1: {vector1.shape} (это вектор-столбец)")
print(f"Размер вектора 2: {vector2.shape} (это вектор-столбец)")

vectorized = sum(vector1[i][0] * vector2[i][0] for i in range(len(vector1)))
print(f"Скалярное произведение (1 способ): {vectorized}")

byElements = 0
for i in range(len(vector1)):
    byElements += vector1[i][0] * vector2[i][0]
print(f"Скалярное произведение (2 способ): {byElements}")

dot_product_vectorized = np.dot(vector1.T, vector2)[0][0]
print(f"Скалярное произведение (3 способ): {dot_product_vectorized}")

