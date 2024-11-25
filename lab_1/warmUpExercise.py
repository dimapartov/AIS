import numpy as np


def warmUpExercise(n):
    # Способ с использованием встроенной функции
    matrix_builtin = np.eye(n)
    print(f'Матрица с использованием numpy: \n{matrix_builtin}')

    # Способ без встроенных функций
    matrix_manual = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    print(f'Матрица без использования numpy: \n{np.array(matrix_manual)}')
