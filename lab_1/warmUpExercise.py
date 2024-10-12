import numpy as np


def warmUpExercise(n):
    # Способ с использованием встроенной функции
    matrix_builtin = np.eye(n)
    print(f'Identity matrix using numpy: \n{matrix_builtin}')

    # Способ без встроенных функций
    matrix_manual = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    print(f'Identity matrix manually: \n{np.array(matrix_manual)}')
