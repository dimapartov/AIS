import matplotlib.pyplot as plt


def plot(X, Y):
    plt.scatter(X, Y, marker='x', color='r')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль СТО')
    plt.show()
