import csv
import math
import random


# Загрузить набор данных
def load_dataset(filepath):
    data = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Пропустить заголовок
        for row in reader:
            data.append([float(value) for value in row])
    return data


# Разделить набор данных на обучающую и тестовую выборки
def train_test_split(data, test_size=0.2):
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]


# Сигмоидная функция для логистической регрессии
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Предсказание с использованием весов логистической регрессии
def predict(features, weights):
    z = sum(w * x for w, x in zip(weights, [1] + features))  # Включить смещение (bias)
    return sigmoid(z)


# Обучение логистической регрессии с использованием градиентного спуска
def train_logistic_regression(X, y, epochs, learning_rate):
    n_features = len(X[0])
    weights = [0.0] * (n_features + 1)  # +1 для смещения (bias)

    for _ in range(epochs):
        for i in range(len(X)):
            prediction = predict(X[i], weights)
            error = y[i] - prediction
            # Обновить веса и смещение
            for j in range(n_features):
                weights[j + 1] += learning_rate * error * X[i][j]
            weights[0] += learning_rate * error  # Обновление смещения

    return weights


# Оценка точности модели
def compute_accuracy(X, y, weights):
    correct_predictions = 0
    for i in range(len(X)):
        prediction = predict(X[i], weights) >= 0.5  # Порог 0.5 для бинарной классификации
        if prediction == y[i]:
            correct_predictions += 1
    return correct_predictions / len(X)


# Предсказание статуса для заданных пользователем признаков
def predict_status(input_features, weights):
    if len(input_features) != len(weights) - 1:
        raise ValueError(f"Ожидалось {len(weights) - 1} признаков, но получено {len(input_features)}.")

    # Вычислить предсказание с использованием сигмоидной функции
    z = weights[0] + sum(w * x for w, x in zip(weights[1:], input_features))
    return 1 if sigmoid(z) >= 0.5 else 0


# Основная функция для обучения и оценки модели
if __name__ == "__main__":
    # Загрузить и обработать набор данных
    file_path = '../data/data_frame_normalized_trimmed_shuffled.csv'
    data = load_dataset(file_path)

    # Разделить на признаки и целевую переменную
    features = [row[:-1] for row in data]
    labels = [int(row[-1]) for row in data]  # Преобразовать в целые числа (0 или 1)

    # Разделить на обучающую и тестовую выборки
    train_data, test_data = train_test_split(data)
    X_train = [row[:-1] for row in train_data]
    y_train = [int(row[-1]) for row in train_data]
    X_test = [row[:-1] for row in test_data]
    y_test = [int(row[-1]) for row in test_data]

    # Обучить модель
    weights = train_logistic_regression(X_train, y_train, 1000, 0.01)

    # Оценить модель
    accuracy = compute_accuracy(X_test, y_test, weights)
    print(f"Точность модели: {accuracy:.2f}")

    # Интерактивное предсказание
    print("\nВведите значения признаков (через запятую, соответствующие первым 6 столбцам):")
    user_input = input("Формат: Hour(1-24),Category(1-4),Tyres(1-2),Pit Stops, Best Lap Time, Best Lap Kph: ")

    try:
        # Преобразовать ввод в список чисел с плавающей точкой
        features = list(map(float, user_input.split(',')))

        # Сделать предсказание с использованием обученных весов
        result = predict_status(features, weights)
        print(f"Предсказанный статус: {result}")
    except ValueError as e:
        print(f"Ошибка: {e}")
