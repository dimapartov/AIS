import csv
import math
import random


# Загрузить набор данных с преобразованием меток:
# Статус 1 преобразуем в 0, статус 2 преобразуем в 1
def load_dataset(filepath):
    data = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Пропустить заголовок
        for row in reader:
            features = [float(value) for value in row[:-1]]
            label = int(float(row[-1]))
            if label == 1:
                label = 0
            elif label == 2:
                label = 1
            else:
                raise ValueError("Найдено неожиданное значение статуса")
            data.append(features + [label])
    return data


# Разделить набор данных на обучающую и тестовую выборки
def train_test_split(data, test_size=0.2):
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]


# Численно устойчивая сигмоидная функция
def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)


# Предсказание с использованием весов логистической регрессии
def predict(features, weights):
    z = weights[0] + sum(w * x for w, x in zip(weights[1:], features))
    return sigmoid(z)


# Обучение логистической регрессии с учетом весовой корректировки для дисбаланса классов
def train_logistic_regression(X, y, epochs, learning_rate):
    n_features = len(X[0])
    weights = [0.0] * (n_features + 1)  # +1 для смещения (bias)

    # Подсчет количества примеров для каждого класса
    count_0 = sum(1 for label in y if label == 0)
    count_1 = sum(1 for label in y if label == 1)
    # Коэффициент для примеров минорного класса (label == 1)
    weight_factor = count_0 / count_1 if count_1 != 0 else 1.0

    for _ in range(epochs):
        for i in range(len(X)):
            prediction = predict(X[i], weights)
            # Если пример относится к минорному классу, усиливаем его вклад
            sample_weight = weight_factor if y[i] == 1 else 1.0
            error = sample_weight * (y[i] - prediction)
            for j in range(n_features):
                weights[j + 1] += learning_rate * error * X[i][j]
            weights[0] += learning_rate * error

    return weights


# Оценка точности модели
def compute_accuracy(X, y, weights):
    correct_predictions = 0
    for i in range(len(X)):
        prediction = 1 if predict(X[i], weights) >= 0.5 else 0
        if prediction == y[i]:
            correct_predictions += 1
    return correct_predictions / len(X)


# Предсказание статуса для заданных пользователем признаков
def predict_status(input_features, weights):
    if len(input_features) != len(weights) - 1:
        raise ValueError(f"Ожидалось {len(weights) - 1} признаков, но получено {len(input_features)}.")
    z = weights[0] + sum(w * x for w, x in zip(weights[1:], input_features))
    # Возвращаем оригинальный статус: если вероятность для класса 1 (после преобразования, т.е. статус 2) >= 0.5,
    # возвращаем 2, иначе возвращаем 1.
    return 2 if sigmoid(z) >= 0.5 else 1


# Основная функция для обучения и оценки модели
if __name__ == "__main__":
    file_path = '../main_task/data/data_frame_normalized_trimmed_shuffled.csv'
    data = load_dataset(file_path)

    train_data, test_data = train_test_split(data)
    X_train = [row[:-1] for row in train_data]
    y_train = [row[-1] for row in train_data]
    X_test = [row[:-1] for row in test_data]
    y_test = [row[-1] for row in test_data]

    weights = train_logistic_regression(X_train, y_train, epochs=1000, learning_rate=0.01)

    accuracy = compute_accuracy(X_test, y_test, weights)
    print(f"Точность модели: {accuracy:.2f}")

    print("\nВведите значения признаков (через запятую, соответствующие первым 6 столбцам):")
    user_input = input("Формат: Hour(1-24),Category(1-4),Tyres(1-2),Pit Stops, Best Lap Time, Best Lap Kph: ")

    try:
        input_features = list(map(float, user_input.split(',')))
        result = predict_status(input_features, weights)
        print(f"Предсказанный статус: {result}")
    except ValueError as e:
        print(f"Ошибка: {e}")
