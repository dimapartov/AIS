import math

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

# Загрузить веса модели из файла
def load_weights(filepath):
    with open(filepath, 'r') as f:
        return list(map(float, f.read().strip().split(',')))

# Предсказать статус на основе входных признаков и весов модели
def predict_status(input_features, weights):
    if len(input_features) != len(weights) - 1:
        raise ValueError(f"Ожидалось {len(weights) - 1} признаков, но получено {len(input_features)}.")
    z = weights[0] + sum(w * x for w, x in zip(weights[1:], input_features))
    # Если вероятность для класса 1 (после преобразования, т.е. статус 2) >= 0.5, возвращаем 2, иначе 1
    return 2 if sigmoid(z) >= 0.5 else 1

if __name__ == "__main__":
    # Загрузить веса обученной модели
    weights = load_weights("model_weights.txt")
    print("Модель загружена.")

    # Запрос значений признаков у пользователя
    print("Введите значения признаков (через запятую, соответствующие первым 6 столбцам):")
    user_input = input("Формат: Hour(1-24),Category(1-4),Tyres(1-2),Pit Stops, Best Lap Time, Best Lap Kph: ")
    try:
        input_features = list(map(float, user_input.split(',')))
        result = predict_status(input_features, weights)
        print(f"Предсказанный статус: {result}")
    except ValueError as e:
        print(f"Ошибка: {e}")
