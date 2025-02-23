import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath):
    """Загрузить набор данных и разделить его на признаки и целевую переменную."""
    # Загрузить набор данных
    data = pd.read_csv(filepath)

    # Разделить на признаки и целевую переменную
    X = data.iloc[:, :-1]  # Все столбцы, кроме последнего
    y = data.iloc[:, -1]  # Последний столбец (целевая переменная)

    return X, y


def split_and_normalize_data(X, y, test_size=0.2, random_state=42):
    """Разделить данные на обучающую и тестовую выборки и нормализовать их."""
    # Разделить данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Нормализовать данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train):
    """Обучить RandomForestClassifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Оценить модель и вывести метрики."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Точность модели с использованием библиотек: {accuracy:.2f}')


def predict_status(model, scaler, input_features, feature_names):
    """Предсказать статус на основе входных признаков."""
    # Создать DataFrame для входных признаков
    input_df = pd.DataFrame([input_features], columns=feature_names)

    # Нормализовать входные признаки
    input_scaled = scaler.transform(input_df)

    # Сделать предсказание
    prediction = model.predict(input_scaled)
    return prediction[0]


if __name__ == "__main__":
    # Путь к набору данных
    filepath = '../data/data_frame_normalized_trimmed_shuffled.csv'

    # Загрузить и предобработать данные
    X, y = load_and_preprocess_data(filepath)

    # Разделить и нормализовать данные
    X_train, X_test, y_train, y_test, scaler = split_and_normalize_data(X, y)

    # Обучить модель
    model = train_model(X_train, y_train)

    # Оценить модель
    evaluate_model(model, X_test, y_test)

    # Интерактивное предсказание
    print("\nВведите значения признаков (через запятую, соответствующие первым 6 столбцам):")
    user_input = input("Формат: Hour(1-24),Category(1-4),Tyres(1-2),Pit Stops, Best Lap Time, Best Lap Kph: ")

    try:
        # Преобразовать ввод в список чисел с плавающей точкой
        input_features = list(map(float, user_input.split(',')))

        # Проверить, соответствует ли количество входных признаков ожидаемому
        if len(input_features) != len(X.columns):
            print(f"Ошибка: ожидалось {len(X.columns)} признаков, но получено {len(input_features)}.")
        else:
            # Предсказать статус
            result = predict_status(model, scaler, input_features, X.columns)
            print(f'Предсказанный статус: {result}')
    except ValueError as e:
        print(f"Ошибка: {e}")
