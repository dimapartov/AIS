import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Загрузка обработанного датасета
data = pd.read_csv('../data/data_frame_normalized_trimmed_shuffled.csv')

# Разделение на признаки и целевую переменную
X = data.iloc[:, :-1]  # Все колонки, кроме последней
y = data.iloc[:, -1]  # Последняя колонка

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка эффективности модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# Функция для предсказания статуса
def predict_status(input_features):
    # Преобразование входных данных в DataFrame
    input_df = pd.DataFrame([input_features], columns=data.columns[:-1])

    # Нормализация введенных данных
    input_scaled = scaler.transform(input_df)

    # Получение прогноза
    prediction = model.predict(input_scaled)
    return prediction[0]  # Возвращаем только одно значение прогноза


# Пример использования функции для получения прогноза
if __name__ == "__main__":
    print("\nВведите значения для признаков (все колонки, кроме последней) через запятую:")
    user_input = input("Формат: 'value1,value2,value3,...': ")

    # Преобразование пользовательского ввода в список чисел
    features = list(map(float, user_input.split(',')))

    # Проверка, соответствует ли количество введенных значений количеству признаков
    if len(features) != len(data.columns) - 1:
        print(f"Ошибка: должно быть {len(data.columns) - 1} значений.")
    else:
        result = predict_status(features)
        print(f'Предсказанный статус: {result}')
