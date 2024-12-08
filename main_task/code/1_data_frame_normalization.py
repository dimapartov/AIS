import pandas as pd

# Путь к CSV файлу
csv_file_path = '../data/data_frame_original.csv'
dataFrame = pd.read_csv(csv_file_path)

# Заменяем точку на двоеточие в колонке 'Best Lap Time'
dataFrame['Best Lap Time'] = dataFrame['Best Lap Time'].str.replace('.', ':', regex=False)

# Функция для преобразования времени в секунды
def convert_time_to_seconds(time_str):
    if pd.isna(time_str) or time_str == '':
        return None  # Возвращаем None для пустых значений
    hours, minutes, seconds = map(float, time_str.split(':'))  # Разделяем на часы, минуты и секунды
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)  # Переводим в секунды
    return total_seconds

# Нормализация времени лучшего круга
dataFrame['Best Lap Time'] = dataFrame['Best Lap Time'].apply(convert_time_to_seconds)

print("Unique Categories:", dataFrame['Category'].unique())
print("Unique Tyres:", dataFrame['Tyres'].unique())
print("Unique Status:", dataFrame['Status'].unique())

# Нормализация категорий
category_mapping = {category: idx + 1 for idx, category in enumerate(dataFrame['Category'].unique())}
dataFrame['Category'] = dataFrame['Category'].map(category_mapping)

# Нормализация типов шин
tyres_mapping = {tyres: idx + 1 for idx, tyres in enumerate(dataFrame['Tyres'].unique())}
dataFrame['Tyres'] = dataFrame['Tyres'].map(tyres_mapping)

# Нормализация статуса
status_mapping = {'Running': 1, 'Retired': 2}
dataFrame['Status'] = dataFrame['Status'].map(status_mapping)

# Сохранение нормализованных данных в новый CSV файл
dataFrame.to_csv('../data/data_frame_normalized.csv', index=False)
