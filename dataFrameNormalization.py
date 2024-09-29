import pandas as pd

csv_file_path = 'data/DataFrame.csv'
# Загрузка данных из CSV файла
dataFrame = pd.read_csv(csv_file_path)  # Укажите путь к вашему файлу

# Выводим уникальные значения для проверки
print("Unique Categories:", dataFrame['Category'].unique())
print("Unique Tyres:", dataFrame['Tyres'].unique())
print("Unique Status:", dataFrame['Status'].unique())

# Заменяем категорию на числа
category_mapping = {category: idx + 1 for idx, category in enumerate(dataFrame['Category'].unique())}
dataFrame['Category'] = dataFrame['Category'].map(category_mapping)

# Заменяем шины на числа
tyres_mapping = {tyres: idx + 1 for idx, tyres in enumerate(dataFrame['Tyres'].unique())}
dataFrame['Tyres'] = dataFrame['Tyres'].map(tyres_mapping)

# Заменяем статус на числа
status_mapping = {'Running': 1, 'Retired': 2}
dataFrame['Status'] = dataFrame['Status'].map(status_mapping)

# Сохраняем обработанный датасет обратно в CSV (если необходимо)
dataFrame.to_csv('data/DataFrameNormalized.csv', index=False)
