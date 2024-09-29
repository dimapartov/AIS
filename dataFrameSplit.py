import pandas as pd
from sklearn.model_selection import train_test_split

csv_file_path = 'data/DataFrame.csv'

data = pd.read_csv(csv_file_path, encoding='ISO-8859-1', usecols=['Hour', 'Category', 'Tyres', 'Pit Stops', 'Best Lap Kph', 'Best Lap Time', 'Status'])

X = data.drop(['Best Lap Time', 'Status'], axis=1)
Y = data[['Best Lap Time', 'Status']]

X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Further splitting train+val into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

DataFrame_train = X_train.copy()
DataFrame_train[['Best Lap Time', 'Status']] = Y_train.values

DataFrame_valid = X_val.copy()
DataFrame_valid[['Best Lap Time', 'Status']] = Y_val.values

DataFrame_test = X_test.copy()
DataFrame_test[['Best Lap Time', 'Status']] = Y_test.values

DataFrame_train.to_csv('data/train.csv', index=False)
DataFrame_valid.to_csv('data/validation.csv', index=False)
DataFrame_test.to_csv('data/test.csv', index=False)
