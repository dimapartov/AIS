import pandas as pd
from sklearn.model_selection import train_test_split

csv_file_path = 'data/DataFrame.csv'
data = pd.read_csv(csv_file_path, encoding='ISO-8859-1', usecols = ['Category', 'Tyres', 'Pit Stops', 'Best Lap Kph', 'Best Lap Time'])
X = data.drop('Best Lap Time', axis = 1)
Y = data['Best Lap Time']
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

DataFrame_train = pd.DataFrame(X_train)
DataFrame_train['Best Lap Time'] = Y_train.values

DataFrame_valid = pd.DataFrame(X_val)
DataFrame_valid['Best Lap Time'] = Y_val.values

DataFrame_test = pd.DataFrame(X_test)
DataFrame_test['Best Lap Time'] = Y_test.values

DataFrame_train.to_csv('data/train.csv', index=False)
DataFrame_valid.to_csv('data/validation.csv', index=False)
DataFrame_test.to_csv('data/test.csv', index=False)
