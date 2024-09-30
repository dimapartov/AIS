import pandas as pd

csv_file_path = '../data/data_frame_normalized.csv'
dataFrame = pd.read_csv(csv_file_path, encoding='ISO-8859-1', usecols=['Hour', 'Category', 'Tyres', 'Pit Stops', 'Best Lap Kph', 'Best Lap Time', 'Status'])
dataFrame.to_csv('../data/data_frame_normalized_trimmed.csv', index=False)