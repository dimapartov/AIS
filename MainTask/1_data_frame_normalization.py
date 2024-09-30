import pandas as pd


csv_file_path = '../data/data_frame_original.csv'
dataFrame = pd.read_csv(csv_file_path)

print("Unique Categories:", dataFrame['Category'].unique())
print("Unique Tyres:", dataFrame['Tyres'].unique())
print("Unique Status:", dataFrame['Status'].unique())

category_mapping = {category: idx + 1 for idx, category in enumerate(dataFrame['Category'].unique())}
dataFrame['Category'] = dataFrame['Category'].map(category_mapping)

tyres_mapping = {tyres: idx + 1 for idx, tyres in enumerate(dataFrame['Tyres'].unique())}
dataFrame['Tyres'] = dataFrame['Tyres'].map(tyres_mapping)

status_mapping = {'Running': 1, 'Retired': 2}
dataFrame['Status'] = dataFrame['Status'].map(status_mapping)

dataFrame.to_csv('../data/data_frame_normalized.csv', index=False)
