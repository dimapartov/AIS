import csv
import random


def read_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]
    return header, data

def write_csv(file_path, header, data):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

def custom_shuffle(data):
    n = len(data)
    indices = list(range(n))

    for a in range(1000):
        for i in range(n):
            j = random.randint(0, n - 1)
            indices[i], indices[j] = indices[j], indices[i]

    shuffled_data = [data[i] for i in indices]
    return shuffled_data


input_file = '../data/data_frame_normalized_trimmed.csv'
output_file = '../data/data_frame_normalized_trimmed_shuffled.csv'

header, data = read_csv(input_file)

shuffled_data = custom_shuffle(data)

write_csv(output_file, header, shuffled_data)