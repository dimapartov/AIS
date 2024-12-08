import csv
import math
import random


# Load the dataset
def load_dataset(filepath):
    data = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            data.append([float(value) for value in row])
    return data


# Split dataset into training and testing sets
def train_test_split(data, test_size=0.2):
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]


# Sigmoid function for logistic regression
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Predict using logistic regression weights
def predict(features, weights):
    z = sum(w * x for w, x in zip(weights, [1] + features))  # Include bias term
    return sigmoid(z)


# Train logistic regression using gradient descent
def train_logistic_regression(X, y, epochs=1000, learning_rate=0.01):
    n_features = len(X[0])
    weights = [0.0] * (n_features + 1)  # +1 for bias term

    for _ in range(epochs):
        for i in range(len(X)):
            prediction = predict(X[i], weights)
            error = y[i] - prediction
            # Update weights and bias
            for j in range(n_features):
                weights[j + 1] += learning_rate * error * X[i][j]
            weights[0] += learning_rate * error  # Bias update

    return weights


# Evaluate model accuracy
def compute_accuracy(X, y, weights):
    correct_predictions = 0
    for i in range(len(X)):
        prediction = predict(X[i], weights) >= 0.5  # Threshold 0.5 for binary classification
        if prediction == y[i]:
            correct_predictions += 1
    return correct_predictions / len(X)


# Predict status for user-provided features
def predict_status(input_features, weights):
    if len(input_features) != len(weights) - 1:
        raise ValueError(f"Expected {len(weights) - 1} features, but got {len(input_features)}.")

    # Compute prediction using sigmoid function
    z = weights[0] + sum(w * x for w, x in zip(weights[1:], input_features))
    return 1 if sigmoid(z) >= 0.5 else 0


# Main function to train and evaluate the model
if __name__ == "__main__":
    # Load and process dataset
    file_path = '../data/data_frame_normalized_trimmed_shuffled.csv'
    data = load_dataset(file_path)

    # Split features and target
    features = [row[:-1] for row in data]
    labels = [int(row[-1]) for row in data]  # Convert to integers (0 or 1)

    # Split into training and testing sets
    train_data, test_data = train_test_split(data)
    X_train = [row[:-1] for row in train_data]
    y_train = [int(row[-1]) for row in train_data]
    X_test = [row[:-1] for row in test_data]
    y_test = [int(row[-1]) for row in test_data]

    # Train model
    weights = train_logistic_regression(X_train, y_train, epochs=1000, learning_rate=0.01)

    # Evaluate model
    accuracy = compute_accuracy(X_test, y_test, weights)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Interactive prediction
    print("\nEnter feature values (comma-separated, corresponding to the first 6 columns):")
    user_input = input("Format: 'value1,value2,value3,...': ")

    try:
        # Convert input into a list of floats
        features = list(map(float, user_input.split(',')))

        # Make prediction using trained weights
        result = predict_status(features, weights)
        print(f"Predicted Status: {result}")
    except ValueError as e:
        print(f"Error: {e}")
