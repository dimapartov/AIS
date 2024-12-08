import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath):
    """Load the dataset and split it into features and target."""
    # Load the dataset
    data = pd.read_csv(filepath)

    # Split features and target
    X = data.iloc[:, :-1]  # All columns except the last
    y = data.iloc[:, -1]  # The last column (target)

    return X, y


def split_and_normalize_data(X, y, test_size=0.2, random_state=42):
    """Split the data into train and test sets and normalize it."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train):
    """Train a RandomForestClassifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))


def predict_status(model, scaler, input_features, feature_names):
    """Predict the status given input features."""
    # Create a DataFrame for input features
    input_df = pd.DataFrame([input_features], columns=feature_names)

    # Normalize input features
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    return prediction[0]


if __name__ == "__main__":
    # Filepath to the dataset
    filepath = '../data/data_frame_normalized_trimmed_shuffled.csv'

    # Load and preprocess the data
    X, y = load_and_preprocess_data(filepath)

    # Split and normalize the data
    X_train, X_test, y_train, y_test, scaler = split_and_normalize_data(X, y)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Interactive prediction
    print("\nEnter feature values (comma-separated, corresponding to the first 6 columns):")
    user_input = input("Format: 'value1,value2,value3,...': ")

    try:
        # Convert input into a list of floats
        input_features = list(map(float, user_input.split(',')))

        # Check if the number of input features matches
        if len(input_features) != len(X.columns):
            print(f"Error: Expected {len(X.columns)} features, but got {len(input_features)}.")
        else:
            # Predict the status
            result = predict_status(model, scaler, input_features, X.columns)
            print(f'Predicted Status: {result}')
    except ValueError as e:
        print(f"Error: {e}")
