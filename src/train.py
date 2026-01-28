from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from data import load_data, split_data


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def fit_model(X_train, y_train):
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


def save_model(clf, scaler):
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "wine_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    clf = fit_model(X_train_scaled, y_train)
    accuracy = clf.score(X_test_scaled, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    save_model(clf, scaler)
    print("Model and scaler saved successfully!")