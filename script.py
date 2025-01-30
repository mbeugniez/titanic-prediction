pip install flake8 black isort pytest

# data_preprocessing.py
import pandas as pd


def load_data(train_path, test_path):
    """Charge les fichiers CSV contenant les données Titanic."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def preprocess_data(data):
    """Effectue le prétraitement des données : encodage des variables catégorielles."""
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    return pd.get_dummies(data[features])

if __name__ == "__main__":
    train_data, test_data = load_data("train.csv", "test.csv")
    X = preprocess_data(train_data)
    X_test = preprocess_data(test_data)
    print("Prétraitement terminé.")

import joblib
# model_training.py
from sklearn.ensemble import RandomForestClassifier


def train_model(X, y, model_path="model.pkl"):
    """Entraîne un modèle Random Forest et le sauvegarde."""
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé sous {model_path}")
    return model

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    
    train_data, _ = load_data("train.csv", "test.csv")
    X = preprocess_data(train_data)
    y = train_data["Survived"]
    train_model(X, y)

# model_evaluation.py
import joblib
import pandas as pd


def predict(model_path, X_test, output_path="submission.csv"):
    """Charge le modèle, effectue des prédictions et enregistre les résultats."""
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': X_test.index, 'Survived': predictions})
    output.to_csv(output_path, index=False)
    print(f"Prédictions sauvegardées sous {output_path}")

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    
    _, test_data = load_data("train.csv", "test.csv")
    X_test = preprocess_data(test_data)
    predict("model.pkl", X_test)

# utils.py
import os


def list_files(directory):
    """Affiche la liste des fichiers dans un répertoire donné."""
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            print(os.path.join(dirname, filename))

if __name__ == "__main__":
    list_files("./")