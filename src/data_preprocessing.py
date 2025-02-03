import pandas as pd

def load_data(train_path, test_path):
    """Charge les fichiers CSV contenant les données Titanic."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def preprocess_data(data):
    """Effectue le prétraitement des données en
     encodant les variables catégorielles."""
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    return pd.get_dummies(data[features], drop_first=False)



if __name__ == "__main__":
    train_data, test_data = load_data("train.csv", "test.csv")
    X = preprocess_data(train_data)
    X_test = preprocess_data(test_data)
    print("Prétraitement terminé.")
