import joblib
from sklearn.ensemble import RandomForestClassifier


def train_model(X, y, model_path="model.pkl"):
    """Entraîne un modèle Random Forest et le sauvegarde."""

    # Création du modèle avec une indentation correcte
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=1
    )

    # Entraînement du modèle
    model.fit(X, y)

    # Sauvegarde du modèle
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé sous {model_path}")

    return model


if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data

    # Chargement des données d'entraînement
    train_data, _ = load_data("train.csv", "test.csv")
    X = preprocess_data(train_data)
    y = train_data["Survived"]

    # Entraînement du modèle
    train_model(X, y)
