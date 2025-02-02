# tests/test_data_preprocessing.py
import pandas as pd
import joblib
from data_preprocessing import preprocess_data
from model_training import train_model
from model_evaluation import predict


def test_preprocess_data():
    """Vérifie que le prétraitement des données fonctionne correctement."""
    data = pd.DataFrame({
        "Pclass": [1, 3],
        "Sex": ["male", "female"],
        "SibSp": [0, 1],
        "Parch": [0, 2]
    })

    processed_data = preprocess_data(data)

    assert "Sex_male" in processed_data.columns
    assert "Sex_female" in processed_data.columns
    assert processed_data.shape[1] == 4


# tests/test_model_training.py
def test_train_model():
    """Vérifie que l'entraînement du modèle fonctionne sans erreur."""
    X = pd.DataFrame({
        "Pclass": [1, 3],
        "Sex_male": [1, 0],
        "Sex_female": [0, 1],
        "SibSp": [0, 1]
    })
    y = pd.Series([1, 0])

    model = train_model(X, y)

    assert model is not None
    assert hasattr(model, "predict")


# tests/test_model_evaluation.py
def test_predict(tmp_path):
    """Vérifie que la prédiction s'exécute correctement
    et génère un fichier CSV."""
    model_path = tmp_path / "test_model.pkl"
    output_path = tmp_path / "test_predictions.csv"

    X_test = pd.DataFrame({
        "Pclass": [1, 3],
        "Sex_male": [1, 0],
        "Sex_female": [0, 1],
        "SibSp": [0, 1]
    })

    model = joblib.load("model.pkl")
    joblib.dump(model, model_path)

    predict(model_path, X_test, output_path)

    assert output_path.exists()
    df_output = pd.read_csv(output_path)
    assert "Survived" in df_output.columns
