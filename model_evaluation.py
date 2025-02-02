import joblib
import joblib
import pandas as pd


def predict(model_path, X_test, output_path="submission.csv"):
    """Charge le modèle, effectue des prédictions et enregistre les résultats."""
    
    # Chargement du modèle
    model=joblib.load(model_path)
    
    # Prédictions
    predictions=model.predict(X_test)
    
    # Création du DataFrame de résultats
    output=pd.DataFrame({"PassengerId":X_test.index,"Survived":predictions})
    
    # Sauvegarde des résultats
    output.to_csv(output_path,index=False)
    print(f"Prédictions sauvegardées sous {output_path}")


if __name__=="__main__":
    from data_preprocessing import load_data,preprocess_data

    # Chargement des données de test
    _,test_data=load_data("train.csv","test.csv")
    X_test=preprocess_data(test_data)
    
    # Effectuer les prédictions
    predict("model.pkl",X_test)

