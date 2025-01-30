---
title: "Rapport du Projet Titanic Survival Prediction"
author: "Équipe BUT 3 Sciences des Données"
date: "2025-01-30"
output: github_document
---

# 1. Introduction

Dans le cadre de notre formation en **BUT 3 Sciences des Données**, nous avons réalisé un projet d’ingénierie logicielle appliqué à un problème de **Machine Learning**. Ce projet s’inscrit dans la mise en pratique des compétences acquises en **modularisation du code, documentation, tests unitaires et intégration continue**.

Nous avons travaillé sur le **Titanic Survival Prediction**, un projet classique en **Data Science**, consistant à prédire la survie des passagers du Titanic à partir de diverses caractéristiques (**classe, sexe, nombre de proches à bord, etc.**). Notre objectif était non seulement d’améliorer la qualité du modèle, mais aussi de **refactoriser le code en suivant les bonnes pratiques d’ingénierie logicielle**.

---

# 2. Objectifs du Projet

Ce projet nous a permis d’atteindre plusieurs objectifs pédagogiques et techniques :

- **Structurer un projet de Machine Learning** en modules indépendants et réutilisables.
- **Appliquer les bonnes pratiques de développement** (conformité à PEP 8, modularisation, docstrings, etc.).
- **Mettre en place des tests unitaires** pour garantir la fiabilité du code.
- **Automatiser les processus avec une pipeline CI/CD** (tests automatiques via GitHub Actions).
- **Collaborer efficacement en utilisant Git et GitHub**.

L’approche adoptée repose sur la séparation des différentes tâches en modules bien définis :

1. **Prétraitement des données** (nettoyage, transformation des variables catégorielles, encodage).
2. **Entraînement du modèle** (Random Forest avec ajustement des hyperparamètres).
3. **Évaluation du modèle** (prédictions sur les données test et soumission du fichier de résultats).

---

# 3. Instructions d’Installation et d’Utilisation

## 3.1 Prérequis

Avant de commencer, assurez-vous d’avoir :

- **Python 3.8+** installé sur votre machine.
- Un **environnement virtuel** pour isoler les dépendances (*optionnel mais recommandé*).
- Les **bibliothèques requises** listées dans `requirements.txt`.

---

## 3.2 Installation

###  ** Cloner le dépôt GitHub**
```bash
git clone https://github.com/votre-repo/titanic-prediction.git
cd titanic-prediction
```

###  ** Installer les dépendances**
Nous recommandons l'utilisation d'un **environnement virtuel** :

```bash
python -m venv venv
source venv/bin/activate  # Sur Mac/Linux
venv\Scripts\activate  # Sur Windows
pip install -r requirements.txt
```

###  **Télécharger les données**
Les fichiers `train.csv` et `test.csv` doivent être placés à la racine du projet.

**Lien vers le dataset :** [🔗 Titanic Dataset - Kaggle](https://www.kaggle.com/competitions/titanic/data)

---

## 3.3 Utilisation

###  **Exécuter le prétraitement des données**
```bash
python src/data_preprocessing.py
```

###  **Entraîner le modèle**
```bash
python src/model_training.py
```

###  **Faire des prédictions et générer le fichier de soumission**
```bash
python src/model_evaluation.py
```

###  **Effectuer les tests unitaires**
```bash
pytest tests/
```

---

# 4. Contributions des Membres de l’Équipe

Ce projet a été réalisé en équipe, avec une répartition des rôles selon les **compétences et préférences** de chacun :

- **[Nom 1]** : Responsable du **prétraitement des données** et de leur structuration.
- **[Nom 2]** : Implémentation et **entraînement du modèle Random Forest**.
- **[Nom 3]** : Évaluation du modèle, génération des **prédictions et soumission des résultats**.
- **[Nom 4]** : Mise en place de la **CI/CD avec GitHub Actions** et rédaction de la documentation.

Chaque membre a également contribué à la **relecture et à l’amélioration du code** en respectant les bonnes pratiques d’ingénierie logicielle.

---

# 5. Conclusion

Ce projet nous a permis d’acquérir une **expérience concrète en ingénierie logicielle appliquée à la Data Science**. Nous avons appris à **structurer un projet ML de manière rigoureuse**, à **automatiser des tâches essentielles** et à **travailler efficacement en équipe avec Git/GitHub**.

Grâce à l’implémentation de **tests unitaires et d’un pipeline CI/CD**, notre projet est désormais **plus robuste et plus fiable**, garantissant une **meilleure reproductibilité des résultats**.

 **Ce projet nous a préparés aux exigences du monde professionnel en combinant Data Science et Ingénierie Logicielle.** 