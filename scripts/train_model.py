"""
scripts/train_model.py
----------------------
Entraîne un modèle de classification sur le dataset Pima Indians Diabetes
et sauvegarde l'artefact dans artifacts/model.joblib.

Différence code / artefact :
  - Le CODE (ce fichier) contient la logique d'entraînement.
  - L'ARTEFACT (model.joblib) est le résultat concret : le modèle entraîné,
    prêt à l'emploi, sans avoir besoin de ré-exécuter le code d'entraînement.
    On peut le partager, le versionner, le déployer indépendamment du code.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Chargement des données ───────────────────────────────────────────────────
print("Chargement des données...")

# Dataset Pima Indians Diabetes (fallback synthétique si pas de réseau)
try:
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    df.columns = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "dpf", "age", "diabetes"
    ]
    print(f"  Dataset chargé : {df.shape[0]} lignes")
except Exception:
    print("  Réseau indisponible — génération d'un dataset synthétique")
    from sklearn.datasets import make_classification
    X_s, y_s = make_classification(n_samples=768, n_features=8, random_state=42)
    cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "dpf", "age"]
    df = pd.DataFrame(X_s, columns=cols)
    df["diabetes"] = y_s

# ── Features sélectionnées ───────────────────────────────────────────────────
# On choisit les 3 features les plus prédictives pour garder l'API simple
FEATURES = ["glucose", "bmi", "age"]
TARGET   = "diabetes"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Entraînement ─────────────────────────────────────────────────────────────
print("Entraînement du modèle...")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
pipeline.fit(X_train, y_train)

# ── Évaluation ───────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print("\nPerformances sur le jeu de test :")
print(classification_report(y_test, y_pred, target_names=["Non diabétique", "Diabétique"]))

# ── Sauvegarde de l'artefact ─────────────────────────────────────────────────
# On sauvegarde un dictionnaire complet : modèle + métadonnées
artifact = {
    "model":    pipeline,
    "features": FEATURES,
    "version":  "v1.0",
    "target":   TARGET,
    "classes":  ["Non diabétique", "Diabétique"],
}

os.makedirs("artifacts", exist_ok=True)
artifact_path = "artifacts/model.joblib"
joblib.dump(artifact, artifact_path)
print(f"Artefact sauvegardé : {artifact_path}")
print(f"  Features : {FEATURES}")
print(f"  Version  : v1.0")
