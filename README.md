# 🩺 ML Serving TP — Déploiement d'un modèle ML avec FastAPI & Streamlit

API de prédiction du risque de diabète + Interface Web + bonnes pratiques MLOps.

---

## Structure du projet

```
ml-serving-tp/
├── app/
│   ├── __init__.py
│   ├── api.py          # API FastAPI (/health, /predict)
│   ├── schemas.py      # Schémas Pydantic (validation entrées/sorties)
│   └── model.py        # Chargement de l'artefact (une seule fois)
├── ui/
│   └── streamlit_app.py  # Interface utilisateur Streamlit
├── artifacts/
│   └── model.joblib    # Modèle entraîné + métadonnées (généré par train_model.py)
├── scripts/
│   └── train_model.py  # Script d'entraînement et sauvegarde du modèle
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### Étape 1 — Entraîner le modèle (générer l'artefact)

```bash
python scripts/train_model.py
```

Cela crée `artifacts/model.joblib`.

### Étape 2 — Lancer l'API FastAPI

```bash
uvicorn app.api:app --reload
```

L'API est disponible sur : http://127.0.0.1:8000

### Étape 3 — Tester l'API

**Via Swagger UI (interface graphique) :**
```
http://127.0.0.1:8000/docs
```

**Via curl — endpoint /health :**
```bash
curl http://127.0.0.1:8000/health
```

**Via curl — endpoint /predict :**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"glucose": 150.0, "bmi": 32.5, "age": 50}'
```

### Étape 4 — Lancer l'interface Streamlit

Dans un second terminal :
```bash
streamlit run ui/streamlit_app.py
```

L'UI est disponible sur : http://localhost:8501

---

## Endpoints API

| Méthode | Endpoint  | Description                              |
|---------|-----------|------------------------------------------|
| GET     | /health   | Statut API + version modèle + features   |
| POST    | /predict  | Prédiction à partir des données patient  |
| GET     | /docs     | Documentation Swagger interactive        |

### Exemple de requête `/predict`

```json
{
  "glucose": 120.0,
  "bmi": 28.5,
  "age": 45
}
```

### Exemple de réponse

```json
{
  "prediction": 0,
  "prediction_label": "Non diabétique",
  "probability_diabetic": 0.2341,
  "model_version": "v1.0"
}
```

---

## Questions conceptuelles — Réponses

**Pourquoi séparer API (backend) et UI (frontend) ?**  
La séparation permet à chaque composant d'évoluer indépendamment.
L'API peut être consommée par plusieurs clients (Streamlit, mobile, autre service)
sans duplication du code ML. On peut aussi scaler chaque partie séparément.

**Différence entre code et artefact ?**  
Le code (`train_model.py`) contient la logique d'entraînement.
L'artefact (`model.joblib`) est le résultat concret : le modèle entraîné, prêt à l'emploi,
versionné et déployable sans ré-exécuter l'entraînement.

**Pourquoi ne pas recharger le modèle à chaque requête ?**  
Charger un fichier joblib prend du temps (I/O disque + désérialisation).
En le chargeant une seule fois au démarrage, toutes les requêtes partagent
le même objet en mémoire : réponse instantanée, sans overhead.

**Pourquoi respecter l'ordre des features au moment du predict ?**  
sklearn attend un tableau numpy dans le même ordre qu'à l'entraînement.
Si l'ordre change, les colonnes sont mal associées aux poids appris →
prédictions silencieusement incorrectes. L'artefact stocke la liste `features`
pour garantir cet ordre à chaque prédiction.

---

## Modèle

- **Algorithme :** Logistic Regression (Pipeline avec StandardScaler)
- **Dataset :** Pima Indians Diabetes (768 patientes)
- **Features :** glucose, bmi, age
- **Cible :** diabète (0 = Non diabétique, 1 = Diabétique)
- **Accuracy test :** ~82%
