"""
app/api.py
----------
API FastAPI exposant deux endpoints :
  - GET  /health  → vérifie que l'API et le modèle sont opérationnels
  - POST /predict → retourne une prédiction à partir des features patient

Pourquoi séparer API (backend) et UI (frontend) ?
  - Séparation des responsabilités : l'API gère la logique ML,
    l'UI gère l'affichage. Chacune peut évoluer indépendamment.
  - L'API peut être consommée par plusieurs clients (Streamlit, mobile,
    autre service) sans duplication de code ML.
  - On peut scaler l'API et l'UI indépendamment selon la charge.

Pourquoi respecter l'ordre des features au moment du predict ?
  - sklearn attend un tableau numpy dans le même ordre que lors de
    l'entraînement. Si l'ordre change, les colonnes sont mal associées
    aux poids appris → prédictions incorrectes (sans erreur explicite !).
  - C'est pourquoi on stocke la liste `features` dans l'artefact et
    on reconstruit le tableau dans cet ordre exact.
"""

import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictRequest, PredictResponse, HealthResponse
from app.model import load_artifact

# ── État global partagé entre les requêtes ───────────────────────────────────
# L'artefact est chargé UNE SEULE FOIS au démarrage (pattern singleton)
_artifact: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement du modèle au démarrage, libération à l'arrêt."""
    global _artifact
    print("[API] Démarrage — chargement de l'artefact...")
    _artifact = load_artifact()
    print("[API] Modèle prêt ✓")
    yield
    print("[API] Arrêt — libération des ressources.")
    _artifact.clear()


# ── Création de l'application ────────────────────────────────────────────────
app = FastAPI(
    title="ML Serving API — Prédiction Diabète",
    description=(
        "API de prédiction du risque de diabète basée sur une "
        "Régression Logistique entraînée sur le dataset Pima Indians."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Autoriser les appels depuis Streamlit (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoint /health ─────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérification de l'état de l'API",
    tags=["Monitoring"],
)
def health() -> HealthResponse:
    """
    Retourne le statut de l'API, la version du modèle et les features attendues.
    Utile pour vérifier que le service est opérationnel avant d'envoyer des requêtes.
    """
    if not _artifact:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return HealthResponse(
        status="ok",
        version=_artifact["version"],
        features=_artifact["features"],
    )


# ── Endpoint /predict ────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Prédire le risque de diabète",
    tags=["Prédiction"],
)
def predict(payload: PredictRequest) -> PredictResponse:
    """
    Reçoit les données cliniques d'un patient et retourne :
    - la classe prédite (0 ou 1),
    - le libellé humain,
    - la probabilité d'être diabétique,
    - la version du modèle utilisé.

    Les features sont ordonnées selon la liste stockée dans l'artefact
    pour garantir la cohérence avec l'entraînement.
    """
    if not _artifact:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    model    = _artifact["model"]
    features = _artifact["features"]   # ordre exact de l'entraînement
    classes  = _artifact["classes"]
    version  = _artifact["version"]

    # Construire le vecteur dans le bon ordre de features
    # payload est un objet Pydantic → on accède aux champs par nom
    input_values = [getattr(payload, feat) for feat in features]
    X = np.array(input_values).reshape(1, -1)

    # Prédiction
    pred       = int(model.predict(X)[0])
    proba      = float(model.predict_proba(X)[0][1])   # P(diabétique)
    label      = classes[pred]

    return PredictResponse(
        prediction=pred,
        prediction_label=label,
        probability_diabetic=round(proba, 4),
        model_version=version,
    )
