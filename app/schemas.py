"""
app/schemas.py
--------------
Schémas Pydantic pour la validation des entrées/sorties de l'API.

Pydantic garantit que :
  - les types sont corrects (ex. glucose doit être un float)
  - les valeurs sont dans des plages médicalement cohérentes
  - un message d'erreur clair est renvoyé si un champ est invalide
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class PredictRequest(BaseModel):
    """
    Données d'entrée pour la prédiction du risque de diabète.
    Les 3 features correspondent exactement à celles utilisées à l'entraînement.
    """
    glucose: float = Field(
        ...,
        ge=0, le=300,
        description="Glycémie à jeun (mg/dL) — entre 0 et 300",
        examples=[120.0]
    )
    bmi: float = Field(
        ...,
        ge=0, le=100,
        description="Indice de masse corporelle (kg/m²) — entre 0 et 100",
        examples=[28.5]
    )
    age: int = Field(
        ...,
        ge=1, le=120,
        description="Âge du patient (années) — entre 1 et 120",
        examples=[45]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"glucose": 120.0, "bmi": 28.5, "age": 45}
            ]
        }
    }


class PredictResponse(BaseModel):
    """
    Réponse de l'API après prédiction.
    """
    prediction: int = Field(
        ...,
        description="Classe prédite : 0 = Non diabétique, 1 = Diabétique"
    )
    prediction_label: str = Field(
        ...,
        description="Libellé lisible de la prédiction"
    )
    probability_diabetic: float = Field(
        ...,
        description="Probabilité d'être diabétique (entre 0 et 1)"
    )
    model_version: str = Field(
        ...,
        description="Version du modèle utilisé pour la prédiction"
    )


class HealthResponse(BaseModel):
    """
    Réponse de l'endpoint /health.
    """
    status: str = Field(..., description="Statut de l'API : 'ok' si tout fonctionne")
    version: str = Field(..., description="Version du modèle chargé")
    features: list[str] = Field(..., description="Features attendues par le modèle")
