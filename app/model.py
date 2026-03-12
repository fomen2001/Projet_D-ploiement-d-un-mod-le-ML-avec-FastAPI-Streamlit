"""
app/model.py
------------
Chargement de l'artefact modèle — une seule fois au démarrage de l'API.

Pourquoi ne pas recharger le modèle à chaque requête ?
  - Charger un fichier joblib prend du temps (I/O disque + désérialisation).
  - Sous charge (ex. 100 req/s), recharger à chaque fois multiplierait
    la latence et consommerait inutilement le CPU/disque.
  - En chargeant une seule fois en mémoire au démarrage, toutes les requêtes
    partagent le même objet : réponse instantanée, sans overhead.
"""

import joblib
import os

# Chemin vers l'artefact (relatif à la racine du projet)
ARTIFACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "model.joblib"
)

def load_artifact() -> dict:
    """
    Charge et retourne l'artefact depuis le fichier .joblib.
    À appeler une seule fois au démarrage (via lifespan FastAPI).
    
    Returns:
        dict avec les clés : model, features, version, target, classes
    
    Raises:
        FileNotFoundError si l'artefact est absent.
    """
    path = os.path.abspath(ARTIFACT_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artefact introuvable : {path}\n"
            "Lancez d'abord : python scripts/train_model.py"
        )
    artifact = joblib.load(path)
    print(f"[model.py] Artefact chargé — version {artifact.get('version', '?')}, "
          f"features : {artifact.get('features', [])}")
    return artifact
