"""
ui/streamlit_app.py
-------------------
Interface utilisateur Streamlit connectée à l'API FastAPI.

Fonctionnalités :
  - Champs de saisie pour les données cliniques du patient
  - Appel à POST /predict avec requests
  - Affichage du résultat avec jauge de probabilité
  - Gestion robuste des erreurs (API indisponible, timeout, mauvais format)
"""

import streamlit as st
import requests

# ── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction Diabète — ML Serving",
    page_icon="🩺",
    layout="centered",
)

API_URL = "http://127.0.0.1:8000"
TIMEOUT = 5  # secondes avant timeout

# ── Titre & Description ──────────────────────────────────────────────────────
st.title("🩺 Prédiction du risque de diabète")
st.markdown(
    """
    Cette application utilise un modèle de **Régression Logistique** 
    entraîné sur le dataset *Pima Indians Diabetes*.  
    Renseignez les données cliniques du patient, puis cliquez sur **Prédire**.
    """
)

# ── Vérification de l'état de l'API ─────────────────────────────────────────
st.sidebar.header("📡 État de l'API")
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        st.sidebar.success("✅ API connectée")
        st.sidebar.write(f"**Version modèle :** `{health_data['version']}`")
        st.sidebar.write(f"**Features :** `{', '.join(health_data['features'])}`")
    else:
        st.sidebar.error(f"⚠️ API répond avec le code {health_resp.status_code}")
except requests.exceptions.ConnectionError:
    st.sidebar.error("❌ API inaccessible — lancez d'abord :\n`uvicorn app.api:app --reload`")
except requests.exceptions.Timeout:
    st.sidebar.warning("⏱️ API trop lente (timeout)")

st.sidebar.markdown("---")
st.sidebar.markdown("📖 [Documentation Swagger](http://127.0.0.1:8000/docs)")

# ── Formulaire de saisie ─────────────────────────────────────────────────────
st.subheader("📋 Données cliniques du patient")

col1, col2, col3 = st.columns(3)

with col1:
    glucose = st.number_input(
        "🩸 Glycémie (mg/dL)",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0,
        help="Glycémie à jeun — valeur normale : 70–100 mg/dL",
    )

with col2:
    bmi = st.number_input(
        "⚖️ IMC (kg/m²)",
        min_value=0.0,
        max_value=100.0,
        value=28.5,
        step=0.1,
        help="Indice de Masse Corporelle — valeur normale : 18.5–25",
    )

with col3:
    age = st.number_input(
        "🎂 Âge (années)",
        min_value=1,
        max_value=120,
        value=45,
        step=1,
        help="Âge du patient en années",
    )

# ── Bouton de prédiction ─────────────────────────────────────────────────────
st.markdown("---")

if st.button("🔍 Prédire le risque de diabète", use_container_width=True, type="primary"):

    payload = {"glucose": glucose, "bmi": bmi, "age": age}

    # ── Appel à l'API avec gestion d'erreurs ─────────────────────────────────
    try:
        with st.spinner("Analyse en cours..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=TIMEOUT,
            )

        # ── Traitement de la réponse ──────────────────────────────────────────
        if response.status_code == 200:
            result = response.json()
            pred      = result["prediction"]
            label     = result["prediction_label"]
            proba     = result["probability_diabetic"]
            version   = result["model_version"]

            st.markdown("---")
            st.subheader("📊 Résultat de la prédiction")

            # Affichage selon le résultat
            if pred == 1:
                st.error(f"🔴 **{label}** — risque détecté")
            else:
                st.success(f"🟢 **{label}** — pas de risque détecté")

            # Jauge de probabilité
            st.metric(
                label="Probabilité d'être diabétique",
                value=f"{proba * 100:.1f} %",
                delta=f"{'⚠️ Élevé' if proba > 0.5 else '✅ Faible'}",
            )
            st.progress(proba)

            # Détails
            with st.expander("📦 Détails techniques"):
                st.json({
                    "payload_envoyé": payload,
                    "réponse_API": result,
                })
                st.caption(f"Modèle version : `{version}` | Seuil de décision : θ = 0.5")

        elif response.status_code == 422:
            # Erreur de validation Pydantic
            st.error("❌ Données invalides — vérifiez les valeurs saisies.")
            with st.expander("Détails de l'erreur"):
                st.json(response.json())

        elif response.status_code == 503:
            st.error("⚠️ Le modèle n'est pas chargé côté serveur. Relancez l'API.")

        else:
            st.error(f"❌ Erreur inattendue de l'API (code {response.status_code})")
            with st.expander("Détails"):
                st.text(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ **Impossible de contacter l'API.**\n\n"
            "Vérifiez que l'API est bien lancée :\n"
            "```\nuvicorn app.api:app --reload\n```"
        )

    except requests.exceptions.Timeout:
        st.warning(
            f"⏱️ **L'API n'a pas répondu en {TIMEOUT} secondes.**\n\n"
            "Elle est peut-être surchargée ou en cours de démarrage."
        )

    except Exception as e:
        st.error(f"💥 Erreur inattendue : {e}")

# ── Pied de page ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "TP Déploiement ML — FastAPI & Streamlit | "
    "Modèle : Logistic Regression | Dataset : Pima Indians Diabetes"
)
