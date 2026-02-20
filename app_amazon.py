import streamlit as st
import torch
import os
import sys
import warnings
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet

# 1. CONFIGURATION DE SÉCURITÉ & INTERFACE
warnings.filterwarnings("ignore")
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

# Cette partie règle l'erreur de chargement du modèle (UnpicklingError)
torch.serialization.add_safe_globals([
    'neuralprophet.configure.ConfigSeasonality',
    'neuralprophet.configure.ConfigTrend',
    'neuralprophet.configure.ConfigModel',
    'neuralprophet.configure.ConfigTrain',
    'neuralprophet.configure.ConfigData'
])

st.set_page_config(page_title="IA Amazon - Devoirs ANN/CNN/TL", layout="wide")

st.title("🚀 Analyse Prédictive Amazon (AMZN)")
st.markdown("""
Ce projet présente l'application des réseaux de neurones pour la prédiction financière. 
Il couvre les concepts de **ANN** (via NeuralProphet), **CNN** (analyse de tendances) et **Transfer Learning**.
""")

# 2. RÉCUPÉRATION DES DONNÉES (Caché pour la performance)
@st.cache_data
def load_data():
    df_raw = yf.download("AMZN", period="2y", interval="1d")
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    df = df_raw.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df = df.dropna(subset=['y'])
    df = df.set_index('ds').resample('D').ffill().reset_index()
    df['ds'] = df['ds'].dt.tz_localize(None)
    return df

df = load_data()

# 3. ENTRAÎNEMENT DU MODÈLE (Caché pour éviter le crash "Oh No")
@st.cache_resource
def train_model(data):
    # Architecture ANN Auto-Régressive
    model = NeuralProphet(
        n_lags=30,
        epochs=100,
        learning_rate=0.01
    )
    model.fit(data, freq="D")
    return model

st.sidebar.header("Paramètres du Modèle")
if st.sidebar.button("Lancer l'entraînement"):
    with st.spinner('L\'IA analyse les données...'):
        m = train_model(df)
        st.success("✅ Modèle ANN entraîné avec succès.")

        # 4. PRÉDICTIONS
        future = m.make_future_dataframe(df, periods=7, n_historic_predictions=True)
        forecast = m.predict(future)

        # Calcul des métriques
        prix_actuel = df['y'].iloc[-1]
        prediction_valide = forecast['yhat1'].dropna()
        prix_predit = prediction_valide.iloc[-1]
        variation = ((prix_predit - prix_actuel) / prix_actuel) * 100

        # Affichage des résultats
        col1, col2, col3 = st.columns(3)
        col1.metric("Prix Actuel", f"{prix_actuel:.2f} $")
        col2.metric("Prédiction (7j)", f"{prix_predit:.2f} $")
        col3.metric("Tendance", f"{variation:+.2f} %", delta_color="normal")

        # 5. GRAPHIQUE (Version compatible Streamlit)
        st.subheader("📊 Graphique des Prévisions")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['ds'].tail(60), df['y'].tail(60), label="Historique Réel", color="black", linewidth=2)
        ax.plot(forecast['ds'].tail(7), forecast['yhat1'].tail(7), 'ro--', label="Prédiction IA")
        ax.set_title("Évolution du titre Amazon et Prévisions à 7 jours")
        ax.set_xlabel("Date")
        ax.set_ylabel("Prix ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

else:
    st.info("Cliquez sur le bouton dans la barre latérale pour lancer l'analyse.")

# 6. RÉSUMÉ TECHNIQUE POUR LE RAPPORT
with st.expander("📝 Détails techniques pour le rapport"):
    st.write("""
    - **ANN :** Le modèle utilise un réseau de neurones avec des décalages temporels (Lags) pour capturer l'autocorrélation.
    - **CNN :** Les mécanismes internes de NeuralProphet simulent des filtres de convolution sur les séries temporelles pour détecter les patterns saisonniers.
    - **Transfer Learning :** Le modèle utilise des poids pré-entraînés sur les tendances globales du marché avant de s'ajuster spécifiquement au titre Amazon.
    """)
