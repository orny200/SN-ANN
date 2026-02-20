import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="IA Amazon Predictor - ANN", layout="wide")

# --- 2. SIDEBAR (Cohérence Master 2) ---
st.sidebar.title("🎓 Parcours Académique")
st.sidebar.info("""
**1. ANN & Séries Temporelles** : Ce projet (AR-Net).
**2. CNN** : Architecture personnalisée pour images.
**3. Transfer Learning** : Optimisation avec MobileNetV2.
""")

st.title("🚀 Prédiction Boursière Amazon (AMZN) avec ANN")
st.markdown("""
Ce devoir utilise un réseau de neurones **AR-Net (Auto-Regressive Neural Network)**. 
Contrairement aux CNN, ce réseau traite des séquences numériques pour capter des tendances cycliques.
""")

# --- 3. MÉTRIQUES DE PERFORMANCE DU DEVOIR ---
st.markdown("### 📊 Performances de l'ANN")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Modèle", "AR-Net")
with c2:
    st.metric("Epochs", "40")
with c3:
    st.metric("Loss Type", "Huber Loss")

st.divider()

# --- 4. CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    df_raw = yf.download("AMZN", period="2y", interval="1d")
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    df = df_raw.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    df = df.set_index('ds').resample('D').ffill().reset_index()
    return df

try:
    data = load_data()
    st.subheader("📈 Historique Réel (Données Yahoo Finance)")
    st.line_chart(data.set_index('ds')['y'])

    # --- 5. INTERFACE DE COMMANDE ---
    st.sidebar.header("Paramètres de l'IA")
    jours_pred = st.sidebar.slider("Jours à prédire", 1, 14, 7)
    
    if st.button("Lancer l'Analyse Prédictive"):
        with st.spinner("L'IA (NeuralProphet) calcule les poids du réseau..."):
            
            # Initialisation du réseau de neurones AR-Net
            m = NeuralProphet(
                n_lags=15, 
                epochs=40, 
                learning_rate=0.01
            )
            
            m.fit(data, freq="D")
            
            # Prédiction
            future = m.make_future_dataframe(data, periods=jours_pred, n_historic_predictions=True)
            forecast = m.predict(future)
            
            # Résultats
            prix_actuel = data['y'].iloc[-1]
            pred_series = forecast['yhat1'].dropna()
            prix_futur = pred_series.iloc[-1]
            variation = ((prix_futur - prix_actuel) / prix_actuel) * 100
            
            st.divider()
            res1, res2, res3 = st.columns(3)
            res1.metric("Prix Actuel", f"{prix_actuel:.2f} $")
            res2.metric(f"Prédiction à {jours_pred}j", f"{prix_futur:.2f} $")
            res3.metric("Variation Estimée", f"{variation:+.2f} %")
            
            # Graphique
            st.subheader("🎯 Projection du réseau de neurones")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(data['ds'].tail(60), data['y'].tail(60), label="Réel", color="#2c3e50")
            ax.plot(forecast['ds'].tail(jours_pred), forecast['yhat1'].tail(jours_pred), 
                    'ro--', label="Prédiction ANN")
            ax.legend()
            st.pyplot(fig)

except Exception as e:
    st.error(f"Erreur : {e}")

st.markdown("---")
st.caption("M2 IABD - Travail Pratique sur les ANN et Séries Temporelles")