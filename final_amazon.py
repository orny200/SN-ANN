import torch

# Cette ligne règle ton erreur de "UnpicklingError"
torch.serialization.add_safe_globals([
    'neuralprophet.configure.ConfigSeasonality',
    'neuralprophet.configure.ConfigTrend',
    'neuralprophet.configure.ConfigModel',
    'neuralprophet.configure.ConfigTrain',
    'neuralprophet.configure.ConfigData'
])
import os
import sys
import warnings

# 1. CONFIGURATION
warnings.filterwarnings("ignore")
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

print("--- 🚀 LANCEMENT DU PROJET IA : AMAZON ---")

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from neuralprophet import NeuralProphet
    import matplotlib.pyplot as plt
    print("✅ Bibliothèques chargées.")
except ImportError:
    print("❌ Erreur : Bibliothèques manquantes.")
    sys.exit()

# 2. DONNÉES
print("\n--- 1. RÉCUPÉRATION DES DONNÉES ---")
df_raw = yf.download("AMZN", period="2y", interval="1d")

if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)

df = df_raw.reset_index()[['Date', 'Close']]
df.columns = ['ds', 'y']

# Nettoyage anti-NaN (combler les week-ends)
df = df.dropna(subset=['y'])
df = df.set_index('ds').resample('D').ffill().reset_index()
df['ds'] = df['ds'].dt.tz_localize(None)

print(f"✅ Données prêtes : {len(df)} jours.")

# 3. MODÈLE (VERSION ÉPURÉE SANS ERREUR)
print("\n--- 2. ENTRAÎNEMENT DE L'IA ---")
# Ici, on n'utilise que les arguments de base reconnus par toutes les versions
m = NeuralProphet(
    n_lags=30,
    epochs=100
)

# Entraînement
m.fit(df, freq="D")
print("✅ Entraînement terminé.")

# 4. PRÉDICTION
print("\n--- 3. CALCUL DES PRÉDICTIONS ---")
future = m.make_future_dataframe(df, periods=7, n_historic_predictions=True)
forecast = m.predict(future)

prix_actuel = df['y'].iloc[-1]
# On récupère la dernière prédiction non vide
prediction_valide = forecast['yhat1'].dropna()
prix_predit = prediction_valide.iloc[-1]
variation = ((prix_predit - prix_actuel) / prix_actuel) * 100

print(f"📈 Prix actuel : {prix_actuel:.2f} $")
print(f"🔮 Prédiction à 7 jours : {prix_predit:.2f} $")
print(f"📊 Tendance : {variation:+.2f} %")

# 5. GRAPHIQUE
print("\n--- 4. GÉNÉRATION DU GRAPHIQUE ---")
plt.figure(figsize=(12, 6))
plt.plot(df['ds'].tail(60), df['y'].tail(60), label="Réel", color="black")
plt.plot(forecast['ds'].tail(7), forecast['yhat1'].tail(7), 'ro--', label="Prédiction")
plt.title(f"Amazon (AMZN) - Prédiction : {prix_predit:.2f}$")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()