import streamlit as st

# ... (garder tout ton début de script jusqu'à la création du dataframe df) ...

st.title("🚀 Prédiction Amazon - Deep Learning")

# On utilise le cache pour éviter de ré-entraîner à chaque clic
@st.cache_resource
def entrainer_modele(data):
    model = NeuralProphet(n_lags=30, epochs=100)
    model.fit(data, freq="D")
    return model

with st.spinner('Entraînement de l\'IA en cours...'):
    m = entrainer_modele(df)
    st.success("✅ Modèle entraîné !")

# --- PRÉDICTIONS ---
future = m.make_future_dataframe(df, periods=7, n_historic_predictions=True)
forecast = m.predict(future)

# --- AFFICHAGE DES MÉTRIQUES ---
col1, col2, col3 = st.columns(3)
prix_actuel = df['y'].iloc[-1]
prix_predit = forecast['yhat1'].iloc[-1]
variation = ((prix_predit - prix_actuel) / prix_actuel) * 100

col1.metric("Prix Actuel", f"{prix_actuel:.2f} $")
col2.metric("Prédiction (7j)", f"{prix_predit:.2f} $")
col3.metric("Tendance", f"{variation:+.2f} %")

# --- GRAPHIQUE STREAMLIT ---
st.subheader("📊 Visualisation des résultats")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['ds'].tail(60), df['y'].tail(60), label="Réel", color="black")
ax.plot(forecast['ds'].tail(7), forecast['yhat1'].tail(7), 'ro--', label="Prédiction")
ax.legend()
st.pyplot(fig) # <-- Indispensable sur Streamlit !
