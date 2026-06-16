import streamlit as st
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt
from scipy.stats import poisson as scipy_poisson
import numpy as np

# ── Page Config ───────────────────────────────────────────
st.set_page_config(page_title="WM 2026 Predictor", page_icon="🏆", layout="centered")

# ── Daten laden ───────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("xgb_wm2026.pkl")

@st.cache_resource
def load_features():
    return joblib.load("features.pkl")

@st.cache_data
def load_team_stats():
    with open("team_stats.json") as f:
        return json.load(f)

@st.cache_data
def load_schedule():
    with open("wm2026_schedule.json") as f:
        return json.load(f)

@st.cache_resource
def load_poisson():
    ph = joblib.load("poisson_home.pkl")
    pa = joblib.load("poisson_away.pkl")
    fh = joblib.load("features_home.pkl")
    fa = joblib.load("features_away.pkl")
    return ph, pa, fh, fa

poisson_home, poisson_away, features_home, features_away = load_poisson()

model      = load_model()
features   = load_features()
team_stats = load_team_stats()
schedule   = load_schedule()

alle_teams = sorted(team_stats.keys())

# ── Predict Score Funktion ────────────────────────────────
def predict_score(home, away, elo_diff):
    x_h = pd.DataFrame([{
        "avg_goals_scored_home":   home["avg_goals_scored"],
        "avg_goals_conceded_away": away["avg_goals_conceded"],
        "elo_diff":                elo_diff,
        "home_boost":              home["home_boost"],
        "wc_exp_home":             home["wc_exp"]
    }])
    x_a = pd.DataFrame([{
        "avg_goals_scored_away":   away["avg_goals_scored"],
        "avg_goals_conceded_home": home["avg_goals_conceded"],
        "elo_diff":                elo_diff,
        "wc_exp_away":             away["wc_exp"]
    }])

    lh = poisson_home.predict(x_h)[0]
    la = poisson_away.predict(x_a)[0]

    matrix = np.outer(
        scipy_poisson.pmf(range(6), lh),
        scipy_poisson.pmf(range(6), la)
    )
    idx = np.unravel_index(matrix.argmax(), matrix.shape)
    return lh, la, idx[0], idx[1], matrix[idx]

# ── Header ────────────────────────────────────────────────
st.title("🏆 WM 2026 — Match Predictor")
st.markdown("Wähle zwei Teams und erhalte eine KI-basierte Spielprognose mit Erklärung.")
st.divider()

# ── Team Auswahl ──────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 Home Team", alle_teams, index=alle_teams.index("Germany"))
with col2:
    away_team = st.selectbox("✈️ Away Team", alle_teams, index=alle_teams.index("Brazil"))

predict_btn = st.button("⚽ Vorhersage berechnen", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────
if predict_btn:
    if home_team == away_team:
        st.warning("Bitte zwei verschiedene Teams wählen.")
        st.stop()

    home = team_stats[home_team]
    away = team_stats[away_team]

    X = pd.DataFrame([{
        "avg_goals_scored_home":   home["avg_goals_scored"],
        "avg_goals_conceded_home": home["avg_goals_conceded"],
        "avg_goals_scored_away":   away["avg_goals_scored"],
        "avg_goals_conceded_away": away["avg_goals_conceded"],
        "wc_exp_home":             home["wc_exp"],
        "wc_exp_away":             away["wc_exp"],
        "elo_diff":                home["elo"] - away["elo"],
        "home_boost":              home["home_boost"]
    }])

    probs = model.predict_proba(X)[0]

    # ── Ergebnis anzeigen ─────────────────────────────────
    st.divider()
    st.subheader(f"{home_team} vs. {away_team}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🔴 Loss", f"{probs[0]:.1%}")
        st.progress(float(probs[0]))
    with c2:
        st.metric("🟡 Draw", f"{probs[1]:.1%}")
        st.progress(float(probs[1]))
    with c3:
        st.metric("🟢 Win", f"{probs[2]:.1%}")
        st.progress(float(probs[2]))

    st.caption(f"ELO: {home_team} {home['elo']:.0f} — {away_team} {away['elo']:.0f}  |  Differenz: {home['elo'] - away['elo']:.0f}")

    # ── Torvorhersage ─────────────────────────────────────
    st.divider()
    st.subheader("⚽ Torvorhersage")

    elo_diff = home["elo"] - away["elo"]
    lh, la, g_h, g_a, prob = predict_score(home, away, elo_diff)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Erwartete Tore {home_team}", f"{lh:.2f}")
    with col2:
        st.metric(f"Erwartete Tore {away_team}", f"{la:.2f}")
    with col3:
        st.metric("Wahrscheinlichstes Ergebnis", f"{g_h}:{g_a}", f"{prob:.1%}")

    # ── SHAP Waterfall ────────────────────────────────────
    st.divider()
    st.subheader("🔍 Feature Importance (SHAP)")
    st.caption("Warum hat das Modell diese Vorhersage getroffen?")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0, :, 2], show=False)
    st.pyplot(fig)
    plt.close()