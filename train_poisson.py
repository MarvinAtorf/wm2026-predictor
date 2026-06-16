import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import PoissonRegressor

# Daten laden
URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
result_df = pd.read_csv(URL)
result_df["date"] = pd.to_datetime(result_df["date"])

wm_df = result_df[result_df["tournament"] == "FIFA World Cup"].copy()
wm_df.reset_index(drop=True, inplace=True)
wm_df.drop(columns=["index"], errors="ignore", inplace=True)
wm_df = wm_df.sort_values("date").reset_index(drop=True)

# Features berechnen (gleiche Logik wie Colab)
fallback_scored   = wm_df["home_score"].mean()
fallback_conceded = wm_df["away_score"].mean()

def get_team_stats(df, team, before_date, n=10):
    home_games = df[(df["home_team"] == team) & (df["date"] < before_date)]
    away_games = df[(df["away_team"] == team) & (df["date"] < before_date)]
    scored   = pd.concat([home_games["home_score"], away_games["away_score"]]).tail(n)
    conceded = pd.concat([home_games["away_score"], away_games["home_score"]]).tail(n)
    if len(scored) == 0:
        return fallback_scored, fallback_conceded
    return scored.mean(), conceded.mean()

avg_gs_home, avg_gc_home, avg_gs_away, avg_gc_away = [], [], [], []
for _, row in wm_df.iterrows():
    gs_h, gc_h = get_team_stats(wm_df, row["home_team"], row["date"])
    gs_a, gc_a = get_team_stats(wm_df, row["away_team"], row["date"])
    avg_gs_home.append(gs_h); avg_gc_home.append(gc_h)
    avg_gs_away.append(gs_a); avg_gc_away.append(gc_a)

wm_df["avg_goals_scored_home"]   = avg_gs_home
wm_df["avg_goals_conceded_home"] = avg_gc_home
wm_df["avg_goals_scored_away"]   = avg_gs_away
wm_df["avg_goals_conceded_away"] = avg_gc_away

# ELO berechnen
K, DEFAULT_ELO = 32, 1500
elo_ratings = {}
elo_home_list, elo_away_list = [], []
alle_spiele = result_df.sort_values("date").reset_index(drop=True)

for _, row in alle_spiele.iterrows():
    home, away = row["home_team"], row["away_team"]
    r_home = elo_ratings.get(home, DEFAULT_ELO)
    r_away = elo_ratings.get(away, DEFAULT_ELO)
    e_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
    if row["home_score"] > row["away_score"]:   s_home, s_away = 1.0, 0.0
    elif row["home_score"] == row["away_score"]: s_home, s_away = 0.5, 0.5
    else:                                        s_home, s_away = 0.0, 1.0
    elo_home_list.append(r_home); elo_away_list.append(r_away)
    elo_ratings[home] = r_home + K * (s_home - e_home)
    elo_ratings[away] = r_away + K * (1 - s_home - (1 - e_home))

alle_spiele["date"]     = pd.to_datetime(alle_spiele["date"])
alle_spiele["elo_home"] = elo_home_list
alle_spiele["elo_away"] = elo_away_list
elo_merge = alle_spiele[["date", "home_team", "away_team", "elo_home", "elo_away"]]
wm_df = wm_df.merge(elo_merge, on=["date", "home_team", "away_team"], how="left")
wm_df["elo_diff"]   = wm_df["elo_home"] - wm_df["elo_away"]
wm_df["home_boost"] = wm_df["home_team"].isin(["United States", "Canada", "Mexico"]).astype(int)

# Clean
wm_df_clean = wm_df.dropna(subset=["home_score", "away_score"])

# Poisson Training
features_home = ["avg_goals_scored_home", "avg_goals_conceded_away", "elo_diff", "home_boost", "wc_exp_home"]
features_away = ["avg_goals_scored_away", "avg_goals_conceded_home", "elo_diff", "wc_exp_away"]

# wc_exp berechnen
def get_wc_exp(df, team, before_date):
    h = df[(df["home_team"] == team) & (df["date"] < before_date)]
    a = df[(df["away_team"] == team) & (df["date"] < before_date)]
    return pd.concat([h, a])["date"].dt.year.nunique()

wm_df_clean = wm_df_clean.copy()
wm_df_clean["wc_exp_home"] = wm_df_clean.apply(lambda r: get_wc_exp(wm_df, r["home_team"], r["date"]), axis=1)
wm_df_clean["wc_exp_away"] = wm_df_clean.apply(lambda r: get_wc_exp(wm_df, r["away_team"], r["date"]), axis=1)

X_home = wm_df_clean[features_home]
X_away = wm_df_clean[features_away]
y_home = wm_df_clean["home_score"]
y_away = wm_df_clean["away_score"]

poisson_home = PoissonRegressor(max_iter=1000)
poisson_home.fit(X_home, y_home)

poisson_away = PoissonRegressor(max_iter=1000)
poisson_away.fit(X_away, y_away)

# Speichern
joblib.dump(poisson_home,  "poisson_home.pkl")
joblib.dump(poisson_away,  "poisson_away.pkl")
joblib.dump(features_home, "features_home.pkl")
joblib.dump(features_away, "features_away.pkl")

print("✓ Poisson Modelle lokal gespeichert.")