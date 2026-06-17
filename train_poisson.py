import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import PoissonRegressor

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
result_df = pd.read_csv(URL)
result_df["date"] = pd.to_datetime(result_df["date"])

wm_df = result_df[result_df["tournament"] == "FIFA World Cup"].copy()
wm_df.reset_index(drop=True, inplace=True)
wm_df = wm_df.sort_values("date").reset_index(drop=True)

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

def get_recent_form(df, team, before_date, n=5):
    home_games = df[(df["home_team"] == team) & (df["date"] < before_date)]
    away_games = df[(df["away_team"] == team) & (df["date"] < before_date)]
    home_pts = home_games.apply(
        lambda r: 3 if r["home_score"] > r["away_score"]
        else 1 if r["home_score"] == r["away_score"] else 0, axis=1)
    away_pts = away_games.apply(
        lambda r: 3 if r["away_score"] > r["home_score"]
        else 1 if r["away_score"] == r["home_score"] else 0, axis=1)
    alle_punkte = pd.concat([home_pts, away_pts]).sort_index().tail(n)
    if len(alle_punkte) == 0:
        return 1.5
    return alle_punkte.mean()

def get_wc_experience(df, team, before_date):
    home = df[(df["home_team"] == team) & (df["date"] < before_date)]
    away = df[(df["away_team"] == team) & (df["date"] < before_date)]
    return pd.concat([home, away])["date"].dt.year.nunique()

# Feature-Loops
avg_gs_home, avg_gc_home = [], []
avg_gs_away, avg_gc_away = [], []
form_home, form_away     = [], []

for _, row in wm_df.iterrows():
    gs_h, gc_h = get_team_stats(result_df, row["home_team"], row["date"])
    gs_a, gc_a = get_team_stats(result_df, row["away_team"], row["date"])
    avg_gs_home.append(gs_h); avg_gc_home.append(gc_h)
    avg_gs_away.append(gs_a); avg_gc_away.append(gc_a)
    form_home.append(get_recent_form(result_df, row["home_team"], row["date"]))
    form_away.append(get_recent_form(result_df, row["away_team"], row["date"]))

wm_df["avg_goals_scored_home"]   = avg_gs_home
wm_df["avg_goals_conceded_home"] = avg_gc_home
wm_df["avg_goals_scored_away"]   = avg_gs_away
wm_df["avg_goals_conceded_away"] = avg_gc_away
wm_df["recent_form_home"]        = form_home
wm_df["recent_form_away"]        = form_away

# ELO
K, DEFAULT_ELO = 32, 1500
elo_ratings = {}
elo_home_list, elo_away_list = [], []
alle_spiele = result_df.sort_values("date").reset_index(drop=True)

for _, row in alle_spiele.iterrows():
    home, away = row["home_team"], row["away_team"]
    r_home = elo_ratings.get(home, DEFAULT_ELO)
    r_away = elo_ratings.get(away, DEFAULT_ELO)
    e_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
    e_away = 1 - e_home
    if row["home_score"] > row["away_score"]:    s_home, s_away = 1.0, 0.0
    elif row["home_score"] == row["away_score"]: s_home, s_away = 0.5, 0.5
    else:                                        s_home, s_away = 0.0, 1.0
    elo_home_list.append(r_home); elo_away_list.append(r_away)
    elo_ratings[home] = r_home + K * (s_home - e_home)
    elo_ratings[away] = r_away + K * (s_away - e_away)

alle_spiele["elo_home"] = elo_home_list
alle_spiele["elo_away"] = elo_away_list

elo_merge = alle_spiele[["date", "home_team", "away_team", "elo_home", "elo_away"]]
wm_df = wm_df.merge(elo_merge, on=["date", "home_team", "away_team"], how="left")
wm_df["elo_diff"]   = wm_df["elo_home"] - wm_df["elo_away"]
wm_df["home_boost"] = wm_df["home_team"].isin(["United States", "Canada", "Mexico"]).astype(int)

# wc_exp
wm_df["wc_exp_home"] = wm_df.apply(
    lambda row: get_wc_experience(wm_df, row["home_team"], row["date"]), axis=1)
wm_df["wc_exp_away"] = wm_df.apply(
    lambda row: get_wc_experience(wm_df, row["away_team"], row["date"]), axis=1)

# Clean + nur ab 1990
wm_df_clean = wm_df.dropna(subset=["home_score", "away_score"])
wm_df_poisson = wm_df_clean[wm_df_clean["date"].dt.year >= 1990].copy()

features_home = ["avg_goals_scored_home", "avg_goals_conceded_away",
                 "elo_diff", "home_boost", "recent_form_home", "wc_exp_home"]
features_away = ["avg_goals_scored_away", "avg_goals_conceded_home",
                 "elo_diff", "recent_form_away", "wc_exp_away"]

poisson_home = PoissonRegressor(max_iter=1000)
poisson_home.fit(wm_df_poisson[features_home], wm_df_poisson["home_score"])

poisson_away = PoissonRegressor(max_iter=1000)
poisson_away.fit(wm_df_poisson[features_away], wm_df_poisson["away_score"])

joblib.dump(poisson_home,  "poisson_home.pkl")
joblib.dump(poisson_away,  "poisson_away.pkl")
joblib.dump(features_home, "features_home.pkl")
joblib.dump(features_away, "features_away.pkl")

print("Poisson Modelle gespeichert.")
print("Home MAE:", round(abs(poisson_home.predict(wm_df_poisson[features_home]) - wm_df_poisson["home_score"]).mean(), 3))
print("Away MAE:", round(abs(poisson_away.predict(wm_df_poisson[features_away]) - wm_df_poisson["away_score"]).mean(), 3))