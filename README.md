# WM 2026 Match Predictor

A Streamlit app that predicts FIFA World Cup 2026 matches. Pick two teams and get win/draw/loss probabilities, expected goals, the most likely scorelines, and a SHAP explanation of why the model predicted what it did.

<img width="1035" height="1612" alt="screenshot" src="https://github.com/user-attachments/assets/dd747c8f-786c-497d-922c-5c11a51e909d" />


## What it does

- **Outcome probabilities:** an XGBoost classifier predicts loss, draw and win for the home team.
- **Expected goals:** two Poisson regression models estimate the goals of each team.
- **Likely scorelines:** 10,000 Monte Carlo draws from the two Poisson distributions give the three most probable results.
- **Explainability:** a SHAP waterfall plot shows which features pushed the prediction up or down.

## How it works

Training data are historical results from the [international results dataset](https://github.com/martj42/international_results). Only FIFA World Cup matches are used as training samples; Elo ratings are computed over all international matches.

| Feature | Description |
|---|---|
| `elo_diff` | Elo difference between the teams (K=32, start rating 1500) |
| `avg_goals_scored` / `avg_goals_conceded` | Average over each team's last 10 matches |
| `recent_form` | Average points over the last 5 matches |
| `wc_exp` | Number of different years a team has played at a World Cup |
| `home_boost` | 1 for the host nations (USA, Canada, Mexico) |

All features are calculated strictly from matches played before the match date, so no future information leaks into the training data. The Poisson models are trained on World Cup matches from 1990 onwards (`train_poisson.py`).

## Project structure

```
app.py                  Streamlit app
train_poisson.py        Feature engineering and Poisson model training
team_stats.json         Current team statistics used for predictions
wm2026_schedule.json    World Cup 2026 schedule
*.pkl                   Trained models and feature lists
tests/                  Tests
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To retrain the Poisson models:

```bash
python train_poisson.py
```

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, XGBoost, SHAP, SciPy, Matplotlib

## Limitations

- **Probabilities, not certainties.** Football is highly random, and the app is a modelling exercise, not betting advice.
- **Two independent models.** The XGBoost classifier and the Poisson regressions are trained separately, so they can disagree. In the example above (Germany vs. Brazil), the classifier gives Germany a low win probability, while the Poisson models expect slightly more goals for Germany.
- **World Cup experience carries a lot of weight.** `wc_exp` acts as a proxy for historical strength, and in SHAP plots it can outweigh the Elo difference. Its influence has not been validated yet.
- **Home/away orientation.** The models were trained with home and away roles taken from the dataset. At a neutral venue, swapping the teams may change the prediction.
- **Small feature set.** Line-ups, injuries, tournament stage and travel are not modelled.
