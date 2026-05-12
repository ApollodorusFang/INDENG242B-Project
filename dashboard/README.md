# CryptoLOB — interactive dashboard

Live companion to our INDENG 242B final report. Lets anyone inspect our
test-set results and **interactively re-run the cost-aware backtest** on
our six model families.

## Pages

1. **🏠 Overview** — hero + abstract + headline numbers.
2. **📊 Dataset** — feature schema, mid-price trajectory, label distribution.
3. **🤖 Model Performance** — test-set regression table + predicted-vs-realized scatter.
4. **🧪 Backtest Sandbox** — *the meaningful-interaction page*.
   Sliders for signal threshold τ and per-turnover cost (bps); the dashboard
   recomputes NAV, Sharpe, MaxDD, and per-bet win rate for every selected
   model in real time. This is the live version of Table 3 in the report.
5. **ℹ️ About** — team, methodology, repo link, AI disclosure.

## Run locally

```bash
pip install -r requirements.txt

# (Optional) regenerate the consolidated CSV the dashboard reads.
# Only needed if you re-ran the replication pipeline.
python -m replication.export_dashboard_data

streamlit run streamlit_app.py
```

Open <http://localhost:8501>.

## Data the dashboard reads

| Path                                          | What it provides                                |
| --------------------------------------------- | ----------------------------------------------- |
| `data/processed/dataset_metadata.json`        | Split sizes, lookback, horizon                  |
| `replication/results/_traditional_metrics.csv`| RMSE / MAE / R² / DirAcc / AUC table            |
| `dashboard/data/test_predictions.csv`         | Consolidated test-set predictions (preferred)   |
| `replication/results/{family}_predictions.csv`| Per-family fallback if the consolidated CSV is missing |

All four are optional; the dashboard falls back to the numbers from the
report and a synthetic demo panel if nothing is committed.

## Deployment (Streamlit Community Cloud)

1. Push this branch to GitHub.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click
   **New app**.
3. Repository: `ApollodorusFang/INDENG242B-Project` · Branch: `main` ·
   Main file path: `streamlit_app.py`.
4. Click **Deploy**. First build takes ~2 minutes; subsequent pushes
   redeploy automatically.

The free tier (1 GB RAM, sleeps after inactivity) is plenty for this app —
it only reads small CSVs and runs the backtest in NumPy.
